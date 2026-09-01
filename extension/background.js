/**
 * Service worker.
 *
 * Two jobs:
 *  1. Keep a single offscreen document alive (offscreen.html) and relay
 *     face-detection requests to it. MediaPipe's WASM runtime has to run
 *     inside that offscreen document rather than in the content script --
 *     see the comment at the top of face-detector.js for why.
 *  2. On MEETING_STARTED (sent once by content.js when the first video
 *     tile appears), start tab audio capture and relay periodic voice
 *     check results back to the tab that asked for them. getUserMedia on a
 *     tab stream has to run in the offscreen document too (a service
 *     worker has no media pipeline of its own), so this is just
 *     start/stop plumbing -- the actual capture loop lives in offscreen.js.
 */

async function ensureOffscreenDocument() {
    try {
        await chrome.offscreen.createDocument({
            url: "offscreen.html",
            reasons: ["WORKERS", "USER_MEDIA"],
            justification:
                "Runs the MediaPipe face-detection WASM model and captures/analyzes tab audio for voice deepfake detection, both of which need a full window/DOM context not available to content scripts or the background service worker.",
        });
    } catch (err) {
        if (!String(err).includes("single offscreen document")) {
            throw err;
        }
        // an offscreen document already exists -- fine.
    }
}

// Which tab's audio is currently being captured, so voice check results
// coming back from the offscreen document (which has no notion of "tabs")
// can be routed to the right content script.
let voiceCaptureTabId = null;

async function startVoiceCapture(tabId) {
    if (voiceCaptureTabId === tabId) {
        return; // already capturing this tab
    }

    const streamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: tabId });
    await ensureOffscreenDocument();
    voiceCaptureTabId = tabId;

    await chrome.runtime.sendMessage({
        type: "OFFSCREEN_START_VOICE_CAPTURE",
        streamId,
    });
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "MEETING_STARTED") {
        const tabId = sender.tab?.id;
        if (tabId != null) {
            startVoiceCapture(tabId).catch((err) => {
                console.error("[Background] failed to start voice capture:", err);
            });
        }
        return false;
    }

    if (message.type === "VOICE_CHECK_RESULT") {
        // Relayed from offscreen.js -- forward to whichever tab started
        // capture, matching content.js's listener for this same message type.
        if (voiceCaptureTabId != null) {
            chrome.tabs.sendMessage(voiceCaptureTabId, message).catch(() => {
                // tab navigated away/closed -- nothing to relay to anymore.
            });
        }
        return false;
    }

    if (message.type !== "DETECT_FACES") {
        return false;
    }

    (async () => {
        try {
            await ensureOffscreenDocument();

            const response = await chrome.runtime.sendMessage({
                type: "OFFSCREEN_DETECT_FACES",
                imageDataUrl: message.imageDataUrl,
                width: message.width,
                height: message.height,
            });

            sendResponse(response);
        } catch (err) {
            sendResponse({ error: String(err) });
        }
    })();

    return true; // keep the message channel open for the async response
});
