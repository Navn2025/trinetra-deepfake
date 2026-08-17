/**
 * Service worker.
 *
 * Its only job: keep a single offscreen document alive (offscreen.html) and
 * relay face-detection requests to it. MediaPipe's WASM runtime has to run
 * inside that offscreen document rather than in the content script -- see
 * the comment at the top of face-detector.js for why.
 */

async function ensureOffscreenDocument() {
    try {
        await chrome.offscreen.createDocument({
            url: "offscreen.html",
            reasons: ["WORKERS"],
            justification:
                "Runs the MediaPipe face-detection WASM model, which needs a full window/DOM context not available to content scripts or the background service worker.",
        });
    } catch (err) {
        if (!String(err).includes("single offscreen document")) {
            throw err;
        }
        // an offscreen document already exists -- fine.
    }
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
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
