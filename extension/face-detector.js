/**
 * Face detection — MediaPipe Tasks Vision (BlazeFace, short-range).
 *
 * The actual MediaPipe model runs inside an offscreen document
 * (offscreen.html / offscreen.js), not here. Reason: MediaPipe's WASM
 * loader injects a <script> tag into document.body to load its runtime
 * (since importScripts() isn't available outside a Worker). Content
 * scripts share the page's DOM but execute in an isolated JS world
 * separate from that injected script's world, so the loader's
 * self.ModuleFactory side-effect is invisible here and MediaPipe throws
 * "ModuleFactory not set." An offscreen document is a normal, single-world
 * extension page, so the same loading mechanism works correctly there.
 *
 * This file just captures the frame as a data URL and forwards it to
 * background.js, which relays it to the offscreen document and returns
 * the result.
 *
 * Coordinates returned are normalized (0 -> 1) relative to the input
 * canvas, largest face first — same contract the original mock detector
 * used, so nothing downstream (content.js, overlay.js) needs to change.
 */

async function detectFaces(canvas) {

    if (!canvas) {
        return [];
    }

    console.log(
        "[FaceDetector] Detecting face..."
    );

    const imageDataUrl = canvas.toDataURL("image/jpeg", 0.85);

    let response;
    try {
        response = await chrome.runtime.sendMessage({
            type: "DETECT_FACES",
            imageDataUrl,
            width: canvas.width,
            height: canvas.height,
        });
    } catch (err) {
        console.error("[FaceDetector] message to background failed:", err);
        return [];
    }

    if (!response || response.error) {
        console.error("[FaceDetector] detection failed:", response && response.error);
        return [];
    }

    return response.faces || [];
}
