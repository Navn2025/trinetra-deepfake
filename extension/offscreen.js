/**
 * Runs inside the offscreen document (offscreen.html). This is a normal,
 * single-world extension page -- unlike a content script, it has no
 * isolated/main-world split, so MediaPipe's WASM loader (which injects a
 * <script> tag and expects to set a global afterwards) works correctly here.
 *
 * Receives frames from background.js (relayed from face-detector.js in the
 * content script), runs MediaPipe FaceDetector on them, and responds with
 * normalized face boxes.
 */

let faceDetectorPromise = null;

async function getFaceDetector() {
    if (!faceDetectorPromise) {
        faceDetectorPromise = (async () => {
            const { FaceDetector, FilesetResolver } = await import(
                chrome.runtime.getURL("lib/mediapipe/vision_bundle.mjs")
            );
            const filesetResolver = await FilesetResolver.forVisionTasks(
                chrome.runtime.getURL("lib/mediapipe/wasm")
            );
            return FaceDetector.createFromOptions(filesetResolver, {
                baseOptions: {
                    modelAssetPath: chrome.runtime.getURL(
                        "models/blaze_face_short_range.tflite"
                    ),
                    // 'GPU' is faster but not guaranteed available everywhere --
                    // switch to 'CPU' here if you see delegate-init errors.
                    delegate: "GPU",
                },
                runningMode: "IMAGE",
                minDetectionConfidence: 0.5,
            });
        })();
    }
    return faceDetectorPromise;
}

async function detectFacesFromDataUrl(imageDataUrl, width, height) {
    const detector = await getFaceDetector();

    const img = new Image();
    await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
        img.src = imageDataUrl;
    });

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    canvas.getContext("2d").drawImage(img, 0, 0, width, height);

    const result = detector.detect(canvas);
    if (!result.detections || result.detections.length === 0) {
        return [];
    }

    return result.detections
        .map((d) => ({
            x: d.boundingBox.originX / width,
            y: d.boundingBox.originY / height,
            width: d.boundingBox.width / width,
            height: d.boundingBox.height / height,
            confidence: d.categories?.[0]?.score ?? 0,
        }))
        .sort((a, b) => b.width * b.height - a.width * a.height);
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message.type !== "OFFSCREEN_DETECT_FACES") {
        return false;
    }

    detectFacesFromDataUrl(message.imageDataUrl, message.width, message.height)
        .then((faces) => sendResponse({ faces }))
        .catch((err) => sendResponse({ error: String(err) }));

    return true; // async response
});
