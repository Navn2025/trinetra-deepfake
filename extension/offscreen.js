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
    if (message.type === "OFFSCREEN_START_VOICE_CAPTURE") {
        startVoiceCapture(message.streamId).catch((err) => {
            console.error("[Offscreen] failed to start voice capture:", err);
        });
        return false;
    }

    if (message.type !== "OFFSCREEN_DETECT_FACES") {
        return false;
    }

    detectFacesFromDataUrl(message.imageDataUrl, message.width, message.height)
        .then((faces) => sendResponse({ faces }))
        .catch((err) => sendResponse({ error: String(err) }));

    return true; // async response
});


/**
 * Tab audio capture for voice deepfake detection.
 *
 * Continuously collects VOICE_CHUNK_SECONDS of audio and POSTs each chunk
 * to the voice service (voice-integrity/src/server.py, port 8001) as a WAV
 * file as soon as it's full, then immediately starts collecting the next
 * one -- back-to-back short checks instead of a long chunk on a slow fixed
 * interval, so a verdict comes back roughly every VOICE_CHUNK_SECONDS.
 * WAV rather than MediaRecorder's default webm/opus output deliberately --
 * the server reads audio with `soundfile` (libsndfile), which can't decode
 * Opus, so this builds a PCM WAV by hand from raw samples instead of
 * transcoding server-side.
 *
 * Runs continuously once started (no stop message exists yet -- this is a
 * v1: capture starts on MEETING_STARTED and stops when the tab/offscreen
 * document closes).
 */
const VOICE_API_URL = "http://127.0.0.1:8001";
const VOICE_CHUNK_SECONDS = 15;

let voiceCaptureState = null;

function mergeFloat32(chunks, totalLength) {
    const merged = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
        merged.set(chunk, offset);
        offset += chunk.length;
    }
    return merged;
}

function encodeWav(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    function writeString(offset, str) {
        for (let i = 0; i < str.length; i++) {
            view.setUint8(offset + i, str.charCodeAt(i));
        }
    }

    writeString(0, "RIFF");
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    view.setUint32(16, 16, true); // fmt chunk size
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); // byte rate (mono, 16-bit)
    view.setUint16(32, 2, true); // block align
    view.setUint16(34, 16, true); // bits per sample
    writeString(36, "data");
    view.setUint32(40, samples.length * 2, true);

    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }

    return new Blob([buffer], { type: "audio/wav" });
}

async function sendVoiceChunk(samples, sampleRate) {
    const blob = encodeWav(samples, sampleRate);

    try {
        const formData = new FormData();
        formData.append("file", blob, "chunk.wav");

        const response = await fetch(`${VOICE_API_URL}/voice/check`, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`voice API returned ${response.status}`);
        }

        const result = await response.json();
        chrome.runtime.sendMessage({ type: "VOICE_CHECK_RESULT", result });
    } catch (err) {
        console.error("[Offscreen] voice check failed:", err);
        chrome.runtime.sendMessage({ type: "VOICE_CHECK_RESULT", error: String(err) });
    }
}

async function startVoiceCapture(streamId) {
    if (voiceCaptureState) {
        return; // already running
    }

    const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
            mandatory: {
                chromeMediaSource: "tab",
                chromeMediaSourceId: streamId,
            },
        },
    });

    const audioContext = new AudioContext();
    const source = audioContext.createMediaStreamSource(stream);

    // getUserMedia on a tab stream silences the tab's normal audio output
    // by default -- route it back to the speakers so capturing doesn't mute
    // the call for the local user.
    source.connect(audioContext.destination);

    // ScriptProcessorNode is deprecated in favor of AudioWorklet, but it
    // doesn't need a separate module file to load, and this only runs in a
    // background offscreen document the user never sees or interacts with
    // -- the deprecation's main-thread-jank concern doesn't apply here.
    const bufferSize = 4096;
    const processor = audioContext.createScriptProcessor(bufferSize, 1, 1);

    let collected = [];
    let collectedSamples = 0;
    const samplesPerChunk = audioContext.sampleRate * VOICE_CHUNK_SECONDS;

    processor.onaudioprocess = (event) => {
        const input = event.inputBuffer.getChannelData(0);
        collected.push(new Float32Array(input));
        collectedSamples += input.length;

        if (collectedSamples >= samplesPerChunk) {
            const merged = mergeFloat32(collected, collectedSamples);
            sendVoiceChunk(merged, audioContext.sampleRate); // fire-and-forget
            // Next window starts immediately -- capture never pauses to
            // wait on the previous chunk's network round trip.
            collected = [];
            collectedSamples = 0;
        }
    };

    source.connect(processor);
    // Silent output (onaudioprocess never writes to event.outputBuffer) --
    // this connection just keeps the node alive, Chrome can stop calling
    // onaudioprocess for nodes with no downstream destination.
    processor.connect(audioContext.destination);

    voiceCaptureState = { audioContext, processor, source, stream };
}
