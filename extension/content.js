console.log(
    "========================================"
);

console.log(
    " Meet Deepfake Detector"
);

console.log(
    " Extension initialized"
);

console.log(
    "========================================"
);


const PROCESS_INTERVAL = 3000;

let processing = false;

// Fires once per page load, the first time any video tile is seen -- used
// as the "meeting started" signal for automatic voice capture (see
// background.js/offscreen.js). Video tiles are a reasonable proxy across
// every supported platform (Meet/Zoom/Teams/Discord/WhatsApp) without
// needing per-platform "call connected" selectors.
let meetingStartSignaled = false;


async function processVideo(video, index) {

    console.log(
        `[Pipeline] Processing video ${index}`
    );


    // --------------------------------
    // 1. Capture frame
    // --------------------------------

    const frameCanvas =
        captureFrame(video);

    if (!frameCanvas) {

        console.log(
            `[Pipeline] Could not capture video ${index}`
        );

        return;
    }


    // --------------------------------
    // 2. Detect face
    // --------------------------------

    const faces =
        await detectFaces(frameCanvas);

    console.log(
        `[Pipeline] Faces detected: ${faces.length}`
    );


    if (faces.length === 0) {

        console.log(
            `[Pipeline] No face found in video ${index}`
        );

        updateFaceBoxes(video, []);

        return;
    }

    // Position every face's box this cycle, independent of whether
    // prediction below succeeds -- updateOverlay() only recolors them once
    // verdicts come back.
    updateFaceBoxes(video, faces);


    // --------------------------------
    // 3. Crop + predict every detected face, concurrently
    // --------------------------------

    const predictions =
        await Promise.all(
            faces.map(async (face, faceIndex) => {

                const faceCanvas =
                    cropFaceFromCanvas(frameCanvas, face);

                if (!faceCanvas) {

                    console.log(
                        `[Pipeline] Could not crop face ${faceIndex} for video ${index}`
                    );

                    return null;
                }

                const faceBlob =
                    await canvasToBlob(faceCanvas);

                return await predictFace(faceBlob);
            })
        );

    console.log(
        `[Pipeline] Predictions for video ${index}:`,
        predictions
    );


    // --------------------------------
    // 4. Update UI
    // --------------------------------

    updateOverlay(
        video,
        faces,
        predictions
    );
}


async function processAllVideos() {

    if (processing) {
        return;
    }

    processing = true;

    try {

        const videos =
            getMeetVideos();

        console.log(
            `[Pipeline] Found ${videos.length} active videos`
        );

        if (!meetingStartSignaled && videos.length > 0) {
            meetingStartSignaled = true;
            // Fire-and-forget -- background.js starts tab audio capture;
            // any failure (no mic/tab-capture permission granted yet, etc.)
            // shows up in its own console, not here.
            chrome.runtime.sendMessage({ type: "MEETING_STARTED" }).catch(() => {});
        }

        // Process every tile concurrently instead of one at a time --
        // each tile's own latency (detect -> crop -> predict) no longer
        // stacks up across tiles, so total time per cycle drops from the
        // sum of all tiles' latencies to roughly the slowest one.
        const results =
            await Promise.allSettled(
                videos.map((video, i) => processVideo(video, i))
            );

        results.forEach((result, i) => {
            if (result.status === "rejected") {
                console.error(
                    `[Pipeline] Unexpected error on video ${i}:`,
                    result.reason
                );
            }
        });

    } catch (error) {

        console.error(
            "[Pipeline] Unexpected error:",
            error
        );

    } finally {

        processing = false;
    }
}


setInterval(
    processAllVideos,
    PROCESS_INTERVAL
);


// Voice check results relayed from background.js -- they originate in
// offscreen.js's tab-audio capture loop, started once via MEETING_STARTED
// above. Whole-tab audio, so this updates one page-level banner rather than
// a per-tile badge (see overlay.js's updateVoiceBanner).
chrome.runtime.onMessage.addListener((message) => {
    if (message.type === "VOICE_CHECK_RESULT") {
        updateVoiceBanner(message.result, message.error);
    }
});