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

        return;
    }


    // --------------------------------
    // 3. Crop to the largest detected face
    // --------------------------------

    const faceCanvas =
        cropFaceFromCanvas(frameCanvas, faces[0]);

    if (!faceCanvas) {

        console.log(
            `[Pipeline] Could not crop face for video ${index}`
        );

        return;
    }

    const faceBlob =
        await canvasToBlob(faceCanvas);

    console.log(
        `[Pipeline] Face crop captured: ${faceBlob.size} bytes`
    );


    // --------------------------------
    // 4. Predict
    // --------------------------------

    const prediction =
        await predictFace(faceBlob);


    if (!prediction) {

        console.log(
            `[Pipeline] No prediction for video ${index}`
        );

        return;
    }


    console.log(
        `[Pipeline] Prediction for video ${index}:`,
        prediction
    );


    // --------------------------------
    // 5. Update UI
    // --------------------------------

    updateOverlay(
        video,
        prediction
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