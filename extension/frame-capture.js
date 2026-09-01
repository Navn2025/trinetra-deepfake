function captureFrame(video)
{
    if (!(video instanceof HTMLVideoElement))
    {
        console.error('[FrameCapture] Invalid video element:', video);
        return null;
    }

    if (
        video.readyState<HTMLVideoElement.HAVE_CURRENT_DATA)
    {
        console.warn('[FrameCapture] Video not ready for frame capture:', video);
        return null;
    }
    const width = video.videoWidth;
    const height = video.videoHeight;

    if (width === 0 || height === 0) {
        console.warn(
            "[FrameCapture] Invalid video dimensions"
        );
        return null;
    }
    const canvas = document.createElement("canvas");

    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext("2d");

    if (!ctx) {
        console.error(
            "[FrameCapture] Could not create canvas context"
        );
        return null;
    }
    ctx.translate(width, 0);
    ctx.scale(-1, 1);
     ctx.drawImage(
        video,
        0,
        0,
        width,
        height
    );

    return canvas;
        
    
}
function canvasToBlob(
    canvas,
    quality = 0.85
) {
    return new Promise((resolve) => {

        canvas.toBlob(
            (blob) => {
                resolve(blob);
            },
            "image/jpeg",
            quality
        );

    });
}
async function captureFrameBlob(video) {

    const canvas = captureFrame(video);

    if (!canvas) {
        return null;
    }

    return await canvasToBlob(canvas);
}

/**
 * Crops a square, margin-padded region around a detected face out of
 * `sourceCanvas` and returns it as a new canvas resized to `outputSize`.
 *
 * `face` uses the normalized (0 -> 1) box format returned by detectFaces()
 * in face-detector.js. The square-crop-with-margin shape mirrors
 * DeepfakeBench's own preprocessing (preprocess.py: extract_aligned_face_dlib,
 * default res=256) so a model trained on that pipeline sees a similar input
 * shape here.
 */
function cropFaceFromCanvas(sourceCanvas, face, outputSize = 256, marginRatio = 0.3) {

    if (!sourceCanvas || !face) {
        return null;
    }

    const sw = sourceCanvas.width;
    const sh = sourceCanvas.height;

    const boxW = face.width * sw;
    const boxH = face.height * sh;
    const cx = (face.x + face.width / 2) * sw;
    const cy = (face.y + face.height / 2) * sh;
    const side = Math.max(boxW, boxH) * (1 + marginRatio * 2);

    let sx = Math.round(cx - side / 2);
    let sy = Math.round(cy - side / 2);
    let s = Math.round(side);

    sx = Math.max(0, Math.min(sx, sw - 1));
    sy = Math.max(0, Math.min(sy, sh - 1));
    s = Math.min(s, sw - sx, sh - sy);

    if (s <= 0) {
        return null;
    }

    const cropCanvas = document.createElement("canvas");
    cropCanvas.width = outputSize;
    cropCanvas.height = outputSize;

    const ctx = cropCanvas.getContext("2d");
    ctx.drawImage(sourceCanvas, sx, sy, s, s, 0, 0, outputSize, outputSize);

    return cropCanvas;
}