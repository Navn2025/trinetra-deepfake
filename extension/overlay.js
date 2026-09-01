const BADGE_CLASS =
    "deepfake-detector-badge";

const FACE_BOX_CLASS = "deepfake-detector-facebox";

// video -> {box, badge}[], one pair per detected face, indexed by that
// face's position in detectFaces()'s (largest-first) result array. Reused
// across cycles rather than recreated -- detectFaces() has no cross-frame
// identity tracking (unlike the desktop app's FaceTracker), so index 0 is
// just "the biggest face this cycle," not a stable per-person id. Good
// enough for a lightweight per-tile overlay; a face count that shrinks
// just hides the extra pairs rather than destroying them, so they're ready
// to reappear on the next cycle without recreating DOM nodes every 3s.
const facePairPool = new WeakMap();

/**
 * Maps the video's *intrinsic* content (video.videoWidth/videoHeight --
 * the space detectFaces()'s normalized boxes are defined in, see
 * face-detector.js) onto its actual on-screen rendered box, accounting for
 * `object-fit` (Meet/Zoom/etc tiles commonly use "cover", which crops the
 * video rather than letterboxing it -- a plain percentage scale would
 * misplace the box whenever the intrinsic and displayed aspect ratios
 * differ). Assumes centered object-position (50% 50%), the CSS default.
 */
function computeVideoContentRect(video) {

    const vw = video.videoWidth;
    const vh = video.videoHeight;
    const ow = video.offsetWidth;
    const oh = video.offsetHeight;

    if (!vw || !vh || !ow || !oh) {
        return null;
    }

    const fit = getComputedStyle(video).objectFit || "fill";

    let scaleX, scaleY;

    if (fit === "cover") {
        scaleX = scaleY = Math.max(ow / vw, oh / vh);
    } else if (fit === "contain" || fit === "scale-down") {
        scaleX = scaleY = Math.min(ow / vw, oh / vh);
    } else {
        // "fill" (or anything unrecognized) stretches to the box exactly,
        // independent axes, no letterboxing.
        scaleX = ow / vw;
        scaleY = oh / vh;
    }

    const contentW = vw * scaleX;
    const contentH = vh * scaleY;

    return {
        left: video.offsetLeft + (ow - contentW) / 2,
        top: video.offsetTop + (oh - contentH) / 2,
        width: contentW,
        height: contentH,
    };
}


/**
 * Ensures `video`'s parent has exactly `count` box+badge pairs (creating
 * new ones or hiding/reusing extras from the pool), and returns them in
 * face-index order.
 */
function getFacePairs(video, count) {

    const parent = video.parentElement;

    if (!parent) {
        return [];
    }

    parent.style.position = "relative";

    let pool = facePairPool.get(video);
    if (!pool) {
        pool = [];
        facePairPool.set(video, pool);
    }

    while (pool.length < count) {
        const box = document.createElement("div");
        box.className = FACE_BOX_CLASS;

        const badge = document.createElement("div");
        badge.className = BADGE_CLASS;

        parent.appendChild(box);
        parent.appendChild(badge);

        pool.push({ box, badge });
    }

    pool.forEach((pair, i) => {
        const visible = i < count;
        pair.box.style.display = visible ? "block" : "none";
        pair.badge.style.display = visible ? "block" : "none";
    });

    return pool;
}


/**
 * Positions a box (and its badge, anchored just above it) around each
 * detected face, in the video's own on-screen coordinates -- called every
 * detection cycle independent of whether predictions come back yet, so
 * boxes track faces even on cycles where the API calls fail. `faces` is
 * the array of normalized (0 -> 1) boxes from detectFaces()
 * (face-detector.js); pass [] to hide everything (no face found).
 */
function updateFaceBoxes(video, faces) {

    const pairs =
        getFacePairs(video, faces.length);

    const content =
        computeVideoContentRect(video);

    if (!content) {
        return;
    }

    faces.forEach((face, i) => {
        const { box, badge } = pairs[i];

        const left = content.left + face.x * content.width;
        const top = content.top + face.y * content.height;
        const width = face.width * content.width;
        const height = face.height * content.height;

        box.style.left = `${left}px`;
        box.style.top = `${top}px`;
        box.style.width = `${width}px`;
        box.style.height = `${height}px`;

        // Badge sits just above its own box (falling back to just inside
        // the top edge for a face near the top of the tile) instead of a
        // single fixed tile corner, since there can now be several.
        badge.style.left = `${left}px`;
        badge.style.top = `${Math.max(0, top - 28)}px`;
    });
}


/**
 * Sets each face's badge text/color (and matching box border color) from
 * its prediction. `predictions[i]` corresponds to `faces[i]`; a null entry
 * (failed API call, or no crop produced) leaves that face's badge showing
 * "…" rather than a stale verdict.
 */
function updateOverlay(video, faces, predictions) {

    const pairs =
        getFacePairs(video, faces.length);

    faces.forEach((_face, i) => {
        const { box, badge } = pairs[i];
        const prediction = predictions[i];

        if (!prediction) {
            badge.textContent = "…";
            badge.dataset.type = "uncertain";
            box.dataset.type = "uncertain";
            return;
        }

        // A crop the backend rejected as too low quality (prediction.fake_probability
        // === null, prediction.quality_issue set) carries no numeric signal.
        if (prediction.fake_probability === null || prediction.fake_probability === undefined) {
            badge.textContent = "… low-quality frame";
            badge.dataset.type = "uncertain";
            box.dataset.type = "uncertain";
            return;
        }

        // Shows the exact value from this cycle's prediction (same object
        // logged to the console in content.js) -- no smoothing/averaging,
        // so the badge always matches the latest API response 1:1.
        const fake = prediction.fake_probability;
        const percentage = Math.round(fake * 100);

        if (fake >= 0.65) {
            badge.textContent = `⚠ POSSIBLE DEEPFAKE ${percentage}%`;
            badge.dataset.type = "fake";
        } else if (fake <= 0.35) {
            badge.textContent = `✓ REAL ${100 - percentage}%`;
            badge.dataset.type = "real";
        } else {
            badge.textContent = `? UNCERTAIN ${percentage}%`;
            badge.dataset.type = "uncertain";
        }

        box.dataset.type = badge.dataset.type;
    });
}


const VOICE_BANNER_ID = "deepfake-detector-voice-banner";

// Whole-tab voice check status, injected once into the page (not per-tile
// -- tab audio capture can't isolate a single speaker, see content.js).
function updateVoiceBanner(result, error) {

    let banner =
        document.getElementById(VOICE_BANNER_ID);

    if (!banner) {
        banner = document.createElement("div");
        banner.id = VOICE_BANNER_ID;
        document.body.appendChild(banner);
    }

    const time =
        new Date().toLocaleTimeString();

    if (error || !result) {
        banner.textContent = `🎙 Voice check failed (${time})`;
        banner.dataset.type = "uncertain";
        return;
    }

    if (result.verdict === "FAKE/SPOOFED") {
        banner.textContent = `🎙 ⚠ VOICE MAY BE SYNTHETIC (${time})`;
        banner.dataset.type = "fake";
    } else {
        banner.textContent = `🎙 Voice sounds genuine (${time})`;
        banner.dataset.type = "real";
    }
}