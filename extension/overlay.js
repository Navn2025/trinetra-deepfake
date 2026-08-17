const BADGE_CLASS =
    "deepfake-detector-badge";

// Number of recent per-tile predictions averaged together before choosing
// a label. Classification runs on a single independent frame every cycle
// with no temporal context, so a single bad-angle/motion-blur/occlusion
// frame can swing the label on its own -- averaging recent history smooths
// that out at the cost of a few seconds of lag before a real change shows.
const HISTORY_SIZE = 5;

const predictionHistory = new WeakMap(); // video -> number[] of recent fake_probability values


function getBadgeForVideo(video) {

    const parent = video.parentElement;

    if (!parent) {
        return null;
    }

    let badge =
        parent.querySelector(
            `.${BADGE_CLASS}`
        );

    if (!badge) {

        badge =
            document.createElement("div");

        badge.className =
            BADGE_CLASS;

        parent.style.position =
            "relative";

        parent.appendChild(badge);
    }

    return badge;
}


function updateOverlay(
    video,
    prediction
) {

    if (!prediction) {
        return;
    }

    const badge =
        getBadgeForVideo(video);

    if (!badge) {
        return;
    }

    // A crop the backend rejected as too low quality (prediction.fake_probability
    // === null, prediction.quality_issue set) carries no numeric signal --
    // pushing it into the running average would silently coerce to 0 in JS
    // arithmetic (null + number === number), biasing the smoothed score
    // toward "real" for no real reason. Show it plainly instead and leave
    // the averaged history untouched so the next good frame isn't diluted.
    if (prediction.fake_probability === null || prediction.fake_probability === undefined) {
        badge.textContent = "… low-quality frame";
        badge.dataset.type = "uncertain";
        return;
    }

    const history =
        predictionHistory.get(video) || [];

    history.push(prediction.fake_probability);
    if (history.length > HISTORY_SIZE) {
        history.shift();
    }
    predictionHistory.set(video, history);

    const fake =
        history.reduce((sum, v) => sum + v, 0) / history.length;

    const percentage =
        Math.round(fake * 100);

    if (fake >= 0.65) {

        badge.textContent =
            `⚠ POSSIBLE DEEPFAKE ${percentage}%`;

        badge.dataset.type =
            "fake";

    } else if (fake <= 0.35) {

        badge.textContent =
            `✓ REAL ${100 - percentage}%`;

        badge.dataset.type =
            "real";

    } else {

        badge.textContent =
            `? UNCERTAIN ${percentage}%`;

        badge.dataset.type =
            "uncertain";
    }
}