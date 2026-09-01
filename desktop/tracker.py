"""
Matches detected faces across detection cycles by position, so predictions
can be smoothed over a rolling window the same way the browser extension
smooths per-<video>-tile predictions (extension/overlay.js) -- there's no
stable DOM element to key off of here, just raw frame coordinates, so
faces are matched to the nearest previous face center within a distance
threshold.

Split into two phases (assign_track_ids then record_prediction) rather than
one update() call: main.py needs a face's track_id before it can call the
model (the prediction is recorded against that id), so track-id assignment
happens first and prediction recording is a separate step.
"""
import math
from collections import deque

HISTORY_SIZE = 3  # number of recent predictions averaged per tracked face
MAX_MATCH_DISTANCE = 0.15  # fraction of frame diagonal; tune if faces move fast


class FaceTracker:
    def __init__(self):
        self._tracks: dict[int, dict] = {}
        self._next_id = 0

    def assign_track_ids(self, detections: list[dict]) -> list[dict]:
        """
        detections: list of {"bbox_fraction": (x, y, w, h), ...}
        Returns the same list with "track_id" added (position-matched
        against tracks from the previous cycle). Drops tracks not matched
        this cycle (that face is no longer on screen). Does not touch
        prediction history -- call record_prediction separately once each
        face's prediction for this cycle is in hand.
        """
        used_ids: set[int] = set()
        results = []

        for det in detections:
            x, y, w, h = det["bbox_fraction"]
            center = (x + w / 2, y + h / 2)

            best_id, best_dist = None, MAX_MATCH_DISTANCE
            for track_id, track in self._tracks.items():
                if track_id in used_ids:
                    continue
                dist = math.hypot(center[0] - track["center"][0], center[1] - track["center"][1])
                if dist < best_dist:
                    best_id, best_dist = track_id, dist

            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
                self._tracks[best_id] = {"center": center, "history": deque(maxlen=HISTORY_SIZE)}

            used_ids.add(best_id)
            self._tracks[best_id]["center"] = center
            results.append({**det, "track_id": best_id})

        # drop tracks not matched this cycle -- that face is no longer on screen
        self._tracks = {tid: t for tid, t in self._tracks.items() if tid in used_ids}

        return results

    def record_prediction(self, track_id: int, fake_probability: float | None) -> float | None:
        """Appends a new prediction to `track_id`'s rolling history and
        returns the smoothed average. `fake_probability=None` (a failed or
        quality-rejected API call) is skipped rather than folded into the
        average -- sum() can't add None, and a fabricated placeholder would
        just bias the average toward "real" for no real reason. Returns
        None if the track has no scored history yet (still waiting on its
        first prediction)."""
        track = self._tracks.get(track_id)
        if track is None:
            return None
        if fake_probability is not None:
            track["history"].append(fake_probability)
        return sum(track["history"]) / len(track["history"]) if track["history"] else None
