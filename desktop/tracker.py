"""
Matches detected faces across detection cycles by position, so predictions
can be smoothed over a rolling window the same way the browser extension
smooths per-<video>-tile predictions (extension/overlay.js) -- there's no
stable DOM element to key off of here, just raw frame coordinates, so
faces are matched to the nearest previous face center within a distance
threshold.
"""
import math
from collections import deque

HISTORY_SIZE = 3  # number of recent predictions averaged per tracked face
MAX_MATCH_DISTANCE = 0.15  # fraction of frame diagonal; tune if faces move fast


class FaceTracker:
    def __init__(self):
        self._tracks: dict[int, dict] = {}
        self._next_id = 0

    def update(self, detections: list[dict]) -> list[dict]:
        """
        detections: list of {"bbox_fraction": (x, y, w, h), "fake_probability": float}
        Returns the same list with "smoothed_fake_probability" and "track_id" added.
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
            track = self._tracks[best_id]
            track["center"] = center
            # None means "no numeric signal this cycle" (failed API call, or
            # a low-quality crop the backend rejected) -- skip it rather than
            # folding it into the rolling average (would crash: sum() can't
            # add None) or silently pulling the average toward whatever a
            # fabricated placeholder value would be.
            if det["fake_probability"] is not None:
                track["history"].append(det["fake_probability"])

            smoothed = (
                sum(track["history"]) / len(track["history"]) if track["history"] else None
            )
            results.append(
                {
                    **det,
                    "track_id": best_id,
                    "smoothed_fake_probability": smoothed,
                }
            )

        # drop tracks not matched this cycle -- that face is no longer on screen
        self._tracks = {tid: t for tid, t in self._tracks.items() if tid in used_ids}

        return results
