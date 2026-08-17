"""Writes a small synthetic .mp4 (no real face content -- just for exercising
the decode/sampling/aggregation mechanics of video_pipeline.py end to end,
not the model's actual accuracy on real footage)."""
import sys
from pathlib import Path

import cv2
import numpy as np

OUT_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "test_video.mp4"
DURATION_SECONDS = 6
FPS = 24
SIZE = (320, 240)


def main():
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT_PATH), fourcc, FPS, SIZE)
    rng = np.random.default_rng(0)
    for _ in range(DURATION_SECONDS * FPS):
        frame = rng.integers(0, 255, (SIZE[1], SIZE[0], 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    print(f"wrote {OUT_PATH} ({DURATION_SECONDS}s @ {FPS}fps)")


if __name__ == "__main__":
    main()
