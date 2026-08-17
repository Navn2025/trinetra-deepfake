"""
Prints the real Laplacian-variance sharpness measurements config.py's
MIN_SHARPNESS_VARIANCE was calibrated against -- run this again if you want
to sanity-check or re-tune that threshold, rather than trusting the number
in the comment blindly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image

from preprocessing import sharpness_variance

rng = np.random.default_rng(0)


def main():
    cases = []

    cases.append(("solid color (worst case, zero texture)", Image.new("RGB", (256, 256), (120, 100, 150))))

    checker = np.zeros((256, 256, 3), dtype=np.uint8)
    checker[::8, :] = 255
    checker[:, ::8] = 255
    cases.append(("checkerboard (very sharp)", Image.fromarray(checker)))

    cases.append(
        ("random noise (unrealistically high, sanity check)", Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)))
    )

    tiny16 = Image.fromarray(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8))
    cases.append(("16x16 source upsampled 16x to 256 (bilinear)", tiny16.resize((256, 256), Image.BILINEAR)))

    tiny32 = Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8))
    cases.append(("32x32 source upsampled 8x to 256 (bilinear)", tiny32.resize((256, 256), Image.BILINEAR)))

    checker_gray = cv2.cvtColor(checker, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(checker_gray, (15, 15), 5)
    cases.append(("sharp image heavily gaussian-blurred (defocus, not upsampling)", Image.fromarray(blurred)))

    gradient = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        gradient[i, :, :] = i
    cases.append(("smooth gradient (no hard edges)", Image.fromarray(gradient)))

    print(f"{'case':<55} sharpness")
    for label, img in cases:
        print(f"{label:<55} {sharpness_variance(img):.2f}")


if __name__ == "__main__":
    main()
