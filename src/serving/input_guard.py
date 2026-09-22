"""Lightweight heuristic screen for whether an uploaded image plausibly looks
like an H&E-stained histopathology slide.

This is not an out-of-distribution detector. The classifier was trained on
exactly two classes (benign/malignant histology patches) and has no notion
of "not applicable" -- fed a photo, a solid color, or anything else, it will
still produce a confident-looking prediction, and that confidence is
meaningless outside the training distribution (see docs/model_card.md,
Limitations). This check catches the obvious, common cases -- ordinary
photos, cartoons, solid/blank images -- by testing for two properties real
H&E slides reliably have and most non-histology images don't:

  1. A meaningful fraction of pixels in the purple/magenta/pink hue range
     that hematoxylin (nuclei) and eosin (cytoplasm/stroma) staining produce.
  2. Enough local texture to be a real tissue image rather than a flat or
     near-flat image.

Thresholds were picked empirically against 60 random real BreakHis samples
across all magnifications/subtypes (worst case: hue fraction 0.026, texture
8.03) with generous margin below those observed minimums, and against a set
of representative non-histology images (photos, solid colors, cartoons).
It will not catch everything -- e.g. synthetic random noise can spuriously
pass -- that tradeoff is intentional: a heuristic this cheap cannot be a
real OOD detector, only a screen for the common, honest-mistake cases.
"""

import numpy as np
from PIL import Image

_HUE_MIN = 250.0
_HUE_MAX = 350.0
_MIN_SATURATION = 0.12
_MIN_HUE_FRACTION = 0.02
_MIN_TEXTURE = 3.0


def looks_like_histology(image: Image.Image) -> bool:
    """Return True if the image plausibly looks like an H&E histology slide."""
    small = image.convert("RGB").resize((150, 150))

    hsv = np.asarray(small.convert("HSV")).astype("float32")
    hue = hsv[..., 0] / 255.0 * 360.0
    saturation = hsv[..., 1] / 255.0
    in_stain_range = (hue >= _HUE_MIN) & (hue <= _HUE_MAX) & (saturation > _MIN_SATURATION)
    hue_fraction = float(in_stain_range.mean())

    gray = np.asarray(small.convert("L")).astype("float32")
    texture = float(np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean())

    return hue_fraction >= _MIN_HUE_FRACTION and texture >= _MIN_TEXTURE
