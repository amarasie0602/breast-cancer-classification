"""Macenko H&E stain normalization.

BreakHis slides were stained and scanned over time, and their pink/purple
balance varies from patient to patient. With only ~60 training patients per
magnification, a classifier can learn "this shade of purple" as a shortcut
for a patient's label instead of learning tissue structure. Macenko's method
(Macenko et al., ISBI 2009) estimates each image's own haematoxylin and eosin
colour vectors and re-renders it with a fixed reference stain, so every image
reaches the model in the same colour space.

The reference stain vectors and concentrations are the ones published with
the method's widely used reference implementation, so no particular BreakHis
image (and therefore no particular patient or split) defines the target.
"""

from pathlib import Path

import numpy as np
from PIL import Image

# A stain-normalized copy of the dataset is marked, once every image has been
# converted, by an empty file at its root named after the method (e.g.
# "STAIN_NORMALIZATION.macenko"). Training and evaluation check for it, so a
# model can't be trained on one colour space and evaluated on another without
# an error. Only known methods are looked for, and nothing is read from the
# dataset directory.
MACENKO = "macenko"
KNOWN_STAIN_NORMALIZATIONS = (MACENKO,)

# Reference H (column 0) and E (column 1) optical-density vectors, and the
# 99th-percentile concentration of each stain in the reference image.
HE_REFERENCE = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
MAX_CONCENTRATION_REFERENCE = np.array([1.9705, 1.0308])

# Each image's own background ("white" glass) brightness, per colour channel,
# is taken as this percentile. BreakHis backgrounds range from bright white to
# a dim green or pink cast; assuming a fixed white point renders those as
# faint stain, turning empty lumens lilac after normalization.
_BACKGROUND_PERCENTILE = 99.0
# Output background brightness, as in the reference implementation.
_TRANSMITTED_LIGHT = 240.0
_OD_THRESHOLD = 0.15  # pixels this transparent in any channel are background
_ANGLE_PERCENTILE = 1.0  # robust extremes of the stain angle distribution
# Below this many tissue pixels the stain estimate is noise (e.g. a mostly
# blank field), and the image is returned unchanged rather than distorted.
_MIN_TISSUE_PIXELS = 500
# Every Nth tissue pixel is enough to estimate two colour vectors, and keeps
# normalization of a 700x460 BreakHis image to about 0.1s on CPU.
_ESTIMATION_STRIDE = 4


def _stain_vectors(tissue_od: np.ndarray) -> np.ndarray:
    """The image's own 3x2 H&E optical-density matrix (H in column 0)."""
    _, eigenvectors = np.linalg.eigh(np.cov(tissue_od.T))
    plane = eigenvectors[:, 1:3]  # the two largest-variance directions
    projected = tissue_od @ plane
    angles = np.arctan2(projected[:, 1], projected[:, 0])
    low, high = np.percentile(angles, [_ANGLE_PERCENTILE, 100 - _ANGLE_PERCENTILE])
    v_low = plane @ np.array([np.cos(low), np.sin(low)])
    v_high = plane @ np.array([np.cos(high), np.sin(high)])
    # Haematoxylin absorbs more red than eosin does, so it has the larger
    # first (R) component; eigenvector signs are arbitrary, so fix them too.
    vectors = np.array([v_low, v_high] if v_low[0] > v_high[0] else [v_high, v_low]).T
    return vectors * np.sign(vectors.sum(axis=0))


def macenko_normalize(rgb: np.ndarray) -> np.ndarray:
    """Return ``rgb`` (HxWx3 uint8) re-rendered with the reference H&E stain.

    Images without enough tissue to estimate a stain are returned unchanged.
    """
    height, width, _ = rgb.shape
    pixels = rgb.reshape(-1, 3).astype(np.float64)
    background = np.percentile(pixels, _BACKGROUND_PERCENTILE, axis=0)
    od = np.maximum(-np.log((pixels + 1.0) / (background + 1.0)), 0.0)

    tissue = od[np.all(od >= _OD_THRESHOLD, axis=1)]
    if len(tissue) < _MIN_TISSUE_PIXELS:
        return rgb
    stains = _stain_vectors(tissue[::_ESTIMATION_STRIDE])

    concentrations, *_ = np.linalg.lstsq(stains, od.T, rcond=None)
    # A pixel can't hold a negative amount of stain; off-palette tints (e.g. a
    # green cast) otherwise come out brighter than the background.
    concentrations = np.maximum(concentrations, 0.0)
    max_concentration = np.percentile(concentrations, 99, axis=1)
    if np.any(max_concentration <= 0) or not np.all(np.isfinite(max_concentration)):
        return rgb
    concentrations *= (MAX_CONCENTRATION_REFERENCE / max_concentration)[:, None]

    normalized = _TRANSMITTED_LIGHT * np.exp(-HE_REFERENCE @ concentrations)
    return np.clip(normalized.T, 0, 255).astype(np.uint8).reshape(height, width, 3)


class StainNormalize:
    """torchvision-style transform: PIL image in, stain-normalized PIL image out."""

    def __call__(self, image: Image.Image) -> Image.Image:
        return Image.fromarray(macenko_normalize(np.asarray(image.convert("RGB"))))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(method={MACENKO!r})"


def stain_marker_name(method: str) -> str:
    """File name that marks a dataset copy as normalized with ``method``."""
    return f"STAIN_NORMALIZATION.{method}"


def dataset_stain_normalization(root: str | Path) -> str | None:
    """The stain normalization a dataset copy was built with, or None for raw data."""
    for method in KNOWN_STAIN_NORMALIZATIONS:
        if (Path(root) / stain_marker_name(method)).is_file():
            return method
    return None


def require_stain_normalization(data_root: str | Path, expected: str | None) -> None:
    """Raise if ``data_root`` isn't in the colour space ``expected`` (None = raw)."""
    actual = dataset_stain_normalization(data_root)
    if actual != expected:
        raise ValueError(
            f"expected {expected or 'raw (unnormalized)'} images, but {data_root} holds "
            f"{actual or 'raw (unnormalized)'} images"
        )
