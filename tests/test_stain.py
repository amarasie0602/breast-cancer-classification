import numpy as np
import pytest
from PIL import Image

from src.data.stain import (
    HE_REFERENCE,
    STAIN_MARKER_FILE,
    StainNormalize,
    dataset_stain_normalization,
    macenko_normalize,
    require_stain_normalization,
)


def _render(stains: np.ndarray, background: float = 235.0, seed: int = 0) -> np.ndarray:
    """A synthetic H&E field: blobby haematoxylin and eosin concentration maps
    rendered through the given 3x2 stain matrix, on a glass background."""
    rng = np.random.default_rng(seed)
    height, width = 96, 128
    yy, xx = np.mgrid[0:height, 0:width]
    h = np.zeros((height, width))
    for cy, cx in rng.uniform([0, 0], [height, width], size=(25, 2)):
        h += 1.4 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 30.0)
    e = 0.3 + 0.5 * (np.sin(xx / 9.0) + 1) / 2
    e[:, :12] = 0  # a strip of bare glass
    h[:, :12] = 0
    concentrations = np.stack([h.ravel(), e.ravel()])
    rgb = background * np.exp(-stains @ concentrations)
    return np.clip(rgb.T, 0, 255).astype(np.uint8).reshape(height, width, 3)


# Same tissue, stained differently: a bluer haematoxylin and a paler, more
# orange eosin, as between two labs or two staining batches.
OTHER_STAIN = np.array([[0.70, 0.10], [0.62, 0.72], [0.35, 0.69]])
OTHER_STAIN /= np.linalg.norm(OTHER_STAIN, axis=0)


def test_normalizing_brings_differently_stained_copies_together():
    a = _render(HE_REFERENCE)
    b = _render(OTHER_STAIN, background=215.0)  # also a dimmer white point

    before = np.abs(a.astype(float) - b.astype(float)).mean()
    after = np.abs(
        macenko_normalize(a).astype(float) - macenko_normalize(b).astype(float)
    ).mean()

    assert after < before / 2, f"mean abs difference {before:.1f} -> {after:.1f}"


def test_background_stays_background():
    normalized = macenko_normalize(_render(OTHER_STAIN, background=215.0))
    glass = normalized[:, :12].reshape(-1, 3)
    # A dim or tinted white point must not be re-rendered as faint stain.
    assert glass.mean() > 225
    assert np.ptp(glass.mean(axis=0)) < 10  # neutral, not lilac


def test_image_without_tissue_is_returned_unchanged():
    blank = np.full((64, 64, 3), 238, dtype=np.uint8)
    assert np.array_equal(macenko_normalize(blank), blank)


def test_transform_keeps_size_and_mode():
    image = Image.fromarray(_render(OTHER_STAIN)).convert("RGBA")
    out = StainNormalize()(image)
    assert out.size == image.size
    assert out.mode == "RGB"


def test_raw_dataset_has_no_stain_normalization(tmp_path):
    assert dataset_stain_normalization(tmp_path) is None
    require_stain_normalization(tmp_path, None)


def test_mismatched_colour_space_is_an_error(tmp_path):
    (tmp_path / STAIN_MARKER_FILE).write_text("macenko\n")
    assert dataset_stain_normalization(tmp_path) == "macenko"
    require_stain_normalization(tmp_path, "macenko")
    with pytest.raises(ValueError, match="holds macenko images"):
        require_stain_normalization(tmp_path, None)
