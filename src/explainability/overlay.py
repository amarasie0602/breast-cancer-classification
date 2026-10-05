"""Render a Grad-CAM heatmap as an overlay on the original image."""

import numpy as np
import numpy.typing as npt
from PIL import Image

# matplotlib's "jet" colormap, from its segment data, as a 256-entry lookup
# table. Defined here so serving doesn't need matplotlib (and its seven
# dependencies) just for one colour scale; tests check it matches matplotlib.
_JET_SEGMENTS = {
    "red": ((0.0, 0.0), (0.35, 0.0), (0.66, 1.0), (0.89, 1.0), (1.0, 0.5)),
    "green": ((0.0, 0.0), (0.125, 0.0), (0.375, 1.0), (0.64, 1.0), (0.91, 0.0), (1.0, 0.0)),
    "blue": ((0.0, 0.5), (0.11, 1.0), (0.34, 1.0), (0.65, 0.0), (1.0, 0.0)),
}
_LUT_SIZE = 256
_JET_LUT = np.stack(
    [
        np.interp(np.linspace(0.0, 1.0, _LUT_SIZE), *zip(*_JET_SEGMENTS[channel], strict=True))
        for channel in ("red", "green", "blue")
    ],
    axis=-1,
)


def jet(values: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    """RGB in [0, 1] for values in [0, 1], as matplotlib's "jet" maps them."""
    index = np.clip((np.asarray(values, dtype=np.float64) * _LUT_SIZE).astype(int), 0, _LUT_SIZE - 1)
    return _JET_LUT[index]


def cam_to_overlay(
    cam: npt.NDArray[np.floating], original_image: Image.Image, alpha: float = 0.4
) -> Image.Image:
    """cam: 2D array in [0, 1]. original_image: PIL Image. Returns a PIL Image."""
    cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        original_image.size, resample=Image.BILINEAR
    )
    cam_norm = np.asarray(cam_resized).astype(np.float32) / 255.0

    heatmap = (jet(cam_norm) * 255).astype(np.uint8)

    base = np.asarray(original_image.convert("RGB")).astype(np.float32)
    blended = (1 - alpha) * base + alpha * heatmap.astype(np.float32)
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))
