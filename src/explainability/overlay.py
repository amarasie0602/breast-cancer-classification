"""Render a Grad-CAM heatmap as an overlay on the original image."""

import numpy as np
from matplotlib import colormaps
from PIL import Image


def cam_to_overlay(cam, original_image, alpha=0.4, colormap="jet"):
    """cam: 2D array in [0, 1]. original_image: PIL Image. Returns a PIL Image."""
    cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        original_image.size, resample=Image.BILINEAR
    )
    cam_norm = np.asarray(cam_resized).astype(np.float32) / 255.0

    heatmap = colormaps[colormap](cam_norm)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)

    base = np.asarray(original_image.convert("RGB")).astype(np.float32)
    blended = (1 - alpha) * base + alpha * heatmap.astype(np.float32)
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))
