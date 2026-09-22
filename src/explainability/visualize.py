"""Batch Grad-CAM visualization: grid of original vs. overlay for a batch of images."""

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from src.explainability.overlay import cam_to_overlay


def plot_gradcam_grid(
    images: Sequence[Image.Image],
    cams: Sequence[np.ndarray],
    labels: Optional[Sequence] = None,
    preds: Optional[Sequence] = None,
    ncols: int = 4,
) -> Figure:
    """images: list of PIL Images. cams: list of 2D numpy arrays in [0, 1]."""
    n = len(images)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, (image, cam) in enumerate(zip(images, cams, strict=True)):
        overlay = cam_to_overlay(cam, image)
        axes[i].imshow(overlay)
        axes[i].axis("off")

        title_parts = []
        if labels is not None:
            title_parts.append(f"true={labels[i]}")
        if preds is not None:
            title_parts.append(f"pred={preds[i]}")
        if title_parts:
            axes[i].set_title(" ".join(title_parts), fontsize=9)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    return fig
