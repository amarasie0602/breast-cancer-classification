import matplotlib

matplotlib.use("Agg")

import numpy as np
from PIL import Image

from src.explainability.visualize import plot_gradcam_grid


def test_plot_gradcam_grid_runs_without_error():
    images = [Image.new("RGB", (32, 32), color=(i * 20, 0, 0)) for i in range(5)]
    cams = [np.random.rand(7, 7).astype(np.float32) for _ in range(5)]
    labels = [0, 1, 0, 1, 0]
    preds = [0, 1, 1, 1, 0]

    fig = plot_gradcam_grid(images, cams, labels=labels, preds=preds, ncols=3)

    assert fig is not None
