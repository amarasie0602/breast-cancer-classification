"""Log a batch of Grad-CAM overlays to the active MLflow run as an artifact."""

import tempfile
from pathlib import Path
from typing import Optional, Sequence

from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor, nn

from src.explainability.gradcam import GradCAM
from src.explainability.visualize import plot_gradcam_grid


def log_gradcam_batch(
    model: nn.Module,
    target_layer: nn.Module,
    images: Sequence[Image.Image],
    tensors: Sequence[Tensor],
    labels: Optional[Sequence] = None,
    preds: Optional[Sequence] = None,
    artifact_path: str = "gradcam",
) -> Figure:
    import mlflow

    cam_extractor = GradCAM(model, target_layer)
    cams = [
        cam_extractor(t.unsqueeze(0).requires_grad_())[0].detach().cpu().numpy() for t in tensors
    ]

    fig = plot_gradcam_grid(images, cams, labels=labels, preds=preds)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "gradcam_grid.png"
        fig.savefig(out_path)
        mlflow.log_artifact(str(out_path), artifact_path=artifact_path)

    return fig
