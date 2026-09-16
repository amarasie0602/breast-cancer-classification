"""Log a batch of Grad-CAM overlays to the active MLflow run as an artifact."""

import tempfile
from pathlib import Path

from src.explainability.gradcam import GradCAM
from src.explainability.visualize import plot_gradcam_grid


def log_gradcam_batch(model, target_layer, images, tensors, labels=None, preds=None, artifact_path="gradcam"):
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
