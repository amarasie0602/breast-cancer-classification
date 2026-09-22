import matplotlib

matplotlib.use("Agg")

import mlflow

from src.data.dataset import BreakHisDataset
from src.data.transforms import eval_transform
from src.explainability.mlflow_logging import log_gradcam_batch
from src.models.classifier import BreakHisClassifier


def test_log_gradcam_batch_writes_mlflow_artifact(breakhis_root, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")

    ds = BreakHisDataset(breakhis_root, transform=eval_transform())
    raw_ds = BreakHisDataset(breakhis_root)

    model = BreakHisClassifier(pretrained=False)
    model.eval()
    target_layer = model.backbone.layer4[-1]

    n = 3
    tensors = [ds[i][0] for i in range(n)]
    images = [raw_ds[i][0] for i in range(n)]
    labels = [ds[i][1] for i in range(n)]

    with mlflow.start_run() as run:
        log_gradcam_batch(model, target_layer, images, tensors, labels=labels)
        run_id = run.info.run_id

    client = mlflow.MlflowClient()
    artifacts = client.list_artifacts(run_id, path="gradcam")
    assert any(a.path.endswith("gradcam_grid.png") for a in artifacts)
