from fastapi.testclient import TestClient
from torch import optim

from src.models.classifier import BreakHisClassifier
from src.serving.app import app
from src.serving.model_loader import get_model
from src.training.checkpoint import save_checkpoint
from tests.test_app import _fake_histology_bytes

client = TestClient(app)


def test_health_response_has_latency_header():
    resp = client.get("/health")
    assert "x-response-time-ms" in resp.headers


def test_metrics_reflects_recorded_predictions(monkeypatch, tmp_path):
    import src.serving.metrics as metrics_module

    metrics_module.prediction_counts.clear()

    checkpoint_path = tmp_path / "model.pt"
    model = BreakHisClassifier(pretrained=False)
    optimizer = optim.Adam(model.parameters())
    save_checkpoint(checkpoint_path, model, optimizer, epoch=0, metrics={})
    get_model.cache_clear()
    monkeypatch.setattr("src.serving.app.CHECKPOINT_PATH", str(checkpoint_path))

    client.post(
        "/predict",
        files={"file": ("sample.png", _fake_histology_bytes(), "image/png")},
        data={"magnification": "40"},
    )

    resp = client.get("/metrics")
    assert resp.status_code == 200
    distribution = resp.json()["prediction_distribution"]
    assert sum(distribution.values()) == 1
