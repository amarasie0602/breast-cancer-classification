import io

from fastapi.testclient import TestClient
from PIL import Image
from torch import optim

from src.models.classifier import BreakHisClassifier
from src.serving.app import app
from src.serving.model_loader import get_model
from src.training.checkpoint import save_checkpoint

client = TestClient(app)


def test_health_returns_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def _fake_image_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(120, 50, 50)).save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_predict_without_checkpoint_returns_503(monkeypatch):
    monkeypatch.setattr("src.serving.app.CHECKPOINT_PATH", "nonexistent_checkpoint.pt")
    resp = client.post(
        "/predict",
        files={"file": ("sample.png", _fake_image_bytes(), "image/png")},
        data={"magnification": "40"},
    )
    assert resp.status_code == 503


def test_predict_with_invalid_image_returns_400(monkeypatch, tmp_path):
    fake_checkpoint = tmp_path / "fake.pt"
    fake_checkpoint.write_bytes(b"not a real checkpoint")
    monkeypatch.setattr("src.serving.app.CHECKPOINT_PATH", str(fake_checkpoint))

    resp = client.post(
        "/predict",
        files={"file": ("not_an_image.txt", io.BytesIO(b"hello"), "text/plain")},
        data={"magnification": "40"},
    )
    assert resp.status_code == 400


def test_predict_happy_path_returns_valid_response(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "model.pt"
    model = BreakHisClassifier(pretrained=False)
    optimizer = optim.Adam(model.parameters())
    save_checkpoint(checkpoint_path, model, optimizer, epoch=0, metrics={})
    get_model.cache_clear()
    monkeypatch.setattr("src.serving.app.CHECKPOINT_PATH", str(checkpoint_path))

    resp = client.post(
        "/predict",
        files={"file": ("sample.png", _fake_image_bytes(), "image/png")},
        data={"magnification": "100"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] in ("benign", "malignant")
    assert 0.0 <= body["probability"] <= 1.0
    assert body["magnification"] == "100"
