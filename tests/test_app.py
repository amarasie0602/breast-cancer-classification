import base64
import io

import numpy as np
import torch
from fastapi.testclient import TestClient
from PIL import Image
from torch import optim

from src.data.dataset import SUBTYPE_NAMES
from src.models.classifier import BreakHisClassifier, MalignantSubtypeClassifier
from src.serving.app import app
from src.serving.model_loader import get_model, get_subtype_model
from src.training.checkpoint import save_checkpoint

client = TestClient(app)


def _make_forced_binary_checkpoint(tmp_path, force_malignant: bool):
    """A real BreakHisClassifier (so GradCAM's layer4 hook still works)
    with its fc layer overridden to ignore the input and always predict
    the same class, for deterministic benign/malignant test outcomes."""
    model = BreakHisClassifier(pretrained=False)
    with torch.no_grad():
        model.backbone.fc.weight.zero_()
        model.backbone.fc.bias.fill_(50.0 if force_malignant else -50.0)
    path = tmp_path / "forced_binary.pt"
    save_checkpoint(path, model, optim.Adam(model.parameters()), epoch=0, metrics={})
    return path


def _make_forced_subtype_checkpoint(tmp_path, forced_index: int):
    """A real MalignantSubtypeClassifier with its fc layer overridden to
    always predict the same subtype, regardless of input."""
    model = MalignantSubtypeClassifier(num_classes=len(SUBTYPE_NAMES), pretrained=False)
    with torch.no_grad():
        head = model.backbone.classifier[-1]
        head.weight.zero_()
        head.bias.zero_()
        head.bias[forced_index] = 50.0
    path = tmp_path / "forced_subtype.pt"
    save_checkpoint(path, model, optim.Adam(model.parameters()), epoch=0, metrics={})
    return path


def test_index_serves_html_page():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Breast Cancer Histopathology Classifier" in resp.text


def test_health_returns_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def _fake_histology_bytes():
    """A synthetic image textured and colored enough to pass the
    histology input guard, standing in for a real H&E slide in tests
    that exercise the inference path, not input validation."""
    rng = np.random.default_rng(42)
    base = np.array([170, 90, 150], dtype="float32")
    noise = rng.normal(0, 35, size=(64, 64, 3))
    arr = np.clip(base + noise, 0, 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    buf.seek(0)
    return buf


def _fake_photo_bytes():
    """A flat, non-histology-colored image that the input guard should reject."""
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(120, 200, 90)).save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_predict_without_checkpoint_returns_503(monkeypatch):
    monkeypatch.setattr("src.serving.app.CHECKPOINT_PATH", "nonexistent_checkpoint.pt")
    resp = client.post(
        "/predict",
        files={"file": ("sample.png", _fake_histology_bytes(), "image/png")},
        data={"magnification": "40"},
    )
    assert resp.status_code == 503


def test_predict_with_non_histology_image_returns_422(monkeypatch, tmp_path):
    fake_checkpoint = tmp_path / "fake.pt"
    fake_checkpoint.write_bytes(b"not a real checkpoint")
    monkeypatch.setattr("src.serving.app.CHECKPOINT_PATH", str(fake_checkpoint))

    resp = client.post(
        "/predict",
        files={"file": ("photo.png", _fake_photo_bytes(), "image/png")},
        data={"magnification": "40"},
    )
    assert resp.status_code == 422
    assert resp.json()["detail"] == "Invalid Image — Please upload a valid breast histology image."


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
    monkeypatch.setattr("src.serving.app.SUBTYPE_CHECKPOINT_PATH", "nonexistent_subtype.pt")

    resp = client.post(
        "/predict",
        files={"file": ("sample.png", _fake_histology_bytes(), "image/png")},
        data={"magnification": "100"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] in ("benign", "malignant")
    assert 0.0 <= body["probability"] <= 1.0
    assert body["magnification"] == "100"

    overlay_bytes = base64.b64decode(body["gradcam_overlay_base64"])
    overlay_image = Image.open(io.BytesIO(overlay_bytes))
    assert overlay_image.size == (64, 64)


def test_predict_benign_has_no_subtype_fields(monkeypatch, tmp_path):
    checkpoint_path = _make_forced_binary_checkpoint(tmp_path, force_malignant=False)
    get_model.cache_clear()
    monkeypatch.setattr("src.serving.app.CHECKPOINT_PATH", str(checkpoint_path))
    monkeypatch.setattr("src.serving.app.SUBTYPE_CHECKPOINT_PATH", "nonexistent_subtype.pt")

    resp = client.post(
        "/predict",
        files={"file": ("sample.png", _fake_histology_bytes(), "image/png")},
        data={"magnification": "40"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "benign"
    assert body["subtype"] is None
    assert body["subtype_display_name"] is None
    assert body["subtype_confidence"] is None


def test_predict_malignant_without_subtype_checkpoint_omits_subtype(monkeypatch, tmp_path):
    checkpoint_path = _make_forced_binary_checkpoint(tmp_path, force_malignant=True)
    get_model.cache_clear()
    monkeypatch.setattr("src.serving.app.CHECKPOINT_PATH", str(checkpoint_path))
    monkeypatch.setattr("src.serving.app.SUBTYPE_CHECKPOINT_PATH", "nonexistent_subtype.pt")

    resp = client.post(
        "/predict",
        files={"file": ("sample.png", _fake_histology_bytes(), "image/png")},
        data={"magnification": "40"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "malignant"
    assert body["subtype"] is None


def test_predict_malignant_with_subtype_checkpoint_returns_subtype(monkeypatch, tmp_path):
    checkpoint_path = _make_forced_binary_checkpoint(tmp_path, force_malignant=True)
    subtype_checkpoint_path = _make_forced_subtype_checkpoint(tmp_path, forced_index=2)  # mucinous_carcinoma
    get_model.cache_clear()
    get_subtype_model.cache_clear()
    monkeypatch.setattr("src.serving.app.CHECKPOINT_PATH", str(checkpoint_path))
    monkeypatch.setattr("src.serving.app.SUBTYPE_CHECKPOINT_PATH", str(subtype_checkpoint_path))

    resp = client.post(
        "/predict",
        files={"file": ("sample.png", _fake_histology_bytes(), "image/png")},
        data={"magnification": "40"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "malignant"
    assert body["subtype"] == "mucinous_carcinoma"
    assert body["subtype_display_name"] == "Mucinous Carcinoma"
    assert body["subtype_confidence"] > 0.99
