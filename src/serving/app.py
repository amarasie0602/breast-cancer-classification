"""FastAPI inference service for the BreakHis classifier."""

import base64
import io
import os
from pathlib import Path
from typing import Dict

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from src.data.dataset import SUBTYPE_DISPLAY_NAMES, SUBTYPE_NAMES
from src.data.transforms import eval_transform
from src.explainability.gradcam import GradCAM
from src.explainability.overlay import cam_to_overlay
from src.serving.input_guard import looks_like_histology
from src.serving.logging_middleware import RequestLoggingMiddleware
from src.serving.metrics import get_prediction_distribution, record_prediction
from src.serving.model_loader import get_model, get_subtype_model
from src.serving.schemas import PredictionResponse

app = FastAPI(title="Breast Cancer Histopathology Classifier")
app.add_middleware(RequestLoggingMiddleware)

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "checkpoints/best_mag40.pt")
SUBTYPE_CHECKPOINT_PATH = os.environ.get("SUBTYPE_CHECKPOINT_PATH", "checkpoints/best_subtype.pt")
STATIC_DIR = Path(__file__).parent / "static"

# Operating point for benign/malignant. 0.5 is where sigmoid happens to
# cross, not a chosen threshold: at 0.5 this model runs at sensitivity
# 0.88-0.99 but specificity 0.46-0.73 (see docs/model_card.md). Raising it
# trades caught cancers for fewer false alarms. Deliberately left at the
# documented default rather than silently tuned, since where it belongs is
# a clinical cost judgement; `python -m scripts.tune_threshold` prints the
# whole curve to inform it.
DECISION_THRESHOLD = float(os.environ.get("DECISION_THRESHOLD", "0.5"))


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Dict[str, Dict[str, int]]:
    return {"prediction_distribution": get_prediction_distribution()}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...), magnification: str = Form("40")
) -> PredictionResponse:
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as e:
        raise HTTPException(status_code=400, detail="File is not a valid image") from e

    if not os.path.exists(CHECKPOINT_PATH):
        raise HTTPException(status_code=503, detail="Model checkpoint not available")

    if not looks_like_histology(image):
        raise HTTPException(
            status_code=422,
            detail="Invalid Image — Please upload a valid breast histology image.",
        )

    model = get_model(CHECKPOINT_PATH)
    tensor = eval_transform()(image).unsqueeze(0)

    with torch.no_grad():
        logit = model(tensor)
        probability = torch.sigmoid(logit).item()

    label = "malignant" if probability >= DECISION_THRESHOLD else "benign"
    record_prediction(label)

    with GradCAM(model, model.backbone.layer4[-1]) as cam_extractor:
        cam = cam_extractor(tensor.clone().requires_grad_())[0].detach().cpu().numpy()
    overlay = cam_to_overlay(cam, image)

    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    overlay_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    subtype = subtype_display_name = None
    subtype_confidence = None
    if label == "malignant" and os.path.exists(SUBTYPE_CHECKPOINT_PATH):
        subtype_model = get_subtype_model(SUBTYPE_CHECKPOINT_PATH)
        with torch.no_grad():
            subtype_logits = subtype_model(tensor)
            subtype_probs = torch.softmax(subtype_logits, dim=1)[0]
        subtype_idx = int(subtype_probs.argmax().item())
        subtype = SUBTYPE_NAMES[subtype_idx]
        subtype_display_name = SUBTYPE_DISPLAY_NAMES[subtype]
        subtype_confidence = float(subtype_probs[subtype_idx].item())

    return PredictionResponse(
        label=label,
        probability=probability,
        magnification=magnification,
        gradcam_overlay_base64=overlay_base64,
        subtype=subtype,
        subtype_display_name=subtype_display_name,
        subtype_confidence=subtype_confidence,
    )
