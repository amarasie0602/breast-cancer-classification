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

from src.data.transforms import eval_transform
from src.explainability.gradcam import GradCAM
from src.explainability.overlay import cam_to_overlay
from src.serving.input_guard import looks_like_histology
from src.serving.logging_middleware import RequestLoggingMiddleware
from src.serving.metrics import get_prediction_distribution, record_prediction
from src.serving.model_loader import get_model
from src.serving.schemas import PredictionResponse

app = FastAPI(title="Breast Cancer Histopathology Classifier")
app.add_middleware(RequestLoggingMiddleware)

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "checkpoints/best_mag40.pt")
STATIC_DIR = Path(__file__).parent / "static"


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
            detail=(
                "This doesn't look like an H&E-stained histopathology image. "
                "The model only recognizes breast tissue histology slides and "
                "has no way to reject unrelated images gracefully, so results "
                "on other images would be meaningless."
            ),
        )

    model = get_model(CHECKPOINT_PATH)
    tensor = eval_transform()(image).unsqueeze(0)

    with torch.no_grad():
        logit = model(tensor)
        probability = torch.sigmoid(logit).item()

    label = "malignant" if probability >= 0.5 else "benign"
    record_prediction(label)

    with GradCAM(model, model.backbone.layer4[-1]) as cam_extractor:
        cam = cam_extractor(tensor.clone().requires_grad_())[0].detach().cpu().numpy()
    overlay = cam_to_overlay(cam, image)

    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    overlay_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return PredictionResponse(
        label=label,
        probability=probability,
        magnification=magnification,
        gradcam_overlay_base64=overlay_base64,
    )
