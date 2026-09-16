"""FastAPI inference service for the BreakHis classifier."""

import io
import os

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.data.transforms import eval_transform
from src.serving.model_loader import get_model
from src.serving.schemas import PredictionResponse

app = FastAPI(title="Breast Cancer Histopathology Classifier")

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "checkpoints/best_mag40.pt")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), magnification: str = Form("40")):
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File is not a valid image")

    if not os.path.exists(CHECKPOINT_PATH):
        raise HTTPException(status_code=503, detail="Model checkpoint not available")

    model = get_model(CHECKPOINT_PATH)
    tensor = eval_transform()(image).unsqueeze(0)

    with torch.no_grad():
        logit = model(tensor)
        probability = torch.sigmoid(logit).item()

    label = "malignant" if probability >= 0.5 else "benign"
    return PredictionResponse(label=label, probability=probability, magnification=magnification)
