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

from src.data.dataset import SUBTYPE_SCHEMES
from src.data.transforms import eval_transform
from src.explainability.gradcam import GradCAM
from src.explainability.overlay import cam_to_overlay
from src.serving.input_guard import looks_like_histology
from src.serving.logging_middleware import RequestLoggingMiddleware
from src.serving.metrics import get_prediction_distribution, record_prediction
from src.serving.model_loader import (
    get_model,
    get_subtype_model,
    subtype_checkpoint_macro_f1,
    subtype_checkpoint_scheme,
)
from src.serving.schemas import PredictionResponse

app = FastAPI(title="Breast Cancer Histopathology Classifier")
app.add_middleware(RequestLoggingMiddleware)

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "checkpoints/best_mag40.pt")

# One binary model is trained per magnification, and they are not
# interchangeable: on the held-out test set the 40x model's specificity is
# 0.544 while the 200x model's is 0.732. Serving always used the 40x model no
# matter what magnification the user selected, so /predict now picks
# best_mag{magnification}.pt from this directory when it exists and falls back
# to CHECKPOINT_PATH when it doesn't (e.g. a container that only ships the
# 40x checkpoint).
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "checkpoints"))
ALLOWED_MAGNIFICATIONS = ("40", "100", "200", "400")
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

# Stage 3 only reports a subtype if its checkpoint actually cleared this
# validation macro F1. Random guessing over 4 classes scores ~0.25, and
# both subtype training attempts landed at 0.08-0.19 -- so without this
# gate the UI would render something like "Mucinous Carcinoma, 87%
# confidence" out of what is effectively a coin toss. Below the bar,
# /predict returns the benign/malignant result with subtype fields null
# and says why, rather than dressing up noise as a finding.
MIN_SUBTYPE_MACRO_F1 = float(os.environ.get("MIN_SUBTYPE_MACRO_F1", "0.55"))

# The bar has to sit well above what guessing scores, and that depends on the
# number of classes: random guessing gets ~0.25 macro F1 over 4 classes but
# ~0.50 over 2. A 2-class model at 0.55 would be barely better than a coin, so
# the ductal-vs-other scheme is held to a higher bar.
MIN_MACRO_F1_BY_SCHEME = {
    "four_subtypes": MIN_SUBTYPE_MACRO_F1,
    "ductal_vs_other": float(os.environ.get("MIN_DUCTAL_VS_OTHER_MACRO_F1", "0.70")),
}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Dict[str, Dict[str, int]]:
    return {"prediction_distribution": get_prediction_distribution()}


def _binary_checkpoint_for(magnification: str):
    """Return (checkpoint path, magnification of the model actually used).

    ``magnification`` has already been validated against
    ALLOWED_MAGNIFICATIONS, so it can't be used to reach an arbitrary path.
    """
    matched = CHECKPOINT_DIR / f"best_mag{magnification}.pt"
    if matched.is_file():
        return str(matched), magnification
    return CHECKPOINT_PATH, None


def _classify_subtype(tensor):
    """Stage 3. Returns (subtype, display_name, confidence) when a good
    enough model is available, a string explaining why not when there is a
    checkpoint but it didn't clear MIN_SUBTYPE_MACRO_F1, or None when no
    subtype checkpoint is deployed at all."""
    if not os.path.exists(SUBTYPE_CHECKPOINT_PATH):
        return None

    scheme = subtype_checkpoint_scheme(SUBTYPE_CHECKPOINT_PATH)
    names = SUBTYPE_SCHEMES[scheme]["names"]
    minimum = MIN_MACRO_F1_BY_SCHEME[scheme]
    chance = 1 / len(names)
    recorded_macro_f1 = subtype_checkpoint_macro_f1(SUBTYPE_CHECKPOINT_PATH)
    if recorded_macro_f1 < minimum:
        return (
            "Subtype classification is unavailable: the available model scores "
            f"{recorded_macro_f1:.2f} macro F1 on validation, below the {minimum:.2f} "
            f"minimum (guessing at random across {len(names)} classes scores about "
            f"{chance:.2f}). BreakHis has only 4-6 training patients for 3 of its 4 "
            "malignant subtypes, too few to learn a subtype classifier that "
            "generalizes to a patient it has never seen."
        )

    model = get_subtype_model(SUBTYPE_CHECKPOINT_PATH)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    idx = int(probs.argmax().item())
    name = names[idx]
    return name, SUBTYPE_SCHEMES[scheme]["display"][name], float(probs[idx].item())


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...), magnification: str = Form("40")
) -> PredictionResponse:
    if magnification not in ALLOWED_MAGNIFICATIONS:
        raise HTTPException(
            status_code=400,
            detail=f"magnification must be one of {', '.join(ALLOWED_MAGNIFICATIONS)}",
        )

    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as e:
        raise HTTPException(status_code=400, detail="File is not a valid image") from e

    checkpoint_path, model_magnification = _binary_checkpoint_for(magnification)
    if not os.path.exists(checkpoint_path):
        raise HTTPException(status_code=503, detail="Model checkpoint not available")

    if not looks_like_histology(image):
        raise HTTPException(
            status_code=422,
            detail="Invalid Image — Please upload a valid breast histology image.",
        )

    model = get_model(checkpoint_path)
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

    subtype = subtype_display_name = subtype_unavailable_reason = None
    subtype_confidence = None
    if label == "malignant":
        subtype_result = _classify_subtype(tensor)
        if isinstance(subtype_result, str):
            subtype_unavailable_reason = subtype_result
        elif subtype_result is not None:
            subtype, subtype_display_name, subtype_confidence = subtype_result

    return PredictionResponse(
        label=label,
        probability=probability,
        magnification=magnification,
        model_magnification=model_magnification,
        gradcam_overlay_base64=overlay_base64,
        subtype=subtype,
        subtype_display_name=subtype_display_name,
        subtype_confidence=subtype_confidence,
        subtype_unavailable_reason=subtype_unavailable_reason,
    )
