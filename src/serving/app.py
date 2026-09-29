"""FastAPI inference service for the BreakHis classifier."""

import base64
import io
import os
from pathlib import Path
from typing import Annotated, Dict

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from src.data.dataset import SUBTYPE_SCHEMES
from src.data.stain import MACENKO, StainNormalize
from src.data.transforms import eval_transform
from src.explainability.gradcam import GradCAM
from src.explainability.overlay import cam_to_overlay
from src.serving.input_guard import looks_like_histology
from src.serving.logging_middleware import RequestLoggingMiddleware
from src.serving.metrics import get_prediction_distribution, record_prediction
from src.serving.ood import load_histology_screen
from src.serving.model_loader import (
    binary_checkpoint_stain_normalization,
    get_model,
    get_subtype_model,
    subtype_checkpoint_macro_f1,
    subtype_checkpoint_scheme,
)
from src.serving.schemas import PredictionResponse

app = FastAPI(title="Breast Cancer Histopathology Classifier")
app.add_middleware(RequestLoggingMiddleware)

# The app serves the weights-only copies in serving_checkpoints/, both locally
# and in the image, so what runs here is exactly what ships. They are written
# from the training checkpoints in checkpoints/ by
# `python -m scripts.export_serving_checkpoints`; re-run it after retraining.
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "serving_checkpoints/best_mag40.pt")

# One binary model is trained per magnification, and they are not
# interchangeable: on the held-out test set the 40x model's specificity is
# 0.544 while the 200x model's is 0.732. Serving always used the 40x model no
# matter what magnification the user selected, so /predict now picks
# best_mag{magnification}.pt from this directory when it exists and falls back
# to CHECKPOINT_PATH when it doesn't (e.g. a container that only ships the
# 40x checkpoint).
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "serving_checkpoints"))
ALLOWED_MAGNIFICATIONS = ("40", "100", "200", "400")
# Stage 1's second check (src/serving/ood.py): skipped if it hasn't been
# fitted, so the app still runs without it.
HISTOLOGY_SCREEN_PATH = os.environ.get(
    "HISTOLOGY_SCREEN_PATH", "serving_checkpoints/histology_screen.pt"
)
SUBTYPE_CHECKPOINT_PATH = os.environ.get(
    "SUBTYPE_CHECKPOINT_PATH", "serving_checkpoints/best_subtype.pt"
)
STATIC_DIR = Path(__file__).parent / "static"

# BreakHis images are 700x460 (0.3 megapixels). Much larger uploads are refused
# before decoding: Pillow only warns below ~179M pixels, and decoding an image
# that size allocates gigabytes. The byte limit stops an oversized upload from
# being read into memory at all.
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
MAX_IMAGE_PIXELS = 40_000_000

# Operating point for benign/malignant. 0.5 is where sigmoid happens to
# cross, not a chosen threshold: at 0.5 this model runs at sensitivity
# 0.88-0.99 but specificity 0.46-0.73 (see docs/model_card.md). Raising it
# trades caught cancers for fewer false alarms. Deliberately left at the
# documented default rather than silently tuned, since where it belongs is
# a clinical cost judgement; `python -m scripts.tune_threshold` prints the
# whole curve to inform it.
DECISION_THRESHOLD = float(os.environ.get("DECISION_THRESHOLD", "0.5"))

# Probabilities in this band are reported as uncertain. Measured on the 1,388
# validation images: inside 0.2-0.8 (15% of images) the served models are
# right 68% of the time, outside it 94%. A 52%-malignant result shown in the
# same red box as a 99% one reads as a diagnosis; it's closer to a coin toss.
UNCERTAIN_BAND = (
    float(os.environ.get("UNCERTAIN_LOW", "0.2")),
    float(os.environ.get("UNCERTAIN_HIGH", "0.8")),
)

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


def _decode_image(image_bytes: bytes) -> Image.Image:
    """Decode an upload to RGB, turning every way it can be unreadable into a 400."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise HTTPException(
                status_code=400,
                detail=f"Image is too large ({image.width}x{image.height} pixels)",
            )
        # Image.open only reads the header; a truncated or corrupt body only
        # fails here, when the pixels are actually decoded.
        return image.convert("RGB")
    except Image.DecompressionBombError as e:
        raise HTTPException(status_code=400, detail="Image is too large") from e
    except (OSError, SyntaxError, ValueError) as e:  # UnidentifiedImageError is an OSError
        raise HTTPException(status_code=400, detail="File is not a valid image") from e


def _is_histology(image: Image.Image) -> bool:
    """Stage 1: the cheap colour-and-texture screen, then (only if that passes)
    the check that the image resembles the training slides at all."""
    if not looks_like_histology(image):
        return False
    screen = load_histology_screen(HISTOLOGY_SCREEN_PATH)
    return screen is None or not screen.is_unfamiliar(image)


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

    try:
        scheme = subtype_checkpoint_scheme(SUBTYPE_CHECKPOINT_PATH)
    except Exception:  # noqa: BLE001 - any unreadable file means the same thing here
        # A corrupt file, a Git LFS pointer checked out without `lfs: true`,
        # or a checkpoint from an incompatible version. Stage 3 is optional,
        # so this must not turn a perfectly good stage-2 result into a 500.
        return (
            "Subtype classification is unavailable: the subtype model file could "
            "not be loaded."
        )
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


@app.post(
    "/predict",
    responses={
        400: {"description": "Unsupported magnification, or the file is not a usable image"},
        413: {"description": "The upload is larger than MAX_UPLOAD_BYTES"},
        422: {"description": "Not a breast histology image (stage 1 rejected it)"},
        503: {"description": "No usable model checkpoint is available to serve"},
    },
)
async def predict(
    file: Annotated[UploadFile, File()],
    magnification: Annotated[str, Form()] = "40",
) -> PredictionResponse:
    if magnification not in ALLOWED_MAGNIFICATIONS:
        raise HTTPException(
            status_code=400,
            detail=f"magnification must be one of {', '.join(ALLOWED_MAGNIFICATIONS)}",
        )

    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
        )
    image = _decode_image(image_bytes)

    checkpoint_path, model_magnification = _binary_checkpoint_for(magnification)
    if not os.path.exists(checkpoint_path):
        raise HTTPException(status_code=503, detail="Model checkpoint not available")

    if not _is_histology(image):
        raise HTTPException(
            status_code=422,
            detail="Invalid Image — Please upload a valid breast histology image.",
        )

    stain_normalization = binary_checkpoint_stain_normalization(checkpoint_path)
    if stain_normalization not in (None, MACENKO):
        raise HTTPException(
            status_code=503,
            detail=f"Model needs unsupported preprocessing: {stain_normalization}",
        )

    model = get_model(checkpoint_path)
    # A model trained on stain-normalized images has to see the upload
    # normalized the same way. The subtype model was trained on raw images, so
    # it keeps getting the raw tensor, and Grad-CAM is drawn over the original.
    raw_tensor = eval_transform()(image).unsqueeze(0)
    tensor = (
        eval_transform()(StainNormalize()(image)).unsqueeze(0)
        if stain_normalization == MACENKO
        else raw_tensor
    )

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
        subtype_result = _classify_subtype(raw_tensor)
        if isinstance(subtype_result, str):
            subtype_unavailable_reason = subtype_result
        elif subtype_result is not None:
            subtype, subtype_display_name, subtype_confidence = subtype_result

    return PredictionResponse(
        label=label,
        probability=probability,
        uncertain=UNCERTAIN_BAND[0] <= probability <= UNCERTAIN_BAND[1],
        magnification=magnification,
        model_magnification=model_magnification,
        gradcam_overlay_base64=overlay_base64,
        subtype=subtype,
        subtype_display_name=subtype_display_name,
        subtype_confidence=subtype_confidence,
        subtype_unavailable_reason=subtype_unavailable_reason,
    )
