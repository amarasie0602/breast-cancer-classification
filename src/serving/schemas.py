"""Request/response schemas for the inference API."""


from pydantic import BaseModel


class PredictionResponse(BaseModel):
    label: str  # "benign" or "malignant"
    probability: float  # malignancy probability, 0-1
    # True when the probability is in the band where the models are right only
    # about two times in three (see UNCERTAIN_BAND in app.py); the label is
    # still the side of the threshold it falls on.
    uncertain: bool = False
    magnification: str  # the magnification whose model was used
    # "detected" when the magnification came from the image, "selected" when
    # the request named one.
    magnification_source: str = "selected"
    detected_magnification: str | None = None
    detected_magnification_confidence: float | None = None
    # Set when the detected magnification is unsure, or confidently disagrees
    # with the one selected.
    magnification_warning: str | None = None
    # Magnification of the model that actually ran, or None when no
    # magnification-matched checkpoint was available and serving fell back to
    # its default model. Lets the UI say plainly when those differ.
    model_magnification: str | None = None
    gradcam_overlay_base64: str
    subtype: str | None = None  # e.g. "ductal_carcinoma"; only set when label == "malignant"
    subtype_display_name: str | None = None  # e.g. "Invasive Ductal Carcinoma (IDC)"
    subtype_confidence: float | None = None
    # Set when the malignant branch ran but no trustworthy subtype model
    # was available, so the UI can explain the absence instead of just
    # silently omitting stage 3.
    subtype_unavailable_reason: str | None = None
