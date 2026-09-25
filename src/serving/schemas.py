"""Request/response schemas for the inference API."""

from typing import Optional

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    label: str  # "benign" or "malignant"
    probability: float  # malignancy probability, 0-1
    magnification: str
    # Magnification of the model that actually ran, or None when no
    # magnification-matched checkpoint was available and serving fell back to
    # its default model. Lets the UI say plainly when those differ.
    model_magnification: Optional[str] = None
    gradcam_overlay_base64: str
    subtype: Optional[str] = None  # e.g. "ductal_carcinoma"; only set when label == "malignant"
    subtype_display_name: Optional[str] = None  # e.g. "Invasive Ductal Carcinoma (IDC)"
    subtype_confidence: Optional[float] = None
    # Set when the malignant branch ran but no trustworthy subtype model
    # was available, so the UI can explain the absence instead of just
    # silently omitting stage 3.
    subtype_unavailable_reason: Optional[str] = None
