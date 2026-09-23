"""Request/response schemas for the inference API."""

from typing import Optional

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    label: str  # "benign" or "malignant"
    probability: float  # malignancy probability, 0-1
    magnification: str
    gradcam_overlay_base64: str
    subtype: Optional[str] = None  # e.g. "ductal_carcinoma"; only set when label == "malignant"
    subtype_display_name: Optional[str] = None  # e.g. "Invasive Ductal Carcinoma (IDC)"
    subtype_confidence: Optional[float] = None
