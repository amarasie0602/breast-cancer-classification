"""Request/response schemas for the inference API."""

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    label: str
    probability: float
    magnification: str
    gradcam_overlay_base64: str
