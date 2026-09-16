"""Request/response schemas for the inference API."""

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    label: str
    probability: float
    magnification: str
