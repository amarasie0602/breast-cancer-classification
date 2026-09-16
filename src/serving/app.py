"""FastAPI inference service for the BreakHis classifier."""

from fastapi import FastAPI

app = FastAPI(title="Breast Cancer Histopathology Classifier")


@app.get("/health")
def health():
    return {"status": "ok"}
