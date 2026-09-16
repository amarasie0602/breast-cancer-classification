# Breast Cancer MLOps Pipeline

Binary (benign vs. malignant) breast cancer histopathology classifier trained on
the [BreakHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)
dataset, with an MLOps pipeline around it: experiment tracking, data/model
versioning, automated testing, containerized serving, and CI/CD.

BreakHis provides each sample at four magnification levels (40x, 100x, 200x,
400x). This project evaluates the model per-magnification to compare how
classification performance varies with zoom level.

## Stack

- PyTorch (transfer learning backbone), Grad-CAM for explainability
- MLflow for experiment tracking
- DVC for dataset/model versioning
- pytest for testing
- FastAPI + Docker for serving
- GitHub Actions for CI/CD

## Setup

Instructions will be added as the pipeline components land.
