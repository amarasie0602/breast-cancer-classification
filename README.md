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

## Architecture

```
data/BreaKHis_v1/          Raw dataset (DVC-tracked, not in Git)
src/
  data/                    Dataset loader, patient-level splits, augmentation, EDA helpers
  models/                  ResNet50 transfer-learning classifier
  training/                Train/eval loop, metrics, checkpointing, MLflow logging, CLI
  explainability/          Grad-CAM, overlay rendering, MLflow artifact logging
  serving/                 FastAPI inference app (predict + Grad-CAM overlay in response)
tests/                     pytest suite (unit + integration + model validation gate)
configs/                   YAML configs for data splits and training hyperparameters
notebooks/                 EDA
docs/                      Model card
.github/workflows/         CI (lint, test, model-validation, Docker build) and CD
```

Training runs are tracked in MLflow (SQLite-backed locally); the dataset and
best model checkpoints are versioned with DVC. The API loads a checkpoint at
startup and returns both a benign/malignant prediction and a Grad-CAM overlay
showing which region of the image drove that prediction.

## Setup

Instructions will be added as the pipeline components land.
