# Breast Cancer MLOps Pipeline

[![CI](https://github.com/amarasie0602/breast-cancer-diagosis/actions/workflows/ci.yml/badge.svg)](https://github.com/amarasie0602/breast-cancer-diagosis/actions/workflows/ci.yml)
[![CD](https://github.com/amarasie0602/breast-cancer-diagosis/actions/workflows/cd.yml/badge.svg)](https://github.com/amarasie0602/breast-cancer-diagosis/actions/workflows/cd.yml)

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

```bash
python -m venv .venv
.venv/Scripts/activate  # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt

# Place the BreakHis dataset at data/BreaKHis_v1/ (see data/README.md)

pytest -q                        # run the test suite
python -m src.training.train --data-root data/BreaKHis_v1 --magnification 40
python -m src.training.compare_runs   # compare val F1 across magnifications
uvicorn src.serving.app:app --reload  # run the API locally
```

Or via Docker:

```bash
docker compose up --build
```

## Results

Per-magnification comparison — the project's headline finding. ResNet50,
transfer learning, patient-level 70/15/15 split, held-out **test** set
(never used for training or checkpoint selection):

| Magnification | Test F1 | Test Accuracy | Test Precision | Test Recall |
| -------------- | ------- | -------------- | --------------- | ----------- |
| 200x           | 0.952   | 0.925          | 0.916            | 0.990       |
| 40x            | 0.901   | 0.848          | 0.838            | 0.974       |
| 400x           | 0.882   | 0.819          | 0.796            | 0.988       |
| 100x           | 0.861   | 0.796          | 0.841            | 0.882       |

Recall is consistently high (0.88-0.99) across magnifications, but the
ranking is noisy — it flips depending on whether you look at validation
or test metrics, since each patient-level split has only ~11-13 patients
per magnification. See [docs/model_card.md](docs/model_card.md#results)
for the full breakdown (including the validation-set numbers) and why
that instability matters more than which magnification "wins."

## CI/CD

**CI** (`.github/workflows/ci.yml`) runs on every push/PR: lint (ruff),
the full pytest suite, a model-validation job that gates on minimum F1
(skips gracefully if no checkpoint is present, but runs for real and
enforces the threshold once one exists — see Results below), a smoke-train
job that runs the actual training CLI end-to-end on a tiny synthetic
dataset, and a Docker build check to catch container-breaking changes
before merge.

**CD** (`.github/workflows/cd.yml`) builds the serving image and pushes it to
GitHub Container Registry (`ghcr.io/<owner>/breast-cancer-classifier`) on
merge to `main` — no external account needed. The final deploy step is
gated on an optional `DEPLOY_HOOK_URL` repository secret; add a deploy-hook
URL from Render, Railway, or a similar free-tier host to enable automatic
deployment. Without it, the step is a no-op and the image can still be run
locally with `docker compose up --build` or `docker run` against the
pushed ghcr.io image.

## Limitations

See [docs/model_card.md](docs/model_card.md) for the full breakdown. In
short: this is a coursework/portfolio project, not a clinical tool — trained
on a small single-institution dataset (82 patients), with known class and
subtype imbalance, and no external validation. Grad-CAM overlays are
illustrative, not proof the model attends to clinically meaningful features.
