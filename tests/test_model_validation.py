"""Model validation gate: fails if a trained checkpoint's val F1 drops below threshold.

Skipped when no checkpoint is present (e.g. CI without a trained model yet).
This is the quality gate referenced by the CI pipeline's model-validation step.
"""

import os

import pytest
import torch

from src.data.dataset import BreakHisDataset
from src.data.transforms import eval_transform
from src.models.classifier import BreakHisClassifier
from src.training.checkpoint import load_checkpoint
from src.training.loop import evaluate

MIN_F1 = 0.75

CHECKPOINT_PATH = os.environ.get("VALIDATION_CHECKPOINT_PATH", "checkpoints/best_mag40.pt")
VAL_DATA_ROOT = os.environ.get("VALIDATION_DATA_ROOT", "data/BreaKHis_v1")


@pytest.mark.skipif(
    not os.path.exists(CHECKPOINT_PATH), reason="No trained checkpoint available to validate"
)
def test_checkpoint_meets_minimum_f1_threshold():
    model = BreakHisClassifier(pretrained=False)
    load_checkpoint(CHECKPOINT_PATH, model)
    model.eval()

    ds = BreakHisDataset(VAL_DATA_ROOT, magnification="40", transform=eval_transform())
    dataloader = torch.utils.data.DataLoader(ds, batch_size=32)

    metrics = evaluate(model, dataloader, torch.nn.BCEWithLogitsLoss(), "cpu")

    assert metrics["f1"] >= MIN_F1, f"val F1 {metrics['f1']:.3f} below threshold {MIN_F1}"
