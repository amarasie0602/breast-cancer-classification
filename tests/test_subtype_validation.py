"""Model validation gate for the malignant-subtype classifier: fails if a
trained checkpoint's macro F1 drops below threshold.

Skipped when no checkpoint is present (e.g. CI without a trained model yet).
Mirrors tests/test_model_validation.py's binary-classifier gate.
"""

import os

import pytest
import torch

from src.data.dataset import SUBTYPE_NAMES, BreakHisSubtypeDataset
from src.data.transforms import eval_transform
from src.models.classifier import MalignantSubtypeClassifier
from src.training.checkpoint import load_checkpoint
from src.training.subtype_loop import evaluate

# Lower than the binary gate's 0.75: this is a much harder 4-class problem
# with real class imbalance (ductal_carcinoma outnumbers papillary_carcinoma
# ~6:1), so a lower macro-F1 bar is a realistic sanity check, not a rubber
# stamp -- see docs/model_card.md for the actual achieved numbers.
MIN_MACRO_F1 = 0.5
_MIN_REAL_CHECKPOINT_BYTES = 1_000_000

CHECKPOINT_PATH = os.environ.get("VALIDATION_SUBTYPE_CHECKPOINT_PATH", "checkpoints/best_subtype.pt")
VAL_DATA_ROOT = os.environ.get("VALIDATION_DATA_ROOT", "data/BreaKHis_v1")


def _checkpoint_available() -> bool:
    return (
        os.path.exists(CHECKPOINT_PATH)
        and os.path.getsize(CHECKPOINT_PATH) >= _MIN_REAL_CHECKPOINT_BYTES
    )


@pytest.mark.skipif(
    not _checkpoint_available(), reason="No trained subtype checkpoint available to validate"
)
def test_subtype_checkpoint_meets_minimum_macro_f1_threshold():
    model = MalignantSubtypeClassifier(num_classes=len(SUBTYPE_NAMES), pretrained=False)
    load_checkpoint(CHECKPOINT_PATH, model)
    model.eval()

    ds = BreakHisSubtypeDataset(VAL_DATA_ROOT, transform=eval_transform())
    dataloader = torch.utils.data.DataLoader(ds, batch_size=32)

    metrics = evaluate(
        model, dataloader, torch.nn.CrossEntropyLoss(), "cpu", num_classes=len(SUBTYPE_NAMES)
    )

    assert metrics["macro_f1"] >= MIN_MACRO_F1, (
        f"subtype macro F1 {metrics['macro_f1']:.3f} below threshold {MIN_MACRO_F1}"
    )
