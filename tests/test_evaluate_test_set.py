import torch
from torch import optim

from scripts.evaluate_test_set import evaluate_binary_checkpoint, evaluate_subtype_checkpoint
from src.data.dataset import SUBTYPE_NAMES
from src.models.classifier import BreakHisClassifier, MalignantSubtypeClassifier
from src.training.checkpoint import save_checkpoint


def test_evaluate_binary_checkpoint_reports_full_metric_set(
    breakhis_root_multi_patient, tmp_path
):
    checkpoint = tmp_path / "binary.pt"
    model = BreakHisClassifier(pretrained=False)
    save_checkpoint(checkpoint, model, optim.Adam(model.parameters()), epoch=0, metrics={})

    metrics = evaluate_binary_checkpoint(
        checkpoint,
        breakhis_root_multi_patient,
        magnification="40",
        ratios=(0.5, 0.25, 0.25),
        seed=42,
        batch_size=2,
    )

    for key in ("accuracy", "precision", "recall", "f1", "sensitivity", "specificity"):
        assert 0.0 <= metrics[key] <= 1.0
    assert metrics["confusion_matrix"].shape == (2, 2)


def test_evaluate_subtype_checkpoint_reports_per_class_metrics(
    breakhis_root_all_subtypes, tmp_path
):
    checkpoint = tmp_path / "subtype.pt"
    model = MalignantSubtypeClassifier(num_classes=len(SUBTYPE_NAMES), pretrained=False)
    save_checkpoint(checkpoint, model, optim.Adam(model.parameters()), epoch=0, metrics={})

    metrics = evaluate_subtype_checkpoint(
        checkpoint,
        breakhis_root_all_subtypes,
        ratios=(0.5, 0.25, 0.25),
        seed=42,
        batch_size=2,
    )

    assert len(metrics["per_class_f1"]) == len(SUBTYPE_NAMES)
    assert metrics["confusion_matrix"].shape == (len(SUBTYPE_NAMES), len(SUBTYPE_NAMES))
    assert torch.is_tensor(metrics["confusion_matrix"])
