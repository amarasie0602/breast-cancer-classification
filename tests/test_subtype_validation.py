"""Model validation gate for the subtype classifier.

The score is recomputed on validation patients. The macro F1 stored inside
the checkpoint is not the pass/fail decision. Missing weights or missing
data skip locally and fail when ``CI_REQUIRE_MODEL_GATE=1``.
"""

import os
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image
from torch import optim

from src.data.dataset import SUBTYPE_NAMES, SUBTYPE_SCHEMES, BreakHisSubtypeDataset
from src.data.manifest import patient_id_of, resolve_patient_split
from src.data.splits import filter_samples_by_patients
from src.data.transforms import eval_transform
from src.models.classifier import MalignantSubtypeClassifier
from src.serving.app import MIN_SUBTYPE_MACRO_F1
from src.serving.model_loader import subtype_checkpoint_macro_f1, subtype_checkpoint_scheme
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.subtype_loop import evaluate
from tests.gateutil import gate_missing

_MIN_REAL_CHECKPOINT_BYTES = 1_000_000
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _checkpoint_path() -> str:
    return os.environ.get("VALIDATION_SUBTYPE_CHECKPOINT_PATH", "serving_checkpoints/best_subtype.pt")


def _data_root() -> Path:
    return Path(os.environ.get("VALIDATION_DATA_ROOT", "data/BreaKHis_v1"))


def _data_config() -> dict:
    return yaml.safe_load((_REPO_ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))


def _checkpoint_reason(path: str) -> str | None:
    if not os.path.exists(path) or os.path.getsize(path) < _MIN_REAL_CHECKPOINT_BYTES:
        return (
            "No trained subtype checkpoint available to validate "
            f"({path} is missing or is a Git LFS pointer). Check out with git lfs pull."
        )
    return None


def _dataset_reason(root: Path) -> str | None:
    slides = root / "histology_slides" / "breast"
    if not slides.is_dir() or not any(slides.rglob("*.png")):
        return (
            "BreakHis data is not available to validate "
            f"({root} has no histology PNGs). Run dvc pull after the Drive remote is configured."
        )
    return None


def test_subtype_checkpoint_meets_minimum_macro_f1_threshold():
    checkpoint = _checkpoint_path()
    missing_checkpoint = _checkpoint_reason(checkpoint)
    if missing_checkpoint:
        gate_missing(missing_checkpoint)
    root = _data_root()
    missing_data = _dataset_reason(root)
    if missing_data:
        gate_missing(missing_data)

    scheme = subtype_checkpoint_scheme(checkpoint)
    class_names = SUBTYPE_SCHEMES[scheme]["names"]
    config = _data_config()
    ratios_cfg = config["split_ratios"]
    ratios = (ratios_cfg["train"], ratios_cfg["val"], ratios_cfg["test"])

    full = BreakHisSubtypeDataset(root, scheme=scheme)
    _, val_patients, _ = resolve_patient_split(
        full.samples,
        ratios=ratios,
        seed=int(config["seed"]),
        manifest_key=("subtype", scheme),
        min_per_split=1,
    )
    dataset = BreakHisSubtypeDataset(root, transform=eval_transform(), scheme=scheme)
    dataset.samples = filter_samples_by_patients(dataset.samples, val_patients)
    if not dataset.samples:
        gate_missing(f"Subtype validation split for {scheme} has no images at {root}.")

    model = MalignantSubtypeClassifier(num_classes=len(class_names), pretrained=False)
    load_checkpoint(checkpoint, model)
    model.eval()
    patient_ids = [patient_id_of(sample) for sample in dataset.samples]
    metrics = evaluate(
        model,
        torch.utils.data.DataLoader(dataset, batch_size=32),
        torch.nn.CrossEntropyLoss(),
        "cpu",
        num_classes=len(class_names),
        patient_ids=patient_ids,
    )
    # Recomputed from the validation images. The number written into the
    # checkpoint by training is reported and then ignored.
    stored = subtype_checkpoint_macro_f1(checkpoint)
    recomputed = metrics["macro_f1"]
    patient_f1 = metrics["patient_macro_f1"]
    assert recomputed >= MIN_SUBTYPE_MACRO_F1, (
        f"recomputed validation macro F1 is {recomputed:.3f} "
        f"(patient-level {patient_f1:.3f}), below {MIN_SUBTYPE_MACRO_F1:.2f}. "
        f"The checkpoint file records {stored:.3f}, which is not the decision."
    )
    assert patient_f1 >= MIN_SUBTYPE_MACRO_F1, (
        f"patient-level validation macro F1 is {patient_f1:.3f}, below {MIN_SUBTYPE_MACRO_F1:.2f}"
    )


def test_subtype_gate_does_not_trust_the_stored_macro_f1(tmp_path, monkeypatch):
    """A checkpoint that claims 0.99 still fails when validation patients recompute below the bar."""
    root = tmp_path / "data"
    for index, subtype in enumerate(SUBTYPE_NAMES):
        for patient_index in range(6):
            directory = (
                root / "histology_slides" / "breast" / "malignant" / "SOB" / subtype
                / f"P{index}-{patient_index}" / "40X"
            )
            directory.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(directory / "a.png")

    model = MalignantSubtypeClassifier(num_classes=4, pretrained=False)
    with torch.no_grad():
        head = model.backbone.classifier[-1]
        head.weight.zero_()
        head.bias.zero_()
        head.bias[0] = 50.0
    checkpoint = tmp_path / "best_subtype.pt"
    save_checkpoint(
        checkpoint,
        model,
        optim.Adam(model.parameters()),
        epoch=0,
        metrics={"macro_f1": 0.99, "label_scheme": "four_subtypes"},
    )
    monkeypatch.setenv("VALIDATION_SUBTYPE_CHECKPOINT_PATH", str(checkpoint))
    monkeypatch.setenv("VALIDATION_DATA_ROOT", str(root))

    with pytest.raises(AssertionError, match="recomputed validation macro F1"):
        test_subtype_checkpoint_meets_minimum_macro_f1_threshold()


def test_ci_require_model_gate_fails_when_the_subtype_checkpoint_is_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("CI_REQUIRE_MODEL_GATE", "1")
    monkeypatch.setenv("VALIDATION_SUBTYPE_CHECKPOINT_PATH", str(tmp_path / "missing.pt"))
    try:
        test_subtype_checkpoint_meets_minimum_macro_f1_threshold()
    except BaseException as exc:
        assert exc.__class__.__name__ == "Failed"
        assert "No trained subtype checkpoint" in str(exc)
    else:
        raise AssertionError("a missing subtype checkpoint must fail the gate in CI")
