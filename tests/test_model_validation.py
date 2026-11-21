"""Model validation gate: validation patients only.

Fails when image-level F1, sensitivity or specificity on the validation
patients drops below the floor, or when patient-level sensitivity does.
Skipped when the checkpoint or the dataset is missing. Set
``CI_REQUIRE_MODEL_GATE=1`` to turn those skips into failures. The workflow
does that. A skip is not a passing gate.
"""

import os
from pathlib import Path

import pytest
import torch
import yaml

from src.data.dataset import BreakHisDataset
from src.data.manifest import patient_id_of, resolve_patient_split
from src.data.splits import filter_samples_by_patients
from src.data.transforms import eval_transform
from src.models.classifier import BreakHisClassifier
from src.training.checkpoint import load_checkpoint
from src.training.loop import evaluate
from tests.gateutil import gate_missing

MIN_F1 = 0.75
MIN_SENSITIVITY = 0.85
MIN_SPECIFICITY = 0.50
MIN_PATIENT_SENSITIVITY = 0.85
# A real checkpoint is ~95MB. A Git LFS pointer is a few hundred bytes.
_MIN_REAL_CHECKPOINT_BYTES = 1_000_000

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _checkpoint_path() -> str:
    return os.environ.get("VALIDATION_CHECKPOINT_PATH", "serving_checkpoints/best_mag40.pt")


def _data_root() -> Path:
    return Path(os.environ.get("VALIDATION_DATA_ROOT", "data/BreaKHis_v1"))


def _magnification(checkpoint: str) -> str:
    override = os.environ.get("VALIDATION_MAGNIFICATION")
    if override:
        return override
    name = Path(checkpoint).name
    for magnification in ("400", "200", "100", "40"):
        if magnification in name:
            return magnification
    return "40"


def _data_config() -> dict:
    return yaml.safe_load((_REPO_ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))


def _checkpoint_reason(path: str) -> str | None:
    if not os.path.exists(path) or os.path.getsize(path) < _MIN_REAL_CHECKPOINT_BYTES:
        return (
            "No trained checkpoint available to validate "
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


def validation_samples(root: Path, magnification: str) -> list:
    """Images whose patient is in the validation split, and that split only."""
    config = _data_config()
    ratios_cfg = config["split_ratios"]
    ratios = (ratios_cfg["train"], ratios_cfg["val"], ratios_cfg["test"])
    full = BreakHisDataset(root, magnification=magnification)
    train_patients, val_patients, _ = resolve_patient_split(
        full.samples,
        ratios=ratios,
        seed=int(config["seed"]),
        manifest_key=("binary", str(magnification)),
    )
    if not val_patients.isdisjoint(train_patients):
        raise AssertionError("validation patients overlap the training patients")
    chosen = filter_samples_by_patients(full.samples, val_patients)
    if not chosen:
        raise AssertionError(f"validation split for magnification {magnification} has no images")
    return chosen


def test_checkpoint_meets_minimum_f1_threshold():
    checkpoint = _checkpoint_path()
    missing_checkpoint = _checkpoint_reason(checkpoint)
    if missing_checkpoint:
        gate_missing(missing_checkpoint)
    root = _data_root()
    missing_data = _dataset_reason(root)
    if missing_data:
        gate_missing(missing_data)

    magnification = _magnification(checkpoint)
    model = BreakHisClassifier(pretrained=False)
    load_checkpoint(checkpoint, model)
    model.eval()

    dataset = BreakHisDataset(root, magnification=magnification, transform=eval_transform())
    dataset.samples = validation_samples(root, magnification)
    patient_ids = [patient_id_of(sample) for sample in dataset.samples]
    metrics = evaluate(
        model,
        torch.utils.data.DataLoader(dataset, batch_size=32),
        torch.nn.BCEWithLogitsLoss(),
        "cpu",
        patient_ids=patient_ids,
    )

    assert metrics["f1"] >= MIN_F1, f"val image F1 {metrics['f1']:.3f} below {MIN_F1}"
    assert metrics["sensitivity"] >= MIN_SENSITIVITY, (
        f"val image sensitivity {metrics['sensitivity']:.3f} below {MIN_SENSITIVITY}"
    )
    assert metrics["specificity"] >= MIN_SPECIFICITY, (
        f"val image specificity {metrics['specificity']:.3f} below {MIN_SPECIFICITY}"
    )
    assert metrics["patient_sensitivity"] >= MIN_PATIENT_SENSITIVITY, (
        f"val patient sensitivity {metrics['patient_sensitivity']:.3f} below {MIN_PATIENT_SENSITIVITY}"
    )


def test_binary_gate_rejects_high_f1_when_sensitivity_is_low(monkeypatch):
    """Image F1 alone must not pass the gate."""
    monkeypatch.setattr("tests.test_model_validation._checkpoint_reason", lambda path: None)
    monkeypatch.setattr("tests.test_model_validation._dataset_reason", lambda root: None)
    monkeypatch.setattr("tests.test_model_validation.load_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "tests.test_model_validation.validation_samples",
        lambda root, magnification: [{"path": Path("root/patient/40X/a.png"), "label": 1}],
    )
    monkeypatch.setattr(
        "tests.test_model_validation.evaluate",
        lambda *args, **kwargs: {
            "f1": 0.99,
            "sensitivity": 0.10,
            "specificity": 0.95,
            "patient_sensitivity": 0.99,
        },
    )

    with pytest.raises(AssertionError, match="sensitivity"):
        test_checkpoint_meets_minimum_f1_threshold()


def test_validation_samples_exclude_training_patients(breakhis_root_multi_patient):
    chosen = validation_samples(breakhis_root_multi_patient, "40")
    full = BreakHisDataset(breakhis_root_multi_patient, magnification="40")
    config = _data_config()
    ratios_cfg = config["split_ratios"]
    train, val, _ = resolve_patient_split(
        full.samples,
        ratios=(ratios_cfg["train"], ratios_cfg["val"], ratios_cfg["test"]),
        seed=int(config["seed"]),
        manifest_key=("binary", "40"),
    )
    chosen_patients = {patient_id_of(sample) for sample in chosen}
    assert chosen_patients == val
    assert chosen_patients.isdisjoint(train)


def test_ci_require_model_gate_fails_when_the_checkpoint_is_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("CI_REQUIRE_MODEL_GATE", "1")
    monkeypatch.setenv("VALIDATION_CHECKPOINT_PATH", str(tmp_path / "missing.pt"))
    try:
        test_checkpoint_meets_minimum_f1_threshold()
    except BaseException as exc:
        assert exc.__class__.__name__ == "Failed"
        assert "No trained checkpoint" in str(exc)
    else:
        raise AssertionError("a missing checkpoint must fail the gate when CI_REQUIRE_MODEL_GATE=1")


def test_missing_data_skips_outside_ci(monkeypatch, tmp_path):
    monkeypatch.delenv("CI_REQUIRE_MODEL_GATE", raising=False)
    checkpoint = tmp_path / "best_mag40.pt"
    checkpoint.write_bytes(b"x" * _MIN_REAL_CHECKPOINT_BYTES)
    monkeypatch.setenv("VALIDATION_CHECKPOINT_PATH", str(checkpoint))
    monkeypatch.setenv("VALIDATION_DATA_ROOT", str(tmp_path / "no-dataset"))
    with pytest.raises(BaseException, match="BreakHis data is not available") as caught:
        test_checkpoint_meets_minimum_f1_threshold()
    assert caught.value.__class__.__name__ == "Skipped"
