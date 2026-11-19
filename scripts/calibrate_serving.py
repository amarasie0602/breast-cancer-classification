"""Decide test-time augmentation and fit probability calibration for serving.

Two changes to how the served binary models turn an image into a probability,
both decided on the validation patients and only reported on the held-out
test patients:

1. Test-time augmentation (TTA): score the image in its 8 rotations and
   flips (tissue has no "up") and average the logits. Adopt it only when
   the validation-patient utility says so: each magnification contributes
   one patient-level sensitivity and specificity, those four are averaged,
   and TTA wins when ``selection_rank`` is higher than the plain view.
   The rank clears a sensitivity floor and then maximises specificity.
   The held-out test set is printed afterwards and is not an input.

2. Temperature scaling: one temperature T per magnification, fitted on
   validation logits (minimising log loss), so that the probability shown
   matches how often the model is right. sigmoid(logit / T) keeps the sign
   of the logit, so no label at the 0.5 threshold changes.

Writes serving_checkpoints/calibration.json, which serving reads. Each entry
records the SHA-256 of the checkpoint it was fitted for, since a temperature
is only valid for that exact model.

    python -m scripts.calibrate_serving
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import BreakHisDataset
from src.data.manifest import patient_id_of, resolve_patient_split
from src.data.splits import filter_samples_by_patients
from src.data.transforms import eval_transform
from src.serving.calibration import dihedral_views
from src.serving.model_loader import get_model
from src.training.patient_metrics import (
    mean_rates,
    patient_sensitivity_specificity,
    prefer_alternative,
    sensitivity_floor_from_config,
)
from src.training.train import NUM_WORKERS, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "BreaKHis_v1"
SERVING_DIR = REPO_ROOT / "serving_checkpoints"
OUTPUT = SERVING_DIR / "calibration.json"
MAGNIFICATIONS = ("40", "100", "200", "400")
# Logits are cached so re-running the decision doesn't rescore every image;
# the cache is keyed by the checkpoints' hashes.
LOGIT_CACHE = REPO_ROOT / "checkpoints" / "experiments" / "calibration_logits.npz"
UNCERTAIN_BAND = (0.2, 0.8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def logits(model, dataset):
    """Plain and TTA-averaged logits, labels, and patient ids in dataset order."""
    patient_ids = [patient_id_of(sample) for sample in dataset.samples]
    plain, averaged, labels = [], [], []
    for images, y in DataLoader(dataset, batch_size=16, num_workers=NUM_WORKERS):
        views = dihedral_views(images)  # (8, B, C, H, W)
        out = model(views.flatten(0, 1)).view(len(views), len(images))
        plain.append(out[0])
        averaged.append(out.mean(0))
        labels.append(y)
    labels_np = torch.cat(labels).numpy()
    if len(labels_np) != len(patient_ids):
        raise RuntimeError("logit dump and patient ids fell out of order; the loader must not shuffle")
    return (
        torch.cat(plain).numpy(),
        torch.cat(averaged).numpy(),
        labels_np,
        np.asarray(patient_ids),
    )


def fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    """Temperature minimising validation log loss (1-D search on log T)."""
    zt, yt = torch.tensor(z, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=1.0, max_iter=200)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(zt / log_t.exp(), yt)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp())


def rates(z: np.ndarray, y: np.ndarray) -> dict:
    pred = z >= 0  # sigmoid >= 0.5
    sens = float((pred & (y == 1)).sum() / max((y == 1).sum(), 1))
    spec = float((~pred & (y == 0)).sum() / max((y == 0).sum(), 1))
    return {"sensitivity": sens, "specificity": spec, "balanced_accuracy": (sens + spec) / 2}


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error of the predicted class's confidence."""
    confidence = np.maximum(p, 1 - p)
    correct = (p >= 0.5) == (y == 1)
    edges = np.linspace(0.5, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (confidence >= lo) & ((confidence < hi) | np.isclose(hi, 1.0))
        if in_bin.any():
            total += in_bin.mean() * abs(confidence[in_bin].mean() - correct[in_bin].mean())
    return float(total)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _load_cached_logits(hashes: dict) -> dict:
    if not LOGIT_CACHE.is_file():
        return {}
    data = np.load(LOGIT_CACHE)
    cached = {}
    for m in MAGNIFICATIONS:
        patients_cached = f"{m}_val_patients" in data and f"{m}_test_patients" in data
        if f"{m}_sha256" in data and str(data[f"{m}_sha256"]) == hashes[m] and patients_cached:
            cached[m] = {
                split: tuple(data[f"{m}_{split}_{part}"] for part in ("plain", "tta", "labels", "patients"))
                for split in ("val", "test")
            }
    return cached


def _save_cached_logits(per_mag: dict, hashes: dict) -> None:
    arrays = {}
    for m, splits in per_mag.items():
        arrays[f"{m}_sha256"] = np.array(hashes[m])
        for split, parts in splits.items():
            for part, values in zip(("plain", "tta", "labels", "patients"), parts, strict=True):
                arrays[f"{m}_{split}_{part}"] = values
    LOGIT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(LOGIT_CACHE, **arrays)


def main() -> None:
    config = load_config(REPO_ROOT / "configs" / "data.yaml")
    r = config["split_ratios"]
    hashes = {m: sha256(SERVING_DIR / f"best_mag{m}.pt") for m in MAGNIFICATIONS}
    per_mag = _load_cached_logits(hashes)
    for magnification in MAGNIFICATIONS:
        if magnification in per_mag:
            continue
        ds = BreakHisDataset(DATA_ROOT, magnification=magnification)
        _, val, test = resolve_patient_split(
            ds.samples,
            ratios=(r["train"], r["val"], r["test"]),
            seed=config["seed"],
            manifest_key=("binary", magnification),
        )
        model = get_model(str(SERVING_DIR / f"best_mag{magnification}.pt"))
        splits = {}
        for name, patients in (("val", val), ("test", test)):
            split_ds = BreakHisDataset(DATA_ROOT, magnification=magnification, transform=eval_transform())
            split_ds.samples = filter_samples_by_patients(split_ds.samples, patients)
            splits[name] = logits(model, split_ds)
        per_mag[magnification] = splits
        counts = f"{len(splits['val'][2])} val, {len(splits['test'][2])} test"
        print(f"{magnification}x: scored {counts}", flush=True)
    _save_cached_logits(per_mag, hashes)

    def pooled(split, which):  # which: 0 = plain, 1 = TTA. Report only; not the TTA decision.
        z = np.concatenate([per_mag[m][split][which] for m in MAGNIFICATIONS])
        y = np.concatenate([per_mag[m][split][2] for m in MAGNIFICATIONS])
        return z, y

    def patient_rates(split, which):
        """One (sensitivity, specificity) pair per magnification, then an unweighted mean."""
        per_magnification = []
        for magnification in MAGNIFICATIONS:
            logits_m = per_mag[magnification][split][which]
            labels = per_mag[magnification][split][2]
            patients = per_mag[magnification][split][3]
            per_magnification.append(patient_sensitivity_specificity(patients, logits_m >= 0, labels))
        return mean_rates(per_magnification)

    # Validation patients only. Test logits are scored later as a report.
    floor = sensitivity_floor_from_config(load_config(REPO_ROOT / "configs" / "train.yaml"))
    plain_patient = patient_rates("val", 0)
    tta_patient = patient_rates("val", 1)
    use_tta = prefer_alternative(plain_patient, tta_patient, floor)
    print(
        f"\nvalidation patients, plain sensitivity/specificity: {plain_patient}\n"
        f"validation patients, TTA:                          {tta_patient}\n"
        f"sensitivity floor {floor:.2f} -> use TTA: {use_tta}"
    )
    which = 1 if use_tta else 0

    calibration = {"tta": use_tta, "models": {}}
    print("\ncalibration (expected calibration error, lower is better):")
    for magnification in MAGNIFICATIONS:
        z_val, y_val = per_mag[magnification]["val"][which], per_mag[magnification]["val"][2]
        temperature = fit_temperature(z_val, y_val)
        z_test, y_test = per_mag[magnification]["test"][which], per_mag[magnification]["test"][2]
        print(
            f"  {magnification}x: T = {temperature:.2f}; validation ECE {ece(sigmoid(z_val), y_val):.3f} -> "
            f"{ece(sigmoid(z_val / temperature), y_val):.3f}; test ECE {ece(sigmoid(z_test), y_test):.3f} -> "
            f"{ece(sigmoid(z_test / temperature), y_test):.3f}"
        )
        checkpoint = SERVING_DIR / f"best_mag{magnification}.pt"
        calibration["models"][checkpoint.name] = {"temperature": temperature, "sha256": hashes[magnification]}

    print("\nHeld-out test rates are a report. They do not choose TTA or the temperature.")
    for name, which_ in (("plain", 0), ("TTA", 1)):
        print(f"held-out test, {name}: {rates(*pooled('test', which_))}")

    print(f"\nuncertain band {UNCERTAIN_BAND} on calibrated probabilities:")
    for split in ("val", "test"):
        p = np.concatenate([
            sigmoid(per_mag[m][split][which] / calibration["models"][f"best_mag{m}.pt"]["temperature"])
            for m in MAGNIFICATIONS
        ])
        y = np.concatenate([per_mag[m][split][2] for m in MAGNIFICATIONS])
        band = (p >= UNCERTAIN_BAND[0]) & (p <= UNCERTAIN_BAND[1])
        correct = (p >= 0.5) == (y == 1)
        errors = ~correct
        print(
            f"  {split}: {band.mean():.1%} flagged; right {correct[band].mean():.0%} inside, "
            f"{correct[~band].mean():.1%} outside; {band[errors].sum()} of {errors.sum()} errors flagged"
        )

    OUTPUT.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUTPUT.name}")


if __name__ == "__main__":
    main()
