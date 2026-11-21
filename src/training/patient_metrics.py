"""Patient-level scores for selection and reporting.

BreakHis patients are the unit that generalises. Image-level F1 lets a
patient with many similar tiles count many times, so checkpoint selection
and the test-time-augmentation decision use one vote per patient:

- a patient's label is malignant when at least half of that patient's
  images are malignant (BreakHis patients are single-label; this is the
  tie-safe reading of that label);
- a patient's prediction is malignant when at least half of that patient's
  images are predicted malignant.

Selection keeps a model only when it is better than the one already saved
under ``selection_rank``: clear the sensitivity floor first, then maximise
specificity. Image-level F1 stays a logged secondary metric.
"""

import math
from collections import defaultdict

import numpy as np

DEFAULT_SENSITIVITY_FLOOR = 0.95


def as_int_array(values) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values).astype(int).reshape(-1)


def patient_votes(patient_ids, preds, labels) -> tuple[list[str], np.ndarray, np.ndarray]:
    """One prediction and one label per patient, in first-seen order."""
    preds = as_int_array(preds)
    labels = as_int_array(labels)
    patient_ids = [str(patient_id) for patient_id in patient_ids]
    if not (len(patient_ids) == len(preds) == len(labels)):
        raise ValueError(
            f"patient_ids ({len(patient_ids)}), preds ({len(preds)}) and "
            f"labels ({len(labels)}) differ in length"
        )
    grouped: dict[str, dict[str, list[int]]] = {}
    order: list[str] = []
    for patient_id, pred, label in zip(patient_ids, preds.tolist(), labels.tolist(), strict=True):
        if patient_id not in grouped:
            grouped[patient_id] = {"preds": [], "labels": []}
            order.append(patient_id)
        grouped[patient_id]["preds"].append(int(pred))
        grouped[patient_id]["labels"].append(int(label))

    patient_preds = []
    patient_labels = []
    for patient_id in order:
        group = grouped[patient_id]
        patient_labels.append(1 if np.mean(group["labels"]) >= 0.5 else 0)
        patient_preds.append(1 if np.mean(group["preds"]) >= 0.5 else 0)
    return order, np.asarray(patient_preds, dtype=int), np.asarray(patient_labels, dtype=int)


def patient_sensitivity_specificity(patient_ids, preds, labels) -> tuple[float, float]:
    """Sensitivity and specificity across patients. A class with no patients is NaN."""
    _, patient_preds, patient_labels = patient_votes(patient_ids, preds, labels)
    positives = patient_labels == 1
    negatives = ~positives
    sensitivity = float("nan")
    specificity = float("nan")
    if positives.any():
        sensitivity = float(((patient_preds == 1) & positives).sum() / positives.sum())
    if negatives.any():
        specificity = float(((patient_preds == 0) & negatives).sum() / negatives.sum())
    return sensitivity, specificity


def patient_level_predictions(patient_ids, preds, labels, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Majority class per patient. Ties break toward the smaller class index."""
    preds = as_int_array(preds)
    labels = as_int_array(labels)
    patient_ids = [str(patient_id) for patient_id in patient_ids]
    if not (len(patient_ids) == len(preds) == len(labels)):
        raise ValueError("patient_ids, preds and labels differ in length")
    grouped: dict[str, dict[str, list[int]]] = {}
    order: list[str] = []
    for patient_id, pred, label in zip(patient_ids, preds.tolist(), labels.tolist(), strict=True):
        if patient_id not in grouped:
            grouped[patient_id] = {"preds": [], "labels": []}
            order.append(patient_id)
        grouped[patient_id]["preds"].append(int(pred))
        grouped[patient_id]["labels"].append(int(label))

    def _mode(values: list[int]) -> int:
        counts = np.bincount(np.asarray(values, dtype=int), minlength=num_classes)
        return int(counts.argmax())

    return (
        np.asarray([_mode(grouped[patient_id]["preds"]) for patient_id in order], dtype=int),
        np.asarray([_mode(grouped[patient_id]["labels"]) for patient_id in order], dtype=int),
    )


def selection_rank(
    sensitivity: float, specificity: float, sensitivity_floor: float
) -> tuple[int, float, float]:
    """Higher is better. Clearing the sensitivity floor outranks missing it.

    Among models that clear the floor, higher specificity wins and sensitivity
    is the tie-break. Among models that miss it, higher sensitivity wins, so
    training still keeps the least-bad epoch instead of saving nothing.
    """
    if not (np.isfinite(sensitivity) and np.isfinite(specificity)):
        return (0, -1.0, -1.0)
    if sensitivity + 1e-12 >= sensitivity_floor:
        return (1, float(specificity), float(sensitivity))
    return (0, float(sensitivity), float(specificity))


def sensitivity_floor_from_config(config: dict, default: float = DEFAULT_SENSITIVITY_FLOOR) -> float:
    selection = config.get("selection") or {}
    return float(selection.get("sensitivity_floor", default))


def mean_rates(per_magnification: list[tuple[float, float]]) -> tuple[float, float]:
    """Unweighted mean across magnifications. A magnification with more images does not count more."""
    if not per_magnification:
        raise ValueError("mean_rates received no magnifications")
    sensitivity = float(np.nanmean([sensitivity for sensitivity, _ in per_magnification]))
    specificity = float(np.nanmean([specificity for _, specificity in per_magnification]))
    return sensitivity, specificity


def prefer_alternative(
    baseline: tuple[float, float],
    alternative: tuple[float, float],
    sensitivity_floor: float,
) -> bool:
    """True when ``alternative`` outranks ``baseline`` on the selection utility."""
    return selection_rank(*alternative, sensitivity_floor) > selection_rank(*baseline, sensitivity_floor)


def is_logged_metric(key: str, value) -> bool:
    """Scalars MLflow can store. NaN is rejected by the client, and tensors are not metrics."""
    if key == "confusion_matrix" or key.startswith("per_class"):
        return False
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def bootstrap_patient_ci(
    patient_ids,
    preds,
    labels,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float] | float | int]:
    """95% interval for patient-level sensitivity and specificity.

    Each replicate draws patients with replacement. A patient drawn twice
    counts twice; grouping by the original id would collapse those draws.
    Replicates that lack a class are ignored for that class's interval.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    preds = as_int_array(preds)
    labels = as_int_array(labels)
    patient_ids = [str(patient_id) for patient_id in patient_ids]
    point_sens, point_spec = patient_sensitivity_specificity(patient_ids, preds, labels)

    groups: dict[str, list[int]] = defaultdict(list)
    for index, patient_id in enumerate(patient_ids):
        groups[patient_id].append(index)
    patients = np.asarray(sorted(groups))
    rng = np.random.default_rng(seed)
    sens_draws: list[float] = []
    spec_draws: list[float] = []
    for _ in range(n_bootstrap):
        drawn = rng.choice(patients, size=len(patients), replace=True)
        expanded_ids: list[str] = []
        expanded_preds: list[int] = []
        expanded_labels: list[int] = []
        for draw_index, patient_id in enumerate(drawn.tolist()):
            for index in groups[patient_id]:
                expanded_ids.append(f"{draw_index}:{patient_id}")
                expanded_preds.append(int(preds[index]))
                expanded_labels.append(int(labels[index]))
        sens, spec = patient_sensitivity_specificity(expanded_ids, expanded_preds, expanded_labels)
        if np.isfinite(sens):
            sens_draws.append(sens)
        if np.isfinite(spec):
            spec_draws.append(spec)

    def _interval(draws: list[float]) -> tuple[float, float]:
        if not draws:
            return (float("nan"), float("nan"))
        low = 100 * (alpha / 2)
        high = 100 * (1 - alpha / 2)
        return (float(np.percentile(draws, low)), float(np.percentile(draws, high)))

    return {
        "patient_sensitivity": point_sens,
        "patient_specificity": point_spec,
        "patient_sensitivity_ci95": _interval(sens_draws),
        "patient_specificity_ci95": _interval(spec_draws),
        "n_patients": len(patients),
        "n_bootstrap": n_bootstrap,
    }
