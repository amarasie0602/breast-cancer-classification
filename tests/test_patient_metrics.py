import math

import numpy as np

from src.training.patient_metrics import (
    bootstrap_patient_ci,
    patient_level_predictions,
    patient_sensitivity_specificity,
    selection_rank,
)


def test_each_patient_casts_one_vote():
    # Patient A: 3 malignant images, 2 called malignant -> patient call malignant.
    # Patient B: 1 benign image called malignant -> patient call malignant (false positive).
    # Patient C: 2 benign images, both called benign -> true negative.
    ids = ["A", "A", "A", "B", "C", "C"]
    preds = [1, 1, 0, 1, 0, 0]
    labels = [1, 1, 1, 0, 0, 0]
    sensitivity, specificity = patient_sensitivity_specificity(ids, preds, labels)
    assert sensitivity == 1.0
    assert specificity == 0.5


def test_a_tie_inside_a_patient_counts_as_malignant():
    sensitivity, specificity = patient_sensitivity_specificity(
        ["A", "A"], preds=[1, 0], labels=[0, 0]
    )
    assert sensitivity != sensitivity  # no malignant patient
    assert specificity == 0.0


def test_mean_rates_weights_each_magnification_once():
    from src.training.patient_metrics import mean_rates

    sensitivity, specificity = mean_rates([(1.0, 0.0), (0.0, 1.0)])
    assert sensitivity == 0.5
    assert specificity == 0.5


def test_prefer_alternative_uses_the_selection_rank():
    from src.training.patient_metrics import prefer_alternative

    floor = 0.95
    # Higher specificity above the floor beats a higher-sensitivity model that is less specific.
    assert prefer_alternative((0.99, 0.40), (0.96, 0.80), floor)
    # Missing the floor loses to a model that clears it, even with lower specificity.
    assert not prefer_alternative((0.96, 0.40), (0.50, 0.99), floor)


def test_non_finite_metrics_are_not_logged():
    from src.training.patient_metrics import is_logged_metric

    assert is_logged_metric("f1", 0.9)
    assert not is_logged_metric("patient_specificity", float("nan"))
    assert not is_logged_metric("confusion_matrix", 1.0)
    assert not is_logged_metric("per_class_f1", [0.1, 0.2])


def test_selection_rank_clears_the_sensitivity_floor_before_specificity():
    floor = 0.95
    clears = selection_rank(0.96, 0.40, floor)
    misses = selection_rank(0.90, 0.99, floor)
    assert clears > misses
    assert selection_rank(0.97, 0.70, floor) > selection_rank(0.99, 0.60, floor)
    assert selection_rank(0.80, 0.10, floor) > selection_rank(0.70, 0.90, floor)


def test_bootstrap_interval_contains_the_point_estimate():
    ids = ["A", "A", "B", "C", "C", "D"]
    preds = [1, 1, 0, 0, 1, 0]
    labels = [1, 1, 1, 0, 0, 0]
    result = bootstrap_patient_ci(ids, preds, labels, n_bootstrap=200, seed=1)
    low, high = result["patient_sensitivity_ci95"]
    assert math.isfinite(low) and low <= result["patient_sensitivity"] <= high
    spec_low, spec_high = result["patient_specificity_ci95"]
    assert spec_low <= result["patient_specificity"] <= spec_high
    assert result["n_patients"] == 4


def test_patient_level_predictions_break_ties_toward_the_smaller_index():
    preds, labels = patient_level_predictions(
        ["A", "A", "B"], preds=[0, 1, 2], labels=[0, 0, 2], num_classes=3
    )
    assert preds.tolist() == [0, 2]
    assert labels.tolist() == [0, 2]
    assert preds.dtype == np.int64 or preds.dtype == np.int32 or preds.dtype == int
