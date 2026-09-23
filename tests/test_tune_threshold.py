import torch

from scripts.tune_threshold import sweep


def test_sweep_reports_the_sensitivity_specificity_tradeoff():
    # Malignant (label 1) scored high, benign (label 0) scored mid --
    # so a low threshold catches everything but flags benign too.
    probs = torch.tensor([0.9, 0.8, 0.7, 0.45, 0.4, 0.3])
    labels = torch.tensor([1, 1, 1, 0, 0, 0])

    rows = sweep(probs, labels)
    by_threshold = {r["threshold"]: r for r in rows}

    # At 0.35, one benign (0.4, 0.45) still slips through as malignant.
    assert by_threshold[0.35]["sensitivity"] == 1.0
    assert by_threshold[0.35]["specificity"] < 1.0

    # At 0.5, every benign is correctly rejected and every malignant caught.
    assert by_threshold[0.5]["sensitivity"] == 1.0
    assert by_threshold[0.5]["specificity"] == 1.0
    assert by_threshold[0.5]["youden_j"] == 1.0


def test_sweep_specificity_rises_as_threshold_rises():
    probs = torch.tensor([0.9, 0.6, 0.55, 0.2])
    labels = torch.tensor([1, 0, 0, 0])

    rows = {r["threshold"]: r for r in sweep(probs, labels)}
    assert rows[0.5]["specificity"] < rows[0.7]["specificity"]
    assert rows[0.7]["specificity"] == 1.0
