import torch

from src.training.metrics import accuracy, logits_to_preds, precision_recall_f1


def test_logits_to_preds_thresholds_at_zero_point_five():
    logits = torch.tensor([-2.0, 2.0, 0.0])
    preds = logits_to_preds(logits)
    assert preds.tolist() == [0, 1, 1]


def test_accuracy_all_correct():
    preds = torch.tensor([1, 0, 1, 0])
    labels = torch.tensor([1, 0, 1, 0])
    assert accuracy(preds, labels) == 1.0


def test_accuracy_all_wrong():
    preds = torch.tensor([1, 0])
    labels = torch.tensor([0, 1])
    assert accuracy(preds, labels) == 0.0


def test_precision_recall_f1_perfect():
    preds = torch.tensor([1, 1, 0, 0])
    labels = torch.tensor([1, 1, 0, 0])
    precision, recall, f1 = precision_recall_f1(preds, labels)
    assert precision == 1.0
    assert recall == 1.0
    assert f1 == 1.0


def test_precision_recall_f1_no_positive_predictions():
    preds = torch.tensor([0, 0, 0])
    labels = torch.tensor([1, 0, 1])
    precision, recall, f1 = precision_recall_f1(preds, labels)
    assert precision == 0.0
    assert recall == 0.0
    assert f1 == 0.0
