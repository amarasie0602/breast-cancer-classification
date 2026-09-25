import torch

from src.training.metrics import (
    accuracy,
    confusion_matrix,
    logits_to_preds,
    multiclass_precision_recall_f1,
    precision_recall_f1,
    sensitivity_specificity,
)


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


def test_sensitivity_specificity_perfect():
    preds = torch.tensor([1, 1, 0, 0])
    labels = torch.tensor([1, 1, 0, 0])
    sensitivity, specificity = sensitivity_specificity(preds, labels)
    assert sensitivity == 1.0
    assert specificity == 1.0


def test_sensitivity_specificity_all_predicted_positive():
    preds = torch.tensor([1, 1, 1, 1])
    labels = torch.tensor([1, 1, 0, 0])
    sensitivity, specificity = sensitivity_specificity(preds, labels)
    assert sensitivity == 1.0  # caught every real positive
    assert specificity == 0.0  # but flagged every real negative too


def test_confusion_matrix_counts_correctly():
    preds = torch.tensor([0, 1, 1, 2])
    labels = torch.tensor([0, 0, 1, 2])
    cm = confusion_matrix(preds, labels, num_classes=3)
    # row = true class, col = predicted class
    assert cm[0, 0].item() == 1  # true 0, pred 0
    assert cm[0, 1].item() == 1  # true 0, pred 1 (the one mistake)
    assert cm[1, 1].item() == 1  # true 1, pred 1
    assert cm[2, 2].item() == 1  # true 2, pred 2
    assert cm.sum().item() == 4


def test_multiclass_precision_recall_f1_perfect():
    preds = torch.tensor([0, 1, 2, 0, 1, 2])
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    precisions, recalls, f1s, macro_f1 = multiclass_precision_recall_f1(preds, labels, num_classes=3)
    assert precisions == [1.0, 1.0, 1.0]
    assert recalls == [1.0, 1.0, 1.0]
    assert f1s == [1.0, 1.0, 1.0]
    assert macro_f1 == 1.0


def test_multiclass_precision_recall_f1_one_class_never_predicted():
    preds = torch.tensor([0, 0, 0])
    labels = torch.tensor([0, 1, 2])
    precisions, recalls, f1s, macro_f1 = multiclass_precision_recall_f1(preds, labels, num_classes=3)
    assert recalls[1] == 0.0
    assert recalls[2] == 0.0
    assert macro_f1 < 1.0
