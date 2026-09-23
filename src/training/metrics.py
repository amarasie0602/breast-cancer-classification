"""Binary classification metrics from logits."""

from typing import Tuple

import torch
from torch import Tensor


def logits_to_preds(logits: Tensor, threshold: float = 0.5) -> Tensor:
    probs = torch.sigmoid(logits)
    return (probs >= threshold).long()


def accuracy(preds: Tensor, labels: Tensor) -> float:
    return (preds == labels).float().mean().item()


def precision_recall_f1(preds: Tensor, labels: Tensor) -> Tuple[float, float, float]:
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def sensitivity_specificity(preds: Tensor, labels: Tensor) -> Tuple[float, float]:
    """Sensitivity (= recall, true positive rate) and specificity (true
    negative rate) for the binary (benign=0, malignant=1) classifier."""
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sensitivity, specificity


def confusion_matrix(preds: Tensor, labels: Tensor, num_classes: int) -> Tensor:
    """Rows = true class, columns = predicted class."""
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for true_label, pred_label in zip(labels.tolist(), preds.tolist(), strict=True):
        matrix[true_label, pred_label] += 1
    return matrix


def multiclass_precision_recall_f1(
    preds: Tensor, labels: Tensor, num_classes: int
) -> Tuple[list, list, list, float]:
    """Per-class precision/recall/F1 (one-vs-rest) plus macro-averaged F1.

    Macro (not micro/weighted) averaging is used for the summary metric so
    that the smaller subtypes -- lobular_carcinoma and papillary_carcinoma,
    both under 15% of the malignant-subtype dataset -- aren't washed out by
    ductal_carcinoma's ~64% share when judging overall performance.
    """
    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    macro_f1 = sum(f1s) / num_classes
    return precisions, recalls, f1s, macro_f1
