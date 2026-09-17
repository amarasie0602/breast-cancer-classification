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
