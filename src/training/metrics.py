"""Binary classification metrics from logits."""

import torch


def logits_to_preds(logits, threshold=0.5):
    probs = torch.sigmoid(logits)
    return (probs >= threshold).long()


def accuracy(preds, labels):
    return (preds == labels).float().mean().item()
