"""Training and evaluation loops for the malignant-subtype (multi-class)
classifier -- separate from src/training/loop.py because the binary
classifier's single-logit BCEWithLogitsLoss shape (float labels, unsqueeze)
is fundamentally different from CrossEntropyLoss's multi-class shape
(integer labels, argmax over class logits)."""

from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.metrics import accuracy, confusion_matrix, multiclass_precision_recall_f1


def train_one_epoch(
    model: nn.Module, dataloader: DataLoader, optimizer, criterion: nn.Module, device: str
) -> float:
    if len(dataloader.dataset) == 0:
        raise ValueError("train_one_epoch received an empty dataset")

    model.train()
    total_loss = 0.0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: str, num_classes: int
) -> Dict[str, object]:
    if len(dataloader.dataset) == 0:
        raise ValueError("evaluate received an empty dataset")

    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)

        all_preds.append(logits.argmax(dim=1))
        all_labels.append(labels)

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    precisions, recalls, f1s, macro_f1 = multiclass_precision_recall_f1(preds, labels, num_classes)

    return {
        "loss": total_loss / len(dataloader.dataset),
        "accuracy": accuracy(preds, labels),
        "macro_f1": macro_f1,
        "per_class_precision": precisions,
        "per_class_recall": recalls,
        "per_class_f1": f1s,
        "confusion_matrix": confusion_matrix(preds, labels, num_classes),
    }
