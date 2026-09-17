"""Training and evaluation loops."""

from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.metrics import accuracy, logits_to_preds, precision_recall_f1


def train_one_epoch(
    model: nn.Module, dataloader: DataLoader, optimizer, criterion: nn.Module, device: str
) -> float:
    if len(dataloader.dataset) == 0:
        raise ValueError("train_one_epoch received an empty dataset")

    model.train()
    total_loss = 0.0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: str) -> Dict[str, float]:
    if len(dataloader.dataset) == 0:
        raise ValueError("evaluate received an empty dataset")

    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        labels_float = labels.to(device).float().unsqueeze(1)

        logits = model(images)
        loss = criterion(logits, labels_float)
        total_loss += loss.item() * images.size(0)

        all_preds.append(logits_to_preds(logits).squeeze(1))
        all_labels.append(labels.to(device))

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    precision, recall, f1 = precision_recall_f1(preds, labels)

    return {
        "loss": total_loss / len(dataloader.dataset),
        "accuracy": accuracy(preds, labels),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
