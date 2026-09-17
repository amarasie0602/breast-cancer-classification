"""Error analysis helpers: run inference over a dataset and collect predictions."""

from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.metrics import logits_to_preds


@torch.no_grad()
def collect_predictions(model: nn.Module, dataloader: DataLoader, device: str = "cpu") -> List[dict]:
    model.eval()
    records = []
    sample_idx = 0

    for images, labels in dataloader:
        images_dev = images.to(device)
        logits = model(images_dev)
        preds = logits_to_preds(logits).squeeze(1)
        probs = torch.sigmoid(logits).squeeze(1)

        for i in range(images.size(0)):
            records.append(
                {
                    "index": sample_idx,
                    "label": labels[i].item(),
                    "pred": preds[i].item(),
                    "probability": probs[i].item(),
                }
            )
            sample_idx += 1

    return records


def filter_misclassified(records: List[dict]) -> List[dict]:
    return [r for r in records if r["pred"] != r["label"]]


def confidence(record: dict) -> float:
    """Distance from 0.5: how confidently the model made its prediction."""
    return abs(record["probability"] - 0.5)


def most_confident_errors(records: List[dict], n: int = 10) -> List[dict]:
    errors = filter_misclassified(records)
    return sorted(errors, key=confidence, reverse=True)[:n]
