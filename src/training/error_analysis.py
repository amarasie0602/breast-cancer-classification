"""Error analysis helpers: run inference over a dataset and collect predictions."""

import torch

from src.training.metrics import logits_to_preds


@torch.no_grad()
def collect_predictions(model, dataloader, device="cpu"):
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


def filter_misclassified(records):
    return [r for r in records if r["pred"] != r["label"]]
