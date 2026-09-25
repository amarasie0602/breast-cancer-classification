"""Pick a decision threshold for the binary classifier on the validation split.

The default 0.5 is not a chosen operating point, it's just where sigmoid
happens to cross. On this data that default yields sensitivity 0.88-0.99 but
specificity 0.46-0.73 -- the model calls roughly half of benign tissue
malignant (see docs/model_card.md). Moving the threshold trades one for the
other, and this sweep shows the whole curve so the trade is explicit.

Tuning happens on the **validation** split, never the test split: choosing a
threshold is model selection, and doing it on test would make the reported
test numbers meaningless.

    python -m scripts.tune_threshold --magnification 40
"""

import argparse

import torch
from torch.utils.data import DataLoader

from src.data.dataset import BreakHisDataset
from src.data.splits import filter_samples_by_patients, stratified_patient_split
from src.data.transforms import eval_transform
from src.models.classifier import BreakHisClassifier
from src.training.checkpoint import load_checkpoint
from src.training.metrics import sensitivity_specificity
from src.training.train import load_config

THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]


@torch.no_grad()
def collect_probabilities(model, dataloader):
    probs, labels = [], []
    for images, batch_labels in dataloader:
        probs.append(torch.sigmoid(model(images)).squeeze(1))
        labels.append(batch_labels)
    return torch.cat(probs), torch.cat(labels)


def sweep(probs, labels):
    """Return one row per threshold: sensitivity, specificity, Youden's J,
    and balanced accuracy."""
    rows = []
    for t in THRESHOLDS:
        preds = (probs >= t).long()
        sens, spec = sensitivity_specificity(preds, labels)
        rows.append(
            {
                "threshold": t,
                "sensitivity": sens,
                "specificity": spec,
                "youden_j": sens + spec - 1.0,
                "balanced_accuracy": (sens + spec) / 2,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/BreaKHis_v1")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--magnification", default="40", choices=["40", "100", "200", "400"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    data_config = load_config(args.data_config)
    r = data_config["split_ratios"]
    ratios = (r["train"], r["val"], r["test"])

    full_ds = BreakHisDataset(args.data_root, magnification=args.magnification)
    _, val_patients, _ = stratified_patient_split(
        full_ds.samples, ratios=ratios, seed=data_config["seed"]
    )

    val_ds = BreakHisDataset(
        args.data_root, magnification=args.magnification, transform=eval_transform()
    )
    val_ds.samples = filter_samples_by_patients(val_ds.samples, val_patients)

    checkpoint = args.checkpoint or f"checkpoints/best_mag{args.magnification}.pt"
    model = BreakHisClassifier(pretrained=False)
    load_checkpoint(checkpoint, model)
    model.eval()

    probs, labels = collect_probabilities(
        model, DataLoader(val_ds, batch_size=args.batch_size)
    )
    rows = sweep(probs, labels)

    print(f"\n=== Threshold sweep on the VALIDATION split — {args.magnification}x ===")
    print(f"  val images: {len(val_ds.samples)}  patients: {len(val_patients)}")
    print(f"\n  {'thresh':>7} {'sens':>7} {'spec':>7} {'youden':>8} {'bal acc':>8}")
    best = max(rows, key=lambda r: r["youden_j"])
    for row in rows:
        marker = "  <-- best Youden J" if row is best else ""
        print(
            f"  {row['threshold']:>7.2f} {row['sensitivity']:>7.3f} {row['specificity']:>7.3f}"
            f" {row['youden_j']:>8.3f} {row['balanced_accuracy']:>8.3f}{marker}"
        )

    default = next(r for r in rows if r["threshold"] == 0.5)
    print(
        f"\n  default 0.50 : sensitivity {default['sensitivity']:.3f}, "
        f"specificity {default['specificity']:.3f}, balanced acc {default['balanced_accuracy']:.3f}"
    )
    print(
        f"  best   {best['threshold']:.2f} : sensitivity {best['sensitivity']:.3f}, "
        f"specificity {best['specificity']:.3f}, balanced acc {best['balanced_accuracy']:.3f}"
    )
    print(
        "\n  Set DECISION_THRESHOLD in the serving environment to apply a\n"
        "  different operating point. Which point is 'right' depends on the\n"
        "  cost of a missed cancer vs a false alarm -- a clinical judgement,\n"
        "  not a modelling one."
    )


if __name__ == "__main__":
    main()
