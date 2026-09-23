"""End-to-end training entrypoint for the malignant-subtype classifier.

Mirrors src/training/train.py's shape (config, checkpointing, MLflow
logging) but trains on malignant-only samples labeled by subtype
(ductal/lobular/mucinous/papillary carcinoma) instead of benign/malignant,
and combines all four magnifications into one training set rather than
training separately per magnification: subtype counts per-magnification are
too small (e.g. papillary_carcinoma has as few as 135 images at a single
magnification) for a meaningful 4-class split, and the binary classifier's
per-magnification evaluation strategy (see docs/model_card.md) doesn't carry
over cleanly to a much harder, much smaller-per-class problem. This is a
deliberate, documented departure, not an oversight.
"""

import argparse
from pathlib import Path
from typing import Tuple, Union

from torch import nn, optim
from torch.utils.data import DataLoader

from src.data.dataset import SUBTYPE_NAMES, BreakHisSubtypeDataset
from src.data.splits import filter_samples_by_patients, stratified_patient_split
from src.data.transforms import eval_transform, train_transform
from src.models.classifier import MalignantSubtypeClassifier
from src.training.checkpoint import save_checkpoint
from src.training.early_stopping import EarlyStopping
from src.training.subtype_loop import evaluate, train_one_epoch
from src.training.train import load_config

NUM_SUBTYPES = len(SUBTYPE_NAMES)


def build_dataloaders(
    data_root: Union[str, Path],
    split_ratios: Tuple[float, float, float],
    seed: int,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    full_ds = BreakHisSubtypeDataset(data_root)
    train_patients, val_patients, _ = stratified_patient_split(
        full_ds.samples, ratios=tuple(split_ratios), seed=seed
    )

    train_ds = BreakHisSubtypeDataset(data_root, transform=train_transform())
    train_ds.samples = filter_samples_by_patients(train_ds.samples, train_patients)

    val_ds = BreakHisSubtypeDataset(data_root, transform=eval_transform())
    val_ds.samples = filter_samples_by_patients(val_ds.samples, val_patients)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
    )


def _class_weights(samples: list) -> list:
    """Inverse-frequency class weights for CrossEntropyLoss, since
    ductal_carcinoma outnumbers papillary_carcinoma roughly 6:1."""
    counts = [0] * NUM_SUBTYPES
    for s in samples:
        counts[s["label"]] += 1
    total = sum(counts)
    return [total / (NUM_SUBTYPES * c) if c > 0 else 0.0 for c in counts]


def run_training(
    config: dict,
    data_root: Union[str, Path],
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    device: str = "cpu",
    pretrained: bool = True,
) -> float:
    import torch
    import mlflow

    train_loader, val_loader = build_dataloaders(
        data_root, split_ratios, seed, config["batch_size"]
    )

    model = MalignantSubtypeClassifier(num_classes=NUM_SUBTYPES, pretrained=pretrained).to(device)
    weights = torch.tensor(_class_weights(train_loader.dataset.samples), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=config["lr_scheduler"]["patience"],
        factor=config["lr_scheduler"]["factor"],
    )
    early_stopping = EarlyStopping(**config["early_stopping"])

    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_macro_f1 = -1.0

    with mlflow.start_run():
        mlflow.set_tags({"model_variant": "resnet50", "task": "malignant_subtype"})
        mlflow.log_params(
            {
                "learning_rate": config["learning_rate"],
                "batch_size": config["batch_size"],
                "weight_decay": config["weight_decay"],
                "subtypes": ",".join(SUBTYPE_NAMES),
            }
        )

        model.freeze_backbone()
        for epoch in range(config["epochs"]):
            if epoch == config.get("freeze_backbone_epochs", 0):
                model.unfreeze_backbone()

            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate(model, val_loader, criterion, device, num_classes=NUM_SUBTYPES)
            scheduler.step(val_metrics["loss"])

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            for key, value in val_metrics.items():
                if key in ("confusion_matrix", "per_class_precision", "per_class_recall", "per_class_f1"):
                    continue  # not scalars; skip MLflow metric logging
                mlflow.log_metric(f"val_{key}", value, step=epoch)

            if val_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = val_metrics["macro_f1"]
                save_checkpoint(
                    checkpoint_dir / "best_subtype.pt",
                    model,
                    optimizer,
                    epoch,
                    val_metrics,
                )

            if early_stopping.step(val_metrics["loss"]):
                break

    return best_macro_f1


def main() -> None:
    import mlflow

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-config", default="configs/train_subtype.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--mlflow-tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--no-pretrained", action="store_true", help="skip downloading ImageNet weights")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    train_config = load_config(args.train_config)
    data_config = load_config(args.data_config)
    ratios = data_config["split_ratios"]

    run_training(
        train_config,
        args.data_root,
        split_ratios=(ratios["train"], ratios["val"], ratios["test"]),
        seed=data_config["seed"],
        pretrained=not args.no_pretrained,
    )


if __name__ == "__main__":
    main()
