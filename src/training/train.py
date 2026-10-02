"""End-to-end training entrypoint: model, loop, checkpointing, MLflow logging."""

import argparse
from pathlib import Path
from typing import Tuple

import yaml
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import BreakHisDataset
from src.data.splits import filter_samples_by_patients, stratified_patient_split
from src.data.stain import require_stain_normalization
from src.data.transforms import eval_transform, train_transform
from src.models.classifier import BreakHisClassifier
from src.training.checkpoint import save_checkpoint
from src.training.early_stopping import EarlyStopping
from src.training.loop import evaluate, train_one_epoch

# Data is loaded in the main process. Every result in docs/model_card.md was
# produced this way; it is stated explicitly so changing it is a deliberate
# choice (worker processes on Windows re-import the whole module).
NUM_WORKERS = 0


def within_project(path: str | Path, what: str) -> Path:
    """Resolve a command-line path, refusing anything outside the project
    directory (the current working directory)."""
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(Path.cwd()):
        raise ValueError(f"{what} must be within the project directory: {path}")
    return resolved


def load_config(path: str | Path) -> dict:
    with open(within_project(path, "config path")) as f:
        return yaml.safe_load(f)


def _checkpoint_metrics(val_metrics: dict, stain_normalization) -> dict:
    """Validation metrics plus the preprocessing, recorded so evaluation and
    serving can prepare images the way the model was trained on them."""
    if not stain_normalization:
        return val_metrics
    return {**val_metrics, "stain_normalization": stain_normalization}


def build_dataloaders(
    data_root: str | Path,
    magnification: str,
    split_ratios: Tuple[float, float, float],
    seed: int,
    batch_size: int,
    balance_classes: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    full_ds = BreakHisDataset(data_root, magnification=magnification)
    train_patients, val_patients, _ = stratified_patient_split(
        full_ds.samples, ratios=tuple(split_ratios), seed=seed
    )

    train_ds = BreakHisDataset(data_root, magnification=magnification, transform=train_transform())
    train_ds.samples = filter_samples_by_patients(train_ds.samples, train_patients)

    val_ds = BreakHisDataset(data_root, magnification=magnification, transform=eval_transform())
    val_ds.samples = filter_samples_by_patients(val_ds.samples, val_patients)

    if balance_classes:
        # BreakHis is ~2.2:1 malignant to benign, and training on that
        # distribution unweighted is why the model over-calls cancer:
        # sensitivity 0.88-0.99 but specificity 0.46-0.73 on test (see
        # docs/model_card.md). Sampling the two classes equally removes the
        # prior that makes "malignant" the cheap guess.
        counts = {0: 0, 1: 0}
        for s in train_ds.samples:
            counts[s["label"]] += 1
        per_class_weight = {
            label: (len(train_ds.samples) / (2 * n) if n else 0.0) for label, n in counts.items()
        }
        sample_weights = [per_class_weight[s["label"]] for s in train_ds.samples]
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=WeightedRandomSampler(
                sample_weights, num_samples=len(train_ds.samples), replacement=True
            ),
            num_workers=NUM_WORKERS,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS
        )

    return train_loader, DataLoader(val_ds, batch_size=batch_size, num_workers=NUM_WORKERS)


def run_training(
    config: dict,
    data_root: str | Path,
    magnification: str,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    device: str = "cpu",
    pretrained: bool = True,
) -> float:
    import mlflow

    # Normalization happens once, when the dataset copy is built
    # (scripts/normalize_dataset.py); the config says which copy this run is
    # meant for, and a mismatch is an error rather than a silently wrong model.
    stain_normalization = config.get("stain_normalization")
    require_stain_normalization(data_root, stain_normalization)

    train_loader, val_loader = build_dataloaders(
        data_root,
        magnification,
        split_ratios,
        seed,
        config["batch_size"],
        balance_classes=config.get("balance_classes", False),
    )

    model = BreakHisClassifier(pretrained=pretrained).to(device)
    criterion = nn.BCEWithLogitsLoss()
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
    best_f1 = -1.0

    with mlflow.start_run():
        mlflow.set_tags({"magnification": magnification, "model_variant": "resnet50"})
        mlflow.log_params(
            {
                "magnification": magnification,
                "learning_rate": config["learning_rate"],
                "batch_size": config["batch_size"],
                "weight_decay": config["weight_decay"],
                "stain_normalization": stain_normalization or "none",
            }
        )

        model.freeze_backbone()
        for epoch in range(config["epochs"]):
            if epoch == config.get("freeze_backbone_epochs", 0):
                model.unfreeze_backbone()

            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_metrics["loss"])

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            for key, value in val_metrics.items():
                if key == "confusion_matrix":  # not a scalar; skip MLflow metric logging
                    continue
                mlflow.log_metric(f"val_{key}", value, step=epoch)

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                save_checkpoint(
                    checkpoint_dir / f"best_mag{magnification}.pt",
                    model,
                    optimizer,
                    epoch,
                    _checkpoint_metrics(val_metrics, stain_normalization),
                )

            if early_stopping.step(val_metrics["loss"]):
                break

    return best_f1


def main() -> None:
    import mlflow

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--magnification", required=True, choices=["40", "100", "200", "400"])
    parser.add_argument("--train-config", default="configs/train.yaml")
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
        within_project(args.data_root, "data root"),
        args.magnification,
        split_ratios=(ratios["train"], ratios["val"], ratios["test"]),
        seed=data_config["seed"],
        pretrained=not args.no_pretrained,
    )


if __name__ == "__main__":
    main()
