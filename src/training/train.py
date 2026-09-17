"""End-to-end training entrypoint: model, loop, checkpointing, MLflow logging."""

import argparse
from pathlib import Path
from typing import Tuple, Union

import yaml
from torch import nn, optim
from torch.utils.data import DataLoader

from src.data.dataset import BreakHisDataset
from src.data.splits import filter_samples_by_patients, stratified_patient_split
from src.data.transforms import eval_transform, train_transform
from src.models.classifier import BreakHisClassifier
from src.training.checkpoint import save_checkpoint
from src.training.early_stopping import EarlyStopping
from src.training.loop import evaluate, train_one_epoch


def load_config(path: Union[str, Path]) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataloaders(
    data_root: Union[str, Path],
    magnification: str,
    split_ratios: Tuple[float, float, float],
    seed: int,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    full_ds = BreakHisDataset(data_root, magnification=magnification)
    train_patients, val_patients, _ = stratified_patient_split(
        full_ds.samples, ratios=tuple(split_ratios), seed=seed
    )

    train_ds = BreakHisDataset(data_root, magnification=magnification, transform=train_transform())
    train_ds.samples = filter_samples_by_patients(train_ds.samples, train_patients)

    val_ds = BreakHisDataset(data_root, magnification=magnification, transform=eval_transform())
    val_ds.samples = filter_samples_by_patients(val_ds.samples, val_patients)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
    )


def run_training(
    config: dict,
    data_root: Union[str, Path],
    magnification: str,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    device: str = "cpu",
    pretrained: bool = True,
) -> float:
    import mlflow

    train_loader, val_loader = build_dataloaders(
        data_root, magnification, split_ratios, seed, config["batch_size"]
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
                mlflow.log_metric(f"val_{key}", value, step=epoch)

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                save_checkpoint(
                    checkpoint_dir / f"best_mag{magnification}.pt",
                    model,
                    optimizer,
                    epoch,
                    val_metrics,
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
        args.data_root,
        args.magnification,
        split_ratios=(ratios["train"], ratios["val"], ratios["test"]),
        seed=data_config["seed"],
        pretrained=not args.no_pretrained,
    )


if __name__ == "__main__":
    main()
