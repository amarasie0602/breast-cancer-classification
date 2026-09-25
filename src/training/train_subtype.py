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
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import DEFAULT_SUBTYPE_SCHEME, SUBTYPE_SCHEMES, BreakHisSubtypeDataset
from src.data.splits import filter_samples_by_patients, stratified_patient_split
from src.data.transforms import eval_transform, strong_train_transform
from src.models.classifier import MalignantSubtypeClassifier
from src.training.checkpoint import save_checkpoint
from src.training.early_stopping import EarlyStopping
from src.training.subtype_loop import evaluate, train_one_epoch
from src.training.train import load_config

def build_dataloaders(
    data_root: Union[str, Path],
    split_ratios: Tuple[float, float, float],
    seed: int,
    batch_size: int,
    scheme: str = DEFAULT_SUBTYPE_SCHEME,
) -> Tuple[DataLoader, DataLoader]:
    num_classes = len(SUBTYPE_SCHEMES[scheme]["names"])
    full_ds = BreakHisSubtypeDataset(data_root, scheme=scheme)
    # min_per_split=1: lobular_carcinoma has only 5 patients in all of
    # BreakHis, and plain ratio rounding leaves it with 0 test patients --
    # i.e. a class the test set literally cannot measure.
    train_patients, val_patients, _ = stratified_patient_split(
        full_ds.samples, ratios=tuple(split_ratios), seed=seed, min_per_split=1
    )

    train_ds = BreakHisSubtypeDataset(data_root, transform=strong_train_transform(), scheme=scheme)
    train_ds.samples = filter_samples_by_patients(train_ds.samples, train_patients)

    val_ds = BreakHisSubtypeDataset(data_root, transform=eval_transform(), scheme=scheme)
    val_ds.samples = filter_samples_by_patients(val_ds.samples, val_patients)

    # Class-balanced sampling on top of the weighted loss: ductal_carcinoma
    # is ~64% of malignant images, so with plain shuffling a batch of 32
    # often contains zero papillary or lobular examples, and the gradient
    # for those classes arrives too sparsely to learn much.
    weights_per_class = _class_weights(train_ds.samples, num_classes)
    sample_weights = [weights_per_class[s["label"]] for s in train_ds.samples]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(train_ds.samples), replacement=True
    )

    return (
        DataLoader(train_ds, batch_size=batch_size, sampler=sampler),
        DataLoader(val_ds, batch_size=batch_size),
    )


def _class_weights(samples: list, num_classes: int) -> list:
    """Inverse-frequency per-class weights, since ductal_carcinoma
    outnumbers papillary_carcinoma roughly 6:1."""
    counts = [0] * num_classes
    for s in samples:
        counts[s["label"]] += 1
    total = sum(counts)
    return [total / (num_classes * c) if c > 0 else 0.0 for c in counts]


def run_training(
    config: dict,
    data_root: Union[str, Path],
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    device: str = "cpu",
    pretrained: bool = True,
) -> float:
    import mlflow

    scheme = config.get("label_scheme", DEFAULT_SUBTYPE_SCHEME)
    class_names = SUBTYPE_SCHEMES[scheme]["names"]
    num_classes = len(class_names)
    checkpoint_name = config.get("checkpoint_name", "best_subtype.pt")

    train_loader, val_loader = build_dataloaders(
        data_root, split_ratios, seed, config["batch_size"], scheme=scheme
    )

    model = MalignantSubtypeClassifier(num_classes=num_classes, pretrained=pretrained).to(device)
    # Plain (unweighted) loss on purpose: the WeightedRandomSampler in
    # build_dataloaders already makes the classes roughly equiprobable in
    # every batch. Applying inverse-frequency weights on top would correct
    # the same imbalance twice and push the model to over-predict the rare
    # subtypes. Label smoothing instead, as mild regularization against
    # overconfidence on a 4-6-training-patient class.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
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
        mlflow.set_tags(
            {"model_variant": "efficientnet_b0", "task": "malignant_subtype", "label_scheme": scheme}
        )
        mlflow.log_params(
            {
                "learning_rate": config["learning_rate"],
                "batch_size": config["batch_size"],
                "weight_decay": config["weight_decay"],
                "subtypes": ",".join(class_names),
            }
        )

        model.freeze_backbone()
        for epoch in range(config["epochs"]):
            if epoch == config.get("freeze_backbone_epochs", 0):
                model.unfreeze_backbone()

            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate(model, val_loader, criterion, device, num_classes=num_classes)
            scheduler.step(val_metrics["loss"])

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            for key, value in val_metrics.items():
                if key in ("confusion_matrix", "per_class_precision", "per_class_recall", "per_class_f1"):
                    continue  # not scalars; skip MLflow metric logging
                mlflow.log_metric(f"val_{key}", value, step=epoch)

            if val_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = val_metrics["macro_f1"]
                # The label scheme travels with the weights: serving needs it
                # to build the right-sized head and name the classes.
                save_checkpoint(
                    checkpoint_dir / checkpoint_name,
                    model,
                    optimizer,
                    epoch,
                    {**val_metrics, "label_scheme": scheme},
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
