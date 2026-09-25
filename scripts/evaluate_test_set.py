"""Evaluate a trained checkpoint on the held-out test split and print the
full metric set: accuracy, precision, recall, F1, sensitivity, specificity,
and the confusion matrix.

The test split is the one neither training nor checkpoint selection ever
saw -- it is rebuilt here with the same patient-level split, seed and ratios
the training run used, so the numbers are reproducible rather than ad hoc.

    python -m scripts.evaluate_test_set --magnification 40
    python -m scripts.evaluate_test_set --subtype
"""

import argparse

import torch
from torch.utils.data import DataLoader

from src.data.dataset import SUBTYPE_SCHEMES, BreakHisDataset, BreakHisSubtypeDataset
from src.data.splits import filter_samples_by_patients, stratified_patient_split
from src.data.transforms import eval_transform
from src.models.classifier import BreakHisClassifier, MalignantSubtypeClassifier
from src.serving.model_loader import subtype_checkpoint_scheme
from src.training.checkpoint import load_checkpoint
from src.training.loop import evaluate as evaluate_binary
from src.training.subtype_loop import evaluate as evaluate_subtype
from src.training.train import load_config

BINARY_CLASS_NAMES = ("benign", "malignant")


def _print_confusion_matrix(matrix, class_names) -> None:
    width = max(len(n) for n in class_names) + 2
    header = " " * (width + 8) + "".join(f"{n:>{width}}" for n in class_names)
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(header)
    for i, name in enumerate(class_names):
        row = "".join(f"{int(matrix[i, j]):>{width}}" for j in range(len(class_names)))
        print(f"  actual {name:<{width}}{row}")


def evaluate_binary_checkpoint(checkpoint_path, data_root, magnification, ratios, seed, batch_size):
    full_ds = BreakHisDataset(data_root, magnification=magnification)
    _, _, test_patients = stratified_patient_split(full_ds.samples, ratios=ratios, seed=seed)

    test_ds = BreakHisDataset(data_root, magnification=magnification, transform=eval_transform())
    test_ds.samples = filter_samples_by_patients(test_ds.samples, test_patients)
    if not test_ds.samples:
        raise SystemExit(f"No test samples for magnification {magnification}")

    model = BreakHisClassifier(pretrained=False)
    load_checkpoint(checkpoint_path, model)
    model.eval()

    metrics = evaluate_binary(
        model,
        DataLoader(test_ds, batch_size=batch_size),
        torch.nn.BCEWithLogitsLoss(),
        "cpu",
    )

    print(f"\n=== Binary (benign vs malignant) — magnification {magnification}x ===")
    print(f"  test images   : {len(test_ds.samples)}")
    print(f"  test patients : {len(test_patients)}")
    for key in ("accuracy", "precision", "recall", "f1", "sensitivity", "specificity"):
        print(f"  {key:<14}: {metrics[key]:.4f}")
    _print_confusion_matrix(metrics["confusion_matrix"], BINARY_CLASS_NAMES)
    return metrics


def evaluate_subtype_checkpoint(checkpoint_path, data_root, ratios, seed, batch_size):
    # The checkpoint records which labelling it was trained on; evaluating a
    # 2-class model against 4-class labels (or vice versa) would be meaningless.
    scheme = subtype_checkpoint_scheme(str(checkpoint_path))
    class_names = SUBTYPE_SCHEMES[scheme]["names"]

    full_ds = BreakHisSubtypeDataset(data_root, scheme=scheme)
    _, _, test_patients = stratified_patient_split(
        full_ds.samples, ratios=ratios, seed=seed, min_per_split=1
    )

    test_ds = BreakHisSubtypeDataset(data_root, transform=eval_transform(), scheme=scheme)
    test_ds.samples = filter_samples_by_patients(test_ds.samples, test_patients)
    if not test_ds.samples:
        raise SystemExit("No test samples for the subtype task")

    model = MalignantSubtypeClassifier(num_classes=len(class_names), pretrained=False)
    load_checkpoint(checkpoint_path, model)
    model.eval()

    metrics = evaluate_subtype(
        model,
        DataLoader(test_ds, batch_size=batch_size),
        torch.nn.CrossEntropyLoss(),
        "cpu",
        num_classes=len(class_names),
    )

    print(f"\n=== Malignant subtype ({scheme}, {len(class_names)} classes) ===")
    print(f"  test images   : {len(test_ds.samples)}")
    print(f"  accuracy      : {metrics['accuracy']:.4f}")
    print(f"  macro F1      : {metrics['macro_f1']:.4f}")
    print("\n  Per class (precision / recall / F1):")
    for i, name in enumerate(class_names):
        n_images = sum(1 for s in test_ds.samples if s["label"] == i)
        patients = {s["path"].parent.parent.name for s in test_ds.samples if s["label"] == i}
        print(
            f"    {name:<22} {metrics['per_class_precision'][i]:.4f} / "
            f"{metrics['per_class_recall'][i]:.4f} / {metrics['per_class_f1'][i]:.4f}"
            f"   ({n_images} images, {len(patients)} patients)"
        )
    _print_confusion_matrix(metrics["confusion_matrix"], class_names)
    print(
        "\n  NOTE: with only 1-2 test patients for three of these classes,\n"
        "  these per-class numbers are indicative, not validated. See\n"
        "  docs/model_card.md."
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/BreaKHis_v1")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--magnification", default="40", choices=["40", "100", "200", "400"])
    parser.add_argument("--checkpoint", default=None, help="defaults per task")
    parser.add_argument("--subtype", action="store_true", help="evaluate the subtype model instead")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    data_config = load_config(args.data_config)
    r = data_config["split_ratios"]
    ratios = (r["train"], r["val"], r["test"])
    seed = data_config["seed"]

    if args.subtype:
        checkpoint = args.checkpoint or "checkpoints/best_subtype.pt"
        evaluate_subtype_checkpoint(checkpoint, args.data_root, ratios, seed, args.batch_size)
    else:
        checkpoint = args.checkpoint or f"checkpoints/best_mag{args.magnification}.pt"
        evaluate_binary_checkpoint(
            checkpoint, args.data_root, args.magnification, ratios, seed, args.batch_size
        )


if __name__ == "__main__":
    main()
