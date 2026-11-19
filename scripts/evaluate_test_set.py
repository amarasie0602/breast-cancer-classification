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
import math

import torch
from torch.utils.data import DataLoader

from src.data.dataset import SUBTYPE_SCHEMES, BreakHisDataset, BreakHisSubtypeDataset
from src.data.manifest import patient_id_of, resolve_patient_split
from src.data.splits import filter_samples_by_patients
from src.data.stain import require_stain_normalization
from src.data.transforms import eval_transform
from src.models.classifier import BreakHisClassifier, MalignantSubtypeClassifier
from src.serving.model_loader import subtype_checkpoint_scheme
from src.training.checkpoint import checkpoint_stain_normalization, load_checkpoint
from src.training.metrics import accuracy, confusion_matrix, precision_recall_f1, sensitivity_specificity
from src.training.patient_metrics import bootstrap_patient_ci
from src.training.subtype_loop import evaluate as evaluate_subtype
from src.training.train import NUM_WORKERS, load_config, within_project

BINARY_CLASS_NAMES = ("benign", "malignant")


def _print_confusion_matrix(matrix, class_names) -> None:
    width = max(len(n) for n in class_names) + 2
    header = " " * (width + 8) + "".join(f"{n:>{width}}" for n in class_names)
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(header)
    for i, name in enumerate(class_names):
        row = "".join(f"{int(matrix[i, j]):>{width}}" for j in range(len(class_names)))
        print(f"  actual {name:<{width}}{row}")


def _format_rate(value) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.4f}"


def _format_ci(interval) -> str:
    low, high = interval
    if not (math.isfinite(low) and math.isfinite(high)):
        return "n/a"
    return f"{low:.4f}-{high:.4f}"


def evaluate_binary_checkpoint(
    checkpoint_path, data_root, magnification, ratios, seed, batch_size, n_bootstrap: int = 1000
):
    require_stain_normalization(data_root, checkpoint_stain_normalization(checkpoint_path))
    full_ds = BreakHisDataset(data_root, magnification=magnification)
    _, _, test_patients = resolve_patient_split(
        full_ds.samples,
        ratios=ratios,
        seed=seed,
        manifest_key=("binary", str(magnification)),
    )

    test_ds = BreakHisDataset(data_root, magnification=magnification, transform=eval_transform())
    test_ds.samples = filter_samples_by_patients(test_ds.samples, test_patients)
    if not test_ds.samples:
        raise SystemExit(f"No test samples for magnification {magnification}")

    model = BreakHisClassifier(pretrained=False)
    load_checkpoint(checkpoint_path, model)
    model.eval()

    patient_ids = [patient_id_of(sample) for sample in test_ds.samples]
    labels = torch.tensor([sample["label"] for sample in test_ds.samples], dtype=torch.long)
    # One forward. The patient interval is computed from these same predictions.
    preds = _binary_preds(model, test_ds, batch_size)
    precision, recall, f1 = precision_recall_f1(preds, labels)
    sensitivity, specificity = sensitivity_specificity(preds, labels)
    metrics = {
        "accuracy": accuracy(preds, labels),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": confusion_matrix(preds, labels, num_classes=2),
    }
    metrics.update(
        bootstrap_patient_ci(patient_ids, preds, labels, n_bootstrap=n_bootstrap, seed=seed)
    )

    print(f"\n=== Binary (benign vs malignant) — magnification {magnification}x ===")
    print("  This split is reported only. It is not used to choose a checkpoint.")
    print(f"  test images   : {len(test_ds.samples)}")
    print(f"  test patients : {len({patient_id_of(sample) for sample in test_ds.samples})}")
    for key in ("accuracy", "precision", "recall", "f1", "sensitivity", "specificity"):
        print(f"  image {key:<12}: {metrics[key]:.4f}")
    print(
        f"  patient sensitivity : {_format_rate(metrics['patient_sensitivity'])} "
        f"(95% CI {_format_ci(metrics['patient_sensitivity_ci95'])}, bootstrap over patients)"
    )
    print(
        f"  patient specificity : {_format_rate(metrics['patient_specificity'])} "
        f"(95% CI {_format_ci(metrics['patient_specificity_ci95'])}, bootstrap over patients)"
    )
    print("  image F1 is secondary. Selection uses patient sensitivity, then specificity.")
    _print_confusion_matrix(metrics["confusion_matrix"], BINARY_CLASS_NAMES)
    return metrics


def _binary_preds(model, dataset, batch_size):
    preds = []
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=NUM_WORKERS)
    with torch.no_grad():
        for images, _labels in loader:
            preds.append((torch.sigmoid(model(images)) >= 0.5).long().view(-1).cpu())
    return torch.cat(preds)


def evaluate_subtype_checkpoint(checkpoint_path, data_root, ratios, seed, batch_size):
    # The checkpoint records which labelling it was trained on; evaluating a
    # 2-class model against 4-class labels (or vice versa) would be meaningless.
    scheme = subtype_checkpoint_scheme(str(checkpoint_path))
    class_names = SUBTYPE_SCHEMES[scheme]["names"]

    full_ds = BreakHisSubtypeDataset(data_root, scheme=scheme)
    _, _, test_patients = resolve_patient_split(
        full_ds.samples,
        ratios=ratios,
        seed=seed,
        manifest_key=("subtype", scheme),
        min_per_split=1,
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
        DataLoader(test_ds, batch_size=batch_size, num_workers=NUM_WORKERS),
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

    # Accuracy alone flatters a model on a skewed test set, so print what
    # always predicting the most common class would score.
    counts = [sum(1 for s in test_ds.samples if s["label"] == i) for i in range(len(class_names))]
    majority = max(range(len(class_names)), key=lambda i: counts[i])
    print(
        f"\n  Baseline: always predicting '{class_names[majority]}' scores "
        f"{counts[majority] / len(test_ds.samples):.4f} accuracy on this test set."
    )
    print(
        f"  NOTE: the test split holds only {len(test_patients)} patients in total, so\n"
        "  these numbers are indicative, not validated. See docs/model_card.md."
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
    data_root = within_project(args.data_root, "data root")

    if args.subtype:
        checkpoint = args.checkpoint or "checkpoints/best_subtype.pt"
        evaluate_subtype_checkpoint(checkpoint, data_root, ratios, seed, args.batch_size)
    else:
        checkpoint = args.checkpoint or f"checkpoints/best_mag{args.magnification}.pt"
        evaluate_binary_checkpoint(
            checkpoint, data_root, args.magnification, ratios, seed, args.batch_size
        )


if __name__ == "__main__":
    main()
