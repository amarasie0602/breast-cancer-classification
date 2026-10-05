"""Run the served app on BACH, a breast histology dataset from another lab.

Every number elsewhere comes from BreakHis (one lab, one scanner). BACH
(ICIAR 2018; CC BY-NC-ND 4.0) has H&E breast microscopy images labelled
Normal, Benign, InSitu and Invasive from a different source, so it shows how
the whole pipeline - stage 1 screening, magnification detection, calibrated
classification, the uncertain band - holds up on images it was never tuned
on. Normal and Benign count as benign, InSitu and Invasive as malignant.

Expects BACH Parquet shards (the 1aurent/BACH mirror's format) in
data/external/BACH/; scripts/download_bach.sh fetches one shard per
class. The data stays out of the repository.

    python -m scripts.evaluate_external_bach
"""

from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from fastapi import HTTPException

from src.serving.app import _analyse

REPO_ROOT = Path(__file__).resolve().parent.parent
BACH_DIR = REPO_ROOT / "data" / "external" / "BACH"
CLASS_NAMES = {0: "Benign", 1: "InSitu", 2: "Invasive", 3: "Normal"}
MALIGNANT_CLASSES = {"InSitu", "Invasive"}


def images():
    for shard in sorted(BACH_DIR.glob("*.parquet")):
        table = pq.read_table(shard, columns=["image", "label"])
        pixels, labels = table.column("image").to_pylist(), table.column("label").to_pylist()
        for image, label in zip(pixels, labels, strict=True):
            if label in CLASS_NAMES:
                yield image["bytes"], CLASS_NAMES[label]


def main() -> None:
    outcomes = []
    for image_bytes, true_class in images():
        try:
            result = _analyse(image_bytes, "auto")
            outcomes.append((true_class, result))
        except HTTPException as e:
            outcomes.append((true_class, e))
        print(".", end="", flush=True)
    print()

    print(f"\n{len(outcomes)} BACH images: {dict(Counter(c for c, _ in outcomes))}")
    rejected = Counter(c for c, r in outcomes if isinstance(r, HTTPException))
    print(f"stage 1 rejected as not histology: {sum(rejected.values())} {dict(rejected)}")

    analysed = [(c, r) for c, r in outcomes if not isinstance(r, HTTPException)]
    print("detected magnification:", dict(Counter(r.magnification for _, r in analysed)))
    print("magnification warnings:", sum(r.magnification_warning is not None for _, r in analysed))

    print("\nper class (of images that reached the classifier):")
    for true_class in ("Normal", "Benign", "InSitu", "Invasive"):
        rows = [r for c, r in analysed if c == true_class]
        if not rows:
            continue
        expected = "malignant" if true_class in MALIGNANT_CLASSES else "benign"
        right = sum(r.label == expected for r in rows)
        uncertain = sum(r.uncertain for r in rows)
        confident_right = sum(r.label == expected and not r.uncertain for r in rows)
        confident_wrong = sum(r.label != expected and not r.uncertain for r in rows)
        print(
            f"  {true_class:<9} n={len(rows):<3} called {expected}: {right}/{len(rows)} "
            f"({right / len(rows):.0%}); uncertain {uncertain}; confident right {confident_right}, "
            f"confident wrong {confident_wrong}"
        )

    malignant = [r for c, r in analysed if c in MALIGNANT_CLASSES]
    benign = [r for c, r in analysed if c not in MALIGNANT_CLASSES]
    if malignant and benign:
        sensitivity = sum(r.label == "malignant" for r in malignant) / len(malignant)
        specificity = sum(r.label == "benign" for r in benign) / len(benign)
        print(
            f"\nsensitivity {sensitivity:.3f}, specificity {specificity:.3f}, "
            f"balanced accuracy {(sensitivity + specificity) / 2:.3f}, "
            f"uncertain {sum(r.uncertain for _, r in analysed) / len(analysed):.0%}"
        )


if __name__ == "__main__":
    main()
