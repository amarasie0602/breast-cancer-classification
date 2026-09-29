"""Fit stage 1's feature-distance check and write it into serving_checkpoints/.

The check describes an image with a general-purpose ImageNet EfficientNet-B0
(not the cancer models, whose features are tuned to histology and don't tell
a photo from a slide) and measures how far that description is from the
training slides' (src/serving/ood.py).

- Statistics are fitted on the training patients' images, all magnifications.
- The threshold is the highest score of any validation patient's image, so
  no validation slide is rejected.
- The held-out test patients' images are only scored, as a check.

Writes serving_checkpoints/histology_screen.pt (the network's weights) and
histology_screen.ood.npz (the statistics and threshold). Re-run it if the
dataset or the patient split changes.

    python -m scripts.fit_histology_screen
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from src.data.dataset import BreakHisDataset
from src.data.splits import stratified_patient_split
from src.serving.ood import FeatureDistance, HistologyScreen, stats_path_for
from src.training.train import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "BreaKHis_v1"
WEIGHTS_PATH = REPO_ROOT / "serving_checkpoints" / "histology_screen.pt"
MAGNIFICATIONS = ("40", "100", "200", "400")
# Principal directions kept. Measured on validation slides against 44
# non-histology images: 128-512 components all rejected no real slide, and
# 512 let the fewest non-histology images through.
N_COMPONENTS = 512


def _images_by_split():
    config = load_config(REPO_ROOT / "configs" / "data.yaml")
    r = config["split_ratios"]
    paths = {"train": [], "val": [], "test": []}
    for magnification in MAGNIFICATIONS:
        ds = BreakHisDataset(DATA_ROOT, magnification=magnification)
        train, val, test = stratified_patient_split(
            ds.samples, ratios=(r["train"], r["val"], r["test"]), seed=config["seed"]
        )
        split_of = {
            **dict.fromkeys(train, "train"),
            **dict.fromkeys(val, "val"),
            **dict.fromkeys(test, "test"),
        }
        for sample in ds.samples:
            paths[split_of[sample["path"].parent.parent.name]].append(sample["path"])
    return paths


def _features(screen: HistologyScreen, paths) -> np.ndarray:
    rows = []
    for start in range(0, len(paths), 64):
        batch = [Image.open(p).convert("RGB") for p in paths[start : start + 64]]
        rows.append(screen.features(batch))
    return np.concatenate(rows)


def main() -> None:
    net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    net.classifier = torch.nn.Identity()
    torch.save(net.state_dict(), WEIGHTS_PATH)
    screen = HistologyScreen(net)

    paths = _images_by_split()
    print({split: len(p) for split, p in paths.items()})
    train, val, test = (_features(screen, paths[s]) for s in ("train", "val", "test"))

    distance = FeatureDistance.fit(train, N_COMPONENTS)
    distance = distance.with_threshold(float(distance.score(val).max()))
    distance.save(stats_path_for(WEIGHTS_PATH))

    test_rejected = int((distance.score(test) > distance.threshold).sum())
    print(f"threshold {distance.threshold:.1f}; held-out test images rejected: {test_rejected}/{len(test)}")
    print(f"wrote {WEIGHTS_PATH.name} and {stats_path_for(WEIGHTS_PATH).name}")


if __name__ == "__main__":
    main()
