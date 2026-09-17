"""Generate a tiny synthetic BreakHis-shaped dataset for CI smoke-testing the training CLI."""

import argparse
from pathlib import Path

from PIL import Image

LAYOUT = [
    ("benign", "adenosis", "SOB_B_A-14-22549AB"),
    ("benign", "adenosis", "SOB_B_A-14-22549CD"),
    ("benign", "fibroadenoma", "SOB_B_F-14-9133"),
    ("benign", "fibroadenoma", "SOB_B_F-14-14134E"),
    ("malignant", "ductal_carcinoma", "SOB_M_DC-14-2523"),
    ("malignant", "ductal_carcinoma", "SOB_M_DC-14-5287"),
    ("malignant", "lobular_carcinoma", "SOB_M_LC-14-15570"),
    ("malignant", "lobular_carcinoma", "SOB_M_LC-14-13412"),
]
MAGNIFICATIONS = ["40", "100", "200", "400"]
IMAGES_PER_MAG = 3


def make_smoke_dataset(root):
    root = Path(root)
    for label, subtype, patient in LAYOUT:
        for mag in MAGNIFICATIONS:
            mag_dir = root / "histology_slides" / "breast" / label / "SOB" / subtype / patient / f"{mag}X"
            mag_dir.mkdir(parents=True, exist_ok=True)
            for i in range(IMAGES_PER_MAG):
                Image.new("RGB", (64, 64)).save(mag_dir / f"{patient}-{mag}-{i}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    make_smoke_dataset(args.root)
