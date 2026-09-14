"""Shared pytest fixtures."""

from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def breakhis_root(tmp_path):
    """Build a tiny synthetic BreakHis-shaped directory tree for testing."""
    layout = [
        ("benign", "adenosis", "SOB_B_A-14-22549AB", "40"),
        ("benign", "adenosis", "SOB_B_A-14-22549AB", "100"),
        ("malignant", "ductal_carcinoma", "SOB_M_DC-14-2523", "40"),
        ("malignant", "ductal_carcinoma", "SOB_M_DC-14-2523", "400"),
    ]
    for label, subtype, patient, mag in layout:
        mag_dir = (
            tmp_path / "histology_slides" / "breast" / label / "SOB" / subtype / patient / f"{mag}X"
        )
        mag_dir.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            img = Image.new("RGB", (8, 8), color=(i * 10, 0, 0))
            img.save(mag_dir / f"{patient}-{mag}-{i:03d}.png")
    return tmp_path
