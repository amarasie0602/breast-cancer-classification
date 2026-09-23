"""Shared pytest fixtures."""

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


@pytest.fixture
def breakhis_root_multi_patient(tmp_path):
    """Multiple patients per class at a single magnification, for split/training tests."""
    patients = [
        ("benign", "adenosis", "SOB_B_A-14-1"),
        ("benign", "adenosis", "SOB_B_A-14-2"),
        ("benign", "fibroadenoma", "SOB_B_F-14-3"),
        ("benign", "fibroadenoma", "SOB_B_F-14-4"),
        ("malignant", "ductal_carcinoma", "SOB_M_DC-14-5"),
        ("malignant", "ductal_carcinoma", "SOB_M_DC-14-6"),
        ("malignant", "lobular_carcinoma", "SOB_M_LC-14-7"),
        ("malignant", "lobular_carcinoma", "SOB_M_LC-14-8"),
    ]
    for label, subtype, patient in patients:
        mag_dir = (
            tmp_path / "histology_slides" / "breast" / label / "SOB" / subtype / patient / "40X"
        )
        mag_dir.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            img = Image.new("RGB", (8, 8), color=(i * 10, 0, 0))
            img.save(mag_dir / f"{patient}-40-{i:03d}.png")
    return tmp_path


@pytest.fixture
def breakhis_root_all_subtypes(tmp_path):
    """Two patients per malignant subtype (all four), for subtype-classifier tests."""
    patients = [
        ("malignant", "ductal_carcinoma", "SOB_M_DC-14-1"),
        ("malignant", "ductal_carcinoma", "SOB_M_DC-14-2"),
        ("malignant", "lobular_carcinoma", "SOB_M_LC-14-3"),
        ("malignant", "lobular_carcinoma", "SOB_M_LC-14-4"),
        ("malignant", "mucinous_carcinoma", "SOB_M_MC-14-5"),
        ("malignant", "mucinous_carcinoma", "SOB_M_MC-14-6"),
        ("malignant", "papillary_carcinoma", "SOB_M_PC-14-7"),
        ("malignant", "papillary_carcinoma", "SOB_M_PC-14-8"),
    ]
    for label, subtype, patient in patients:
        mag_dir = (
            tmp_path / "histology_slides" / "breast" / label / "SOB" / subtype / patient / "40X"
        )
        mag_dir.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            img = Image.new("RGB", (8, 8), color=(i * 10, 0, 0))
            img.save(mag_dir / f"{patient}-40-{i:03d}.png")
    return tmp_path
