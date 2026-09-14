from src.data.dataset import BreakHisDataset
from src.data.eda import class_counts, magnification_counts, subtype_counts


def test_class_counts(breakhis_root):
    ds = BreakHisDataset(breakhis_root)
    counts = class_counts(ds.samples)
    assert counts[0] == 4  # benign
    assert counts[1] == 4  # malignant


def test_magnification_counts(breakhis_root):
    ds = BreakHisDataset(breakhis_root)
    counts = magnification_counts(ds.samples)
    assert counts["40"] == 4
    assert counts["100"] == 2
    assert counts["400"] == 2


def test_subtype_counts(breakhis_root):
    ds = BreakHisDataset(breakhis_root)
    counts = subtype_counts(ds.samples)
    assert counts["adenosis"] == 4
    assert counts["ductal_carcinoma"] == 4
