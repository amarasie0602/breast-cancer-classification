import pytest
from PIL import UnidentifiedImageError

from src.data.dataset import BreakHisDataset


def test_dataset_length_counts_all_images(breakhis_root):
    ds = BreakHisDataset(breakhis_root)
    assert len(ds) == 8


def test_dataset_getitem_returns_image_and_label(breakhis_root):
    ds = BreakHisDataset(breakhis_root)
    image, label = ds[0]
    assert image.size == (8, 8)
    assert label in (0, 1)


def test_magnification_filter_restricts_samples(breakhis_root):
    ds = BreakHisDataset(breakhis_root, magnification=40)
    assert len(ds) == 4
    assert all(s["magnification"] == "40" for s in ds.samples)


def test_magnification_filter_with_no_matches_yields_empty_dataset(breakhis_root):
    ds = BreakHisDataset(breakhis_root, magnification=999)
    assert len(ds) == 0


def test_getitem_on_corrupt_image_raises(breakhis_root):
    ds = BreakHisDataset(breakhis_root)
    ds.samples[0]["path"].write_bytes(b"not a real png")

    with pytest.raises(UnidentifiedImageError):
        ds[0]
