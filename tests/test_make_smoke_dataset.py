from scripts.make_smoke_dataset import LAYOUT, MAGNIFICATIONS, make_smoke_dataset
from src.data.dataset import BreakHisDataset


def test_make_smoke_dataset_is_loadable_by_breakhis_dataset(tmp_path):
    make_smoke_dataset(tmp_path)

    ds = BreakHisDataset(tmp_path)
    assert len(ds) > 0

    ds_40 = BreakHisDataset(tmp_path, magnification=40)
    assert len(ds_40) > 0
    assert len(ds_40) == len(LAYOUT) * 3  # IMAGES_PER_MAG

    for mag in MAGNIFICATIONS:
        assert len(BreakHisDataset(tmp_path, magnification=int(mag))) > 0
