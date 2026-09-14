import pytest

from src.data.dataset import BreakHisDataset
from src.data.splits import stratified_patient_split


def test_split_ratios_must_sum_to_one():
    with pytest.raises(ValueError):
        stratified_patient_split([], ratios=(0.5, 0.4, 0.2))


def test_split_is_disjoint_and_covers_all_patients(breakhis_root):
    ds = BreakHisDataset(breakhis_root)
    train, val, test = stratified_patient_split(ds.samples, ratios=(0.5, 0.25, 0.25))

    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)

    all_patients = {s["path"].parent.parent.name for s in ds.samples}
    assert train | val | test == all_patients
