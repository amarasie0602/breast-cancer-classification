from pathlib import Path

import pytest

from src.data.dataset import BreakHisDataset
from src.data.splits import filter_samples_by_patients, stratified_patient_split


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


def test_filter_samples_by_patients(breakhis_root):
    ds = BreakHisDataset(breakhis_root)
    train, val, test = stratified_patient_split(ds.samples, ratios=(0.5, 0.25, 0.25))

    train_samples = filter_samples_by_patients(ds.samples, train)
    assert all(s["path"].parent.parent.name in train for s in train_samples)
    assert len(train_samples) + len(filter_samples_by_patients(ds.samples, val)) + len(
        filter_samples_by_patients(ds.samples, test)
    ) == len(ds.samples)


def test_min_per_split_guarantees_small_class_reaches_val_and_test():
    """A 5-patient class: ratio rounding alone gives it 0 test patients."""
    samples = [
        {"path": Path(f"root/subtype/PATIENT-{i}/40X/img.png"), "label": 0} for i in range(5)
    ]

    _, val, test = stratified_patient_split(samples, ratios=(0.7, 0.15, 0.15))
    assert len(test) == 0  # the problem this parameter exists to fix

    train2, val2, test2 = stratified_patient_split(
        samples, ratios=(0.7, 0.15, 0.15), min_per_split=1
    )
    assert len(val2) >= 1
    assert len(test2) >= 1
    assert len(train2) + len(val2) + len(test2) == 5


def test_min_per_split_is_ignored_when_class_is_too_small_to_satisfy():
    samples = [{"path": Path(f"root/subtype/P-{i}/40X/img.png"), "label": 0} for i in range(2)]

    train, val, test = stratified_patient_split(
        samples, ratios=(0.7, 0.15, 0.15), min_per_split=1
    )
    assert len(train) + len(val) + len(test) == 2
