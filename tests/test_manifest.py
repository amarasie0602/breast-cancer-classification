import json
import sys

import pytest

from scripts.freeze_patient_splits import build_manifest, main
from src.data.dataset import BreakHisDataset
from src.data.manifest import load_manifest, patient_id_of, resolve_patient_split, write_manifest
from src.data.splits import stratified_patient_split


def test_manifest_round_trip_covers_every_patient(breakhis_root_multi_patient, tmp_path, monkeypatch):
    config = {"split_ratios": {"train": 0.5, "val": 0.25, "test": 0.25}, "seed": 7}
    manifest = build_manifest(breakhis_root_multi_patient, config)
    monkeypatch.chdir(tmp_path)
    path = write_manifest(manifest, "patients.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))

    dataset = BreakHisDataset(breakhis_root_multi_patient, magnification="40")
    present = {patient_id_of(sample) for sample in dataset.samples}
    split = loaded["binary"]["40"]
    listed = set(split["train"] + split["val"] + split["test"])
    assert listed == present
    assert set(loaded["binary"]["40"]["train"]).isdisjoint(loaded["binary"]["40"]["val"])
    assert set(loaded["binary"]["40"]["val"]).isdisjoint(loaded["binary"]["40"]["test"])


def test_resolve_uses_the_manifest_when_it_covers_the_patients(
    breakhis_root_multi_patient, tmp_path, monkeypatch
):
    dataset = BreakHisDataset(breakhis_root_multi_patient, magnification="40")
    patients = sorted({patient_id_of(sample) for sample in dataset.samples})
    manifest = {
        "version": 1,
        "binary": {"40": {"train": patients[:2], "val": patients[2:3], "test": patients[3:]}},
    }
    monkeypatch.chdir(tmp_path)
    path = write_manifest(manifest, "patients.json")
    train, val, test = resolve_patient_split(
        dataset.samples,
        ratios=(0.5, 0.25, 0.25),
        seed=1,
        manifest_key=("binary", "40"),
        manifest_path=path,
    )
    assert train == set(patients[:2])
    assert val == set(patients[2:3])
    assert test == set(patients[3:])


def test_resolve_falls_back_when_the_manifest_is_a_different_dataset(
    breakhis_root_multi_patient, tmp_path, monkeypatch
):
    dataset = BreakHisDataset(breakhis_root_multi_patient, magnification="40")
    manifest = {
        "version": 1,
        "binary": {"40": {"train": ["OTHER-1"], "val": ["OTHER-2"], "test": ["OTHER-3"]}},
    }
    monkeypatch.chdir(tmp_path)
    path = write_manifest(manifest, "patients.json")
    resolved = resolve_patient_split(
        dataset.samples,
        ratios=(0.5, 0.25, 0.25),
        seed=3,
        manifest_key=("binary", "40"),
        manifest_path=path,
    )
    assert resolved == stratified_patient_split(dataset.samples, ratios=(0.5, 0.25, 0.25), seed=3)


def test_resolve_rejects_a_stale_partial_manifest(breakhis_root_multi_patient, tmp_path, monkeypatch):
    dataset = BreakHisDataset(breakhis_root_multi_patient, magnification="40")
    one = patient_id_of(dataset.samples[0])
    manifest = {"version": 1, "binary": {"40": {"train": [one], "val": ["OTHER-2"], "test": ["OTHER-3"]}}}
    monkeypatch.chdir(tmp_path)
    path = write_manifest(manifest, "patients.json")
    with pytest.raises(ValueError, match="split manifest is missing"):
        resolve_patient_split(
            dataset.samples,
            ratios=(0.5, 0.25, 0.25),
            seed=3,
            manifest_key=("binary", "40"),
            manifest_path=path,
        )


def test_write_and_load_reject_a_path_outside_the_project(tmp_path):
    outside = tmp_path / "patients.json"
    with pytest.raises(ValueError, match="manifest path"):
        write_manifest({"version": 1}, outside)
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest path"):
        load_manifest(outside)


def test_write_manifest_rejects_parent_directory_traversal(tmp_path, monkeypatch):
    inside = tmp_path / "inside"
    inside.mkdir()
    monkeypatch.chdir(inside)
    with pytest.raises(ValueError, match="manifest path"):
        write_manifest({"version": 1}, "../patients.json")


def test_freeze_rejects_data_root_outside_the_project(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["freeze_patient_splits", "--data-root", "/tmp/not-in-this-repo"])
    with pytest.raises(ValueError, match="data root"):
        main()


def test_freeze_rejects_output_outside_the_project(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["freeze_patient_splits", "--data-root", "data", "--output", "/tmp/patients.json"],
    )
    with pytest.raises(ValueError, match="manifest path"):
        main()
