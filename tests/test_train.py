import pytest

from src.training.train import load_config, run_training


def test_load_config_reads_yaml_within_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("epochs: 5\n")

    config = load_config("config.yaml")

    assert config == {"epochs": 5}


def test_load_config_rejects_path_outside_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="within the project directory"):
        load_config("../../../../etc/passwd")


def test_run_training_end_to_end_smoke(breakhis_root_multi_patient, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")

    config = {
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "lr_scheduler": {"patience": 1, "factor": 0.5},
        "early_stopping": {"patience": 5, "min_delta": 0.0},
        "freeze_backbone_epochs": 0,
        "checkpoint_dir": str(tmp_path / "checkpoints"),
    }

    best_f1 = run_training(
        config,
        breakhis_root_multi_patient,
        magnification="40",
        split_ratios=(0.5, 0.25, 0.25),
        pretrained=False,
    )

    assert best_f1 >= 0.0
    assert (tmp_path / "checkpoints" / "best_mag40.pt").exists()

    runs = mlflow.search_runs()
    assert runs.loc[0, "tags.magnification"] == "40"
    assert runs.loc[0, "tags.model_variant"] == "resnet50"


def test_build_dataloaders_balances_classes_when_requested(breakhis_root_multi_patient):
    from torch.utils.data import WeightedRandomSampler

    from src.training.train import build_dataloaders

    plain, _ = build_dataloaders(
        breakhis_root_multi_patient, "40", (0.5, 0.25, 0.25), 42, 2, balance_classes=False
    )
    balanced, _ = build_dataloaders(
        breakhis_root_multi_patient, "40", (0.5, 0.25, 0.25), 42, 2, balance_classes=True
    )

    assert plain.sampler is not None and not isinstance(plain.sampler, WeightedRandomSampler)
    assert isinstance(balanced.sampler, WeightedRandomSampler)

    # Rarer class must carry the heavier per-sample weight.
    weights_by_label = {}
    for sample, weight in zip(balanced.dataset.samples, balanced.sampler.weights.tolist(), strict=True):
        weights_by_label.setdefault(sample["label"], set()).add(round(weight, 6))
    for weights in weights_by_label.values():
        assert len(weights) == 1  # one weight per class, not per image
