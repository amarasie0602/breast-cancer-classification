from src.training.train import run_training


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
