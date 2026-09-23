from src.training.train_subtype import run_training


def test_run_training_end_to_end_smoke(breakhis_root_all_subtypes, tmp_path, monkeypatch):
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

    # Only 2 patients per subtype in the fixture: (0.5, 0.25, 0.25) would
    # round the val split down to 0 patients per group (round(0.5) == 0),
    # leaving an empty val set. (0.5, 0.5, 0.0) keeps both train and val
    # non-empty, which is all this smoke test needs.
    best_macro_f1 = run_training(
        config,
        breakhis_root_all_subtypes,
        split_ratios=(0.5, 0.5, 0.0),
        pretrained=False,
    )

    assert best_macro_f1 >= 0.0
    assert (tmp_path / "checkpoints" / "best_subtype.pt").exists()

    runs = mlflow.search_runs()
    assert runs.loc[0, "tags.task"] == "malignant_subtype"
    assert runs.loc[0, "tags.model_variant"] == "resnet50"
