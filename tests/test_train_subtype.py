from src.training.checkpoint import load_checkpoint
from src.training.train_subtype import run_training


def test_run_training_keeps_patient_macro_f1_over_higher_image_macro_f1(
    breakhis_root_all_subtypes, tmp_path, monkeypatch
):
    """Image-level macro F1 is logged and is not what decides the saved epoch."""
    monkeypatch.chdir(tmp_path)
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr("src.training.train_subtype.train_one_epoch", lambda *args, **kwargs: 0.1)
    scripted = [
        {"loss": 0.5, "accuracy": 0.9, "macro_f1": 0.90, "patient_macro_f1": 0.10},
        {"loss": 0.4, "accuracy": 0.2, "macro_f1": 0.20, "patient_macro_f1": 0.60},
    ]
    monkeypatch.setattr("src.training.train_subtype.evaluate", lambda *args, **kwargs: scripted.pop(0))

    run_training(
        {
            "epochs": 2,
            "batch_size": 2,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "lr_scheduler": {"patience": 1, "factor": 0.5},
            "early_stopping": {"patience": 5, "min_delta": 0.0},
            "freeze_backbone_epochs": 0,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        },
        breakhis_root_all_subtypes,
        split_ratios=(0.5, 0.5, 0.0),
        pretrained=False,
    )

    from src.models.classifier import MalignantSubtypeClassifier

    model = MalignantSubtypeClassifier(num_classes=4, pretrained=False)
    _, metrics = load_checkpoint(tmp_path / "checkpoints" / "best_subtype.pt", model)
    assert metrics["macro_f1"] == 0.20
    assert metrics["patient_macro_f1"] == 0.60


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
    assert runs.loc[0, "tags.model_variant"] == "efficientnet_b0"


def test_run_training_records_label_scheme_in_checkpoint(
    breakhis_root_all_subtypes, tmp_path, monkeypatch
):
    import mlflow
    import torch

    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    config = {
        "label_scheme": "ductal_vs_other",
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "lr_scheduler": {"patience": 1, "factor": 0.5},
        "early_stopping": {"patience": 5, "min_delta": 0.0},
        "freeze_backbone_epochs": 0,
        "checkpoint_dir": str(tmp_path / "ckpt"),
        "checkpoint_name": "dvo.pt",
    }

    run_training(config, breakhis_root_all_subtypes, split_ratios=(0.5, 0.5, 0.0), pretrained=False)

    saved = torch.load(tmp_path / "ckpt" / "dvo.pt", map_location="cpu", weights_only=True)
    assert saved["metrics"]["label_scheme"] == "ductal_vs_other"
    assert saved["model_state"]["backbone.classifier.1.weight"].shape[0] == 2
