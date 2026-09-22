import mlflow

from src.training.compare_runs import build_comparison_table
from src.training.train import run_training


def _base_config(tmp_path):
    return {
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "lr_scheduler": {"patience": 1, "factor": 0.5},
        "early_stopping": {"patience": 5, "min_delta": 0.0},
        "freeze_backbone_epochs": 0,
        "checkpoint_dir": str(tmp_path / "checkpoints"),
    }


def test_comparison_table_includes_all_runs(breakhis_root_multi_patient, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)

    run_training(
        _base_config(tmp_path),
        breakhis_root_multi_patient,
        magnification="40",
        split_ratios=(0.5, 0.25, 0.25),
        pretrained=False,
    )

    table = build_comparison_table(tracking_uri=tracking_uri)

    assert len(table) == 1
    assert table.iloc[0]["magnification"] == "40"
    assert "val_f1" in table.columns
