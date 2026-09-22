import mlflow

from src.training.sweep import run_sweep


def test_run_sweep_produces_one_result_per_variant(breakhis_root_multi_patient, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")

    base_config = {
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "lr_scheduler": {"patience": 1, "factor": 0.5},
        "early_stopping": {"patience": 5, "min_delta": 0.0},
        "freeze_backbone_epochs": 0,
        "checkpoint_dir": str(tmp_path / "checkpoints"),
    }
    variants = [{"learning_rate": 1e-4}, {"learning_rate": 1e-3}]

    results = run_sweep(
        base_config,
        breakhis_root_multi_patient,
        magnification="40",
        variants=variants,
        split_ratios=(0.5, 0.25, 0.25),
        pretrained=False,
    )

    assert len(results) == 2
    assert all("best_f1" in r for r in results)

    runs = mlflow.search_runs()
    assert len(runs) == 2
