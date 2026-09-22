"""Run a small grid of training configs, logging each as its own MLflow run."""

import argparse
import copy
from pathlib import Path
from typing import List, Union

from src.training.train import load_config, run_training


def run_sweep(
    base_config: dict, data_root: Union[str, Path], magnification: str, variants: List[dict], **kwargs
) -> List[dict]:
    """variants: list of dicts of config overrides, one per run."""
    results = []
    for variant in variants:
        config = copy.deepcopy(base_config)
        config.update(variant)
        best_f1 = run_training(config, data_root, magnification, **kwargs)
        results.append({"config": variant, "best_f1": best_f1})
    return results


def main() -> None:
    import mlflow

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--magnification", required=True, choices=["40", "100", "200", "400"])
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--sweep-config", default="configs/sweep.yaml")
    parser.add_argument("--mlflow-tracking-uri", default="sqlite:///mlflow.db")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    base_config = load_config(args.train_config)
    data_config = load_config(args.data_config)
    sweep_config = load_config(args.sweep_config)
    ratios = data_config["split_ratios"]

    results = run_sweep(
        base_config,
        args.data_root,
        args.magnification,
        sweep_config["variants"],
        split_ratios=(ratios["train"], ratios["val"], ratios["test"]),
        seed=data_config["seed"],
    )

    for r in sorted(results, key=lambda r: r["best_f1"], reverse=True):
        print(f"{r['config']} -> best_f1={r['best_f1']:.4f}")


if __name__ == "__main__":
    main()
