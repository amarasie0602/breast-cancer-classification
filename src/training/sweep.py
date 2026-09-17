"""Run a small grid of training configs, logging each as its own MLflow run."""

import copy

from src.training.train import run_training


def run_sweep(base_config, data_root, magnification, variants, **kwargs):
    """variants: list of dicts of config overrides, one per run."""
    results = []
    for variant in variants:
        config = copy.deepcopy(base_config)
        config.update(variant)
        best_f1 = run_training(config, data_root, magnification, **kwargs)
        results.append({"config": variant, "best_f1": best_f1})
    return results
