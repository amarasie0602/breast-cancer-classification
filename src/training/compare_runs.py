"""Compare best val metrics across MLflow runs, grouped by magnification."""

import argparse

import mlflow


def build_comparison_table(tracking_uri="sqlite:///mlflow.db", experiment_name="Default"):
    mlflow.set_tracking_uri(tracking_uri)
    runs = mlflow.search_runs(experiment_names=[experiment_name])

    if runs.empty:
        return runs

    columns = [
        c
        for c in ["tags.magnification", "tags.model_variant", "metrics.val_f1", "metrics.val_accuracy"]
        if c in runs.columns
    ]
    table = runs[columns].sort_values("metrics.val_f1", ascending=False)
    return table.rename(columns=lambda c: c.split(".")[-1])
