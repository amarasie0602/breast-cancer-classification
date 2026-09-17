"""Compare best val metrics across MLflow runs, grouped by magnification."""

import argparse

import mlflow
import pandas as pd


def build_comparison_table(
    tracking_uri: str = "sqlite:///mlflow.db", experiment_name: str = "Default"
) -> pd.DataFrame:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment-name", default="Default")
    args = parser.parse_args()

    table = build_comparison_table(args.tracking_uri, args.experiment_name)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
