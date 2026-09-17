"""Compare best val metrics across MLflow runs, grouped by magnification."""

import argparse

import mlflow
import pandas as pd


def build_comparison_table(
    tracking_uri: str = "sqlite:///mlflow.db", experiment_name: str = "Default"
) -> pd.DataFrame:
    """Reports each run's *best* epoch metrics, not its last-logged (final) epoch.

    Early stopping continues a few epochs past the best one, so the final
    logged value understates what the checkpointed model actually achieves.
    """
    mlflow.set_tracking_uri(tracking_uri)
    runs = mlflow.search_runs(experiment_names=[experiment_name])

    if runs.empty:
        return runs

    client = mlflow.MlflowClient()
    rows = []
    for _, run in runs.iterrows():
        f1_history = client.get_metric_history(run["run_id"], "val_f1")
        if not f1_history:
            continue
        best = max(f1_history, key=lambda m: m.value)

        acc_history = client.get_metric_history(run["run_id"], "val_accuracy")
        acc_at_best = next((m.value for m in acc_history if m.step == best.step), None)

        rows.append(
            {
                "magnification": run.get("tags.magnification"),
                "model_variant": run.get("tags.model_variant"),
                "val_f1": best.value,
                "val_accuracy": acc_at_best,
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values("val_f1", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment-name", default="Default")
    args = parser.parse_args()

    table = build_comparison_table(args.tracking_uri, args.experiment_name)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
