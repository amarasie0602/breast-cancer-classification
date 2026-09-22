"""In-memory counters for served predictions (label distribution, latency)."""

from collections import Counter

prediction_counts: Counter[str] = Counter()


def record_prediction(label: str) -> None:
    prediction_counts[label] += 1


def get_prediction_distribution() -> dict[str, int]:
    return dict(prediction_counts)
