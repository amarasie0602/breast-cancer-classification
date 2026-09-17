"""In-memory counters for served predictions (label distribution, latency)."""

from collections import Counter
from typing import Dict

prediction_counts: Counter = Counter()


def record_prediction(label: str) -> None:
    prediction_counts[label] += 1


def get_prediction_distribution() -> Dict[str, int]:
    return dict(prediction_counts)
