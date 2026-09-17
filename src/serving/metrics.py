"""In-memory counters for served predictions (label distribution, latency)."""

from collections import Counter

prediction_counts = Counter()


def record_prediction(label):
    prediction_counts[label] += 1


def get_prediction_distribution():
    return dict(prediction_counts)
