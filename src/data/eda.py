"""EDA helpers: class balance, per-magnification counts, per-subtype breakdown."""

from collections import Counter
from typing import Sequence


def class_counts(samples: Sequence[dict]) -> Counter:
    return Counter(s["label"] for s in samples)


def magnification_counts(samples: Sequence[dict]) -> Counter:
    return Counter(s["magnification"] for s in samples)


def subtype_counts(samples: Sequence[dict]) -> Counter:
    return Counter(s["subtype"] for s in samples)
