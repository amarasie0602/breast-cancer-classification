"""EDA helpers: class balance, per-magnification counts, per-subtype breakdown."""

from collections import Counter


def class_counts(samples):
    return Counter(s["label"] for s in samples)


def magnification_counts(samples):
    return Counter(s["magnification"] for s in samples)


def subtype_counts(samples):
    return Counter(s["subtype"] for s in samples)
