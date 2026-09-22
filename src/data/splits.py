"""Patient-level stratified train/val/test splitting for BreakHis.

Splitting must happen at the patient level, not the image level: multiple
images per patient are highly correlated, so an image-level split would leak
the same patient's tissue into both train and test.
"""

import random
from collections import defaultdict
from typing import Sequence, Tuple


def stratified_patient_split(
    samples: Sequence[dict], ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15), seed: int = 42
) -> Tuple[set, set, set]:
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")

    patients_by_label = defaultdict(set)
    for s in samples:
        patient_id = s["path"].parent.parent.name
        patients_by_label[s["label"]].add(patient_id)

    rng = random.Random(seed)
    train_patients, val_patients, test_patients = set(), set(), set()
    for patients in patients_by_label.values():
        patients = sorted(patients)
        rng.shuffle(patients)  # NOSONAR: reproducible split, not a security context
        n = len(patients)
        n_train = round(n * ratios[0])
        n_val = round(n * ratios[1])
        train_patients.update(patients[:n_train])
        val_patients.update(patients[n_train : n_train + n_val])
        test_patients.update(patients[n_train + n_val :])

    return train_patients, val_patients, test_patients


def filter_samples_by_patients(samples: Sequence[dict], patients: set) -> list:
    return [s for s in samples if s["path"].parent.parent.name in patients]
