"""Feature-distance check: is this image like the slides the model was trained on?

Stage 1's colour-and-texture screen (input_guard.py) catches most photos, but
anything purple and textured enough gets through, and the classifier then
calls it benign or malignant with full confidence. This check looks at the
image the way the model does instead: the 2048-number feature vector ResNet50
computes just before its final layer. Training slides' vectors occupy a
compact region; photos, screenshots and noise land far outside it.

"Far" is measured with probabilistic PCA fitted on the training slides'
vectors: a Mahalanobis distance within the top principal directions, plus the
leftover distance outside them scaled by the average leftover variance. The
threshold is set on validation slides, which the statistics weren't fitted
on (scripts/fit_ood.py).

The statistics are stored next to the model they describe (best_mag200.pt ->
best_mag200.ood.npz), since they are only valid for that model's features.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import numpy as np


@dataclass(frozen=True)
class FeatureDistance:
    mean: np.ndarray  # (d,)
    components: np.ndarray  # (k, d), orthonormal rows
    variances: np.ndarray  # (k,), variance along each component
    residual_variance: float  # average variance outside the components
    threshold: float = float("inf")

    @classmethod
    def fit(cls, features: np.ndarray, n_components: int) -> "FeatureDistance":
        mean = features.mean(axis=0)
        centered = features - mean
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        eigenvalues = singular_values**2 / (len(features) - 1)
        leftover = eigenvalues[n_components:]
        return cls(
            mean=mean,
            components=vt[:n_components],
            variances=eigenvalues[:n_components],
            residual_variance=float(leftover.mean()) if len(leftover) else 1e-12,
        )

    def score(self, features: np.ndarray) -> np.ndarray:
        """Distance of each row from the training slides (larger = less like them)."""
        centered = np.atleast_2d(features) - self.mean
        projected = centered @ self.components.T
        inside = (projected**2 / self.variances).sum(axis=1)
        outside = (centered**2).sum(axis=1) - (projected**2).sum(axis=1)
        return inside + np.maximum(outside, 0.0) / self.residual_variance

    def is_unfamiliar(self, features: np.ndarray) -> bool:
        return bool(self.score(features)[0] > self.threshold)

    def with_threshold(self, threshold: float) -> "FeatureDistance":
        return FeatureDistance(
            self.mean, self.components, self.variances, self.residual_variance, float(threshold)
        )

    def save(self, path: Union[str, Path]) -> None:
        np.savez(
            path,
            mean=self.mean.astype(np.float32),
            components=self.components.astype(np.float32),
            variances=self.variances.astype(np.float32),
            residual_variance=np.float64(self.residual_variance),
            threshold=np.float64(self.threshold),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FeatureDistance":
        with np.load(path) as data:
            return cls(
                mean=data["mean"].astype(np.float64),
                components=data["components"].astype(np.float64),
                variances=data["variances"].astype(np.float64),
                residual_variance=float(data["residual_variance"]),
                threshold=float(data["threshold"]),
            )


def stats_path_for(checkpoint_path: Union[str, Path]) -> Path:
    """best_mag200.pt -> best_mag200.ood.npz, beside the model it describes."""
    return Path(checkpoint_path).with_suffix(".ood.npz")


@lru_cache(maxsize=8)
def feature_distance_for(checkpoint_path: str) -> Optional[FeatureDistance]:
    """The fitted check for a checkpoint, or None if none has been fitted."""
    path = stats_path_for(checkpoint_path)
    return FeatureDistance.load(path) if path.is_file() else None
