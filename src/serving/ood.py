"""Feature-distance check: is this image like the slides the model was trained on?

Stage 1's colour-and-texture screen (input_guard.py) lets through anything
purple and textured enough, and the classifier then calls it benign or
malignant with full confidence. This second check describes the image with a
general-purpose ImageNet EfficientNet-B0 and asks how far that description is
from the training slides'.

It deliberately doesn't use the cancer models' own features: fine-tuned only
on histology, they place wallpapers and screenshots right among the slides.
A network that still knows photos from slides separates them.

"Far" is measured with probabilistic PCA fitted on the training slides'
features: a Mahalanobis distance within the top principal directions, plus the
leftover distance outside them scaled by the average leftover variance. The
threshold is set on validation slides (scripts/fit_histology_screen.py).
Statistics are stored beside the weights they describe
(histology_screen.pt -> histology_screen.ood.npz).
"""

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import efficientnet_b0

from src.data.transforms import eval_transform

logger = logging.getLogger(__name__)


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


def stats_path_for(weights_path: Union[str, Path]) -> Path:
    """histology_screen.pt -> histology_screen.ood.npz, beside the weights."""
    return Path(weights_path).with_suffix(".ood.npz")


class HistologyScreen:
    """A feature extractor plus the fitted distance that decides what's familiar."""

    def __init__(self, network: nn.Module, distance: Optional[FeatureDistance] = None):
        self.network = network.eval()
        self.distance = distance
        self._transform = eval_transform()

    @classmethod
    def load(cls, weights_path: Union[str, Path]) -> "HistologyScreen":
        network = efficientnet_b0(weights=None)
        network.classifier = nn.Identity()
        network.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        return cls(network, FeatureDistance.load(stats_path_for(weights_path)))

    def features(self, images: List[Image.Image]) -> np.ndarray:
        with torch.no_grad():
            batch = torch.stack([self._transform(image.convert("RGB")) for image in images])
            return self.network(batch).numpy()

    def is_unfamiliar(self, image: Image.Image) -> bool:
        return self.distance.is_unfamiliar(self.features([image]))


@lru_cache(maxsize=2)
def load_histology_screen(weights_path: str) -> Optional[HistologyScreen]:
    """The fitted screen, or None if it hasn't been fitted or can't be read
    (e.g. a Git LFS pointer in a checkout without LFS)."""
    if not (Path(weights_path).is_file() and stats_path_for(weights_path).is_file()):
        return None
    try:
        return HistologyScreen.load(weights_path)
    except Exception:  # noqa: BLE001 - any unreadable file means the same thing here
        logger.warning("histology screen at %s could not be loaded; skipping it", weights_path)
        return None
