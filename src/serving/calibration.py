"""How a served binary model's output becomes the probability shown.

Two adjustments, both fitted on validation patients by
scripts/calibrate_serving.py and stored in serving_checkpoints/calibration.json:

- Test-time augmentation: tissue has no "up", so the image can be scored in
  its 8 rotations and flips and the logits averaged.
- Temperature scaling: the raw probabilities were over-confident (images
  scored 70-80% malignant were right about 64% of the time on validation).
  Dividing the logit by a fitted temperature T makes "80%" mean about 80%,
  without changing which side of 0.5 any image falls on.

A temperature is only valid for the exact checkpoint it was fitted on, so the
file records each checkpoint's SHA-256 and a mismatched model gets no
temperature (T = 1) rather than a wrong one.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def dihedral_views(images: torch.Tensor) -> torch.Tensor:
    """The 8 rotations and flips of a (B, C, H, W) batch of square images,
    stacked as (8, B, C, H, W); view 0 is the original."""
    views = []
    for quarter_turns in range(4):
        rotated = torch.rot90(images, quarter_turns, dims=(-2, -1))
        views += [rotated, torch.flip(rotated, dims=(-1,))]
    return torch.stack(views)


@lru_cache(maxsize=8)
def _sha256(path: str, size: int, mtime: float) -> str:
    # size and mtime are part of the cache key, so a replaced file is rehashed.
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Calibration:
    tta: bool
    models: dict  # checkpoint file name -> {"temperature": float, "sha256": str}

    def temperature_for(self, checkpoint_path: str | Path) -> float:
        path = Path(checkpoint_path)
        entry = self.models.get(path.name)
        if entry is None:
            return 1.0
        stat = path.stat()
        if _sha256(str(path), stat.st_size, stat.st_mtime) != entry["sha256"]:
            logger.warning("calibration for %s was fitted on a different checkpoint; not applied", path.name)
            return 1.0
        return float(entry["temperature"])


@lru_cache(maxsize=2)
def load_calibration(path: str) -> Calibration | None:
    """The fitted calibration, or None if there is none (serving then uses
    the plain, uncalibrated output)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return Calibration(tta=bool(data["tta"]), models=dict(data["models"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None
