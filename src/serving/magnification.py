"""Recognise the magnification an uploaded image was taken at.

Each magnification has its own binary model, and a model applied to another
zoom level reads tissue at the wrong scale: the same image scored 52%
malignant with the 40x model, 41% with 100x, 52% with 200x and 24% with
400x. Users often don't know the zoom, so serving detects it and picks the
matching model (scripts/train_magnification.py trains the detector).
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from PIL import Image
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from src.data.transforms import eval_transform

logger = logging.getLogger(__name__)

MAGNIFICATIONS = ("40", "100", "200", "400")


def build_network(pretrained: bool = False) -> nn.Module:
    """EfficientNet-B0 with a 4-way magnification head. Training starts from
    ImageNet weights; serving loads the trained ones over an empty network."""
    network = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    network.classifier[1] = nn.Linear(network.classifier[1].in_features, len(MAGNIFICATIONS))
    return network


class MagnificationDetector:
    def __init__(self, network: nn.Module):
        self.network = network.eval()
        self._transform = eval_transform()

    @classmethod
    def load(cls, weights_path: Union[str, Path]) -> "MagnificationDetector":
        network = build_network()
        network.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        return cls(network)

    def detect(self, image: Image.Image) -> Tuple[str, float]:
        """The most likely magnification and the model's probability for it."""
        with torch.no_grad():
            probs = self.network(self._transform(image.convert("RGB")).unsqueeze(0)).softmax(1)[0]
        index = int(probs.argmax())
        return MAGNIFICATIONS[index], float(probs[index])


@lru_cache(maxsize=2)
def load_magnification_detector(weights_path: str) -> Optional[MagnificationDetector]:
    """The trained detector, or None if it isn't there or can't be read
    (e.g. a Git LFS pointer in a checkout without LFS)."""
    if not Path(weights_path).is_file():
        return None
    try:
        return MagnificationDetector.load(weights_path)
    except Exception:  # noqa: BLE001 - any unreadable file means the same thing here
        logger.warning("magnification detector at %s could not be loaded; skipping it", weights_path)
        return None
