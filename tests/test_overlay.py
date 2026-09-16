import numpy as np
from PIL import Image

from src.explainability.overlay import cam_to_overlay


def test_overlay_matches_original_image_size():
    original = Image.new("RGB", (32, 32), color=(100, 100, 100))
    cam = np.random.rand(7, 7).astype(np.float32)

    overlay = cam_to_overlay(cam, original)

    assert overlay.size == original.size
    assert overlay.mode == "RGB"


def test_overlay_with_zero_alpha_returns_original():
    original = Image.new("RGB", (16, 16), color=(50, 60, 70))
    cam = np.random.rand(4, 4).astype(np.float32)

    overlay = cam_to_overlay(cam, original, alpha=0.0)

    assert np.array_equal(np.asarray(overlay), np.asarray(original))
