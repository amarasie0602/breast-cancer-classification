import numpy as np
from PIL import Image

from src.serving.input_guard import looks_like_histology


def _noisy_image(base_rgb, size=64, std=35, seed=0):
    rng = np.random.default_rng(seed)
    base = np.array(base_rgb, dtype="float32")
    noise = rng.normal(0, std, size=(size, size, 3))
    arr = np.clip(base + noise, 0, 255).astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def test_accepts_textured_purple_pink_image():
    image = _noisy_image((170, 90, 150))
    assert looks_like_histology(image) is True


def test_rejects_flat_solid_color_image():
    image = Image.new("RGB", (64, 64), color=(170, 90, 150))
    assert looks_like_histology(image) is False


def test_rejects_textured_but_wrong_hue_image():
    image = _noisy_image((90, 170, 100))  # green, not purple/pink
    assert looks_like_histology(image) is False


def test_rejects_black_and_white_image():
    assert looks_like_histology(Image.new("RGB", (64, 64), color=(0, 0, 0))) is False
    assert looks_like_histology(Image.new("RGB", (64, 64), color=(255, 255, 255))) is False
