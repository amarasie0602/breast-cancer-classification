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


def test_jet_matches_matplotlib_exactly():
    # overlay.py carries its own copy of the colormap so serving doesn't need
    # matplotlib; it must render the same heat map, byte for byte.
    import numpy as np
    from matplotlib import colormaps

    from src.explainability.overlay import jet

    values = np.linspace(0.0, 1.0, 4097)
    ours = (jet(values) * 255).astype(np.uint8)
    theirs = (colormaps["jet"](values)[:, :3] * 255).astype(np.uint8)
    assert np.array_equal(ours, theirs)
