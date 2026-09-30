import torch
from PIL import Image

from src.serving.magnification import (
    MAGNIFICATIONS,
    MagnificationDetector,
    build_network,
    load_magnification_detector,
)


def test_detect_returns_a_known_magnification_and_a_probability():
    detector = MagnificationDetector(build_network())
    magnification, confidence = detector.detect(Image.new("RGB", (70, 46), (200, 120, 180)))
    assert magnification in MAGNIFICATIONS
    assert 0.0 <= confidence <= 1.0


def test_saved_weights_load_back_into_a_detector(tmp_path):
    torch.manual_seed(0)
    network = build_network()
    path = tmp_path / "magnification.pt"
    torch.save(network.state_dict(), path)

    image = Image.new("RGB", (70, 46), (180, 90, 160))
    assert MagnificationDetector.load(path).detect(image) == MagnificationDetector(network).detect(image)


def test_missing_or_unreadable_detector_is_skipped(tmp_path):
    path = tmp_path / "magnification.pt"
    assert load_magnification_detector(str(path)) is None
    path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\n")
    assert load_magnification_detector(str(path)) is None
