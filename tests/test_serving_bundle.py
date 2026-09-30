"""The image must ship a model for every magnification the UI offers.

When only best_mag40.pt was copied into the image, routing silently fell
back to the 40x model for every request in production. Nothing failed, the
answers just came from the wrong model. These checks turn that into a test
failure. They only look at which files exist, so they also pass in CI jobs
where the checkpoints are Git LFS pointers.
"""

from pathlib import Path

from scripts.export_serving_checkpoints import CHECKPOINT_NAMES
from src.serving.app import ALLOWED_MAGNIFICATIONS

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVING_DIR = REPO_ROOT / "serving_checkpoints"


def test_every_selectable_magnification_has_a_committed_model():
    missing = [
        m for m in ALLOWED_MAGNIFICATIONS if not (SERVING_DIR / f"best_mag{m}.pt").is_file()
    ]
    assert not missing, f"no serving model for magnification(s) {', '.join(missing)}"


def test_export_script_covers_every_selectable_magnification():
    expected = {f"best_mag{m}.pt" for m in ALLOWED_MAGNIFICATIONS}
    assert expected <= set(CHECKPOINT_NAMES)


def test_image_copies_the_whole_serving_directory():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text()
    assert "COPY serving_checkpoints/ serving_checkpoints/" in dockerfile


def test_stage_1_feature_screen_ships_with_the_models():
    # Without these two files serving silently skips the second stage-1 check
    # and falls back to the colour screen alone.
    assert (SERVING_DIR / "histology_screen.pt").is_file()
    assert (SERVING_DIR / "histology_screen.ood.npz").is_file()


def test_magnification_detector_ships_with_the_models():
    # Without it /predict can't auto-detect and asks the user to choose.
    assert (SERVING_DIR / "magnification.pt").is_file()
