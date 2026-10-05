import torch

from src.serving.calibration import dihedral_views, load_calibration


def test_dihedral_views_are_the_8_distinct_rotations_and_flips():
    images = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).view(2, 3, 4, 4)
    views = dihedral_views(images)

    assert views.shape == (8, 2, 3, 4, 4)
    assert torch.equal(views[0], images)
    assert len({tuple(v.flatten().tolist()) for v in views[:, 0]}) == 8


def test_missing_or_malformed_calibration_means_none(tmp_path):
    assert load_calibration(str(tmp_path / "absent.json")) is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert load_calibration(str(bad)) is None


def test_unknown_checkpoint_gets_no_temperature(tmp_path):
    good = tmp_path / "calibration.json"
    good.write_text('{"tta": false, "models": {}}')
    checkpoint = tmp_path / "best_mag40.pt"
    checkpoint.write_bytes(b"weights")

    assert load_calibration(str(good)).temperature_for(checkpoint) == 1.0
