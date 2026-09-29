import numpy as np
import pytest

from src.serving.ood import FeatureDistance, HistologyScreen, load_histology_screen, stats_path_for


def _training_features(n=400, d=32, seed=0):
    """Correlated features living mostly in a 4-dimensional subspace."""
    basis = np.random.default_rng(0).normal(size=(4, d))  # same subspace for every call
    rng = np.random.default_rng(seed + 1)
    return rng.normal(size=(n, 4)) @ basis * 3 + rng.normal(scale=0.3, size=(n, d)) + 5


def test_points_like_the_training_data_score_lower_than_points_unlike_it():
    train = _training_features()
    check = FeatureDistance.fit(train, n_components=4)

    familiar = check.score(_training_features(n=200, seed=1))
    # Same scale, but pointing in directions the training data never varies in.
    unfamiliar = check.score(np.random.default_rng(2).normal(scale=3, size=(200, 32)) + 5)

    assert np.percentile(familiar, 99) < np.percentile(unfamiliar, 1)


def test_threshold_decides_what_counts_as_unfamiliar():
    train = _training_features()
    check = FeatureDistance.fit(train, n_components=4)
    typical = check.score(train)
    check = check.with_threshold(float(np.max(typical)))

    assert not check.is_unfamiliar(train[:1])
    assert check.is_unfamiliar(np.full((1, 32), 500.0))


def test_unfitted_threshold_never_rejects():
    check = FeatureDistance.fit(_training_features(), n_components=4)
    assert not check.is_unfamiliar(np.full((1, 32), 1e6))


def test_round_trips_through_its_file(tmp_path):
    train = _training_features()
    check = FeatureDistance.fit(train, n_components=4).with_threshold(123.5)
    path = tmp_path / "best_mag40.ood.npz"

    check.save(path)
    loaded = FeatureDistance.load(path)

    assert loaded.threshold == pytest.approx(123.5)
    np.testing.assert_allclose(loaded.score(train[:20]), check.score(train[:20]), rtol=1e-4)


def test_stats_live_beside_their_weights(tmp_path):
    weights = tmp_path / "histology_screen.pt"
    assert stats_path_for(weights) == tmp_path / "histology_screen.ood.npz"


def test_screen_is_skipped_when_not_fitted_or_unreadable(tmp_path):
    weights = tmp_path / "histology_screen.pt"
    assert load_histology_screen(str(weights)) is None  # nothing fitted

    weights.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\n")
    FeatureDistance.fit(_training_features(), n_components=4).save(stats_path_for(weights))
    assert load_histology_screen(str(weights)) is None  # an LFS pointer, not weights


def test_screen_flags_what_its_distance_says_is_unfamiliar():
    from PIL import Image
    from torch import nn

    class MeanColour(nn.Module):  # a stand-in network: features = mean RGB
        def forward(self, x):
            return x.mean(dim=(2, 3))

    screen = HistologyScreen(MeanColour())
    rng = np.random.default_rng(0)
    slides = [
        Image.fromarray(np.full((32, 32, 3), (180 + rng.integers(-8, 8), 100, 170), np.uint8))
        for _ in range(60)
    ]
    distance = FeatureDistance.fit(screen.features(slides), n_components=2)
    familiar = distance.score(screen.features(slides))
    screen.distance = distance.with_threshold(float(familiar.max()) * 1.5)

    assert not screen.is_unfamiliar(slides[0])
    assert screen.is_unfamiliar(Image.new("RGB", (32, 32), (20, 200, 40)))
