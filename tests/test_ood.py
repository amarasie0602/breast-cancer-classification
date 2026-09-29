import numpy as np
import pytest

from src.serving.ood import FeatureDistance, feature_distance_for, stats_path_for


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


def test_stats_live_beside_their_checkpoint(tmp_path):
    checkpoint = tmp_path / "best_mag200.pt"
    assert stats_path_for(checkpoint) == tmp_path / "best_mag200.ood.npz"
    assert feature_distance_for(str(checkpoint)) is None  # nothing fitted yet
