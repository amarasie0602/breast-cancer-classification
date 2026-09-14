from src.training.early_stopping import EarlyStopping


def test_does_not_stop_while_improving():
    es = EarlyStopping(patience=2)
    assert es.step(1.0) is False
    assert es.step(0.9) is False
    assert es.step(0.8) is False


def test_stops_after_patience_exceeded():
    es = EarlyStopping(patience=2)
    es.step(1.0)
    assert es.step(1.1) is False
    assert es.step(1.2) is True


def test_min_delta_requires_meaningful_improvement():
    es = EarlyStopping(patience=1, min_delta=0.1)
    es.step(1.0)
    assert es.step(0.95) is True
