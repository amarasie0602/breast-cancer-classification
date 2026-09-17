from torch.utils.data import DataLoader

from src.data.dataset import BreakHisDataset
from src.data.transforms import eval_transform
from src.models.classifier import BreakHisClassifier
from src.training.error_analysis import (
    collect_predictions,
    filter_misclassified,
    most_confident_errors,
)


def test_collect_predictions_returns_one_record_per_sample(breakhis_root):
    ds = BreakHisDataset(breakhis_root, transform=eval_transform())
    dataloader = DataLoader(ds, batch_size=4)
    model = BreakHisClassifier(pretrained=False)

    records = collect_predictions(model, dataloader)

    assert len(records) == len(ds)
    assert all({"index", "label", "pred", "probability"} <= r.keys() for r in records)


def test_filter_misclassified_only_returns_wrong_predictions():
    records = [
        {"index": 0, "label": 0, "pred": 0, "probability": 0.1},
        {"index": 1, "label": 1, "pred": 0, "probability": 0.4},
        {"index": 2, "label": 0, "pred": 1, "probability": 0.9},
    ]
    errors = filter_misclassified(records)
    assert {r["index"] for r in errors} == {1, 2}


def test_most_confident_errors_ranks_by_distance_from_half():
    records = [
        {"index": 0, "label": 1, "pred": 0, "probability": 0.45},  # wrong, low confidence
        {"index": 1, "label": 0, "pred": 1, "probability": 0.99},  # wrong, high confidence
        {"index": 2, "label": 0, "pred": 0, "probability": 0.5},  # correct
    ]
    top = most_confident_errors(records, n=1)
    assert top[0]["index"] == 1
