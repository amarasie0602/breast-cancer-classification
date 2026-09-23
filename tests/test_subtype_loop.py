import pytest
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import SUBTYPE_NAMES, BreakHisSubtypeDataset
from src.data.transforms import eval_transform
from src.training.subtype_loop import evaluate, train_one_epoch

NUM_CLASSES = len(SUBTYPE_NAMES)


class EmptyDataset(Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError


class TinyModel(nn.Module):
    """Small stand-in classifier so tests don't need a real ResNet50 download."""

    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, num_classes))

    def forward(self, x):
        return self.net(x)


def test_train_one_epoch_reduces_loss_over_steps(breakhis_root_all_subtypes):
    ds = BreakHisSubtypeDataset(breakhis_root_all_subtypes, transform=eval_transform())
    dataloader = DataLoader(ds, batch_size=4, shuffle=True)

    model = TinyModel(NUM_CLASSES)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    first_loss = train_one_epoch(model, dataloader, optimizer, criterion, "cpu")
    for _ in range(10):
        train_one_epoch(model, dataloader, optimizer, criterion, "cpu")
    last_loss = train_one_epoch(model, dataloader, optimizer, criterion, "cpu")

    assert last_loss < first_loss


def test_evaluate_returns_expected_metric_keys(breakhis_root_all_subtypes):
    ds = BreakHisSubtypeDataset(breakhis_root_all_subtypes, transform=eval_transform())
    dataloader = DataLoader(ds, batch_size=4)

    model = TinyModel(NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()

    metrics = evaluate(model, dataloader, criterion, "cpu", num_classes=NUM_CLASSES)
    assert set(metrics) == {
        "loss",
        "accuracy",
        "macro_f1",
        "per_class_precision",
        "per_class_recall",
        "per_class_f1",
        "confusion_matrix",
    }
    assert len(metrics["per_class_precision"]) == NUM_CLASSES
    assert metrics["confusion_matrix"].shape == (NUM_CLASSES, NUM_CLASSES)


def test_train_one_epoch_on_empty_dataset_raises_clear_error():
    dataloader = DataLoader(EmptyDataset(), batch_size=4)
    model = TinyModel(NUM_CLASSES)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    with pytest.raises(ValueError, match="empty dataset"):
        train_one_epoch(model, dataloader, optimizer, criterion, "cpu")


def test_evaluate_on_empty_dataset_raises_clear_error():
    dataloader = DataLoader(EmptyDataset(), batch_size=4)
    model = TinyModel(NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()

    with pytest.raises(ValueError, match="empty dataset"):
        evaluate(model, dataloader, criterion, "cpu", num_classes=NUM_CLASSES)
