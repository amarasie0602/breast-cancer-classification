from torch import nn, optim

from src.training.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip(tmp_path):
    model = nn.Linear(4, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    path = tmp_path / "ckpt.pt"

    save_checkpoint(path, model, optimizer, epoch=3, metrics={"f1": 0.9})

    new_model = nn.Linear(4, 1)
    new_optimizer = optim.SGD(new_model.parameters(), lr=0.01)
    epoch, metrics = load_checkpoint(path, new_model, new_optimizer)

    assert epoch == 3
    assert metrics == {"f1": 0.9}
    for p1, p2 in zip(model.parameters(), new_model.parameters(), strict=True):
        assert (p1 == p2).all()
