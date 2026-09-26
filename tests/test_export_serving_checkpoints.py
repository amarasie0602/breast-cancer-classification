import pytest
import torch
from torch import nn

from scripts.export_serving_checkpoints import export_serving_checkpoint
from src.training.checkpoint import load_checkpoint, save_checkpoint


def _trained_checkpoint(path):
    torch.manual_seed(0)
    model = nn.Linear(8, 1)
    optimizer = torch.optim.Adam(model.parameters())
    model(torch.randn(4, 8)).sum().backward()
    optimizer.step()
    save_checkpoint(path, model, optimizer, epoch=3, metrics={"f1": 0.9})
    return model


def test_exported_checkpoint_drops_optimizer_state_and_loads_identically(tmp_path):
    original = _trained_checkpoint(tmp_path / "full.pt")

    export_serving_checkpoint(tmp_path / "full.pt", tmp_path / "out" / "slim.pt")

    slim = torch.load(tmp_path / "out" / "slim.pt", weights_only=True)
    assert "optimizer_state" not in slim

    restored = nn.Linear(8, 1)
    epoch, metrics = load_checkpoint(tmp_path / "out" / "slim.pt", restored)
    assert epoch == 3
    assert metrics == {"f1": 0.9}
    for name, tensor in original.state_dict().items():
        assert torch.equal(restored.state_dict()[name], tensor)


def test_exported_checkpoint_is_smaller(tmp_path):
    _trained_checkpoint(tmp_path / "full.pt")
    export_serving_checkpoint(tmp_path / "full.pt", tmp_path / "slim.pt")
    assert (tmp_path / "slim.pt").stat().st_size < (tmp_path / "full.pt").stat().st_size


def test_refuses_a_checkpoint_without_model_weights(tmp_path):
    torch.save({"epoch": 1, "metrics": {}}, tmp_path / "broken.pt")
    with pytest.raises(ValueError, match="model_state"):
        export_serving_checkpoint(tmp_path / "broken.pt", tmp_path / "slim.pt")
