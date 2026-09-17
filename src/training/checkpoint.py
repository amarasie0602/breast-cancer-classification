"""Model checkpoint save/load helpers."""

from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from torch import nn, optim


def save_checkpoint(
    path: Union[str, Path], model: nn.Module, optimizer: optim.Optimizer, epoch: int, metrics: dict
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    map_location: str = "cpu",
) -> Tuple[int, dict]:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["epoch"], checkpoint["metrics"]
