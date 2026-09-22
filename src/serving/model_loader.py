"""Load a trained checkpoint for inference, cached across requests."""

from functools import lru_cache

from src.models.classifier import BreakHisClassifier
from src.training.checkpoint import load_checkpoint


@lru_cache(maxsize=4)
def get_model(checkpoint_path: str, device: str = "cpu") -> BreakHisClassifier:
    model = BreakHisClassifier(pretrained=False).to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()
    return model
