"""Load a trained checkpoint for inference, cached across requests."""

from functools import lru_cache

from src.data.dataset import SUBTYPE_NAMES
from src.models.classifier import BreakHisClassifier, MalignantSubtypeClassifier
from src.training.checkpoint import load_checkpoint


@lru_cache(maxsize=4)
def get_model(checkpoint_path: str, device: str = "cpu") -> BreakHisClassifier:
    model = BreakHisClassifier(pretrained=False).to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()
    return model


@lru_cache(maxsize=4)
def get_subtype_model(checkpoint_path: str, device: str = "cpu") -> MalignantSubtypeClassifier:
    model = MalignantSubtypeClassifier(num_classes=len(SUBTYPE_NAMES), pretrained=False).to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()
    return model
