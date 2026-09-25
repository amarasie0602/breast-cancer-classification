"""Load a trained checkpoint for inference, cached across requests."""

from functools import lru_cache

import torch

from src.data.dataset import DEFAULT_SUBTYPE_SCHEME, SUBTYPE_SCHEMES
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
    scheme = subtype_checkpoint_scheme(checkpoint_path)
    num_classes = len(SUBTYPE_SCHEMES[scheme]["names"])
    model = MalignantSubtypeClassifier(num_classes=num_classes, pretrained=False).to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()
    return model


@lru_cache(maxsize=4)
def subtype_checkpoint_macro_f1(checkpoint_path: str) -> float:
    """The validation macro F1 recorded in the subtype checkpoint at the
    epoch it was saved, or -1.0 if the checkpoint doesn't carry one.

    Serving uses this to decide whether the model is good enough to show a
    subtype at all. Both subtype training attempts so far sit at or below
    chance on validation (macro F1 0.08-0.19 against a 0.25 random
    baseline), and a UI that renders "Mucinous Carcinoma — 87% confidence"
    from a model like that is misleading no matter what caveat sits next
    to it.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metrics = checkpoint.get("metrics") or {}
    value = metrics.get("macro_f1")
    return float(value) if value is not None else -1.0


@lru_cache(maxsize=4)
def subtype_checkpoint_scheme(checkpoint_path: str) -> str:
    """The label scheme a subtype checkpoint was trained with.

    Checkpoints written before schemes existed carry none; those are all
    4-way models, so that is the fallback. An unrecognised scheme is an
    error rather than a guess, since building the wrong-sized head would
    fail to load at best and mislabel classes at worst.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    scheme = (checkpoint.get("metrics") or {}).get("label_scheme", DEFAULT_SUBTYPE_SCHEME)
    if scheme not in SUBTYPE_SCHEMES:
        raise ValueError(f"checkpoint {checkpoint_path} has unknown label scheme {scheme!r}")
    return scheme
