"""End-to-end training entrypoint: model, loop, checkpointing, MLflow logging."""

import argparse
from pathlib import Path

import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader

from src.data.dataset import BreakHisDataset
from src.data.splits import filter_samples_by_patients, stratified_patient_split
from src.data.transforms import eval_transform, train_transform
from src.models.classifier import BreakHisClassifier
from src.training.checkpoint import save_checkpoint
from src.training.early_stopping import EarlyStopping
from src.training.loop import evaluate, train_one_epoch


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataloaders(data_root, magnification, split_ratios, seed, batch_size):
    full_ds = BreakHisDataset(data_root, magnification=magnification)
    train_patients, val_patients, _ = stratified_patient_split(
        full_ds.samples, ratios=tuple(split_ratios), seed=seed
    )

    train_ds = BreakHisDataset(data_root, magnification=magnification, transform=train_transform())
    train_ds.samples = filter_samples_by_patients(train_ds.samples, train_patients)

    val_ds = BreakHisDataset(data_root, magnification=magnification, transform=eval_transform())
    val_ds.samples = filter_samples_by_patients(val_ds.samples, val_patients)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
    )
