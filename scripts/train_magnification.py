"""Train a classifier that recognises the magnification an image was taken at.

The binary models are magnification-specific, and a result from the wrong
one is meaningless, so the app shouldn't rely on the user knowing the zoom.
A general-purpose ImageNet network's features only reach ~82% on this
(validation), confusing neighbouring zoom levels, so this fine-tunes
EfficientNet-B0 on the task itself.

Writes the chosen epoch's weights to serving_checkpoints/magnification.pt.

Augmentation deliberately excludes random crops and rescaling: apparent
scale is the signal being learned. Training uses the training patients, the
best epoch is chosen on the validation patients, and the held-out test
patients are scored once at the end.

    python -m scripts.train_magnification
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data.dataset import BreakHisDataset
from src.data.splits import stratified_patient_split
from src.data.transforms import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, eval_transform
from src.serving.magnification import MAGNIFICATIONS, build_network
from src.training.train import NUM_WORKERS, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "BreaKHis_v1"
OUTPUT = REPO_ROOT / "checkpoints" / "experiments" / "magnification" / "best.pt"
# Weights only, what serving loads (src/serving/magnification.py).
SERVING_OUTPUT = REPO_ROOT / "serving_checkpoints" / "magnification.pt"
EPOCHS = 6
BATCH_SIZE = 32


def train_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomChoice([transforms.RandomRotation((a, a)) for a in (0, 90, 180, 270)]),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class MagnificationDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples, self.transform = samples, transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        from PIL import Image

        s = self.samples[i]
        image = Image.open(s["path"]).convert("RGB")
        return self.transform(image), MAGNIFICATIONS.index(s["magnification"])


def split_samples():
    config = load_config(REPO_ROOT / "configs" / "data.yaml")
    r = config["split_ratios"]
    splits = {"train": [], "val": [], "test": []}
    for magnification in MAGNIFICATIONS:
        ds = BreakHisDataset(DATA_ROOT, magnification=magnification)
        train, val, test = stratified_patient_split(
            ds.samples, ratios=(r["train"], r["val"], r["test"]), seed=config["seed"]
        )
        split_of = {
            **dict.fromkeys(train, "train"),
            **dict.fromkeys(val, "val"),
            **dict.fromkeys(test, "test"),
        }
        for sample in ds.samples:
            splits[split_of[sample["path"].parent.parent.name]].append(sample)
    return splits


@torch.no_grad()
def predict(model, samples):
    model.eval()
    loader = DataLoader(
        MagnificationDataset(samples, eval_transform()), batch_size=64, num_workers=NUM_WORKERS
    )
    probs, labels = [], []
    for images, y in loader:
        probs.append(model(images).softmax(1))
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def report(name, probs, labels):
    pred = probs.argmax(1)
    matrix = np.zeros((4, 4), int)
    for a, b in zip(labels, pred, strict=True):
        matrix[a, b] += 1
    print(f"{name}: accuracy {(pred == labels).mean():.3f}; confusion (rows = actual 40/100/200/400):")
    print(matrix)
    confidence = probs.max(1)
    for c in (0.8, 0.9, 0.95):
        keep = confidence >= c
        accuracy = (pred[keep] == labels[keep]).mean()
        print(f"  confidence >= {c}: covers {keep.mean():.0%}, accuracy {accuracy:.3f}")


def main() -> None:
    torch.manual_seed(0)
    splits = split_samples()
    print({k: len(v) for k, v in splits.items()}, flush=True)
    model = build_network(pretrained=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loader = DataLoader(
        MagnificationDataset(splits["train"], train_transform()),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for images, y in loader:
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(images), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
        scheduler.step()
        probs, labels = predict(model, splits["val"])
        accuracy = float((probs.argmax(1) == labels).mean())
        train_loss = total / len(splits["train"])
        print(f"epoch {epoch + 1}: train loss {train_loss:.3f}, val accuracy {accuracy:.3f}", flush=True)
        if accuracy > best:
            best = accuracy
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_accuracy": accuracy}, OUTPUT)

    model.load_state_dict(torch.load(OUTPUT, weights_only=True)["model_state"])
    report("validation (selection)", *predict(model, splits["val"]))
    report("held-out test", *predict(model, splits["test"]))
    torch.save(model.state_dict(), SERVING_OUTPUT)
    print(f"wrote {SERVING_OUTPUT.name}")


if __name__ == "__main__":
    main()
