"""Write weights-only copies of the binary checkpoints for the serving image.

A training checkpoint is ~280MB, but two thirds of that is Adam's optimizer
state (two moment buffers per weight), which is only needed to resume
training. Serving never touches it. Stripping it gives a ~95MB file that loads
into the model identically, which is what makes it affordable to ship all
four magnification models through Git LFS instead of only the 40x one.

The full checkpoints stay in checkpoints/ (DVC-tracked) for training; the
slim copies go to serving_checkpoints/ (Git LFS-tracked) for the image.

    python -m scripts.export_serving_checkpoints
"""

import argparse
from pathlib import Path
from typing import Union

import torch

MAGNIFICATIONS = ("40", "100", "200", "400")

# Everything load_checkpoint() reads, minus the optimizer state.
SERVING_KEYS = ("epoch", "model_state", "metrics")


def export_serving_checkpoint(source: Union[str, Path], destination: Union[str, Path]) -> None:
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    missing = [key for key in SERVING_KEYS if key not in checkpoint]
    if missing:
        raise ValueError(f"{source} is missing {', '.join(missing)}")
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: checkpoint[key] for key in SERVING_KEYS}, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="checkpoints")
    parser.add_argument("--output-dir", default="serving_checkpoints")
    args = parser.parse_args()

    for magnification in MAGNIFICATIONS:
        name = f"best_mag{magnification}.pt"
        source = Path(args.source_dir) / name
        destination = Path(args.output_dir) / name
        export_serving_checkpoint(source, destination)
        print(
            f"  {name}: {source.stat().st_size / 1e6:.0f}MB -> "
            f"{destination.stat().st_size / 1e6:.0f}MB"
        )


if __name__ == "__main__":
    main()
