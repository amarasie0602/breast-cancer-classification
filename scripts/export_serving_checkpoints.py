"""Write weights-only copies of the checkpoints the serving image ships.

A training checkpoint is ~280MB, but two thirds of that is Adam's optimizer
state (two moment buffers per weight), which is only needed to resume
training. Serving never touches it. Stripping it gives a ~95MB file that loads
into the model identically, which is what makes it affordable to ship all
four magnification models through Git LFS instead of only the 40x one.

The full checkpoints stay in checkpoints/ (DVC-tracked) for training; the
slim copies go to serving_checkpoints/ (Git LFS-tracked), which is what the
app serves both locally and in the image. Re-run this after retraining, or
the app keeps serving the previous model.

    python -m scripts.export_serving_checkpoints
"""

from pathlib import Path
from typing import Union

import torch

# One binary model per magnification, plus the subtype model. The subtype
# model is gated off, but shipping it is what lets the deployed app say *why*
# there is no subtype rather than leaving the field silently empty.
CHECKPOINT_NAMES = (
    "best_mag40.pt",
    "best_mag100.pt",
    "best_mag200.pt",
    "best_mag400.pt",
    "best_subtype.pt",
)

# Fixed rather than taken from the command line: the only job of this script
# is refreshing the directory the app and the image serve from.
REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = REPO_ROOT / "checkpoints"
OUTPUT_DIR = REPO_ROOT / "serving_checkpoints"

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
    for name in CHECKPOINT_NAMES:
        source = SOURCE_DIR / name
        destination = OUTPUT_DIR / name
        export_serving_checkpoint(source, destination)
        print(
            f"  {name}: {source.stat().st_size / 1e6:.0f}MB -> "
            f"{destination.stat().st_size / 1e6:.0f}MB"
        )


if __name__ == "__main__":
    main()
