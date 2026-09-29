"""Write a Macenko stain-normalized copy of the BreakHis dataset.

Normalization is deterministic and takes ~0.1s per image, so it is done once
here rather than on every epoch. The copy keeps the original directory
layout, so patient IDs, the patient-level split and every sample path line up
with the raw dataset exactly. It is resumable: images already written are
skipped. The marker file that tells training the copy is normalized is only
written once every image has been converted.

    python -m scripts.normalize_dataset --workers 8
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.stain import MACENKO, STAIN_MARKER_FILE, macenko_normalize

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO_ROOT / "data" / "BreaKHis_v1"
OUTPUT_ROOT = REPO_ROOT / "data" / "BreaKHis_v1_macenko"


def _normalize_one(source: Path) -> None:
    destination = OUTPUT_ROOT / source.relative_to(SOURCE_ROOT)
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(Image.open(source).convert("RGB"))
    # Write to a temporary name first so an interrupted run never leaves a
    # truncated PNG that a resumed run would then skip.
    partial = destination.with_suffix(".partial.png")
    Image.fromarray(macenko_normalize(rgb)).save(partial)
    partial.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    sources = sorted((SOURCE_ROOT / "histology_slides").rglob("*.png"))
    print(f"normalizing {len(sources)} images into {OUTPUT_ROOT}")
    with Pool(args.workers) as pool:
        for done, _ in enumerate(pool.imap_unordered(_normalize_one, sources, chunksize=16), 1):
            if done % 500 == 0 or done == len(sources):
                print(f"  {done}/{len(sources)}", flush=True)

    written = sum(1 for _ in (OUTPUT_ROOT / "histology_slides").rglob("*.png"))
    if written != len(sources):
        raise SystemExit(f"only {written} of {len(sources)} images written; not marking complete")
    (OUTPUT_ROOT / STAIN_MARKER_FILE).write_text(MACENKO + "\n", encoding="utf-8")
    print(f"done; marked {OUTPUT_ROOT} as {MACENKO}-normalized")


if __name__ == "__main__":
    main()
