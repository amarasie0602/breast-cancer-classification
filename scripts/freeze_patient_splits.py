"""Write the frozen patient split used by training and evaluation.

The file is the list of patient ids, not the images. Generate it once from
the BreakHis tree and commit ``splits/patients.json`` yourself:

    python -m scripts.freeze_patient_splits --data-root data/BreaKHis_v1

``--check`` exits non-zero when the committed file disagrees with a fresh
split of the same tree. That is the check to run after anybody edits
``src/data/splits.py`` or ``configs/data.yaml``.
"""

import argparse
import json
from pathlib import Path

from src.data.dataset import SUBTYPE_SCHEMES, BreakHisDataset, BreakHisSubtypeDataset
from src.data.manifest import MANIFEST_VERSION, default_manifest_path, patient_id_of, write_manifest
from src.data.splits import stratified_patient_split
from src.training.train import load_config

MAGNIFICATIONS = ("40", "100", "200", "400")


def _as_lists(train: set, val: set, test: set) -> dict:
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def build_manifest(data_root: Path, data_config: dict) -> dict:
    ratios_cfg = data_config["split_ratios"]
    ratios = (ratios_cfg["train"], ratios_cfg["val"], ratios_cfg["test"])
    seed = int(data_config["seed"])
    binary = {}
    for magnification in MAGNIFICATIONS:
        dataset = BreakHisDataset(data_root, magnification=magnification)
        if not dataset.samples:
            continue
        train, val, test = stratified_patient_split(dataset.samples, ratios=ratios, seed=seed)
        binary[magnification] = _as_lists(train, val, test)

    subtype = {}
    for scheme in SUBTYPE_SCHEMES:
        dataset = BreakHisSubtypeDataset(data_root, scheme=scheme)
        if not dataset.samples:
            continue
        train, val, test = stratified_patient_split(
            dataset.samples, ratios=ratios, seed=seed, min_per_split=1
        )
        subtype[scheme] = _as_lists(train, val, test)
        present = {patient_id_of(sample) for sample in dataset.samples}
        listed = set(subtype[scheme]["train"] + subtype[scheme]["val"] + subtype[scheme]["test"])
        if present != listed:
            raise RuntimeError(f"subtype split for {scheme} dropped or invented patients")

    return {
        "version": MANIFEST_VERSION,
        "seed": seed,
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "binary": binary,
        "subtype": subtype,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/BreaKHis_v1")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--output", default=None, help="defaults to splits/patients.json")
    parser.add_argument("--force", action="store_true", help="overwrite an existing manifest")
    parser.add_argument("--check", action="store_true", help="fail if the file differs from a fresh split")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not (data_root / "histology_slides" / "breast").is_dir():
        raise SystemExit(
            f"No BreakHis tree at {data_root}. Place the dataset there before freezing the split."
        )
    manifest = build_manifest(data_root, load_config(args.data_config))
    output = Path(args.output) if args.output else default_manifest_path()

    if args.check:
        if not output.is_file():
            raise SystemExit(f"{output} does not exist. Generate it, then commit it.")
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != manifest:
            raise SystemExit(
                f"{output} does not match a fresh split of {data_root}. Regenerate with --force."
            )
        print(f"{output} matches a fresh split")
        return

    if output.exists() and not args.force:
        raise SystemExit(f"{output} already exists. Pass --force to replace it.")
    write_manifest(manifest, output)
    slots = (
        len(split["train"]) + len(split["val"]) + len(split["test"]) for split in manifest["binary"].values()
    )
    n_binary = sum(slots)
    n_mags = len(manifest["binary"])
    print(f"wrote {output} ({n_binary} binary patient-slots across {n_mags} magnifications)")


if __name__ == "__main__":
    main()
