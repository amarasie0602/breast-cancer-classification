"""Frozen patient splits.

``scripts/freeze_patient_splits.py`` writes ``splits/patients.json`` from the
BreakHis tree. Training, calibration, evaluation and the quality gate read
that file when it covers the patients in front of them, so a later directory
listing cannot move somebody from test into train. A dataset whose ids are
disjoint from the manifest (the synthetic trees in unit tests) still uses
``stratified_patient_split`` directly.
"""

import json
from pathlib import Path

from src.data.splits import stratified_patient_split

MANIFEST_VERSION = 1


def default_manifest_path() -> Path:
    return Path(__file__).resolve().parents[2] / "splits" / "patients.json"


def patient_id_of(sample: dict) -> str:
    return sample["path"].parent.parent.name


def load_manifest(path: Path | None = None) -> dict | None:
    manifest_path = default_manifest_path() if path is None else Path(path)
    if not manifest_path.is_file():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def write_manifest(manifest: dict, path: Path | None = None) -> Path:
    manifest_path = default_manifest_path() if path is None else Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _listed_split(manifest: dict, manifest_key: tuple[str, ...]) -> dict | None:
    node: object = manifest
    for key in manifest_key:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    if not isinstance(node, dict):
        return None
    if not {"train", "val", "test"} <= set(node):
        return None
    return node


def resolve_patient_split(
    samples,
    ratios: tuple[float, float, float],
    seed: int,
    *,
    manifest_key: tuple[str, ...] | None = None,
    min_per_split: int = 0,
    manifest_path: Path | None = None,
) -> tuple[set, set, set]:
    """Train/val/test patient ids for ``samples``.

    Uses the frozen manifest when every patient in ``samples`` is listed
    under ``manifest_key``. A completely different id set falls back to
    ``stratified_patient_split`` so unit-test fixtures keep working before
    ``splits/patients.json`` exists. A partial overlap is an error: the
    manifest is stale and must be regenerated, not silently patched.
    """
    computed = stratified_patient_split(samples, ratios=tuple(ratios), seed=seed, min_per_split=min_per_split)
    if manifest_key is None:
        return computed
    manifest = load_manifest(manifest_path)
    if manifest is None:
        return computed
    listed = _listed_split(manifest, manifest_key)
    if listed is None:
        return computed

    train = set(listed["train"])
    val = set(listed["val"])
    test = set(listed["test"])
    present = {patient_id_of(sample) for sample in samples}
    known = train | val | test
    if present.isdisjoint(known):
        return computed
    missing = present - known
    if missing:
        example = ", ".join(sorted(missing)[:5])
        raise ValueError(
            f"split manifest is missing {len(missing)} patients ({example}). "
            "Regenerate it with python -m scripts.freeze_patient_splits"
        )
    return train & present, val & present, test & present
