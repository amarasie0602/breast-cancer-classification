"""PyTorch Dataset for the BreakHis histopathology dataset.

Expects the standard BreaKHis_v1 directory layout:

    <root>/histology_slides/breast/<benign|malignant>/SOB/<subtype>/<patient_id>/<mag>X/<image>.png
"""

from pathlib import Path
from typing import Callable, Optional, Union

from PIL import Image
from torch.utils.data import Dataset

LABEL_MAP = {"benign": 0, "malignant": 1}
MAGNIFICATIONS = ("40", "100", "200", "400")

# The four malignant subtypes BreakHis actually labels (see docs/model_card.md
# for why the other clinically-named subtypes -- DCIS, LCIS, inflammatory,
# Paget, metaplastic, cribriform, and true tubular carcinoma -- are not
# offered: they simply aren't represented in this dataset). Benign tumors in
# BreakHis have no further subtype breakdown beyond the four benign tumor
# types themselves, so subtype classification only applies to the malignant
# branch of the pipeline.
SUBTYPE_LABEL_MAP = {
    "ductal_carcinoma": 0,
    "lobular_carcinoma": 1,
    "mucinous_carcinoma": 2,
    "papillary_carcinoma": 3,
}
SUBTYPE_NAMES = tuple(SUBTYPE_LABEL_MAP.keys())

# Human-readable names for the UI/API, including the clinical abbreviation
# where BreakHis's folder name maps to a widely-used one.
SUBTYPE_DISPLAY_NAMES = {
    "ductal_carcinoma": "Invasive Ductal Carcinoma (IDC)",
    "lobular_carcinoma": "Invasive Lobular Carcinoma (ILC)",
    "mucinous_carcinoma": "Mucinous Carcinoma",
    "papillary_carcinoma": "Papillary Carcinoma",
}

# Ways of labelling the malignant samples for stage 3. "four_subtypes" is the
# original 4-way split; it does not learn, because lobular, papillary and
# mucinous carcinoma have only 5, 6 and 9 patients in the whole dataset (see
# docs/model_card.md). "ductal_vs_other" asks the coarser question the data
# can actually support: ductal carcinoma (38 patients) versus every other
# malignant subtype pooled (20 patients).
SUBTYPE_SCHEMES = {
    "four_subtypes": {
        "label_map": SUBTYPE_LABEL_MAP,
        "names": SUBTYPE_NAMES,
        "display": SUBTYPE_DISPLAY_NAMES,
    },
    "ductal_vs_other": {
        "label_map": {
            "ductal_carcinoma": 0,
            "lobular_carcinoma": 1,
            "mucinous_carcinoma": 1,
            "papillary_carcinoma": 1,
        },
        "names": ("ductal_carcinoma", "other_malignant"),
        "display": {
            "ductal_carcinoma": "Invasive Ductal Carcinoma (IDC)",
            "other_malignant": "Non-ductal carcinoma (lobular, mucinous or papillary)",
        },
    },
}
DEFAULT_SUBTYPE_SCHEME = "four_subtypes"


class BreakHisDataset(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        magnification: Optional[Union[str, int]] = None,
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.magnification = str(magnification) if magnification else None
        self.transform = transform
        self.samples = self._build_manifest()

    def _build_manifest(self):
        slides_root = self.root / "histology_slides" / "breast"
        samples = []
        for label_name, label_idx in LABEL_MAP.items():
            class_dir = slides_root / label_name / "SOB"
            if not class_dir.is_dir():
                continue
            for subtype_dir in class_dir.iterdir():
                if not subtype_dir.is_dir():
                    continue
                for patient_dir in subtype_dir.iterdir():
                    samples.extend(
                        self._samples_for_patient(patient_dir, label_idx, subtype_dir.name)
                    )
        return samples

    def _samples_for_patient(self, patient_dir, label_idx, subtype):
        samples = []
        for mag_dir in sorted(patient_dir.glob("*X")):
            mag = mag_dir.name.rstrip("X")
            if self.magnification and mag != self.magnification:
                continue
            samples.extend(
                {
                    "path": image_path,
                    "label": label_idx,
                    "magnification": mag,
                    "subtype": subtype,
                }
                for image_path in sorted(mag_dir.glob("*.png"))
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, sample["label"]


class BreakHisSubtypeDataset(Dataset):
    """Malignant-only samples, labeled by subtype instead of benign/malignant.

    Reuses BreakHisDataset's manifest building (which already tracks each
    sample's subtype folder name) and filters/remaps it, rather than
    duplicating the directory-walking logic.
    """

    def __init__(
        self,
        root: Union[str, Path],
        magnification: Optional[Union[str, int]] = None,
        transform: Optional[Callable] = None,
        scheme: str = DEFAULT_SUBTYPE_SCHEME,
    ):
        if scheme not in SUBTYPE_SCHEMES:
            raise ValueError(f"unknown subtype scheme {scheme!r}; expected one of {list(SUBTYPE_SCHEMES)}")
        self.root = Path(root)
        self.magnification = str(magnification) if magnification else None
        self.transform = transform
        self.scheme = scheme
        label_map = SUBTYPE_SCHEMES[scheme]["label_map"]
        base = BreakHisDataset(root, magnification=magnification)
        self.samples = [
            {**s, "label": label_map[s["subtype"]]}
            for s in base.samples
            if s["subtype"] in label_map
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, sample["label"]
