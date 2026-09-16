"""PyTorch Dataset for the BreakHis histopathology dataset.

Expects the standard BreaKHis_v1 directory layout:

    <root>/histology_slides/breast/<benign|malignant>/SOB/<subtype>/<patient_id>/<mag>X/<image>.png
"""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

LABEL_MAP = {"benign": 0, "malignant": 1}
MAGNIFICATIONS = ("40", "100", "200", "400")


class BreakHisDataset(Dataset):
    def __init__(self, root, magnification=None, transform=None):
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, sample["label"]
