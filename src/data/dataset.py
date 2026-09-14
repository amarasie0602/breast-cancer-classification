"""PyTorch Dataset for the BreakHis histopathology dataset.

Expects the standard BreaKHis_v1 directory layout:

    <root>/histology_slides/breast/<benign|malignant>/SOB/<subtype>/<patient_id>/<mag>X/<image>.png
"""

from pathlib import Path

LABEL_MAP = {"benign": 0, "malignant": 1}
MAGNIFICATIONS = ("40", "100", "200", "400")


class BreakHisDataset:
    def __init__(self, root):
        self.root = Path(root)
