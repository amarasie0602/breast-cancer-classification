# Model Card: BreakHis Breast Cancer Classifier

## Intended Use

This model classifies breast tissue histopathology images as benign or
malignant. It is a research/coursework artifact intended to demonstrate an
MLOps pipeline (experiment tracking, versioning, testing, CI/CD) around a
transfer-learning image classifier — it is **not** a clinical diagnostic tool
and must not be used to inform real patient care or treatment decisions.

## Dataset Provenance

Trained on [BreakHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)
(Breast Cancer Histopathological Image Classification), 7,909 microscopic
images of breast tumor tissue from 82 patients, collected via surgical
biopsy (SOB) and stained with hematoxylin and eosin (H&E). Images are
provided at four magnification levels — 40x, 100x, 200x, 400x — and are
labeled benign or malignant across several tumor subtypes.

Splits in this project are made at the **patient** level (not image level)
to prevent leakage, since multiple images from the same patient are highly
correlated.
