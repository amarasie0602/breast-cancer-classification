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

## Limitations

- **Small, single-source dataset.** 82 patients from one lab is not
  representative of the broader population; staining protocols, scanners,
  and patient demographics vary across institutions and are not captured
  here. Performance on external data is unknown until validated.
- **Class and subtype imbalance.** BreakHis has substantially more
  malignant than benign images, and tumor subtypes are unevenly
  represented; reported metrics can look better than real-world recall
  on rare subtypes.
- **Magnification sensitivity.** Performance is evaluated per-magnification
  (see results table below); a model tuned at one magnification may not
  generalize to images captured at another.
- **No calibration guarantee.** Predicted probabilities are not verified to
  be well-calibrated; the single-logit threshold (0.5) is a default, not a
  clinically validated operating point.
- **Subtype-specific failure mode.** Error analysis on the 40x model found
  its most confident mistakes concentrated almost entirely on one benign
  subtype (`tubular_adenoma`), predicted malignant with near-certainty.
  Overall accuracy hides this: a model can look strong in aggregate while
  being specifically unreliable on an underrepresented subtype.

## Ethical Considerations

- **Not a diagnostic device.** This model has not undergone clinical
  validation, regulatory review, or comparison against pathologist
  performance on an independent cohort. Any resemblance to a deployable
  diagnostic tool is unintentional.
- **False negatives carry asymmetric harm.** In a real screening context, a
  missed malignant case is far costlier than a false alarm; this project
  reports precision/recall/F1 separately (not just accuracy) so that
  trade-off is visible rather than hidden behind a single number.
- **Explainability is illustrative, not verification.** Grad-CAM overlays
  show which regions influenced a prediction, but a plausible-looking
  heatmap is not proof the model is reasoning about clinically relevant
  features — it can highlight the right region for the wrong reason.
- **Dataset consent and privacy.** BreakHis images are de-identified by the
  original curators; no attempt is made here to re-identify patients or
  link images to other data.

## Results

ResNet50, transfer learning (ImageNet-pretrained), 10 epochs with early
stopping, patient-level 70/15/15 train/val/test split, evaluated per
magnification. Two views are reported because they disagree in an
instructive way:

**Validation set** (used for checkpoint/model selection during training):

| Magnification | Val F1 | Val Accuracy |
| -------------- | ------ | ------------ |
| 40x            | 0.939  | 0.918        |
| 100x           | 0.935  | 0.917        |
| 400x           | 0.915  | 0.886        |
| 200x           | 0.905  | 0.878        |

**Held-out test set** (never used for training or checkpoint selection —
the honest number):

| Magnification | Test F1 | Test Accuracy | Test Precision | Test Recall | Test Images |
| -------------- | ------- | -------------- | --------------- | ----------- | ----------- |
| 200x           | 0.952   | 0.925           | 0.916            | 0.990       | 281         |
| 40x            | 0.901   | 0.848           | 0.838            | 0.974       | 270         |
| 400x           | 0.882   | 0.819           | 0.796            | 0.988       | 238         |
| 100x           | 0.861   | 0.796           | 0.841            | 0.882       | 284         |

**The ranking flips between validation and test** (40x is best on val, but
200x is best on test). With only ~11-13 patients per magnification in each
split, this is expected sampling variance rather than a robust ordering —
it is itself a limitation, not a bug: at this dataset size, per-magnification
rankings should be treated as noisy, and a claim like "40x magnification is
best for this task" is not statistically well-supported by these splits
alone. Recall is consistently high (0.88-0.99) across all four
magnifications, meaning the model rarely misses a malignant case in this
sample, but precision (and therefore false-positive rate) varies more.

See `notebooks/02_error_analysis.ipynb` for a concrete failure mode found
in the 40x model: nearly all of its most confident errors are the benign
`tubular_adenoma` subtype misclassified as malignant, not a spread of
random mistakes.
