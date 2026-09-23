# Model Card: BreakHis Breast Cancer Classifier

## Intended Use

This is a 3-stage research/coursework artifact demonstrating an MLOps
pipeline (experiment tracking, versioning, testing, CI/CD) around
transfer-learning image classifiers — it is **not** a clinical diagnostic
tool and must not be used to inform real patient care or treatment
decisions. Given a breast tissue histology image, it:

1. **Validates the input** is plausibly an H&E-stained histology image at
   all (`src/serving/input_guard.py`), rejecting photos, screenshots, and
   other unrelated images with a clear "Invalid Image" response rather than
   forcing every input through the classifier.
2. **Classifies benign vs. malignant** (the original binary model).
3. **For malignant results only**, classifies the malignant subtype among
   the four BreakHis actually labels: Invasive Ductal Carcinoma (IDC),
   Invasive Lobular Carcinoma (ILC), Mucinous Carcinoma, Papillary
   Carcinoma.

**"Benign" means non-cancerous tumor, not healthy tissue.** BreakHis
contains no normal/healthy breast tissue images at all — every image in the
dataset is from a tumor biopsy (surgery is why the sample was taken in the
first place). An earlier draft of this pipeline's UI called this stage
"Healthy vs. Cancer"; that framing was corrected because it isn't true of
what the model has ever seen a single training example of.

**This model does not estimate ER, PR, HER2, or triple-negative status.**
Those are immunohistochemistry/biomarker results, not something derivable
from H&E morphology alone without validated biomarker labels, which this
dataset does not have.

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

### Available classes

| Level | Class | Images |
| --- | --- | --- |
| Top | benign (non-cancerous tumor) | 2,480 |
| Top | malignant (cancer) | 5,429 |
| Benign subtype | adenosis | 444 |
| Benign subtype | fibroadenoma | 1,014 |
| Benign subtype | phyllodes_tumor | 453 |
| Benign subtype | tubular_adenoma | 569 |
| Malignant subtype | ductal_carcinoma (IDC) | 3,451 |
| Malignant subtype | lobular_carcinoma (ILC) | 626 |
| Malignant subtype | mucinous_carcinoma | 792 |
| Malignant subtype | papillary_carcinoma | 560 |

This project only trains a **malignant**-subtype classifier (Stage 3),
matching BreakHis's actual structure: benign tumors have no further
subtype breakdown beyond the four benign tumor *types* themselves (which
aren't clinically-named "subtypes" the way the malignant ones are), and
none of the other commonly-referenced breast cancer subtypes — DCIS, LCIS,
inflammatory breast cancer, Paget disease, metaplastic carcinoma, or
cribriform carcinoma — appear in this dataset at all. A dataset-provided
"tubular_adenoma" (benign) should not be confused with tubular
*carcinoma* (malignant, clinically distinct, and also not in this
dataset) despite the similar name.

Unlike the binary classifier (trained and evaluated separately per
magnification, see Results below), the subtype classifier is trained on
all four magnifications combined: per-magnification malignant-subtype
counts are too small for a meaningful 4-class split (papillary_carcinoma
has as few as 135 images at a single magnification).

### Dataset quality checks

- **Corrupted files / unreadable images:** none found (`Image.verify()`
  across all 7,909 images).
- **Image dimensions:** nearly uniform — 7,835 of 7,909 images (99.1%) are
  700×460; the remaining 74 (0.9%) are 700×456, a 4px height difference.
  Immaterial in practice since both `train_transform`/`eval_transform`
  resize every image to a common size before it reaches the model, but
  worth recording rather than assuming uniformity.
- **Exact duplicate images (by content hash):** none found.
- **Class imbalance:** malignant outnumbers benign roughly 2.2:1 overall;
  within malignant subtypes, ductal_carcinoma alone is ~64% of all
  malignant images, a ~6:1 ratio against the smallest subtype
  (papillary_carcinoma). The subtype classifier's training loss is
  class-weighted (inverse frequency) to counteract this; see
  `src/training/train_subtype.py`.
- **Patient/slide-level leakage:** prevented by construction — all splits
  (`src/data/splits.py`) are made by grouping images by patient ID first,
  then splitting patients (not images) into train/val/test, since
  multiple images from the same patient/slide are highly correlated.

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
- **Subtype coverage is partial by dataset necessity, not by choice.** Only
  4 of the 11 clinically-recognized malignant subtypes commonly discussed
  are available (see "Available classes" above). A malignant prediction on
  a genuinely rare subtype not in this list will still be forced into one
  of the four available bins by the subtype classifier — there is no
  "other/unknown subtype" option, since the model was never trained to
  recognize that its coverage is incomplete.
- **Input validation is a heuristic, not a trained classifier.** Stage 1
  (`src/serving/input_guard.py`) screens for H&E-characteristic color and
  texture; it catches ordinary photos, cartoons, and blank images but is
  not a real out-of-distribution detector — an adversarial or unusual image
  could still pass through to stages 2-3 and receive a meaningless
  confident label.
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
