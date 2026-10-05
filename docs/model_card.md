# Model Card: BreakHis Breast Cancer Classifier

## Intended Use

This is a 3-stage research/coursework artifact demonstrating an MLOps
pipeline (experiment tracking, versioning, testing, CI/CD) around
transfer-learning image classifiers — it is **not** a clinical diagnostic
tool and must not be used to inform real patient care or treatment
decisions. Given a breast tissue histology image, it:

1. **Validates the input** is plausibly an H&E-stained histology image at
   all, rejecting photos, screenshots, and other unrelated images with a
   clear "Invalid Image" response rather than forcing every input through the
   classifier. Two checks: a colour-and-texture screen
   (`src/serving/input_guard.py`), then a check that a general-purpose
   network's description of the image resembles the training slides'
   (`src/serving/ood.py`).
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

- **On images from another lab it rarely clears non-cancer tissue.**
  Everything else on this page is measured on BreakHis. To see how the
  served pipeline holds up elsewhere, it was run on 107 images of BACH
  (ICIAR 2018, CC BY-NC-ND 4.0; a different lab and scanner;
  `scripts/evaluate_external_bach.py`):

  | BACH class | n | Correct | Of the wrong ones, flagged "Uncertain" |
  | --- | --- | --- | --- |
  | Normal (healthy) | 26 | **2 (8%)** called benign | 3 of 24 |
  | Benign | 27 | **11 (41%)** called benign | 3 of 16 |
  | In situ carcinoma | 27 | 26 (96%) called malignant | 1 of 1 |
  | Invasive carcinoma | 27 | 27 (100%) called malignant | - |

  Sensitivity 98%, specificity 25%. Stage 1 rejected none of them, and
  magnification detection read most as 100x or 40x (BACH images cover a
  wider field than BreakHis 200x; 48 got the "unsure" warning). Cancer is
  still caught, but the model calls most healthy and benign tissue from
  another lab malignant, and mostly with confidence: calibration was fitted
  on BreakHis and doesn't carry over. Healthy tissue fares worst because
  BreakHis contains none. A "differs from the training images" caution was
  considered and not added: on the stage-1 feature distance, the cut-off
  that flags 5% of BreakHis test images flags only 49% of BACH, so it would
  miss half of exactly the images it is meant to warn about. Training data
  with healthy and benign tissue from other labs is what would address
  this.
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
- **Subtype performance is not statistically validated, and cannot be with
  this dataset.** Image counts look adequate (560-3,451 per subtype) but
  the unit that matters for generalization is the *patient*, and there
  BreakHis is very thin:

  | Malignant subtype | Patients | Train | Val | Test |
  | --- | --- | --- | --- | --- |
  | ductal_carcinoma | 38 | 27 | 6 | 5 |
  | mucinous_carcinoma | 9 | 6 | 1 | 2 |
  | papillary_carcinoma | 6 | 4 | 1 | 1 |
  | lobular_carcinoma | 5 | 3 | 1 | 1 |

  Three of the four classes are validated against a *single patient*. One
  patient's slides being read wrong swings that class's F1 between 0 and
  ~1, so subtype macro F1 is dominated by which patients happened to land
  in which split, not by model quality. (Before `min_per_split` was added
  to the splitter, lobular_carcinoma had **zero** test patients at all.)
  Patient-level k-fold cross-validation would give a more stable estimate
  but cannot manufacture patients that aren't in the dataset.
- **The 4-class subtype model does not learn, and this was measured, not
  assumed.** Random guessing across 4 classes scores about 0.25. Every
  configuration tried lands at or below that on validation:

  | Configuration | val macro F1 by epoch | val loss (chance ≈ 1.386) |
  | --- | --- | --- |
  | ResNet50, inverse-frequency weighted loss, mild augmentation | 0.062 → 0.082 → 0.082 | 1.99 → 2.13 → 2.21 |
  | EfficientNet-B0, balanced sampling, heavy augmentation | 0.157 → 0.163 → 0.185 → 0.131 | 1.49 → 1.55 → 1.51 → 2.04 |

  The second run's fourth epoch is the telling one: it is the first epoch
  after the backbone unfroze, and validation macro F1 *fell* (0.185 →
  0.131, accuracy 0.288 → 0.249, exactly chance) while train loss dropped
  sharply (1.009 → 0.750). The model learns its four training patients per
  class better and generalizes to an unseen patient worse. Varying the
  architecture (25.6M vs 5.3M parameters), the imbalance strategy
  (weighted loss vs balanced sampling), the augmentation strength, the
  learning rate and the freeze schedule did not change this.

  This is a property of the data, not of the hyperparameters: four training
  patients is not enough to learn what distinguishes a subtype from a
  patient. Accordingly, **serving refuses to report a subtype** unless a
  checkpoint clears `MIN_SUBTYPE_MACRO_F1` (default 0.55); below that
  `/predict` returns the benign/malignant result with a
  `subtype_unavailable_reason` instead of dressing up a coin toss as a
  finding.

- **The coarser ductal-vs-other question was also tried, and falls short
  too.** Pooling lobular, mucinous and papillary carcinoma into one
  "other malignant" class gives 38 vs 20 patients instead of 5-9 per class
  (`configs/train_ductal_vs_other.yaml`). It is the first stage-3 model to
  beat chance on validation, but only just:

  | Epoch | Backbone | val macro F1 | val loss | train loss |
  | --- | --- | --- | --- | --- |
  | 0 | frozen | 0.456 | 0.759 | 0.651 |
  | 1 | frozen | 0.454 | 0.769 | 0.605 |
  | 2 | unfrozen | **0.549** | 0.774 | 0.533 |
  | 3 | unfrozen | 0.546 | 0.816 | 0.440 |
  | 4 | unfrozen | 0.516 | 0.886 | 0.385 |

  The same overfitting signature as before: train loss keeps falling while
  validation loss rises, and early stopping ended the run. On the held-out
  test split (8 patients, 692 images) the best checkpoint reaches 0.662
  accuracy and 0.622 macro F1 — against a 0.585 accuracy for simply always
  answering "ductal", since ductal is 58.5% of the test images. It finds
  84% of ductal cases but only **40% of non-ductal ones**, calling the rest
  ductal.

  Its validation macro F1 of 0.549 is well under the 0.70 bar serving sets
  for a 2-class model (chance ≈ 0.50), so it is **not deployed**; stage 3
  continues to report no subtype. The checkpoint is kept under
  `checkpoints/experiments/ductal_vs_other/`.

  With both the 4-way and the pooled 2-way question tried, the limit is the
  number of patients, not the modelling choices. Another training run on
  this data isn't expected to change that; more patients per subtype would.
- **Input validation is measured, but on few examples of what it should
  reject.** Before deployment, every held-out test image was sent through
  the running app. The original colour screen rejected 7 of them (faint H&E
  slides) and, on the training and validation images, 1.5%; it also passed
  11 of 44 non-histology images, one of which the model then called 98.5%
  malignant. It was recalibrated on the 6,836 training and validation
  images, and a second check added: an ImageNet EfficientNet-B0 describes
  the image, and a probabilistic-PCA distance fitted on the training slides
  (threshold: the highest validation score) decides whether it looks like
  one. (The cancer models' own features couldn't do this; they placed
  wallpapers among the slides.)

  | | Original screen | Recalibrated colour screen alone | Colour screen + feature check (served) |
  | --- | --- | --- | --- |
  | Held-out test slides rejected | 7 of 1,073 | 0 of 1,073 | **0 of 1,073** |
  | Non-histology images let through | 11 of 44 | 18 of 44 | **3 of 44** (all synthetic noise) |
  | ...of which real photos, wallpapers, screenshots (37) | 8 | 15 | **0** |

  The non-histology set is small (the images available on this machine),
  so the real false-accept rate is unknown. Random noise still passes: its
  texture is within real slides' range and ImageNet features don't flag it.
  Anything that passes goes on to stages 2-3 and gets a confident label.
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
- **False negatives carry asymmetric harm — and this model is tuned the
  safe way, at a real cost.** A missed malignant case is far costlier than
  a false alarm, and this model errs heavily toward over-calling cancer
  (sensitivity 0.88-0.99, specificity 0.46-0.73). That direction is
  defensible, but the cost is that roughly half of benign tissue gets
  flagged, which would make it impractical without a human reviewing every
  positive. Reporting sensitivity and specificity side by side (not just
  accuracy or F1) is what makes that trade-off visible rather than hidden
  behind a single number — see Results.
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
the honest number). Regenerate with
`python -m scripts.evaluate_test_set --magnification 40`:

| Magnification | F1 | Accuracy | Precision | Sensitivity | **Specificity** | Images | Patients |
| -------------- | ---- | -------- | --------- | ----------- | --------------- | ------ | -------- |
| 200x           | 0.952 | 0.925   | 0.916     | 0.990       | **0.732**       | 281    | 11       |
| 40x            | 0.901 | 0.848   | 0.838     | 0.974       | **0.544**       | 270    | 11       |
| 400x           | 0.882 | 0.819   | 0.796     | 0.988       | **0.461**       | 238    | 11       |
| 100x           | 0.861 | 0.796   | 0.841     | 0.882       | **0.575**       | 284    | 11       |

### The specificity problem

**This is the most important number on the page, and F1 hides it.** The
model catches nearly every malignant case (sensitivity 0.88-0.99) but
misclassifies roughly half of all benign tissue as malignant. Confusion
matrices, rows = actual:

```
        40x                        400x
        benign  malignant          benign  malignant
benign      43         36   |  benign      35         41
malignant    5        186   |  malignant    2        160

        200x                       100x
        benign  malignant          benign  malignant
benign      52         19   |  benign      46         34
malignant    2        208   |  malignant   24        180
```

At 400x, 41 of 76 benign images (54%) are called malignant. A headline F1
of 0.88-0.95 looks strong only because F1 here is computed on the malignant
class, and malignant outnumbers benign roughly 2:1 in the test set — so a
model that says "malignant" too readily is rewarded twice over. Accuracy
is inflated for the same reason. Sensitivity and specificity reported
side by side are what make the actual behavior visible.

In screening terms this model is heavily biased toward over-calling
cancer. That bias is the *safer* direction if a human reviews every
positive (a missed cancer is worse than a false alarm), but a tool that
flags half of healthy tissue would be impractical in real use, and no
aggregate score above should be read as "the model works."

### Each magnification is served by its own model, detected from the image

The four models above are not interchangeable: the same image scored 52%,
41%, 52% and 24% malignant with the 40x, 100x, 200x and 400x models, since
three of them read the tissue at the wrong scale. Users often don't know the
zoom, so by default `/predict` detects it (a fine-tuned EfficientNet-B0,
`scripts/train_magnification.py`) and uses the matching model; a user can
still choose one, and is warned if the image confidently looks like another.

| Magnification detector | Validation (epoch chosen on) | Held-out test |
| --- | --- | --- |
| Accuracy | 98.6% | 96.2% |
| When at least 90% confident | 99.8% (92% of images) | 98.3% (88% of images) |

Below 90% the best guess is still used, with a warning that it may be wrong.
Its mistakes are between neighbouring zoom levels, mostly 400x read as 200x.
Sending all 1,073 held-out test images through the running app with
automatic detection: 1,032 reached the matching model, 25 of the other 41
carried the "unsure" warning, and the overall result was practically the
same as always choosing the right magnification by hand (734 of 767
cancers caught either way; 175 vs 176 of 306 benign images cleared).

### Borderline results are reported as uncertain

A 52%-malignant result used to be shown in the same red box as a 99% one.
Probabilities between 0.2 and 0.8 are now reported as "Uncertain" (leaning
benign or malignant) instead of as a finding. With the calibration below,
the served models are right 69% of the time inside that band and 95%
outside it on the validation images; on the held-out test images, 61%
against 89%. 15.8% of test images fall in the band, and 67 of the 163 wrong
test predictions are flagged there rather than stated.

### Probabilities are calibrated; test-time augmentation is not used

The raw probabilities were over-confident: validation images scored 70-80%
malignant were right about 64% of the time. One temperature per
magnification, fitted on validation logits (`scripts/calibrate_serving.py`;
T = 1.28, 1.64, 1.78 and 0.99 for 40x-400x), rescales them. That never
moves an image across the 0.5 threshold, so no label changes. Expected
calibration error falls for all four models on validation; on the held-out
test images it falls for 40x (0.143 to 0.136) and 100x (0.153 to 0.128),
the two worst, and stays about the same for 200x and 400x. The temperatures
are tied to each checkpoint's SHA-256, and a test fails if a model changes
without recalibrating.

Averaging over each image's 8 rotations and flips (test-time augmentation)
was also tried. The rule, written before running, adopted it if validation
balanced accuracy rose with sensitivity down at most one point, and it
"passed" on a 0.3-point rise (89.1% to 89.4%). The held-out test set moved
the other way (76.6% to 75.1%; benign specificity 57.5% to 53.6%), at 8x the
inference cost. A 0.3-point change on ~1,400 images is noise, and the rule
had no minimum, so it now requires a one-point gain and TTA is not used.
That is a change made after seeing a test result, and is recorded here for
that reason.

### Threshold tuning was tried, and rejected

The 0.5 decision threshold is where sigmoid crosses, not a chosen operating
point, so the obvious next move is to tune it. `python -m
scripts.tune_threshold --magnification 40` sweeps the whole curve on the
**validation** split (never test — picking a threshold is model selection).
Doing that for all four magnifications gives:

| Magnification | Best threshold (Youden J) | Val balanced acc at best | at 0.5 |
| ------------- | ------------------------- | ------------------------ | ------ |
| 40x  | 0.90 | 0.923 | 0.894 |
| 100x | 0.45 | 0.930 | 0.928 |
| 200x | 0.80 | 0.893 | 0.872 |
| 400x | 0.45 | 0.880 | 0.871 |

**The optimum is unstable — 0.45, 0.45, 0.80, 0.90 — and the gains are
0.002 to 0.029 balanced accuracy.** Four models on the same data disagreeing
that widely about where the threshold belongs is the signature of noise from
13-patient validation splits, not a real operating point. Tuning the
threshold to 0.90 on the strength of the 40x split would be fitting 13
patients, and would probably not transfer.

So the default stays at 0.5, and `DECISION_THRESHOLD` is exposed as an
environment variable for anyone who wants to move it deliberately. Where it
belongs depends on the cost of a missed cancer versus a false alarm — a
clinical judgement this project can inform but shouldn't quietly make.

### Class-balanced retraining was tried, and not adopted

Retraining the 200x model with class-balanced sampling
(`configs/train_balanced.yaml`) against the same held-out test split:

| 200x, test set | Original | Balanced |
| --- | --- | --- |
| Sensitivity | **0.990** | 0.929 |
| Specificity | 0.732 | **0.817** |
| Accuracy | **0.925** | 0.900 |
| Missed cancers (of 210) | **2** | 15 |
| False alarms (of 71 benign) | 19 | **13** |

Balancing did raise specificity, but by trading 13 additional missed
cancers for 6 fewer false alarms. For a screening-style task, where a missed
malignancy is the costlier error, that is the wrong direction, so the
original checkpoint stays in service. The balanced checkpoint is kept under
`checkpoints/experiments/balanced/` for reference. With 11 test patients
either difference is also within sampling noise; the honest summary is that
balancing moves the model along the sensitivity/specificity trade-off rather
than improving it.

### Stain normalization was tried, and not adopted

With ~60 training patients per magnification, a patient's slide colour could
stand in for its label, so all four models were retrained on Macenko
stain-normalized images (`src/data/stain.py`, `configs/train_stain_macenko.yaml`):
same hyperparameters, same patient split, only the colour space changed. The
rule for adopting a model was fixed before any result was seen: higher test
specificity, with sensitivity no more than one percentage point lower.

| Test set | Specificity: original → normalized | Sensitivity: original → normalized | False alarms | Missed cancers |
| --- | --- | --- | --- | --- |
| 40x  | **0.544** → 0.367 | **0.974** → 0.942 | 36 → 50 of 79 | 5 → 11 of 191 |
| 100x | **0.575** → 0.475 | 0.882 → **0.902** | 34 → 42 of 80 | 24 → 20 of 204 |
| 200x | **0.732** → 0.549 | **0.990** → 0.857 | 19 → 32 of 71 | 2 → 30 of 210 |
| 400x | **0.461** → 0.342 | **0.988** → 0.957 | 41 → 50 of 76 | 2 → 7 of 162 |

Specificity fell at every magnification, so none was adopted and the served
models are unchanged. Validation agreed (F1 0.83-0.88 against 0.91-0.94), and
it isn't only a short run: the 100x and 200x models trained as many epochs as
the originals. The likeliest reading is that stain colour carries real
diagnostic signal in BreakHis rather than only a patient shortcut, and
re-rendering every image with one reference stain removes some of it; the
original models' colour-jitter augmentation already covers moderate stain
variation. Serving supports normalized models (a checkpoint records its
preprocessing), so a future variant can be dropped in if one ever passes.

### Validation flatters these models relative to test

Specificity at threshold 0.5 is 0.81-0.96 on validation but 0.46-0.73 on
test, for the same checkpoints. The validation patients' benign tissue is
simply easier than the test patients'. With ~12 patients on each side of
that split, this gap is a property of which patients landed where, not of
the model — and it is the clearest single illustration of why every number
on this page should be read with wide error bars.

**The ranking flips between validation and test** (40x is best on val, but
200x is best on test). With only 11 test patients per magnification, this is
expected sampling variance rather than a robust ordering — it is itself a
limitation, not a bug: at this dataset size, per-magnification rankings
should be treated as noisy, and a claim like "40x magnification is best for
this task" is not statistically well-supported by these splits alone.

See `notebooks/02_error_analysis.ipynb` for a concrete failure mode found
in the 40x model: nearly all of its most confident errors are the benign
`tubular_adenoma` subtype misclassified as malignant, not a spread of
random mistakes.
