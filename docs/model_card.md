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

_To be filled in once training runs against the real dataset complete —
see `src/training/compare_runs.py` for the per-magnification comparison
this section will report._
