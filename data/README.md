# Dataset placement

This directory is gitignored (raw images are large and versioned via DVC, not Git).

Download the BreaKHis dataset and extract it here so the layout matches:

    data/BreaKHis_v1/histology_slides/breast/<benign|malignant>/SOB/<subtype>/<patient_id>/<mag>X/<image>.png

Source: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/
(official; requires filling a short access-request form).

Alternatively, via the Kaggle mirror `ambarish/breakhis` (no form required):

    kaggle datasets download -d ambarish/breakhis -p data/_kaggle_download
    # unzip, then move the inner BreaKHis_v1/BreaKHis_v1/ up one level to data/BreaKHis_v1/
    # (the Kaggle archive nests it one directory deeper than the layout above)

Once the data is in place, track it with `dvc add data/BreaKHis_v1` and
`dvc push`.
