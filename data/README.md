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

The DVC remote is Google Drive (15 GB free tier, enough for the ~4.3 GB
dataset and the ~1.1 GB training checkpoints). The folder id is a placeholder
in `.dvc/config`. Credentials never go in git: a laptop login stays in the
local DVC cache, and a service account path belongs in `.dvc/config.local`
(see `.dvc/config.local.example`).

One-time setup, after the dataset is on disk:

```bash
# 1. In Google Drive, create an empty folder. Its URL ends in the folder id:
#    https://drive.google.com/drive/folders/FOLDER_ID
# 2. Point the committed remote at that folder.
dvc remote modify gdrive url gdrive://FOLDER_ID

# 3. Install the Drive backend. It is not in requirements.lock.txt.
pip install 'dvc[gdrive]==3.67.1'

# 4. Upload. The first push opens a browser. Do not commit the token.
dvc push

# 5. Freeze the patient split and commit splits/patients.json yourself.
python -m scripts.freeze_patient_splits --data-root data/BreaKHis_v1
```

For GitHub Actions, create a service account, share the Drive folder with it,
and store the JSON in the secret `GDRIVE_SERVICE_ACCOUNT_JSON`. The
model-validation job checks out Git LFS, pulls this remote when that secret
is set, and fails if the gate tests skip.

If the dataset was not tracked yet:

```bash
dvc add data/BreaKHis_v1
dvc push
```
