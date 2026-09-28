---
title: Breast Cancer Histology Classifier
emoji: 🔬
colorFrom: pink
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
short_description: Research-only BreakHis histology classifier, not diagnosis
---

# Breast Cancer Histology Classifier

Upload an H&E-stained breast histology image (BreakHis-style, 40x/100x/200x/400x)
to get a benign vs. malignant prediction with a Grad-CAM heatmap.

**Research and education only. Not a medical device and not for diagnosis.**
The model was trained on 82 patients from a single institution; it misclassifies
a substantial share of benign tissue as malignant, and it does not report a
malignant subtype because no trained subtype model cleared the quality bar.

Source, model card and measured limitations:
https://github.com/amarasie0602/breast-cancer-classification

This Space is deployed automatically from that repository: every merge to
`main` publishes a Docker image to GitHub Container Registry and points this
Space at it. Don't edit files here; they are overwritten on the next deploy.
