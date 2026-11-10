# Usage & Evaluation Guide

This guide covers running validation diagnostics, generating test predictions, and verifying output CSV formatting.

## Project Artifacts

- Source scripts: `train.py`, `inference_script.py`, `evaluate_validation.py`, `check_submission.py`
- Model checkpoint: `model_weights.pt`
- Test predictions: `predictions.csv`
- Documentation & logs: `README.md`, `TECHNICAL_REPORT.md`, `report.pdf`, `training_log.txt`

## 1. Validation & Robustness Evaluation

Extract the dataset so the folder structure matches:

```text
dataset/
  train/
    real/
    fake/
  test/
    images/
  train_metadata.csv
```

Install the dependencies and run `evaluate_validation.py`:

```bash
pip install -r requirements.txt
python evaluate_validation.py --data-dir ./dataset --weights ./model_weights.pt
```

This evaluates the saved checkpoint on the stratified 80/20 validation split (`random_state=42`) and reports ROC AUC, accuracy, and F1 at threshold `0.5` for clean images as well as JPEG quality 50, Gaussian blur (radius 1), and half-resolution resizing.

## 2. Test Inference & Output Verification

Generate predictions on the test directory:

```bash
python inference_script.py --test_dir ./dataset/test/images --weights ./model_weights.pt --output ./predictions.csv
```

Verify that the generated CSV contains the required `image_id,label,confidence` columns, matches all test image filenames, and has valid binary labels and finite probabilities in `[0, 1]`:

```bash
python check_submission.py --test_dir ./dataset/test/images --predictions ./predictions.csv
```
