# Final Submission Guide

This guide supplements the existing README without changing the model, weights,
predictions, or inference interface.

## Current artifacts

- Source code: this repository
- PyTorch checkpoint: `model_weights.pt`
- Prediction file: `predictions.csv`
- Approach and results: `README.md` and [`TECHNICAL_REPORT.md`](./TECHNICAL_REPORT.md)
- Training log: `training_log.txt`

Use the new technical report for the report deliverable; the older `report.pdf`
is not a substitute for the organizer's minimum four-page report requirement.

The historical training log records an AUC of 0.8925. The checkpoint was
re-evaluated on the organizer archive, and measured validation results are
documented in `README.md` and `TECHNICAL_REPORT.md`. The official dataset is
not included in the public repository; the checks were run locally against
the downloaded organizer archive.

## Evaluate the model on a validation split of the provided training data

Extract `hackathon_dataset.zip` so the organizer's metadata and images are:

```text
hackathon_dataset/
  train/
    real/
    fake/
  test/
    images/
  train_metadata.csv
```

Install the pinned dependencies and run the evaluator from the repository
directory:

```powershell
python -m pip install -r requirements.txt
python .\evaluate_validation.py --data-dir .\hackathon_dataset --weights .\model_weights.pt
```

The evaluator uses the required stratified 80/20 split with
`random_state=42`. It prints measured ROC AUC, accuracy, and F1 at threshold
0.5 for the clean validation images and for JPEG compression, Gaussian blur,
and half-resolution resizing. It also reports clean-validation inference time
and model parameter count. These corruption checks are validation diagnostics;
they do not replace evaluation on the official test set.
Because the training script selected the checkpoint using validation AUC, the
validation results are model-selection diagnostics, not an independent
generalization estimate.

## Generate predictions for the official test set

Keep the required inference arguments and output columns unchanged. Write to a
new file first so the existing `predictions.csv` is preserved:

```powershell
python .\inference_script.py --test_dir .\hackathon_dataset\test\images --weights .\model_weights.pt --output .\submission_predictions.csv
```

Check that the generated CSV has exactly the columns
`image_id,label,confidence` and one row per official test image. Only replace
the submitted prediction file after confirming that the test directory is the
official one and the generated file passes those checks.

You can run the non-destructive checker against the generated file:

```powershell
python .\check_submission.py --test_dir .\hackathon_dataset\test\images --predictions .\submission_predictions.csv
```

It verifies the exact column order, image IDs and row count, binary labels, and
finite confidence values between 0 and 1. It does not modify the CSV or images.
The existing `predictions.csv` has also been checked against the organizer
test images and passes. It can be used as the required prediction artifact;
the separately generated `submission_predictions.csv` was kept as a new file
so the existing CSV remained untouched.

## Prepare the Drive submission

1. Create a Drive folder for the final team submission.
2. Set general access to **Anyone with the link — Viewer**.
3. Include the public repository link, the compatible PyTorch weights, the
   verified official-test prediction CSV, and the technical report.
4. Include any additional files requested by the organizers. Do not include
   unofficial test data or claim metrics that were not measured.
5. Open the folder link in a private/incognito browser window to verify access,
   then submit that link using the organizers' required channel.

The repository is public. Drive-folder access must still be verified by the
team before final submission.
