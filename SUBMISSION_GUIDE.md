# Final Submission Guide

This guide supplements the existing README without changing the model, weights,
predictions, or inference interface.

## Current artifacts

- Source code: this repository
- PyTorch checkpoint: `model_weights.pt`
- Prediction file: `predictions.csv`
- Existing approach and results: `README.md` and `report.pdf`
- Training log: `training_log.txt`

The reported validation AUC is the value recorded in the existing training log.
No additional metrics or official-test performance are claimed here. The
official dataset and test images are not included in this repository, so the
validation and official-test commands below must be run where those files are
available.

## Evaluate the model on a validation split of the provided training data

Arrange the provided training images as:

```text
train/
  real/
  fake/
```

Install the pinned dependencies and run the evaluator from the repository
directory:

```powershell
python -m pip install -r requirements.txt
python .\evaluate_validation.py --data-dir .\train --weights .\model_weights.pt
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
python .\inference_script.py --test_dir .\test\images --weights .\model_weights.pt --output .\submission_predictions.csv
```

Check that the generated CSV has exactly the columns
`image_id,label,confidence` and one row per official test image. Only replace
the submitted prediction file after confirming that the test directory is the
official one and the generated file passes those checks.

## Prepare the Drive submission

1. Create a Drive folder for the final team submission.
2. Set general access to **Anyone with the link — Viewer**.
3. Include the public repository link, the compatible PyTorch weights, the
   verified official-test prediction CSV, and the technical report.
4. Include any additional files requested by the organizers. Do not include
   unofficial test data or claim metrics that were not measured.
5. Open the folder link in a private/incognito browser window to verify access,
   then submit that link using the organizers' required channel.

The repository is public, but Drive-folder access and official-test
predictions must still be verified by the team before final submission.
