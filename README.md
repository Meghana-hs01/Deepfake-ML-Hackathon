# Deepfake ML Hackathon: Team Submission

Binary classifier that labels face images as real (0) or deepfake (1), trained only on the provided dataset.

## Results (own validation split, 80/20 stratified)

The existing training log records a clean validation AUC of 0.8925 at epoch 8.
After the organizer dataset became available, the existing checkpoint was
re-evaluated on the provided training metadata using the required stratified
80/20 split. No weights were changed for this evaluation. The measurements are:

| Validation variant (1,400 images) | ROC AUC | Accuracy (0.5) | F1 (0.5) |
|---|---:|---:|---:|
| Clean | 0.9861 | 0.9636 | 0.9634 |
| JPEG quality 50 | 0.9822 | 0.9571 | 0.9573 |
| Gaussian blur (radius 1) | 0.9840 | 0.9557 | 0.9549 |
| Half-resolution resize | 0.9803 | 0.9457 | 0.9442 |

These are local validation and corruption-diagnostic results, not the
organizers' sealed test score or robustness score. The fixed decision threshold
for accuracy and F1 is 0.5. The split is
`train_test_split(test_size=0.2, stratify=label, random_state=42)`.
The historical training-log AUC could not be reproduced from the current
checkpoint: evaluation using class-directory iteration order produced AUC
0.9823, while the metadata-row-order split above produced 0.9861. The split
seed does not determine membership without a fixed input row order; see
`TECHNICAL_REPORT.md` for details.

The full 3,000-image test inference was run. Both the existing
`predictions.csv` and a newly generated output passed `check_submission.py`
against the organizer test images. Their IDs and labels matched, with a
maximum confidence difference of approximately 0.000003. This verifies
coverage and format only; test labels are not available to measure accuracy.

## Environment

- Training environment recorded in the existing project notes: Python 3.13 (Google Colab), PyTorch 2.11, torchvision 0.26.0
- Validation run in this workspace: Python 3.10.11, PyTorch 2.14.1+cpu, torchvision 0.29.2+cpu; CUDA unavailable
- Install: `pip install -r requirements.txt`

## Reproducing Predictions

```bash
python inference_script.py --test_dir ./hackathon_dataset/test/images/ --weights ./model_weights.pt --output submission_predictions.csv
```

Output: one row per test image, columns `image_id,label,confidence`.
Use a new output filename to preserve the existing `predictions.csv`. Confirm
that `hackathon_dataset/test/images/` contains the official test images before
generating final submission predictions.

To verify that the generated CSV matches the test-image directory without
changing either file, run:

```bash
python check_submission.py --test_dir ./hackathon_dataset/test/images/ --predictions ./submission_predictions.csv
```

## Training

`train.py` expects the current working directory to contain
`train/real/` and `train/fake/`. From the extracted organizer dataset directory
in Colab/Linux, run:

```bash
cd hackathon_dataset
python ../train.py
```

The script writes its checkpoint to the current working directory. Running it
from `hackathon_dataset/` therefore does not overwrite the submitted
`model_weights.pt` in the repository directory.

- Split: 80/20 stratified on `label`, `random_state=42`
- Backbone: EfficientNet-B0, ImageNet pretrained, fine-tuned on the provided dataset only
- Head: Dropout(0.3) + Linear(1280, 1), sigmoid output
- Loss: binary cross-entropy with logits
- Optimizer: AdamW, lr 3e-4, weight decay 1e-4, no scheduler
- Epochs: 10, batch size 32 (best checkpoint saved at epoch 8)
- Augmentation (train only): horizontal flip, brightness/contrast jitter (±0.1), Gaussian blur (20% of images)
- Input: 224x224, ImageNet normalization

## Validation and robustness evaluation

The optional `evaluate_validation.py` script loads the existing checkpoint and
supports the organizer archive layout (`train/real/`, `train/fake/`, and
`train_metadata.csv`). Extract `hackathon_dataset.zip` so the
`hackathon_dataset/` folder is beside the script, then run:

```bash
python evaluate_validation.py --data-dir ./hackathon_dataset --weights ./model_weights.pt
```

It reports measured ROC AUC, accuracy, and F1 at threshold 0.5 for clean
validation images and for JPEG compression, Gaussian blur, and half-resolution
resizing. It also reports clean-validation inference time and parameter count.
The results above were produced by this command on the provided dataset.
The checkpoint was selected using validation AUC during training, so results
from this split are model-selection diagnostics, not an independent estimate
of generalization.

## Technical report

See [`TECHNICAL_REPORT.md`](./TECHNICAL_REPORT.md) for the full methodology,
measured validation and corruption results, test CSV verification, limitations,
and reproduction commands.

## Data

Only the officially provided dataset was used for training. No external data, and no test images were used in training.

## Limitations

- The training log's AUC of 0.8925 does not match the current checkpoint's re-evaluation; the source of the discrepancy is unknown.
- Validation metrics are model-selection diagnostics, not an independent estimate of generalization.
- Local corruption checks cover only JPEG quality 50, Gaussian blur, and half-resolution resizing; organizer test variants are undisclosed.
- The organizers' test labels are private, so official test accuracy and ROC AUC are unavailable.
- Local inference timing was measured on CPU, not on the organizers' NVIDIA T4.
