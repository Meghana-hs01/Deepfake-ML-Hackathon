# Deepfake ML Hackathon: Team Submission

Binary classifier that labels face images as real (0) or deepfake (1), trained only on the provided dataset.

## Results (own validation split, 80/20 stratified)

The AUC below is transcribed from `training_log.txt` (best checkpoint, epoch 8).
Other metrics are intentionally marked as unmeasured; smoke tests on synthetic
images are not model-performance results.

| Metric | Value |
|---|---|
| AUC-ROC (clean validation) | 0.8925 (epoch 8, as recorded in `training_log.txt`) |
| Accuracy / F1 (threshold 0.5) | Not measured |
| Corruption robustness | Not measured on the provided-dataset validation split |
| Decision threshold | 0.5 (fixed) |

## Environment

- Python 3.13 (Google Colab), PyTorch 2.11, torchvision 0.26.0
- Install: `pip install -r requirements.txt`

## Reproducing Predictions

```bash
python inference_script.py --test_dir ./test/images/ --weights ./model_weights.pt --output submission_predictions.csv
```

Output: one row per test image, columns `image_id,label,confidence`.
Use a new output filename to preserve the existing `predictions.csv`. Confirm
that `test/images/` contains the official test images before generating the
final submission predictions.

To verify that the generated CSV matches the test-image directory without
changing either file, run:

```bash
python check_submission.py --test_dir ./test/images/ --predictions ./submission_predictions.csv
```

## Training

```bash
python train.py
```

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
evaluates it on the same stratified 80/20 split (`test_size=0.2`,
`stratify=label`, `random_state=42`). Arrange the provided training images as
`train/real/` and `train/fake/`, then run:

```bash
python evaluate_validation.py --data-dir ./train --weights ./model_weights.pt
```

It reports measured ROC AUC, accuracy, and F1 at threshold 0.5 for clean
validation images and for JPEG compression, Gaussian blur, and half-resolution
resizing. It also reports clean-validation inference time and parameter count.
These results are printed when run; they are not included above because the
dataset was not available to run this evaluation when preparing the submission.
The checkpoint was selected using validation AUC during training, so results
from this split are model-selection diagnostics, not an independent estimate
of generalization.

## Data

Only the officially provided dataset was used for training. No external data, and no test images were used in training.

## Limitations

- Validation AUC plateaued around 0.88–0.89 from epoch 2 onward while training loss kept falling (0.53 to 0.08), which suggests overfitting.
- Robustness to JPEG compression, blur, and resizing has not been measured on the provided-dataset validation split.
- Accuracy and F1 have not been measured on the provided-dataset validation split.
