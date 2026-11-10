# Deepfake ML Hackathon: Team Submission

Binary classifier that labels face images as real (0) or deepfake (1), trained
on the organizer-provided dataset.

## Results (stratified 80/20 validation split)

The existing training log records a clean validation AUC of 0.8925 at epoch 8.
The supplied checkpoint was subsequently evaluated on the organizer's
`train_metadata.csv`, using the required split. No weights were changed for
this evaluation.

| Validation variant (1,400 images) | ROC AUC | Accuracy (0.5) | F1 (0.5) |
|---|---:|---:|---:|
| Clean | 0.9861 | 0.9636 | 0.9634 |
| JPEG quality 50 | 0.9822 | 0.9571 | 0.9573 |
| Gaussian blur (radius 1) | 0.9840 | 0.9557 | 0.9549 |
| Half-resolution resize | 0.9803 | 0.9457 | 0.9442 |

These are local validation and corruption-diagnostic results, not the
organizers' sealed test score or hidden robustness score. Accuracy and F1 use
threshold 0.5. The split is
`train_test_split(test_size=0.2, stratify=label, random_state=42)`.

The historical training-log AUC could not be reproduced from the current
checkpoint: evaluation using class-directory iteration order produced AUC
0.9823, while the metadata-row-order split above produced 0.9861. The split
seed does not determine membership without a fixed input row order. See
[`TECHNICAL_REPORT.md`](./TECHNICAL_REPORT.md) for details.

## Test inference result

Inference ran on all 3,000 organizer test images. The existing
`predictions.csv` and a newly generated CSV both passed the submission checker.
Their IDs and labels matched, with a maximum confidence difference of about
0.000003. This verifies coverage, format, and reproducibility only; private
test labels are unavailable, so test accuracy and test AUC are unknown.

## Environment

- Original training environment recorded in project notes: Python 3.13 (Google Colab), PyTorch 2.11, torchvision 0.26.0.
- Validation and test inference in this workspace: Python 3.10.11, PyTorch 2.14.1+cpu, torchvision 0.29.2+cpu, pandas 2.3.3, scikit-learn 1.7.2, Pillow 12.2.0. CUDA was unavailable.
- Install dependencies with `python -m pip install -r requirements.txt`.

## Run and test on Windows

Download and extract the organizer's `hackathon_dataset.zip`. These commands
assume the extracted folder is at
`%USERPROFILE%\Downloads\datasets\hackathon_dataset`. Change `$Dataset` if you
extracted it elsewhere. Run in PowerShell:

```powershell
Set-Location "C:\path\to\Deepfake_Submission"
$Dataset = Join-Path $env:USERPROFILE "Downloads\datasets\hackathon_dataset"

python -m pip install -r .\requirements.txt
python .\evaluate_validation.py --data-dir $Dataset --weights .\model_weights.pt
python .\inference_script.py --test_dir (Join-Path $Dataset "test\images") --weights .\model_weights.pt --output .\submission_predictions.csv
python .\check_submission.py --test_dir (Join-Path $Dataset "test\images") --predictions .\submission_predictions.csv
```

Replace `C:\path\to\Deepfake_Submission` with the folder containing this
README and the scripts. The evaluator uses the fixed stratified 80/20 split.
Inference processes every image in the test folder and writes to the new
`submission_predictions.csv`, leaving the existing `predictions.csv` unchanged.
The checker verifies the exact CSV column order, every test image ID and row
count, binary labels, and valid confidence values. It does not modify any file.

### Example measured validation output

The evaluation completed on 1,400 held-out training images:

```text
device: cpu
validation_images: 1400
clean ROC AUC: 0.9861
clean accuracy at 0.5: 0.9636
clean F1 at 0.5: 0.9634
JPEG quality 50 ROC AUC: 0.9822
Gaussian blur ROC AUC: 0.9840
Half-resolution ROC AUC: 0.9803
```

The command prints the full measured result as JSON, including timing,
parameter count, accuracy, and F1 for every validation variant. These
validation measurements do not reveal the private test score.

### Example test prediction output

The inference run found 3,000 official test images and saved 3,000 rows. The
checker printed:

```text
Found 3000 images
Saved 3000 rows to submission_predictions.csv
Submission CSV is valid: 3000 predictions match the test image directory.
```

Example rows from the generated CSV:

```csv
image_id,label,confidence
0020cba7-1d58-4784-91f9-6fc47d9f2806,1,0.995018720626831
003fae32-30d9-441b-8fcf-073fc04bf1d1,0,0.000005597865310846828
```

`label` is 0 for real and 1 for fake. `confidence` is the predicted
probability that an image is fake. These sample predictions are not test
accuracy claims.

For Linux or macOS, run inference and check the CSV with:

```bash
python inference_script.py --test_dir ./hackathon_dataset/test/images/ --weights ./model_weights.pt --output ./submission_predictions.csv
python check_submission.py --test_dir ./hackathon_dataset/test/images/ --predictions ./submission_predictions.csv
```

## Training

`train.py` expects the current working directory to contain `train/real/` and
`train/fake/`. From the extracted organizer dataset directory in Colab/Linux:

```bash
cd hackathon_dataset
python ../train.py
```

The script writes its checkpoint to the current working directory. Running it
from `hackathon_dataset/` keeps a new checkpoint separate from the submitted
`model_weights.pt` in the repository directory.

- Split: 80/20 stratified on `label`, `random_state=42`
- Backbone: EfficientNet-B0, ImageNet pretrained, fine-tuned only on the provided training data
- Head: Dropout(0.3) + Linear(1280, 1)
- Loss: binary cross-entropy with logits
- Optimizer: AdamW, learning rate 3e-4, weight decay 1e-4, no scheduler
- Epochs: 10, batch size 32; training log records best checkpoint at epoch 8
- Augmentation (train only): horizontal flip, brightness/contrast jitter (±0.1), Gaussian blur (20% of images)
- Input: 224x224, ImageNet normalization

## Validation and robustness evaluation

`evaluate_validation.py` supports the organizer archive layout containing
`train/real/`, `train/fake/`, and `train_metadata.csv`. It reports ROC AUC,
accuracy, and F1 at threshold 0.5 for clean validation images and the three
documented corruption tests. It also reports clean inference timing and
parameter count. Since the checkpoint was selected using validation AUC,
these are model-selection diagnostics, not an independent estimate of
generalization.

## Technical report

See [`TECHNICAL_REPORT.md`](./TECHNICAL_REPORT.md) for the full methodology,
measured validation and corruption results, test CSV verification, historical
metric discrepancy, limitations, and reproduction commands.

## Data and limitations

Only the organizer-provided dataset was used; no external training data or
test images were used for training. The organizers' test labels and hidden
test variants are private. Local corruption checks cover only JPEG quality
50, Gaussian blur, and half-resolution resizing. Local inference timing was
measured on CPU, not on the organizers' NVIDIA T4.
