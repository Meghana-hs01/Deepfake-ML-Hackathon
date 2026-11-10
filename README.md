# Deepfake ML Hackathon submission

This project classifies face images as real (`0`) or fake (`1`). It uses an
EfficientNet-B0 model trained on the organizer-provided images.

## Validation results

The supplied checkpoint was evaluated on 1,400 images from the training set.
The split was stratified and used `test_size=0.2` and `random_state=42`.
Accuracy and F1 use a fixed threshold of 0.5.

| Validation input | ROC AUC | Accuracy | F1 |
|---|---:|---:|---:|
| Clean | 0.9861 | 0.9636 | 0.9634 |
| JPEG quality 50 | 0.9822 | 0.9571 | 0.9573 |
| Gaussian blur (radius 1) | 0.9840 | 0.9557 | 0.9549 |
| Half-resolution resize | 0.9803 | 0.9457 | 0.9442 |

These are local validation results, not the hackathon's private test score.
The blur, compression, and resize runs are simple checks on this same
validation split; they do not represent the organizers' hidden test variants.

There is a discrepancy worth noting: `training_log.txt` records a best AUC of
0.8925, while evaluating the supplied checkpoint on the metadata-ordered
split produced 0.9861. Using class-directory iteration order produced 0.9823
on a different split. The split seed alone does not determine the selected
images unless the input order is also fixed. The reason for the difference
between the historical log and these later evaluations is unknown. More
details are in [`TECHNICAL_REPORT.md`](./TECHNICAL_REPORT.md).

## Test predictions

The inference script processed all 3,000 organizer test images. The generated
CSV and the existing `predictions.csv` both passed the submission checker.
Their image IDs and labels matched; the largest confidence difference was
about 0.000003. This confirms file coverage and format, not prediction
accuracy: test labels are private.

Each output row contains an image ID, a binary label, and the model's
estimated probability that the image is fake:

```csv
image_id,label,confidence
0020cba7-1d58-4784-91f9-6fc47d9f2806,1,0.995018720626831
003fae32-30d9-441b-8fcf-073fc04bf1d1,0,0.000005597865310846828
```

## Run the checks on Windows

Download and extract the organizer's `hackathon_dataset.zip`. The commands
below assume it is at
`%USERPROFILE%\Downloads\datasets\hackathon_dataset`. Open PowerShell in this
project folder and replace the example project path with your actual path:

```powershell
Set-Location "C:\path\to\Deepfake_Submission"
$Dataset = Join-Path $env:USERPROFILE "Downloads\datasets\hackathon_dataset"

python -m pip install -r .\requirements.txt
python .\evaluate_validation.py --data-dir $Dataset --weights .\model_weights.pt
python .\inference_script.py --test_dir (Join-Path $Dataset "test\images") --weights .\model_weights.pt --output .\submission_predictions.csv
python .\check_submission.py --test_dir (Join-Path $Dataset "test\images") --predictions .\submission_predictions.csv
```

The inference command writes a new `submission_predictions.csv` so the
existing `predictions.csv` is left untouched. The checker verifies the header
and column order, test image IDs and row count, labels, and confidence values.

On Linux or macOS, the equivalent inference and check commands are:

```bash
python inference_script.py --test_dir ./hackathon_dataset/test/images/ --weights ./model_weights.pt --output ./submission_predictions.csv
python check_submission.py --test_dir ./hackathon_dataset/test/images/ --predictions ./submission_predictions.csv
```

## Training setup

The model is EfficientNet-B0 with a dropout layer (`0.3`) and a single-logit
output head. Training uses binary cross-entropy with logits and AdamW
(learning rate `3e-4`, weight decay `1e-4`). Images are resized to 224 by 224
and normalized with ImageNet statistics. Training augmentation includes
horizontal flips, brightness/contrast jitter, and Gaussian blur on 20% of
images.

`train.py` expects `train/real/` and `train/fake/` under the current working
directory. From the extracted dataset folder in Colab/Linux, run:

```bash
cd hackathon_dataset
python ../train.py
```

This writes the resulting checkpoint inside `hackathon_dataset/`, rather than
overwriting the checkpoint beside the project scripts.

## Environment and report

The original project notes list Python 3.13, PyTorch 2.11, and torchvision
0.26.0 for training. The later validation and inference run used Python
3.10.11, PyTorch 2.14.1+cpu, and torchvision 0.29.2+cpu; CUDA was unavailable.
Install dependencies with `python -m pip install -r requirements.txt`.

The longer [`TECHNICAL_REPORT.md`](./TECHNICAL_REPORT.md) describes the
dataset checks, model, evaluation, limitations, and the difference between
the historical training log and later evaluation.

## AI assistance

AI assistance was used in preparing this project and its documentation. The
team should follow the hackathon's rules on AI use and disclose assistance as
required.
