# Deepfake ML Hackathon Technical Report

## 1. Problem, dataset, and compliance

### Objective

This project classifies a face image as real (`label = 0`) or fake (`label = 1`).
The submitted model is an EfficientNet-B0 image classifier implemented in
PyTorch. It returns a confidence value interpreted as the probability that an
image is fake. The hackathon's primary metric is ROC AUC, which evaluates the
ranking of those confidence values rather than only the binary decision at a
single threshold.

### Organizer-provided data

The organizer archive contains 7,000 labeled training images and 3,000
unlabeled test images. The training metadata contains 3,478 real examples and
3,522 fake examples. Each metadata `image_id` was checked against a training
image filename; all 7,000 IDs had matching images. The test image IDs were
checked against the supplied sample-submission IDs; all 3,000 matched.

The data layout is:

```text
hackathon_dataset/
  train/
    real/
    fake/
  test/
    images/
  train_metadata.csv
  sample_submission.csv
```

Only the provided training images were used for validation. No external
training data was introduced. The model uses the permitted ImageNet-pretrained
EfficientNet-B0 architecture and fine-tunes it on the organizer-provided
training set. The unlabeled test images were used only for the final
prediction-generation and format checks, not for fitting or validation.

The archive was inspected before evaluation. All 7,000 metadata IDs resolved
to training image files, and no training image was missing from the metadata.
The images in `train/real/` mapped to label 0, and those in `train/fake/`
mapped to label 1. This cross-check verifies the label convention used by the
evaluator rather than relying only on directory names. The supplied sample
submission contained 3,000 IDs, matching the test image folder exactly.

### Validation protocol

The local validation subset is created from the training metadata using the
organizer-recommended split:

```python
train_test_split(
    metadata,
    test_size=0.2,
    stratify=metadata["label"],
    random_state=42,
)
```

This produces 1,400 validation examples: 696 real and 704 fake. Stratification
keeps the class proportions close to those of the complete training set. The
test set has no public labels, so no test accuracy or test ROC AUC can be
calculated by the team.

The input row order passed to `train_test_split` is part of reproducibility.
The validation script reads records in the order of `train_metadata.csv`.
Using the same split fraction, stratification column, and random seed with a
different input order can select different validation images. Future
comparisons should retain the metadata order or record the validation IDs as
part of the experiment output.

### Label and confidence conventions

The convention is consistent across the training metadata, model head,
inference script, and submission format: real is 0 and fake is 1. The
`confidence` field is the sigmoid output for the fake class, not the
confidence in whichever class was predicted. This matters because the
organizers calculate AUC from the confidence column and hidden labels. The
discrete label is calculated as 1 when confidence is at least 0.5. Changing
this threshold would change accuracy and F1, but not the confidence ranking
used for ROC AUC.

---

## 2. Model and training pipeline

### Architecture

The network uses torchvision's EfficientNet-B0 with its ImageNet-pretrained
feature extractor. The original classification head is replaced with:

| Component | Configuration |
|---|---|
| Feature backbone | EfficientNet-B0 |
| Regularization | Dropout, probability 0.3 |
| Output layer | Linear, 1,280 input features to 1 logit |
| Output interpretation | Sigmoid probability for fake class |
| Total parameters | 4,008,829 |

EfficientNet-B0 is a compact convolutional backbone intended to balance
representational capacity with inference cost. It is the architecture already
present in the submitted checkpoint and inference script; no architecture
comparison was performed for this submission. The ImageNet initialization is
a pre-trained model allowed by the event rules. Fine-tuning and validation
examples come only from the provided dataset.

The final layer emits one logit. Training uses
`BCEWithLogitsLoss`, which combines the sigmoid operation and binary
cross-entropy in a numerically stable loss. At inference, sigmoid converts the
logit to a fake-class confidence in `[0, 1]`; the script assigns label 1 at
confidence `>= 0.5`.

### Image processing and augmentation

Images are converted to RGB, resized to 224 by 224 pixels, converted to
tensors, and normalized using the standard ImageNet channel means and standard
deviations. Training augmentation applies random horizontal flips and
brightness/contrast jitter with magnitude 0.1. Gaussian blur is applied to
20% of training examples, with a radius sampled from 0.5 to 1.5. Validation
and test preprocessing do not apply random augmentation.

### Optimization

| Setting | Value |
|---|---:|
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Weight decay | 0.0001 |
| Batch size | 32 |
| Epochs | 10 |
| Random seed | 42 |
| Checkpoint selection | Highest validation ROC AUC |

AdamW provides adaptive updates with decoupled weight decay. The learning rate
and weight decay are the values recorded in the training script. The dropout
head and image augmentations are intended to regularize fine-tuning. The
submission artifacts do not contain a controlled hyperparameter ablation, so
these choices should not be presented as experimentally optimal.

The training log records the best checkpoint at epoch 8, with a clean
validation AUC of 0.8925. It also shows training loss falling from 0.5270 at
epoch 1 to 0.0811 at epoch 10, while logged validation AUC varies around
0.88-0.89. This pattern motivated additional validation diagnostics and is
consistent with possible overfitting.

The training script seeds Python, NumPy, and PyTorch with 42 and uses a
stratified train/validation split. Random image augmentation is applied only
to the training dataset; the validation transform is deterministic. When a
CUDA device is present, training enables automatic mixed precision with a
gradient scaler. Otherwise the script uses regular precision. The saved
checkpoint is a PyTorch state dictionary and is loaded into the same
EfficientNet-B0 classifier structure for inference.
The recorded configuration has no learning-rate scheduler. The artifacts do
not include repeated-seed runs or an ablation comparing augmentation,
regularization, or optimizer variants. The listed parameters document the
executed pipeline; they are not evidence that each setting is optimal. A
future comparison should preserve one fixed validation-ID list and leave the
organizer's test set untouched.

---

## 3. Measured validation and robustness diagnostics

After obtaining the organizer archive, the existing checkpoint was evaluated
with `evaluate_validation.py` on the 1,400-example split from
`train_metadata.csv`. The checkpoint was not retrained or replaced for this
measurement. ROC AUC, accuracy, and F1 were computed from the model's
probabilities and binary predictions at threshold 0.5.

| Validation input | ROC AUC | Accuracy | F1 |
|---|---:|---:|---:|
| Clean | 0.9861 | 0.9636 | 0.9634 |
| JPEG quality 50 | 0.9822 | 0.9571 | 0.9573 |
| Gaussian blur, radius 1 | 0.9840 | 0.9557 | 0.9549 |
| Downsample by half, then resize | 0.9803 | 0.9457 | 0.9442 |

The corruption variants are local stress tests applied to the validation
images. They indicate that performance on this split remained similar under
these particular transformations. They are not the organizers' undisclosed
test variants and must not be described as the official robustness score.
Results are also subject to validation-set model-selection bias because the
training process selected a checkpoint using validation AUC.

ROC AUC measures how well the fake-class confidence ranks fake examples
above real examples across all possible thresholds. This aligns with the
organizers' primary metric. Accuracy measures the fraction of thresholded
labels that match the known validation labels. F1 is the harmonic mean of
precision and recall for the fake class. Since the training classes are
nearly balanced, accuracy and F1 provide useful context, but neither replaces
ROC AUC as the primary comparison. These are all validation measurements;
none estimates the hidden test score without qualification.

The corruption transformations are deterministic. JPEG quality 50
re-encodes each RGB image in memory. Gaussian blur uses a fixed radius of 1.0.
The half-resolution test reduces each image dimension by half with bilinear
interpolation, then restores the original size before the normal 224-by-224
model transform. Each variant is evaluated on the same validation IDs as the
clean result. These controlled changes make the diagnostics repeatable, but
cover only a small fraction of possible image artifacts, compression
settings, resizing pipelines, and camera conditions.

Relative to the clean validation row, the AUC decreased by about 0.0039 for
JPEG quality 50, 0.0021 for blur, and 0.0058 for half-resolution resizing.
Accuracy decreased by about 0.0064, 0.0079, and 0.0179 respectively. These
are absolute differences calculated from the displayed measurements, not
official robustness scores. The largest of these particular changes affected
thresholded accuracy on half-resolution images; this observation should not
be generalized to undisclosed transformations or other datasets.

### Historical metric discrepancy

The existing `training_log.txt` records a validation AUC of 0.8925. The
re-evaluation of the currently supplied checkpoint on the organizer metadata
split produced 0.9861. Recreating the split from class-directory iteration
order produced a different validation subset and AUC of 0.9823. The split
function and seed alone do not identify the exact sample membership unless
the input row order is also fixed. The available artifacts do not explain why
the historical training score differs from both re-evaluations. Both the
historical log value and the newly measured values are retained with their
provenance; the discrepancy is not hidden or represented as a new training
result.

### Efficiency observations

The model has 4,008,829 parameters. The clean validation evaluator ran on CPU
and measured 54.33 seconds for 1,400 images (about 25.77 images per second,
including data loading and preprocessing). This is a local CPU measurement,
not the hackathon's NVIDIA T4 benchmark. The inference script processes images
in batches of 64 and enables CUDA autocast when CUDA is available. The final
3,000-image inference command completed successfully on the local environment;
a T4 runtime was not measured.

The real validation run used Python 3.10.11, PyTorch 2.14.1+cpu,
torchvision 0.29.2+cpu, pandas 2.3.3, scikit-learn 1.7.2, and Pillow 12.2.0.
CUDA was unavailable. These are the versions used for the measurements in
this workspace and differ from the environment noted in the original README.
The repository's pinned `requirements.txt` remains the declared installation
target. Inference time and small numeric differences may vary with hardware
and library versions.

---

## 4. Test predictions, limitations, and reproduction

### Test-set prediction verification

The inference script was run on all 3,000 organizer test images and wrote a
new `submission_predictions.csv`, leaving the tracked `predictions.csv`
unchanged. The submission checker verified the required ordered columns
`image_id,label,confidence`, all 3,000 image IDs, binary labels, and finite
confidence values between 0 and 1. The existing `predictions.csv` also passes
the checker against the organizer test directory. Its IDs and labels match
the newly generated output; the largest absolute confidence difference was
approximately 0.000003, consistent with minor numerical differences across
inference runs. Since test labels are private, these checks verify format and
coverage only, not prediction correctness.

This distinction is important for interpreting the test result. The checker
can establish that every test filename has exactly one corresponding row,
that the header and column order are correct, and that labels and confidence
values are well-formed. It cannot establish whether any private test label is
correct. Matching the new predictions to the existing submission supports
reproducibility of the artifact with this checkpoint, but does not provide an
independent performance estimate.

### Limitations

- The organizer's test labels and hidden test corruptions are unavailable, so
  public test-set accuracy and official robustness scores cannot be reported.
- Local corruption diagnostics cover only JPEG quality 50, Gaussian blur at
  radius 1, and half-resolution resizing; other distortions may behave
  differently.
- The prior training log's AUC does not match the measured AUC of the currently
  supplied checkpoint. The source of that discrepancy could not be established
  from the repository artifacts.
- No controlled ablation was recorded for architecture, augmentations,
  threshold, or optimizer settings.
- The local speed measurement is CPU-only and cannot establish the score on
  the organizers' T4 hardware.

### Reproduction

From the repository directory, after extracting the provided archive beside
the scripts:

```powershell
python -m pip install -r requirements.txt
python .\evaluate_validation.py --data-dir .\hackathon_dataset --weights .\model_weights.pt
python .\inference_script.py --test_dir .\hackathon_dataset\test\images --weights .\model_weights.pt --output .\submission_predictions.csv
python .\check_submission.py --test_dir .\hackathon_dataset\test\images --predictions .\submission_predictions.csv
```

The evaluator prints actual split metrics; the checker validates output
structure without modifying the predictions. For the Drive submission, include
the compatible PyTorch weights, the verified predictions file, source code,
requirements, and this report. Set Drive access to **Anyone with the link —
Viewer** and verify access in a private browser window before submitting.

To reproduce training in the recorded Linux/Colab-style setup, extract the
archive and run the existing training script with the dataset root as the
working directory:

```bash
cd hackathon_dataset
python ../train.py
```

The training script reads `./train/real/` and `./train/fake/` relative to the
working directory and writes `model_weights.pt` there. Running it from
`hackathon_dataset/` therefore keeps a newly trained checkpoint separate
from the submitted checkpoint in the repository directory. The evaluation and
inference commands above explicitly load the submitted checkpoint and write
predictions to a new file.
