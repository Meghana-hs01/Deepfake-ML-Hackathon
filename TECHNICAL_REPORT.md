# Deepfake Image Classification: Technical Report

This report summarizes the EfficientNet-B0 deepfake classifier, dataset preparation, validation metrics, robustness checks, and inference workflow.

## 1. Problem & Dataset

### Objective

The task is to classify each face image as real (`label = 0`) or fake (`label = 1`). The PyTorch EfficientNet-B0 model outputs a confidence score representing the probability that an image is fake. Model performance is primarily evaluated using ROC AUC, alongside accuracy and F1 score at a `0.5` threshold.

### Dataset Breakdown

The dataset contains 7,000 labeled training images (3,478 real and 3,522 fake) and 3,000 unlabeled test images. All 7,000 IDs in `train_metadata.csv` match the training image filenames (`train/real/` for label `0`, `train/fake/` for label `1`), and all 3,000 test image filenames match the IDs in `predictions.csv`.

```text
dataset/
  train/
    real/            # 3,478 images (label = 0)
    fake/            # 3,522 images (label = 1)
  test/
    images/          # 3,000 images
  train_metadata.csv
```

Only the provided training images were used for training and validation, starting from ImageNet-pretrained EfficientNet-B0 weights. Test images are used solely for generating `predictions.csv`.

### Validation Protocol

Validation is performed on a stratified 20% split (`1,400` images: `696` real and `704` fake):

```python
train_test_split(
    metadata,
    test_size=0.2,
    stratify=metadata["label"],
    random_state=42,
)
```

Because `train_test_split` depends on the row order of the input dataframe, `evaluate_validation.py` uses the ordering in `train_metadata.csv` when present so the validation subset stays consistent across runs.

### Label & Confidence Conventions

Real images use label `0` and fake images use label `1`. The `confidence` column stores the model's sigmoid probability for the fake class, and the predicted `label` is `1` when `confidence >= 0.5`. Adjusting the threshold affects accuracy and F1, while ROC AUC depends only on the continuous confidence scores.

---

## 2. Model & Training Pipeline

### Architecture

The classifier uses `torchvision`'s EfficientNet-B0 with `IMAGENET1K_V1` pretrained weights and a custom binary classification head:

| Component | Configuration |
|---|---|
| Feature backbone | EfficientNet-B0 |
| Regularization | Dropout, probability 0.3 |
| Output layer | Linear, 1,280 input features to 1 logit |
| Output interpretation | Sigmoid probability for fake class |
| Total parameters | 4,008,829 |

Training minimizes `BCEWithLogitsLoss`, which combines sigmoid activation and binary cross-entropy in a numerically stable formulation. At inference, `torch.sigmoid` maps the output logit to `[0, 1]`.

### Image Processing & Augmentation

Images are converted to RGB, resized to `224x224`, converted to tensors, and normalized using standard ImageNet channel means and standard deviations. Training augmentation applies random horizontal flips, brightness/contrast jitter (`0.1`), and Gaussian blur (`p = 0.2`, radius sampled from `0.5` to `1.5`). Validation and test preprocessing are deterministic.

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

In `training_log.txt`, training loss decreases from `0.5270` at epoch 1 to `0.0811` at epoch 10, with the best checkpoint saved at epoch 8 (`0.8925` validation AUC). Training seeds Python, NumPy, and PyTorch with `42` and enables CUDA automatic mixed precision when a GPU is available.

---

## 3. Validation & Robustness Diagnostics

The saved checkpoint (`model_weights.pt`) was evaluated using `evaluate_validation.py` on the 1,400-image stratified split from `train_metadata.csv` under clean conditions and three deterministic image degradations (JPEG re-encoding at quality 50, Gaussian blur with radius 1.0, and half-resolution bilinear downsampling followed by upscaling back before the `224x224` transform):

| Validation input | ROC AUC | Accuracy | F1 |
|---|---:|---:|---:|
| Clean | 0.9861 | 0.9636 | 0.9634 |
| JPEG quality 50 | 0.9822 | 0.9571 | 0.9573 |
| Gaussian blur, radius 1 | 0.9840 | 0.9557 | 0.9549 |
| Downsample by half, then resize | 0.9803 | 0.9457 | 0.9442 |

Relative to the clean validation split, ROC AUC decreases by `0.0039` for JPEG quality 50, `0.0021` for Gaussian blur, and `0.0058` for half-resolution resizing; accuracy decreases by `0.0064`, `0.0079`, and `0.0179`, respectively.

### Split Ordering Comparison

`training_log.txt` records a best validation AUC of `0.8925` during training, whereas evaluating `model_weights.pt` on the `train_metadata.csv`-ordered split gives `0.9861` and evaluating on a split built from class-directory iteration order gives `0.9823`, reflecting differences in how the 80/20 split samples images when input row ordering changes.

### Runtime Observations

On CPU-only hardware (Python 3.10.11, PyTorch 2.14.1+cpu), evaluating the 1,400 validation images took `54.33` seconds (about `25.77` images/sec including loading and preprocessing). Inference uses batch size `64` and enables CUDA autocast when a GPU is available.

---

## 4. Test Predictions, Limitations & Reproduction

### Test Inference & Verification

`inference_script.py` processes all 3,000 test images and writes `predictions.csv` with columns `image_id,label,confidence`, binary labels thresholded at `0.5`, and finite confidence probabilities in `[0, 1]`. Running `check_submission.py` confirms that every test image has one valid row and that all headers, IDs, labels, and confidence values are well-formed.

### Limitations

- Ground-truth test labels are unavailable locally, so test-set accuracy and ROC AUC cannot be computed directly.
- Robustness checks cover JPEG quality 50, Gaussian blur (radius 1), and half-resolution resizing; other distortions or unseen synthesis methods may behave differently.
- Validation AUC varies across split orderings (`0.8925` in `training_log.txt` vs. `0.9823` and `0.9861` in post-training split evaluations).
- Runtime measurements were collected on CPU and will differ on GPU hardware.

### Reproduction

Install dependencies and run evaluation, inference, and output verification from the project root:

```bash
pip install -r requirements.txt
python evaluate_validation.py --data-dir ./dataset --weights ./model_weights.pt
python inference_script.py --test_dir ./dataset/test/images --weights ./model_weights.pt --output ./predictions.csv
python check_submission.py --test_dir ./dataset/test/images --predictions ./predictions.csv
```

To retrain from the dataset root (`./train/real/` and `./train/fake/`), run:

```bash
cd dataset
python ../train.py
```
