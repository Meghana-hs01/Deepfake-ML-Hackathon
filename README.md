# Deepfake Image Classification

Binary image classifier that distinguishes real faces (`0`) from deepfake faces (`1`) using a fine-tuned EfficientNet-B0 model in PyTorch.

## Project structure

- `train.py`: training pipeline using an 80/20 stratified split, data augmentation, and ROC AUC checkpointing.
- `inference_script.py`: batch inference script that writes `image_id`, `label`, and `confidence` to CSV.
- `model_weights.pt`: saved PyTorch state dictionary for the trained EfficientNet-B0 model.
- `predictions.csv`: inference output for the 3,000 test images.
- `training_log.txt`: epoch-by-epoch training loss and validation ROC AUC from training.
- `report.pdf`: technical report covering the dataset, architecture, validation metrics, and usage.
- `requirements.txt`: Python package dependencies.

## Model & training

- **Backbone:** EfficientNet-B0 (`IMAGENET1K_V1` pretrained weights) with `Dropout(0.3)` and a single-logit linear head (4,008,829 parameters).
- **Loss & optimizer:** `BCEWithLogitsLoss` with `AdamW` (`lr=3e-4`, `weight_decay=1e-4`), batch size `32`, `10` epochs, and mixed precision when CUDA is available.
- **Preprocessing & augmentation:** RGB conversion, resize to `224x224`, ImageNet normalization, random horizontal flips, brightness/contrast jitter (`0.1`), and Gaussian blur (`p=0.2`, radius `0.5-1.5`) during training.

## Validation results

Evaluated on a stratified 20% validation split (`1,400` images: `696` real, `704` fake, `random_state=42`) using a `0.5` classification threshold:

| Validation input | ROC AUC | Accuracy | F1 |
|---|---:|---:|---:|
| Clean | 0.9861 | 0.9636 | 0.9634 |
| JPEG quality 50 | 0.9822 | 0.9571 | 0.9573 |
| Gaussian blur (radius 1) | 0.9840 | 0.9557 | 0.9549 |
| Half-resolution resize | 0.9803 | 0.9457 | 0.9442 |

Note: `training_log.txt` records a best validation AUC of `0.8925` at epoch 8 during directory-ordered training, while evaluating `model_weights.pt` on the metadata-ordered split gives `0.9861` (`0.9823` under class-directory ordering) due to `train_test_split` row-order dependence.

## Setup & usage

Install dependencies:

```bash
pip install -r requirements.txt
```

### Inference

Run `inference_script.py` on a directory of test images:

```bash
python inference_script.py --test_dir path/to/test/images --weights model_weights.pt --output predictions.csv
```

Output format:

```csv
image_id,label,confidence
0020cba7-1d58-4784-91f9-6fc47d9f2806,1,0.995018720626831
003fae32-30d9-441b-8fcf-073fc04bf1d1,0,0.000005597865310846828
```

### Training

`train.py` expects `train/real/` and `train/fake/` inside the working directory and saves `model_weights.pt` to the current directory:

```bash
cd path/to/dataset
python ../train.py
```
