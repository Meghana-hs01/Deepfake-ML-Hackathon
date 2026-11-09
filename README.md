# Deepfake ML Hackathon: Team Submission

Binary classifier that labels face images as real (0) or deepfake (1), trained only on the provided dataset.

## Results (own validation split, 80/20 stratified)

| Metric | Value |
|---|---|
| AUC-ROC (clean val) | 0.8925 (best epoch 8 of 10) |
| Degraded val | Not measured |
| F1, Accuracy | Not computed |
| Threshold | 0.5 (fixed) |

## Environment

- Python 3.13 (Google Colab), PyTorch 2.11, torchvision 0.26.0
- Install: `pip install -r requirements.txt`

## Reproducing Predictions

```bash
python inference_script.py --test_dir ./test/images/ --weights ./model_weights.pt --output predictions.csv
```

Output: one row per test image, columns `image_id,label,confidence`.

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

## Data

Only the officially provided dataset was used for training. No external data, and no test images were used in training.

## Limitations

- Validation AUC plateaued around 0.88–0.89 from epoch 2 onward while training loss kept falling (0.53 to 0.08), which suggests overfitting.
- Robustness to JPEG compression, noise, and resizing was not measured.
- F1 and accuracy were not computed.
