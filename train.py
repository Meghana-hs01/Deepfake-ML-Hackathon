import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DATA_DIR = Path("train")
IMAGE_SIZE, BATCH_SIZE, EPOCHS, LR = 224, 32, 10, 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXTS = {".jpg", ".jpeg", ".png"}

records = []
for label, folder in [(0, "real"), (1, "fake")]:
    for p in (DATA_DIR / folder).iterdir():
        if p.suffix.lower() in IMG_EXTS:
            records.append((str(p), label))
df = pd.DataFrame(records, columns=["path", "label"])
print("Images:", len(df))
print(df["label"].value_counts())

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)

NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(), NORM,
])
val_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(), NORM,
])

class DeepfakeDataset(Dataset):
    def __init__(self, frame, transform, degrade=False):
        self.frame, self.transform, self.degrade = frame.reset_index(drop=True), transform, degrade

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        img = Image.open(self.frame.path[i]).convert("RGB")
        if self.degrade and random.random() < 0.2:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        return self.transform(img), float(self.frame.label[i])

def build_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, 1))
    return model

train_loader = DataLoader(DeepfakeDataset(train_df, train_tf, degrade=True),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(DeepfakeDataset(val_df, val_tf),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

model = build_model().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

best_auc = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    total = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
            loss = criterion(model(imgs), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * len(imgs)
    train_loss = total / len(train_df)

    model.eval()
    probs, truths = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            logits = model(imgs.to(DEVICE)).float().view(-1)
            probs.extend(torch.sigmoid(logits).cpu().tolist())
            truths.extend(labels.tolist())
    auc = roc_auc_score(truths, probs)
    print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} | val AUC {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), "model_weights.pt")
        print("  saved model_weights.pt")

print(f"Best validation AUC: {best_auc:.4f}")
