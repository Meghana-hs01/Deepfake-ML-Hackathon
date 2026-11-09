import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, THRESHOLD = 224, 64, 2, 0.5
IMG_EXTS = {".jpg", ".jpeg", ".png"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ID_COL = "image_id"

torch.manual_seed(42)

class TestDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.paths = sorted(p for p in Path(image_dir).iterdir()
                            if p.is_file() and p.suffix.lower() in IMG_EXTS)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.paths[i].stem

def build_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, 1))
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    model = build_model()
    model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
    model.to(DEVICE).eval()

    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = TestDataset(args.test_dir, tf)
    print(f"Found {len(ds)} images")
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    ids, confs = [], []
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
        for imgs, batch_ids in loader:
            logits = model(imgs.to(DEVICE)).float().view(-1)
            confs.extend(torch.sigmoid(logits).cpu().tolist())
            ids.extend(batch_ids)

    confs = np.clip(np.array(confs), 0.0, 1.0)
    out = pd.DataFrame({ID_COL: ids,
                        "label": (confs >= THRESHOLD).astype(int),
                        "confidence": confs})
    out.to_csv(args.output, index=False)
    print(f"Saved {len(out)} rows to {args.output}")

if __name__ == "__main__":
    main()
