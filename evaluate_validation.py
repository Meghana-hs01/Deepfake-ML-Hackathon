import argparse
import io
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SEED = 42
IMAGE_SIZE = 224
IMG_EXTS = {".jpg", ".jpeg", ".png"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


class ValidationDataset(Dataset):
    def __init__(self, frame, variant):
        self.frame = frame.reset_index(drop=True)
        self.variant = variant
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            NORMALIZE,
        ])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row.path) as source:
            image = source.convert("RGB")

        if self.variant == "jpeg_q50":
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=50)
            buffer.seek(0)
            with Image.open(buffer) as compressed:
                image = compressed.convert("RGB")
        elif self.variant == "gaussian_blur":
            image = image.filter(ImageFilter.GaussianBlur(radius=1.0))
        elif self.variant == "half_resolution":
            original_size = image.size
            small_size = (max(1, image.width // 2), max(1, image.height // 2))
            image = image.resize(small_size, Image.Resampling.BILINEAR)
            image = image.resize(original_size, Image.Resampling.BILINEAR)

        return self.transform(image), int(row.label)


def build_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, 1))
    return model


def collect_records(data_dir):
    metadata_path = data_dir / "train_metadata.csv"
    train_dir = data_dir / "train"
    records = []
    if metadata_path.is_file() and train_dir.is_dir():
        metadata = pd.read_csv(metadata_path)
        if not {"image_id", "label"}.issubset(metadata.columns):
            raise ValueError(f"{metadata_path} must contain image_id and label columns.")
        if metadata["image_id"].isna().any() or metadata["image_id"].duplicated().any():
            raise ValueError(f"{metadata_path} contains missing or duplicate image_id values.")
        labels = pd.to_numeric(metadata["label"], errors="raise")
        if not labels.isin((0, 1)).all():
            raise ValueError(f"{metadata_path} labels must be 0 or 1.")

        paths_by_id = {
            path.stem: path
            for path in train_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMG_EXTS
        }
        image_count = sum(
            1
            for path in train_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMG_EXTS
        )
        if len(paths_by_id) != image_count:
            raise ValueError(f"{train_dir} contains duplicate image filename stems.")
        missing_ids = set(metadata["image_id"].astype(str)) - paths_by_id.keys()
        if missing_ids:
            raise FileNotFoundError(
                f"{len(missing_ids)} image_id values in {metadata_path} have no matching image in {train_dir}."
            )
        records = [
            (str(paths_by_id[str(image_id)]), int(label))
            for image_id, label in zip(metadata["image_id"], labels)
        ]
    else:
        for label, folder in ((0, "real"), (1, "fake")):
            image_dir = data_dir / folder
            if not image_dir.is_dir():
                raise FileNotFoundError(
                    f"Expected {metadata_path} with {train_dir}, or class directory: {image_dir}"
                )
            records.extend(
                (str(path), label)
                for path in image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMG_EXTS
            )
    if not records:
        raise ValueError(f"No supported images found under {data_dir}")
    frame = pd.DataFrame(records, columns=["path", "label"])
    if frame["label"].nunique() != 2 or frame["label"].value_counts().min() < 2:
        raise ValueError("Both classes need at least two images for a stratified split.")
    return frame


def evaluate(model, frame, batch_size, num_workers):
    results = {}
    for variant in ("clean", "jpeg_q50", "gaussian_blur", "half_resolution"):
        dataset = ValidationDataset(frame, variant)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=DEVICE.type == "cuda",
        )
        probabilities = []
        labels = []
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            for images, batch_labels in loader:
                with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                    logits = model(images.to(DEVICE)).float().view(-1)
                probabilities.extend(torch.sigmoid(logits).cpu().tolist())
                labels.extend(batch_labels.tolist())
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        predictions = [int(probability >= 0.5) for probability in probabilities]
        results[variant] = {
            "roc_auc": float(roc_auc_score(labels, probabilities)),
            "accuracy_at_0.5": float(accuracy_score(labels, predictions)),
            "f1_at_0.5": float(f1_score(labels, predictions, zero_division=0)),
        }
        if variant == "clean":
            elapsed = time.perf_counter() - start
            results[variant]["inference_seconds"] = elapsed
            results[variant]["images_per_second"] = len(labels) / elapsed
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Measure clean and corruption validation metrics on the fixed 80/20 split."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("train"),
                        help="Folder containing real/ and fake/ class directories.")
    parser.add_argument("--weights", type=Path, default=Path("model_weights.pt"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights file not found: {args.weights}")

    frame = collect_records(args.data_dir)
    _, validation_frame = train_test_split(
        frame,
        test_size=0.2,
        stratify=frame["label"],
        random_state=SEED,
    )

    model = build_model()
    model.load_state_dict(torch.load(args.weights, map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()

    results = evaluate(model, validation_frame, args.batch_size, args.num_workers)
    report = {
        "device": str(DEVICE),
        "validation_images": len(validation_frame),
        "split": "train_test_split(test_size=0.2, stratify=label, random_state=42)",
        "threshold": 0.5,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "metrics": results,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
