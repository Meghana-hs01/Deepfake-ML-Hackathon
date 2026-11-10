import argparse
import csv
import math
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png"}
EXPECTED_COLUMNS = ["image_id", "label", "confidence"]


def main():
    parser = argparse.ArgumentParser(
        description="Verify a prediction CSV against the official test image directory."
    )
    parser.add_argument("--test_dir", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    args = parser.parse_args()

    if not args.test_dir.is_dir():
        parser.error(f"Test image directory does not exist: {args.test_dir}")
    if not args.predictions.is_file():
        parser.error(f"Prediction CSV does not exist: {args.predictions}")

    images = sorted(
        path for path in args.test_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMG_EXTS
    )
    if not images:
        parser.error(f"No supported test images found in: {args.test_dir}")
    expected_ids = [path.stem for path in images]
    if len(set(expected_ids)) != len(expected_ids):
        parser.error("Test image filenames have duplicate stems; image_id values would not be unique.")

    with args.predictions.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != EXPECTED_COLUMNS:
            parser.error(
                f"Expected CSV columns {EXPECTED_COLUMNS}, found {reader.fieldnames}."
            )
        rows = list(reader)

    if len(rows) != len(expected_ids):
        parser.error(
            f"Expected {len(expected_ids)} prediction rows for test images, found {len(rows)}."
        )

    for row_number, (row, expected_id) in enumerate(zip(rows, expected_ids), start=2):
        if None in row or any(row.get(column) is None for column in EXPECTED_COLUMNS):
            parser.error(f"Malformed or incomplete CSV row {row_number}.")
        if row["image_id"] != expected_id:
            parser.error(
                f"Row {row_number} has image_id {row['image_id']!r}; expected {expected_id!r}."
            )
        if row["label"] not in {"0", "1"}:
            parser.error(f"Row {row_number} has invalid label {row['label']!r}; expected 0 or 1.")
        try:
            confidence = float(row["confidence"])
        except ValueError:
            parser.error(f"Row {row_number} has a non-numeric confidence.")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            parser.error(f"Row {row_number} confidence must be finite and between 0 and 1.")

    print(f"Submission CSV is valid: {len(rows)} predictions match the test image directory.")


if __name__ == "__main__":
    main()
