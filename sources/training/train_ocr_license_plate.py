from __future__ import annotations

import argparse
import csv
import re
import shutil
import string
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

try:
    import jiwer
except ImportError:  # pragma: no cover - fallback keeps the script usable.
    jiwer = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "ocr"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "ocr_evaluation.csv"
CLASS_NAMES = [str(i) for i in range(10)] + list(string.ascii_uppercase)
VALID_PROVINCES = [str(i) for i in range(11, 100)] + ["80"]

CHAR_TO_NUM = {
    "Z": "2",
    "B": "8",
    "D": "0",
    "O": "0",
    "Q": "0",
    "S": "5",
    "G": "6",
    "I": "1",
    "L": "1",
    "A": "4",
    "J": "1",
    "T": "1",
}
NUM_TO_CHAR = {
    "2": "Z",
    "8": "B",
    "0": "D",
    "5": "S",
    "6": "G",
    "1": "I",
    "4": "A",
    "7": "T",
}


@dataclass
class OcrPrediction:
    text: str
    note: str
    box_count: int


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def extract_dataset(zip_path: Path, output_dir: Path) -> Path:
    zip_path = _resolve_path(zip_path)
    output_dir = _resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset archive not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_dir)

    return output_dir


def write_data_yaml(
    dataset_dir: Path,
    yaml_path: Path,
    train_images: str = "train/images",
    val_images: str = "val/images",
    class_names: Sequence[str] = CLASS_NAMES,
) -> Path:
    dataset_dir = _resolve_path(dataset_dir)
    yaml_path = _resolve_path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    config = {
        "path": str(dataset_dir),
        "train": train_images,
        "val": val_images,
        "nc": len(class_names),
        "names": list(class_names),
    }
    with yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=False)

    return yaml_path


def standardize_plate_chars(chars: Sequence[str]) -> str:
    normalized: List[str] = []
    for index, char in enumerate(chars):
        if index in (0, 1):
            char = CHAR_TO_NUM.get(char, char)
        elif index == 2:
            char = NUM_TO_CHAR.get(char, char)
        elif index > 2:
            char = CHAR_TO_NUM.get(char, char)
        normalized.append(char)
    return "".join(normalized)


def _group_rows(items: List[Dict[str, float]]) -> List[List[Dict[str, float]]]:
    if not items:
        return []

    items = sorted(items, key=lambda item: item["y"])
    rows = [[items[0]]]
    for item in items[1:]:
        previous = rows[-1][-1]
        if (item["y"] - previous["y"]) > previous["h"] * 0.5:
            rows.append([item])
        else:
            rows[-1].append(item)

    for row in rows:
        row.sort(key=lambda item: item["x"])
    return rows


def _flatten_ordered_chars(items: List[Dict[str, float]]) -> List[str]:
    chars: List[str] = []
    for row in _group_rows(items):
        chars.extend(str(item["char"]) for item in row)
    return chars


def predict_plate_text(result, class_names: Sequence[str], threshold: float = 0.7) -> OcrPrediction:
    candidates: List[Dict[str, float]] = []
    for box in result.boxes:
        confidence = float(box.conf[0])
        if confidence < threshold:
            continue

        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        width = x2 - x1
        height = y2 - y1
        if height <= 0 or width > height * 1.5:
            continue

        class_id = int(box.cls[0])
        if class_id >= len(class_names):
            continue

        candidates.append(
            {
                "char": class_names[class_id],
                "x": x1,
                "y": y1,
                "w": width,
                "h": height,
                "conf": confidence,
            }
        )

    if not candidates:
        return OcrPrediction(text="", note="No text found", box_count=0)

    median_height = float(np.median([item["h"] for item in candidates]))
    valid_boxes = [item for item in candidates if item["h"] > median_height * 0.5]
    if not valid_boxes:
        return OcrPrediction(text="", note="All boxes filtered", box_count=0)

    text = standardize_plate_chars(_flatten_ordered_chars(valid_boxes))
    notes = []
    if text[:2] not in VALID_PROVINCES:
        notes.append("Check province code")
    if not re.match(r"^\d{2}[A-Z]\d{4,5}$", text):
        notes.append("Check plate format")

    return OcrPrediction(text=text, note=", ".join(notes), box_count=len(valid_boxes))


def read_yolo_label_text(label_path: Path, class_names: Sequence[str]) -> str:
    if not label_path.exists():
        return ""

    boxes: List[Dict[str, float]] = []
    with label_path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            if class_id >= len(class_names):
                continue
            boxes.append(
                {
                    "char": class_names[class_id],
                    "x": float(parts[1]),
                    "y": float(parts[2]),
                    "h": float(parts[4]),
                }
            )

    return standardize_plate_chars(_flatten_ordered_chars(boxes))


def _levenshtein_distance(source: str, target: str) -> int:
    previous = list(range(len(target) + 1))
    for i, source_char in enumerate(source, start=1):
        current = [i]
        for j, target_char in enumerate(target, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + int(source_char != target_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    if not reference and not hypothesis:
        return 0.0
    if not reference or not hypothesis:
        return 1.0
    if jiwer is not None:
        return float(jiwer.cer(reference, hypothesis))
    return _levenshtein_distance(reference, hypothesis) / max(len(reference), 1)


def iter_image_files(image_dir: Path) -> Iterable[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in suffixes)


def model_class_names(model: YOLO) -> List[str]:
    names = model.names
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    return [str(name) for name in names]


def train_model(args: argparse.Namespace):
    data_yaml = _resolve_path(args.data_yaml)
    project = _resolve_path(args.project)
    model = YOLO(args.base_model)

    train_kwargs = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(project),
        "name": args.name,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    return model.train(**train_kwargs)


def evaluate_model(args: argparse.Namespace) -> Tuple[float, int]:
    model_path = _resolve_path(args.model)
    val_images = _resolve_path(args.val_images)
    val_labels = _resolve_path(args.val_labels)
    output_csv = _resolve_path(args.output_csv)

    model = YOLO(str(model_path))
    class_names = model_class_names(model)
    rows = []

    for image_path in iter_image_files(val_images):
        label_path = val_labels / f"{image_path.stem}.txt"
        ground_truth = read_yolo_label_text(label_path, class_names)
        result = model(str(image_path), verbose=False)[0]
        prediction = predict_plate_text(result, class_names, threshold=args.conf)
        cer = character_error_rate(ground_truth, prediction.text)
        rows.append(
            {
                "file_name": image_path.name,
                "image_path": str(image_path),
                "ground_truth": ground_truth,
                "prediction": prediction.text,
                "cer": f"{cer:.6f}",
                "status": "wrong" if cer > 0 else "correct",
                "note": prediction.note,
                "box_count": prediction.box_count,
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "file_name",
                "image_path",
                "ground_truth",
                "prediction",
                "cer",
                "status",
                "note",
                "box_count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    mean_cer = float(np.mean([float(row["cer"]) for row in rows])) if rows else 0.0
    print(f"Evaluated {len(rows)} images")
    print(f"Mean CER: {mean_cer:.6f}")
    print(f"Report: {output_csv}")
    return mean_cer, len(rows)


def prepare_dataset(args: argparse.Namespace) -> Path:
    dataset_dir = _resolve_path(args.dataset_dir)
    if args.dataset_zip is not None:
        dataset_dir = extract_dataset(args.dataset_zip, dataset_dir)

    yaml_path = write_data_yaml(
        dataset_dir=dataset_dir,
        yaml_path=args.output_yaml,
        train_images=args.train_images,
        val_images=args.val_images,
    )
    print(f"Wrote data config: {yaml_path}")
    return yaml_path


def copy_best_weight(args: argparse.Namespace) -> Path:
    source = _resolve_path(args.source)
    destination = _resolve_path(args.destination)
    if not source.exists():
        raise FileNotFoundError(f"Best weight not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"Copied weight: {source} -> {destination}")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate license plate OCR models")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create a YOLO data YAML for OCR training")
    prepare.add_argument("--dataset-dir", type=Path, required=True, help="Dataset root directory")
    prepare.add_argument("--dataset-zip", type=Path, help="Optional dataset zip to extract first")
    prepare.add_argument("--output-yaml", type=Path, required=True, help="Output YOLO YAML path")
    prepare.add_argument("--train-images", default="train/images", help="Train images path under dataset root")
    prepare.add_argument("--val-images", default="val/images", help="Validation images path under dataset root")
    prepare.set_defaults(func=prepare_dataset)

    train = subparsers.add_parser("train", help="Train the YOLO OCR detector")
    train.add_argument("--data-yaml", type=Path, required=True, help="YOLO data YAML")
    train.add_argument("--base-model", default="yolov8n.pt", help="Base YOLO model")
    train.add_argument("--project", type=Path, default=DEFAULT_OUTPUT_DIR, help="Training output directory")
    train.add_argument("--name", default="yolov8_ocr", help="Run name")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=16)
    train.add_argument("--device", default=None, help="Ultralytics device value, e.g. 0 or cpu")
    train.set_defaults(func=train_model)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate OCR text quality on a validation split")
    evaluate.add_argument("--model", type=Path, required=True, help="Trained model weight")
    evaluate.add_argument("--val-images", type=Path, required=True, help="Validation images directory")
    evaluate.add_argument("--val-labels", type=Path, required=True, help="Validation labels directory")
    evaluate.add_argument("--output-csv", type=Path, default=DEFAULT_REPORT_PATH, help="CSV report path")
    evaluate.add_argument("--conf", type=float, default=0.7, help="Prediction confidence threshold")
    evaluate.set_defaults(func=evaluate_model)

    promote = subparsers.add_parser("promote", help="Copy a selected best.pt into resources/weight")
    promote.add_argument("--source", type=Path, required=True, help="Source best.pt")
    promote.add_argument("--destination", type=Path, required=True, help="Destination weight path")
    promote.set_defaults(func=copy_best_weight)

    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
