# OCR Training Workflow

The original Colab notebook was moved to:

```text
notebooks/training/train_ocr_license_plate.ipynb
```

Use the notebook for exploration. Use the CLI script for reproducible training and evaluation:

```text
sources/training/train_ocr_license_plate.py
```

## Expected Dataset Layout

The script expects a YOLO-style character dataset:

```text
datasets/data_digit_hino/
|-- train/
|   |-- images/
|   `-- labels/
`-- val/
    |-- images/
    `-- labels/
```

Classes must follow this order:

```text
0 1 2 3 4 5 6 7 8 9 A B C ... Z
```

## Prepare YAML

```bash
python -m sources.training.train_ocr_license_plate prepare \
  --dataset-dir datasets/data_digit_hino \
  --output-yaml resources/config/ocr_license_plate.yaml
```

If the dataset is still zipped:

```bash
python -m sources.training.train_ocr_license_plate prepare \
  --dataset-zip datasets/data_digit_hino.zip \
  --dataset-dir datasets/data_digit_hino \
  --output-yaml resources/config/ocr_license_plate.yaml
```

## Train

```bash
python -m sources.training.train_ocr_license_plate train \
  --data-yaml resources/config/ocr_license_plate.yaml \
  --base-model yolov8n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --project runs/ocr \
  --name yolov8_ocr
```

## Evaluate OCR Text Quality

```bash
python -m sources.training.train_ocr_license_plate evaluate \
  --model runs/ocr/yolov8_ocr/weights/best.pt \
  --val-images datasets/data_digit_hino/val/images \
  --val-labels datasets/data_digit_hino/val/labels \
  --output-csv reports/ocr_evaluation.csv \
  --conf 0.7
```

The report contains per-image ground truth, prediction, CER, status, and notes.

## Promote A Weight

After choosing a trained `best.pt`, copy it into the local weight folder:

```bash
python -m sources.training.train_ocr_license_plate promote \
  --source runs/ocr/yolov8_ocr/weights/best.pt \
  --destination resources/weight/last_digit_custom.pt
```

Then update `resources/config/digit.yaml` to point to the promoted weight.
