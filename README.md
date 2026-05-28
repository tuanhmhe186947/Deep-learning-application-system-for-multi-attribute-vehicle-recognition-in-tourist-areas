# Multi-Task Vehicle Recognition System

FastAPI service for vehicle detection, license plate detection, and license plate OCR. The project combines a YOLO-based vehicle detector, a YOLOv5 license plate detector, and a digit/character detector for Vietnamese license plate recognition.

## Features

- Detect cars: `POST /detect/car`
- Detect motorcycles: `POST /detect/moto`
- Detect license plates: `POST /detect/plate`
- Recognize cropped license plate text: `POST /ocr/plate`
- Health check: `GET /health`

## Repository Layout

```text
.
|-- main_api.py                 # FastAPI entrypoint
|-- test_api.py                 # Local API smoke-test client
|-- notebooks/
|   `-- training/               # Experiment notebooks
|-- resources/
|   |-- config/                 # YAML runtime configuration
|   `-- weight/                 # Local model weights, ignored by Git
`-- sources/
    |-- controller/             # API orchestration and inference wrappers
    |-- models/                 # Request/response/config models
    |-- training/               # Reproducible training/evaluation scripts
    |-- util/                   # Image/base64 and OCR utilities
    `-- yolov5/                 # Vendored YOLO/tracker code
```

## Setup

Use Python 3.10+ for the API service.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA builds, install the correct PyTorch wheel from the official PyTorch selector before installing the rest of the requirements.

## Model Weights

Model binaries are not tracked in Git. Put the required files in `resources/weight/`, or edit the paths in:

- `resources/config/vehicle.yaml`
- `resources/config/plate.yaml`
- `resources/config/digit.yaml`

Current expected filenames are documented in `resources/weight/README.md`.

## Run API

```bash
uvicorn main_api:app --host 127.0.0.1 --port 8484
```

Open API docs:

```text
http://127.0.0.1:8484/docs
```

## Request Format

All inference endpoints accept:

```json
{
  "image": "/9j/..."
}
```

The `image` value may be raw base64 or a `data:image/...;base64,...` URI.

## Smoke Test

```bash
python test_api.py path/to/image.jpg --host http://127.0.0.1:8484
```

Annotated outputs are written to `output/`, which is ignored by Git.

## OCR Training

The OCR training notebook is stored at `notebooks/training/train_ocr_license_plate.ipynb`.
The reproducible CLI version is `sources/training/train_ocr_license_plate.py`.

```bash
python -m sources.training.train_ocr_license_plate prepare \
  --dataset-dir datasets/data_digit_hino \
  --output-yaml resources/config/ocr_license_plate.yaml

python -m sources.training.train_ocr_license_plate train \
  --data-yaml resources/config/ocr_license_plate.yaml \
  --project runs/ocr \
  --name yolov8_ocr

python -m sources.training.train_ocr_license_plate evaluate \
  --model runs/ocr/yolov8_ocr/weights/best.pt \
  --val-images datasets/data_digit_hino/val/images \
  --val-labels datasets/data_digit_hino/val/labels
```

More details: `docs/TRAINING_OCR.md`.

## GitHub Notes

- Keep weights, logs, cache files, and generated images out of commits.
- Publish weights through GitHub Releases, Git LFS, or external model storage.
- Update config files when changing model names, class IDs, image sizes, or confidence thresholds.
- Run static checks before pushing:

```bash
pip install -r requirements-dev.txt
ruff check sources main_api.py test_api.py
python -m compileall sources main_api.py test_api.py
```
