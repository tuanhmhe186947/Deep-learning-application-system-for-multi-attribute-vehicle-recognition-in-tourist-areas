# Project Audit

## Fixed

- Added `.gitignore` for caches, logs, generated outputs, local images, and large model binaries.
- Fixed dependency typo `numy` to `numpy` and added missing runtime packages such as `loguru`, `ultralytics`, `torch`, `torchvision`, `Pillow`, `scipy`, `lap`, and `thop`.
- Added `README.md`, `requirements-dev.txt`, `pyproject.toml`, `.env.example`, model-weight notes, and GitHub CI.
- Made config loading safer with `yaml.safe_load`, clear missing-file errors, and optional env-var overrides.
- Made FastAPI controller lazy-load model objects so importing the app does not immediately initialize every model.
- Added request validation and safer base64 image decoding.
- Replaced app-level debug `print` calls with `loguru` logging.
- Fixed OCR angle math that attempted to divide Python lists by floats.
- Sanitized saved output filenames in socket-style responses.
- Moved OCR training notebook into `notebooks/training/`, cleaned its outputs, and added a reproducible CLI training/evaluation script.

## Still Recommended Before Public Release

- Add dataset cards that document data source, annotation format, split strategy, license, and privacy constraints.
- Add model cards with architecture, training data summary, metrics, known failure cases, and intended use.
- Add reproducible training/evaluation scripts if the repository is meant to be research-grade, not only inference-grade.
- Replace or clearly attribute vendored YOLO/tracker code if required by the original licenses.
- Decide how weights will be distributed: GitHub Releases, Git LFS, Hugging Face Hub, or private storage.
