# Model Weights

Large model binaries are intentionally ignored by Git.

Place the required files in this directory, or update the YAML files in `resources/config/`:

- `last_17h16_24_09_yolov11_detect.pt` for vehicle detection
- `last_plate_hama_sz640.pt` for license plate detection
- `last_digit_s_10012025_sz256.pt` for license plate OCR

For public GitHub releases, publish weights through GitHub Releases, Git LFS, or external model storage, then document the download links in the main README.
