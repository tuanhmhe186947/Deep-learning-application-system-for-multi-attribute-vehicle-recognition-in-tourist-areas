import base64

import cv2
import numpy as np
from loguru import logger

from sources.config import LOG_DIR


def base64_2_img(data) -> np.ndarray:
    if not isinstance(data, str) or not data.strip():
        raise ValueError("Image payload must be a non-empty base64 string")

    payload = data.strip()
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]

    try:
        raw = base64.b64decode(payload.encode("ascii"), validate=True)
    except Exception as exc:
        raise ValueError("Image payload is not valid base64") from exc

    array_px = np.frombuffer(raw, np.uint8)
    image = cv2.imdecode(array_px, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image payload could not be decoded by OpenCV")

    return image


def img_2_base64(data) -> str:
    if data is None:
        raise ValueError("Image is required")

    ok, encoded = cv2.imencode(".jpg", data)
    if not ok:
        raise ValueError("Image could not be encoded as JPEG")

    byte_to_base64 = base64.b64encode(encoded.tobytes())
    return byte_to_base64.decode("ascii")


def set_logger(key=None):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    task = f"api_logger_{key}"
    api_logger = logger.bind(task=task)
    logger_file = LOG_DIR / f"api_logger_{key}_{{time:YYYY_MM_DD}}.log"
    api_logger.add(
        str(logger_file),
        rotation="00:00",
        retention="3 days",
        level="INFO",
        filter=lambda record: record["extra"].get("task") == task,
    )
    return api_logger
