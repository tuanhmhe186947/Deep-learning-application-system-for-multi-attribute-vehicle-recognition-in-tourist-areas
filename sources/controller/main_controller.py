from numbers import Real
from typing import Any, List

from fastapi import HTTPException
from loguru import logger

from ..models import DataRequest
from ..util.common import base64_2_img


class APIController:
    def __init__(self):
        super().__init__()
        self._detect_vehicle = None
        self._detect_plate = None
        self._detect_digit = None
        self.api_logger = logger.bind(task="api")

    @property
    def detect_vehicle(self):
        if self._detect_vehicle is None:
            from sources.controller.detect.object_detect_vehicle import DetectVehicle

            self._detect_vehicle = DetectVehicle()
        return self._detect_vehicle

    @property
    def detect_plate(self):
        if self._detect_plate is None:
            from sources.controller.thread.thread_plate import ThreadPlate

            self._detect_plate = ThreadPlate()
        return self._detect_plate

    @property
    def detect_digit(self):
        if self._detect_digit is None:
            from sources.controller.thread.thread_digit import ThreadDigit

            self._detect_digit = ThreadDigit()
        return self._detect_digit

    @staticmethod
    def _is_number(value: Any) -> bool:
        return isinstance(value, Real) and not isinstance(value, bool)

    @staticmethod
    def _decode_request_image(data: DataRequest):
        try:
            return base64_2_img(data.image)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _format_boxes(self, detections: List, include_type: bool = False):
        boxes = []
        for det in detections:
            if len(det) < 5:
                continue

            x1, y1, x2, y2 = det[:4]
            conf_idx = 4
            if len(det) >= 6 and not self._is_number(det[4]):
                conf_idx = 5

            conf = det[conf_idx]
            type_plate = -1
            if include_type:
                if len(det) > conf_idx + 1 and self._is_number(det[conf_idx + 1]):
                    type_plate = int(det[conf_idx + 1])
                elif len(det) > 4 and isinstance(det[4], str):
                    type_plate = 0

            if not self._is_number(conf):
                try:
                    conf = float(conf)
                except Exception:
                    conf = 0.0

            item = {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": float(conf),
            }
            if include_type:
                item["typePlate"] = type_plate
            boxes.append(item)

        return boxes

    async def detect_car(self, data: DataRequest):
        try:
            image = self._decode_request_image(data)
            detections = self.detect_vehicle.detect(image, class_filter=[0]) or []
            return {"boxes": self._format_boxes(detections)}
        except HTTPException:
            raise
        except Exception as exc:
            self.api_logger.exception(f"detect_car error: {exc}")
            raise HTTPException(status_code=500, detail="Vehicle detection failed") from exc

    async def detect_moto(self, data: DataRequest):
        try:
            image = self._decode_request_image(data)
            detections = self.detect_vehicle.detect(image, class_filter=[3]) or []
            return {"boxes": self._format_boxes(detections)}
        except HTTPException:
            raise
        except Exception as exc:
            self.api_logger.exception(f"detect_moto error: {exc}")
            raise HTTPException(status_code=500, detail="Vehicle detection failed") from exc

    async def detect_plate_api(self, data: DataRequest):
        try:
            image = self._decode_request_image(data)
            detections = self.detect_plate.detect_plate(image) or []
            plates = self._format_boxes(detections, include_type=True)
            return {"plates": plates}
        except HTTPException:
            raise
        except Exception as exc:
            self.api_logger.exception(f"detect_plate_api error: {exc}")
            raise HTTPException(status_code=500, detail="Plate detection failed") from exc

    async def ocr_plate(self, data: DataRequest):
        try:
            image = self._decode_request_image(data)
            text, type_plate, _, conf = self.detect_digit.reg_digit(image)
            return {"text": text, "typePlate": int(type_plate), "confidence": float(conf)}
        except HTTPException:
            raise
        except Exception as exc:
            self.api_logger.exception(f"ocr_plate error: {exc}")
            raise HTTPException(status_code=500, detail="Plate OCR failed") from exc


api_controller = APIController()
