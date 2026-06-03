from numbers import Real
from pathlib import Path
from typing import Any, List, Optional

from fastapi import HTTPException
from loguru import logger

from sources.config import ROOT
from sources.models import AppDataRequest, DataRequest
from sources.models.digit_config import digit_config
from sources.models.plate_config import PlateConfig
from sources.models.vehicle_config import VehicleConfig
from sources.util.common import base64_2_img, img_2_base64
from sources.util.function import crop_bbox, max_size_boundingbox


class APIController:
    def __init__(self):
        self._detect_vehicle = None
        self._detect_plate = None
        self._detect_digit = None
        self.api_logger = logger.bind(task="api")

    @property
    def detect_vehicle(self):
        if self._detect_vehicle is None:
            from sources.controller.thread.thread_vehicle import ThreadVehicle

            self._detect_vehicle = ThreadVehicle()
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

    @staticmethod
    def _decode_optional_image(image: Optional[str]):
        if not image:
            return None
        try:
            return base64_2_img(image)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @staticmethod
    def _require_image(data: AppDataRequest):
        if not data.image:
            raise HTTPException(status_code=400, detail="Image is required")
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

            class_name = det[4] if len(det) > 4 and isinstance(det[4], str) else ""
            class_id = -1
            if len(det) > 6 and self._is_number(det[6]):
                class_id = int(det[6])

            item = {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": float(conf),
            }
            if class_name:
                item["className"] = class_name
            if class_id >= 0:
                item["classId"] = class_id
            if include_type:
                item["typePlate"] = type_plate
            boxes.append(item)

        return boxes

    def _recognize_best_plate(self, image, camera_id: int = 0):
        detections = self.detect_plate.detect_plate(image) or []
        best = max_size_boundingbox(detections)
        if not best:
            return {
                "cameraId": camera_id,
                "plateBox": [],
                "plateText": "",
                "typePlate": -1,
                "confidence": 0.0,
                "imgBs64": "",
                "status": 200,
            }

        crop, box = crop_bbox(image, best, padding=0)
        if crop is None:
            return {
                "cameraId": camera_id,
                "plateBox": [],
                "plateText": "",
                "typePlate": -1,
                "confidence": 0.0,
                "imgBs64": "",
                "status": 200,
            }

        expanded_crop, _ = crop_bbox(image, best, padding=7)
        plate_text, type_plate, _, confidence = self.detect_digit.reg_digit(crop)
        return {
            "cameraId": camera_id,
            "plateBox": box,
            "plateText": plate_text,
            "typePlate": int(type_plate),
            "confidence": float(confidence),
            "imgBs64": img_2_base64(expanded_crop if expanded_crop is not None else crop),
            "status": 200,
        }

    @staticmethod
    def _weight_status(config):
        weight = Path(ROOT) / config.WEIGHT
        return {
            "path": str(weight),
            "exists": weight.exists(),
        }

    def readiness(self):
        vehicle = VehicleConfig()
        plate = PlateConfig()
        return {
            "status": "ok",
            "weights": {
                "vehicle": self._weight_status(vehicle),
                "plate": self._weight_status(plate),
                "digit": self._weight_status(digit_config),
            },
        }

    async def detect_car(self, data: DataRequest):
        try:
            image = self._decode_request_image(data)
            detections = self.detect_vehicle.detect_vehicle(image, class_filter=[0]) or []
            return {"boxes": self._format_boxes(detections), "status": 200}
        except HTTPException:
            raise
        except Exception as exc:
            self.api_logger.exception(f"detect_car error: {exc}")
            raise HTTPException(status_code=500, detail="Vehicle detection failed") from exc

    async def detect_moto(self, data: DataRequest):
        try:
            image = self._decode_request_image(data)
            detections = self.detect_vehicle.detect_vehicle(image, class_filter=[3]) or []
            return {"boxes": self._format_boxes(detections), "status": 200}
        except HTTPException:
            raise
        except Exception as exc:
            self.api_logger.exception(f"detect_moto error: {exc}")
            raise HTTPException(status_code=500, detail="Vehicle detection failed") from exc

    async def detect_vehicle_api(self, data: DataRequest):
        try:
            image = self._decode_request_image(data)
            detections = self.detect_vehicle.detect_vehicle(image) or []
            return {"boxes": self._format_boxes(detections), "status": 200}
        except HTTPException:
            raise
        except Exception as exc:
            self.api_logger.exception(f"detect_vehicle_api error: {exc}")
            raise HTTPException(status_code=500, detail="Vehicle detection failed") from exc

    async def detect_plate_api(self, data: DataRequest):
        try:
            image = self._decode_request_image(data)
            detections = self.detect_plate.detect_plate(image) or []
            plates = self._format_boxes(detections, include_type=True)
            return {"plates": plates, "status": 200}
        except HTTPException:
            raise
        except Exception as exc:
            self.api_logger.exception(f"detect_plate_api error: {exc}")
            raise HTTPException(status_code=500, detail="Plate detection failed") from exc

    async def ocr_plate(self, data: DataRequest):
        try:
            image = self._decode_request_image(data)
            text, type_plate, _, conf = self.detect_digit.reg_digit(image)
            return {
                "text": text,
                "typePlate": int(type_plate),
                "confidence": float(conf),
                "status": 200,
            }
        except HTTPException:
            raise
        except Exception as exc:
            self.api_logger.exception(f"ocr_plate error: {exc}")
            raise HTTPException(status_code=500, detail="Plate OCR failed") from exc

    async def recognize_plate(self, data: AppDataRequest):
        try:
            image = self._require_image(data)
            return self._recognize_best_plate(image, camera_id=data.cameraId or 0)
        except HTTPException:
            raise
        except Exception as exc:
            self.api_logger.exception(f"recognize_plate error: {exc}")
            raise HTTPException(status_code=500, detail="Plate recognition failed") from exc

    async def recognize_plate_and_store(self, data: AppDataRequest):
        try:
            image = self._require_image(data)
            image_overview = self._decode_optional_image(data.imageOverview)
            result = self._recognize_best_plate(image, camera_id=data.cameraId or 0)

            if not result["plateBox"]:
                plate_crop = None
            else:
                plate_crop, _ = crop_bbox(image, result["plateBox"], padding=7)

            from sources.models import MainConfig, SocketResponse

            main_config = MainConfig()
            response = SocketResponse(
                camera_id=data.cameraId or 0,
                plate_text=result["plateText"],
            )
            response.get_and_save_relative_path(
                main_config.ROOT_STORAGE,
                result["plateText"],
                plate_crop,
                image,
                image_overview,
                param="api",
            )
            return response._response()
        except HTTPException:
            raise
        except Exception as exc:
            self.api_logger.exception(f"recognize_plate_and_store error: {exc}")
            raise HTTPException(status_code=500, detail="Plate recognition failed") from exc


api_controller = APIController()
