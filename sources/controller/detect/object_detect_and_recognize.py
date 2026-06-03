from loguru import logger

from sources.controller.thread.thread_digit import ThreadDigit
from sources.controller.thread.thread_plate import ThreadPlate
from sources.models import MainConfig, SocketResponse
from sources.util.function import crop_bbox, max_size_boundingbox


class DetectAndRecognize:
    def __init__(self):
        self.detect_plate = ThreadPlate()
        self.detect_digit = ThreadDigit()
        self.__main_config = MainConfig()
        self.logger = logger.bind(component="detect_and_recognize")

    @staticmethod
    def convert_float_bbox(bbox, w, h) -> list:
        return [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]

    def object_detect_and_recognize(
        self,
        image,
        image_overview,
        img_crop_poly,
        camera_id=0,
        param=None,
    ):
        plate_text = ""
        plate_crop = None
        try:
            detections = self.detect_plate.detect_plate(img_crop_poly)
            best = max_size_boundingbox(detections)
            if best:
                crop, _ = crop_bbox(img_crop_poly, best)
                plate_crop, _ = crop_bbox(img_crop_poly, best, padding=7)
                if crop is not None:
                    plate_text, _, _, _ = self.detect_digit.reg_digit(crop)
        except Exception as exc:
            self.logger.exception(f"Plate detect-and-recognize failed: {exc}")

        response = SocketResponse(camera_id=camera_id, plate_text=plate_text)
        response.get_and_save_relative_path(
            self.__main_config.ROOT_STORAGE,
            plate_text,
            plate_crop,
            image,
            image_overview,
            param,
        )
        return response._response()
