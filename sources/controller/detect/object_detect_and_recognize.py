from loguru import logger

from sources.controller.thread.thread_digit import ThreadDigit
from sources.controller.thread.thread_plate import ThreadPlate
from sources.models import MainConfig, SocketResponse
from sources.util.function import max_size_boundingbox


class DetectAndRecognize:
    def __init__(self):
        super().__init__()

        self.detect_plate = ThreadPlate()
        self.detect_digit = ThreadDigit()
        self.__main_config = MainConfig()
        self.logger = logger.bind(component="detect_and_recognize")

    def convert_float_bbox(self, bbox, w, h) -> list:
        x1 = bbox[0] / w
        y1 = bbox[1] / h
        x2 = bbox[2] / w
        y2 = bbox[3] / h
        return [x1, y1, x2, y2]

    def object_detect_and_recognize(
        self,
        image,
        image_overview,
        img_crop_poly,
        camera_id=0,
        param=None,
    ):
        try:
            detections = self.detect_plate.detect_plate(img_crop_poly)
            best = max_size_boundingbox(detections)
            if not len(best):
                response = SocketResponse(camera_id=camera_id, plate_text="")
                response.get_and_save_relative_path(
                    self.__main_config.ROOT_STORAGE,
                    "",
                    None,
                    image,
                    image_overview,
                    param,
                )
                return response._response()

            h, w = img_crop_poly.shape[:2]
            crop = img_crop_poly[best[1]:best[3], best[0]:best[2]]
            digit, _, _, _ = self.detect_digit.reg_digit(crop)
            expand_crop = img_crop_poly[
                max(0, best[1] - 7):min(h, best[3] + 7),
                max(0, best[0] - 7):min(w, best[2] + 7),
            ]

            response = SocketResponse(camera_id=camera_id, plate_text=digit)
            response.get_and_save_relative_path(
                self.__main_config.ROOT_STORAGE,
                digit,
                expand_crop,
                image,
                image_overview,
                param,
            )
            return response._response()

        except Exception as exc:
            self.logger.exception(f"Plate detect-and-recognize failed: {exc}")
            response = SocketResponse(camera_id=camera_id, plate_text="")
            response.get_and_save_relative_path(
                self.__main_config.ROOT_STORAGE,
                "",
                None,
                image,
                image_overview,
                param,
            )
            return response._response()
