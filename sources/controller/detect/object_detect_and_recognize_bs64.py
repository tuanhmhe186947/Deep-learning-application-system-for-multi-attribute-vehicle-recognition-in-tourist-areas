from sources.controller.thread.thread_digit import ThreadDigit
from sources.controller.thread.thread_plate import ThreadPlate
from sources.models import PlateBs64Response
from sources.util.common import img_2_base64
from sources.util.function import crop_bbox, max_size_boundingbox


class DetectAndRecognizeBs64:
    def __init__(self):
        self.detect_plate = ThreadPlate()
        self.detect_digit = ThreadDigit()

    @staticmethod
    def convert_float_bbox(bbox, w, h) -> list:
        return [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]

    def object_detect_and_recognize(self, image, camera_id=0):
        detections = self.detect_plate.detect_plate(image)
        best = max_size_boundingbox(detections)
        if not best:
            return PlateBs64Response(
                cameraId=camera_id,
                plateBox=[],
                plateText="",
                typePlate=-1,
                confidence=0.0,
                imgBs64="",
                status=200,
            )

        crop, box = crop_bbox(image, best)
        if crop is None:
            return PlateBs64Response(
                cameraId=camera_id,
                plateBox=[],
                plateText="",
                typePlate=-1,
                confidence=0.0,
                imgBs64="",
                status=200,
            )

        expand_crop, _ = crop_bbox(image, best, padding=7)
        digit, type_plate, _, confidence = self.detect_digit.reg_digit(crop)
        return PlateBs64Response(
            cameraId=camera_id,
            plateBox=box,
            plateText=digit,
            typePlate=type_plate,
            confidence=confidence,
            imgBs64=img_2_base64(expand_crop if expand_crop is not None else crop),
            status=200,
        )
