from sources.controller.thread.thread_digit import ThreadDigit
from sources.controller.thread.thread_plate import ThreadPlate
from sources.models import PlateBs64Response
from sources.util.common import img_2_base64
from sources.util.function import max_size_boundingbox


class DetectAndRecognizeBs64:
    def __init__(self):
        super().__init__()

        self.detect_plate = ThreadPlate()
        self.detect_digit = ThreadDigit()

    def convert_float_bbox(self, bbox, w, h) -> list:
        x1 = bbox[0] / w
        y1 = bbox[1] / h
        x2 = bbox[2] / w
        y2 = bbox[3] / h
        return [x1, y1, x2, y2]

    def object_detect_and_recognize(self, image, camera_id=0):
        detections = self.detect_plate.detect_plate(image)
        best = max_size_boundingbox(detections)
        if not len(best):
            return PlateBs64Response(
                cameraId=camera_id,
                plateBox=[],
                plateText="",
                typePlate=-1,
                imgBs64="",
                status=200,
            )

        crop = image[best[1]:best[3], best[0]:best[2]]
        h, w = image.shape[:2]
        expand_crop = image[
            max(0, best[1] - 7):min(h, best[3] + 7),
            max(0, best[0] - 7):min(w, best[2] + 7),
        ]
        bbox = best[0:4]
        digit, type_plate, _, _ = self.detect_digit.reg_digit(crop)
        if len(digit):
            return PlateBs64Response(
                cameraId=camera_id,
                plateBox=bbox,
                plateText=digit,
                typePlate=type_plate,
                imgBs64=img_2_base64(expand_crop),
                status=200,
            )

        return PlateBs64Response(
            cameraId=camera_id,
            plateBox=[],
            plateText="",
            typePlate=-1,
            imgBs64="",
            status=200,
        )
