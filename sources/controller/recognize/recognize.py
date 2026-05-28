from sources.controller.thread.thread_digit import ThreadDigit
from sources.models import DigitResponse


class Recognize:
    def __init__(self):
        super().__init__()
        self.detect_digit = ThreadDigit()

    def recognize_plate(self, image, camera_id=0):
        plate_text, type_plate, _, _ = self.detect_digit.reg_digit(image)
        if len(plate_text):
            return DigitResponse(
                cameraId=camera_id,
                plateText=plate_text,
                typePlate=type_plate,
                status=200,
            )

        return DigitResponse(cameraId=camera_id, plateText="", typePlate=-1, status=200)
