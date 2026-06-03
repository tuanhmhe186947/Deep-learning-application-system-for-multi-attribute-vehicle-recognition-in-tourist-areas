from pathlib import Path

from loguru import logger

from sources.config import ROOT
from sources.models.digit_config import digit_config
from sources.util.common import img_2_base64
from sources.util.function import (
    check_alpha_digit,
    filter_text_digit,
    get_angle,
    is_square_lp,
    process_square_lp,
    rotate,
)
from sources.yolov5.detect import Detection


class ThreadDigit:
    def __init__(self):
        self.logger = logger.bind(component="digit_detector")
        self.detect_digit = Detection(dnn=False)
        self.digit_config = digit_config

        self.plate_style = self.digit_config.PLATE_STYLE
        self.digit_car = self.digit_config.DIGIT_CAR
        self.alpha_car = self.digit_config.ALPHA_ARMY

        self.conf_plate_square = self.digit_config.CONF_PLATE_SQUARE

        self.setup_digit()

    def setup_digit(self):
        weight = Path(ROOT) / self.digit_config.WEIGHT
        if not weight.exists():
            raise FileNotFoundError(f"Digit model weight not found: {weight}")

        self.detect_digit.setup_model(
            str(weight),
            self.digit_config.CLASSES,
            self.digit_config.CONF,
            self.digit_config.IMGSZ,
            self.digit_config.DEVICE,
        )
        self.logger.info("Digit OCR model loaded")

    def digit(self, img_plate):
        if img_plate is None or img_plate.size == 0:
            return "", -1, img_plate, 0.0

        result_cur = self.detect_digit.detect(img=img_plate)
        angle = get_angle(img_plate, result_cur)
        rotated = rotate(img_plate, angle)
        result = self.detect_digit.detect(img=rotated)
        if not len(result):
            return "", -1, img_plate, 0.0

        if is_square_lp(result_cur):
            ordered, type_plate = process_square_lp(result)
        else:
            ordered = sorted(result, key=lambda x: x[0])
            type_plate = 1

        digits = ""
        confs = []
        for item in ordered:
            label = item[4]
            conf_val = item[-1] if len(item) else 0.0
            if not isinstance(conf_val, (int, float)):
                conf_val = 0.0
            confs.append(conf_val)
            digits += str(label)

        conf_score = float(min(confs)) if len(confs) else 0.0
        return digits.upper(), type_plate, img_plate, conf_score

    def reg_digit(self, img_plate):
        try:
            digits, type_plate, img, conf_score = self.digit(img_plate)
            if not digits:
                return "", type_plate, img_2_base64(img_plate), conf_score
            if filter_text_digit(digits, self.plate_style):
                if check_alpha_digit(digits, self.digit_car, self.alpha_car):
                    return digits, type_plate, img_2_base64(img), conf_score
            return f"{digits}_unk", type_plate, img_2_base64(img), conf_score
        except Exception as exc:
            self.logger.exception(f"Plate OCR failed: {exc}")
            return "_unk", -1, img_2_base64(img_plate), 0.0
