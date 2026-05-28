from pathlib import Path

from loguru import logger

from ...config import ROOT
from ...yolov5.detect import Detection
from sources.models import PlateConfig


class ThreadPlate:
    def __init__(self):
        super().__init__()

        self.logger = logger.bind(component="plate_detector")
        self.__detect_plate = Detection(dnn=False)
        self.plate_config = PlateConfig()
        self.setup_plate()

    def setup_plate(self):
        weight = Path(ROOT) / self.plate_config.WEIGHT
        if not weight.exists():
            raise FileNotFoundError(f"Plate model weight not found: {weight}")

        self.__detect_plate.setup_model(
            str(weight),
            self.plate_config.CLASSES,
            self.plate_config.CONF,
            self.plate_config.IMGSZ,
            self.plate_config.DEVICE,
        )
        self.logger.info("Plate detection model loaded")

    def detect_plate(self, image):
        try:
            return self.__detect_plate.detect(image)
        except Exception as exc:
            self.logger.exception(f"Plate detection failed: {exc}")
            return []
