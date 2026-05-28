from ultralytics import YOLO
from pathlib import Path
from loguru import logger

from ...config import ROOT
from sources.models import VehicleConfig


class ThreadVehicle:
    def __init__(self):
        super().__init__()

        self.logger = logger.bind(component="vehicle_detector")
        self.vehicle_config = VehicleConfig()
        self.__detect_vehicle = None
        self.names = {}

        self.setup_vehicle()

    def setup_vehicle(self):
        weight = Path(ROOT) / self.vehicle_config.WEIGHT
        if not weight.exists():
            raise FileNotFoundError(f"Vehicle model weight not found: {weight}")

        self.classes = self.vehicle_config.CLASSES
        self.conf = self.vehicle_config.CONF
        self.imgsz = self.vehicle_config.IMGSZ
        self.device = self.vehicle_config.DEVICE

        self.__detect_vehicle = YOLO(str(weight))
        self.names = self.__detect_vehicle.names
        self.logger.info("Vehicle detection model loaded")

    def setup_plate(self):
        self.setup_vehicle()

    def detect_vehicle(self, image, class_filter=None):
        try:
            results = self.__detect_vehicle.predict(
                image,
                conf=self.conf,
                imgsz=self.imgsz,
                classes=self.classes,
                device=self.device,
                verbose=False
            )

            detect_list = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self.names[cls]
                    detect_list.append([int(x1), int(y1), int(x2), int(y2), class_name, conf])

            if class_filter:
                name_filter = set()
                if isinstance(class_filter, (list, tuple, set)):
                    for c in class_filter:
                        if isinstance(c, str):
                            name_filter.add(c)
                        elif isinstance(c, (int, float)):
                            idx = int(c)
                            if idx in self.names:
                                name_filter.add(self.names[idx])
                if name_filter:
                    detect_list = [r for r in detect_list if len(r) > 4 and r[4] in name_filter]

            return detect_list

        except Exception as exc:
            self.logger.exception(f"Vehicle detection failed: {exc}")
            return []
