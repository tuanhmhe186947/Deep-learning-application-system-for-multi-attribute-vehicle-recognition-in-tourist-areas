from pathlib import Path

from loguru import logger
from ultralytics import YOLO

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

    def _class_name(self, class_id):
        if isinstance(self.names, dict):
            return self.names.get(class_id, str(class_id))
        if isinstance(self.names, (list, tuple)) and 0 <= class_id < len(self.names):
            return self.names[class_id]
        return str(class_id)

    @staticmethod
    def _filter_sets(class_filter):
        filter_ids = set()
        filter_names = set()

        if class_filter is None:
            return filter_ids, filter_names

        values = class_filter if isinstance(class_filter, (list, tuple, set)) else [class_filter]
        for value in values:
            if isinstance(value, str):
                filter_names.add(value)
            elif isinstance(value, (int, float)):
                filter_ids.add(int(value))

        return filter_ids, filter_names

    def detect_vehicle(self, image, class_filter=None):
        try:
            filter_ids, filter_names = self._filter_sets(class_filter)
            results = self.__detect_vehicle.predict(
                image,
                conf=self.conf,
                imgsz=self.imgsz,
                classes=self.classes,
                device=self.device,
                verbose=False,
            )

            detect_list = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self._class_name(cls)
                    if filter_ids and cls not in filter_ids:
                        continue
                    if filter_names and class_name not in filter_names:
                        continue
                    detect_list.append([int(x1), int(y1), int(x2), int(y2), class_name, conf, cls])

            return detect_list

        except Exception as exc:
            self.logger.exception(f"Vehicle detection failed: {exc}")
            return []
