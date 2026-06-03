from sources.controller.thread.thread_vehicle import ThreadVehicle


class DetectVehicle:
    def __init__(self):
        self.detect_vehicle = ThreadVehicle()

    def detect(self, image, class_filter=None):
        return self.detect_vehicle.detect_vehicle(image, class_filter=class_filter)
