__all__ = [
    "DetectAndRecognize",
    "DetectAndRecognizeBs64",
    "DetectAndRecognizePlateBs64",
    "DetectVehicle",
]


def __getattr__(name):
    if name == "DetectAndRecognize":
        from .object_detect_and_recognize import DetectAndRecognize

        return DetectAndRecognize
    if name == "DetectAndRecognizeBs64":
        from .object_detect_and_recognize_bs64 import DetectAndRecognizeBs64

        return DetectAndRecognizeBs64
    if name == "DetectAndRecognizePlateBs64":
        from .object_detect_plate_bs64 import DetectAndRecognizePlateBs64

        return DetectAndRecognizePlateBs64
    if name == "DetectVehicle":
        from .object_detect_vehicle import DetectVehicle

        return DetectVehicle

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
