from dataclasses import dataclass, asdict
from typing import List, Optional

from sources.models.config_loader import load_yaml_config


@dataclass
class VehicleConfig:
    # model detect
    WEIGHT: Optional[str] = None
    CLASSES: Optional[List[int]] = None
    CONF: Optional[float] = None
    IMGSZ: Optional[int] = None
    DEVICE: Optional[str] = None
    AGNOSTIC_NMS: Optional[bool] = None
    HALF: Optional[bool] = None

    def __post_init__(self):
        data = load_yaml_config("vehicle.yaml", env_var="VEHICLE_CONFIG_PATH")
        for key, value in data.items():
            setattr(self, key, value)
    
    def __str__(self):
        return str(asdict(self))


if __name__ == "__main__":
    config = VehicleConfig()
    print(config)
