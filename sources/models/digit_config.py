from dataclasses import dataclass, asdict
from typing import List, Optional

from sources.models.config_loader import load_yaml_config


@dataclass
class DigitConfig:
    # model recogbnize
    WEIGHT: Optional[str] = None
    CLASSES: Optional[List[int]] = None
    CONF: Optional[float] = None
    IMGSZ: Optional[int] = None
    DEVICE: Optional[str] = None
    AGNOSTIC_NMS: Optional[bool] = None
    HALF: Optional[bool] = None

    # Const conf plate square
    CONF_PLATE_SQUARE: Optional[float] = None
    
    # style text recognize
    PLATE_STYLE: Optional[List[str]] = None
    PLATE_STYLE_CAR: Optional[List[str]] = None
    PLATE_STYLE_MOTO: Optional[List[str]] = None
    DIGIT_CAR: Optional[List[str]] = None
    STYLE_ALPHA_CAR: Optional[List[str]] = None
    ALPHA_ARMY: Optional[List[str]] = None

    def __post_init__(self):
        data = load_yaml_config("digit.yaml", env_var="DIGIT_CONFIG_PATH", env_prefix="DIGIT")
        for key, value in data.items():
            setattr(self, key, value)
    
    def __str__(self):
        return str(asdict(self))


digit_config = DigitConfig()

if __name__ == "__main__":
    config = DigitConfig()
    print(config.WEIGHT)
