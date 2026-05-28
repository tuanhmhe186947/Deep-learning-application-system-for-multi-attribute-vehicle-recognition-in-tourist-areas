from dataclasses import dataclass, asdict
from typing import Optional

from sources.models.config_loader import load_yaml_config


@dataclass
class MainConfig:
    """Config for main application"""

    # name of run file application
    NAME: Optional[str] = None

    # config api
    HOST: Optional[str] = None
    PORT: Optional[int] = None
    RELOAD: Optional[bool] = None
    WORKER: Optional[int] = None

    # path file log
    FILE_LOG: Optional[str] = None
    TIMEOUT_KEEP_ALIVE: Optional[int] = None
    
    # storage
    ROOT_STORAGE: Optional[str] = None

    def __post_init__(self):
        data = load_yaml_config("main.yaml", env_var="MAIN_CONFIG_PATH")
        for key, value in data.items():
            setattr(self, key, value)
    
    def __str__(self):
        return str(asdict(self))

if __name__ == "__main__":
    config = MainConfig()
    print(config)
