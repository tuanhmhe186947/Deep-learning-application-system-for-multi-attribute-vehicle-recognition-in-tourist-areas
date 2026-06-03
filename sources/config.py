from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESOURCES_DIR = ROOT / "resources"
CONFIG_DIR = RESOURCES_DIR / "config"
LOG_DIR = RESOURCES_DIR / "logs"
WEIGHT_DIR = RESOURCES_DIR / "weight"


def project_path(*parts) -> Path:
    return ROOT.joinpath(*parts)
