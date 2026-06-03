import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from sources.config import CONFIG_DIR, ROOT


def _parse_env_value(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _apply_env_overrides(data: Dict[str, Any], prefix: Optional[str]) -> Dict[str, Any]:
    if not prefix:
        return data

    override_prefix = prefix.upper()
    for key in list(data.keys()):
        env_name = f"{override_prefix}_{key.upper()}"
        if env_name in os.environ:
            data[key] = _parse_env_value(os.environ[env_name])

    return data


def load_yaml_config(
    file_name: str,
    env_var: Optional[str] = None,
    env_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    override = os.getenv(env_var) if env_var else None
    config_path = Path(override) if override else CONFIG_DIR / file_name

    if not config_path.is_absolute():
        config_path = ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    return _apply_env_overrides(data, env_prefix)
