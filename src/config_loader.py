from dataclasses import dataclass
from typing import Any, Dict
import yaml

def load_config(path: str) -> Dict[str, Any]:
    """Load config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
