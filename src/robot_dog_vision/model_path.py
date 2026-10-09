import os
from pathlib import Path


def load_model_path(env_key: str) -> Path:
    value = os.getenv(env_key)
    if value is None:
        raise RuntimeError(
            f"{env_key} is not configured. "
            "Run scripts/download_checkpoints.py first."
        )

    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"{env_key} points to a missing file: {path}")

    return path
