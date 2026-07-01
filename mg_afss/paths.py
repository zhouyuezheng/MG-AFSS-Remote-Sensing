"""Path helpers for the public MG-AFSS release candidate.

Set ``ULTRALYTICS_PATH`` to a local Ultralytics checkout after applying the
files under ``ultralytics_patch/``. The bundled patch directory is not a full
Ultralytics distribution by itself.
"""
from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_or_path(env_name: str, default: Path) -> Path:
    value = os.environ.get(env_name)
    return Path(value).expanduser().resolve() if value else default.resolve()


def get_ultralytics_path() -> Path:
    """Return the patched Ultralytics runtime path."""
    return _env_or_path("ULTRALYTICS_PATH", PROJECT_ROOT / "ultralytics_runtime")


def get_reference_ultralytics_path() -> Path:
    """Backward-compatible alias used by older scripts."""
    return get_ultralytics_path()


def get_data_root(name: str | None = None) -> Path:
    """Return the dataset root, or a named dataset under it."""
    root = PROJECT_ROOT / "dataset"
    return (root / name).resolve() if name else root.resolve()


def get_weights_root() -> Path:
    """Return the local weights root."""
    return (PROJECT_ROOT / "weights").resolve()


def get_experiments_root() -> Path:
    """Return the experiments output root."""
    return (PROJECT_ROOT / "experiments").resolve()


def require_existing(path: Path, label: str) -> Path:
    """Raise a clear error if a required path is missing."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path
