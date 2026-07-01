"""Quick environment check for a patched MG-AFSS Ultralytics runtime."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

RELEASE_ROOT = Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

from mg_afss.paths import get_ultralytics_path


def check_path(label, path):
    if path.exists():
        print(f"[OK] {label}: {path}")
        return True
    print(f"[FAIL] {label}: {path}")
    return False


def check_import(module_name, label=None):
    label = label or module_name
    try:
        importlib.import_module(module_name)
        print(f"[OK] import {label}")
        return True
    except Exception as exc:
        print(f"[FAIL] import {label}: {exc}")
        return False


def main():
    ok = True

    runtime = get_ultralytics_path()
    ok &= check_path("patched Ultralytics runtime", runtime)
    if str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))

    for module, label in (
        ("torch", "PyTorch"),
        ("pandas", None),
        ("numpy", None),
        ("yaml", "PyYAML"),
        ("cv2", "opencv-python"),
        ("PIL", "Pillow"),
        ("matplotlib", None),
        ("scipy", None),
    ):
        ok &= check_import(module, label)

    try:
        import torch

        print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
        has_sdpa = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        print(f"[INFO] torch scaled_dot_product_attention: {has_sdpa}")
        ok &= has_sdpa
        if torch.cuda.is_available():
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[INFO] CUDA version: {torch.version.cuda}")
    except Exception as exc:
        print(f"[FAIL] torch runtime details: {exc}")
        ok = False

    ok &= check_import("ultralytics", "patched Ultralytics")
    ok &= check_import("ultralytics.models.yolo.detect.afss_train", "AFSS detect trainer")
    ok &= check_import("ultralytics.models.yolo.obb.afss_train", "AFSS OBB trainer")

    if ok:
        print("\nEnvironment check passed.")
        return 0
    print("\nEnvironment check found issues.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
