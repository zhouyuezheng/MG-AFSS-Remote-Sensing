"""Unified entry point for MG-AFSS experiments.

It keeps experiment identity, output paths, AFSS parameters, timing metadata,
and summary files consistent across standard training, fixed AFSS, and MG-AFSS
runs. Internal ``v3`` field names are retained only for compatibility with the
experiment records used by the paper.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from mg_afss.paths import (
    PROJECT_ROOT,
    get_reference_ultralytics_path,
    get_ultralytics_path,
    get_weights_root,
)


RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
METRIC_MAP50_CANDIDATES = ("metrics/mAP50(B)", "metrics/mAP50", "metrics/mAP50-50(B)")
METRIC_MAP5095_CANDIDATES = ("metrics/mAP50-95(B)", "metrics/mAP50-95", "metrics/mAP50-95(M)")
MG_METHODS = {"mg_afss", "mg_afss_safe", "v3", "v3_safe"}
MG_SAFE_METHODS = {"mg_afss_safe", "v3_safe"}


def is_mg_afss_method(method: str) -> bool:
    return method in MG_METHODS


def is_mg_afss_safe_method(method: str) -> bool:
    return method in MG_SAFE_METHODS


def public_method_name(method: str) -> str:
    if method == "v3":
        return "mg_afss"
    if method == "v3_safe":
        return "mg_afss_safe"
    return method


class RuntimeRecorder:
    """Collect lightweight runtime information through Ultralytics callbacks."""

    def __init__(self) -> None:
        self.active_sizes: dict[str, int] = {}
        self.active_ratios: dict[str, float] = {}
        self.difficulty_by_epoch: dict[str, dict[str, int]] = {}
        self.epoch_wall_time_s: dict[str, float] = {}
        self._epoch_start: dict[int, float] = {}

    def on_train_epoch_start(self, trainer) -> None:  # noqa: ANN001
        self._epoch_start[int(getattr(trainer, "epoch", 0))] = time.time()

    def on_train_epoch_end(self, trainer) -> None:  # noqa: ANN001
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        start = self._epoch_start.get(epoch - 1)
        if start is not None:
            self.epoch_wall_time_s[str(epoch)] = round(time.time() - start, 6)

        size = None
        sampler = getattr(getattr(trainer, "train_loader", None), "sampler", None)
        indices = getattr(sampler, "_indices", None)
        if indices is not None:
            size = len(indices)
            self.active_sizes[str(epoch)] = size
        state = getattr(trainer, "afss_state_manager", None)
        total = getattr(state, "num_images", None)
        if total is None:
            ref_state = getattr(trainer, "afss_state", None)
            if isinstance(ref_state, dict) and ref_state:
                total = len(ref_state)

        dataset = getattr(getattr(trainer, "train_loader", None), "dataset", None)
        im_files = getattr(dataset, "im_files", None)
        if im_files is not None:
            size = len(im_files)
            self.active_sizes[str(epoch)] = size
        if size is not None and total:
            self.active_ratios[str(epoch)] = round(size / total, 6)

        state = getattr(trainer, "afss_state_manager", None)
        if state is not None:
            try:
                self.difficulty_by_epoch[str(epoch)] = state.get_difficulty_distribution()
            except Exception:
                pass
        elif hasattr(trainer, "_get_afss_level_counts"):
            try:
                self.difficulty_by_epoch[str(epoch)] = trainer._get_afss_level_counts()
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run standard training, fixed AFSS, or MG-AFSS with stable metadata outputs."
    )
    parser.add_argument("--dataset", required=True, help="Logical dataset name, e.g. nwpu_vhr10.")
    parser.add_argument("--data", required=True, type=Path, help="Dataset yaml path.")
    parser.add_argument("--task", choices=["detect", "obb"], default="detect")
    parser.add_argument("--model-size", choices=["n", "s", "m", "l", "x"], default="n")
    parser.add_argument("--train-mode", choices=["scratch", "pretrained"], required=True)
    parser.add_argument(
        "--method",
        choices=["off", "afss", "mg_afss", "mg_afss_safe", "v3", "v3_safe"],
        required=True,
        help="Training method. 'v3' and 'v3_safe' are deprecated aliases for old records.",
    )
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "experiments" / "mg_afss_runs")
    parser.add_argument("--run-id", required=True)

    parser.add_argument("--model", type=Path, default=None, help="Optional explicit model yaml/pt path.")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--cache", default=False)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--single-cls", action="store_true", help="Train as a single-class dataset.")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-download", action="store_true", help="Allow Ultralytics to resolve missing weights.")

    parser.add_argument("--warmup", type=int, default=None, help="Fixed AFSS warmup epochs.")
    parser.add_argument("--hard-thr", type=float, default=0.55)
    parser.add_argument("--easy-thr", type=float, default=0.85)
    parser.add_argument("--moderate-ratio", type=float, default=0.40)
    parser.add_argument("--easy-ratio", type=float, default=0.02)
    parser.add_argument("--moderate-cover", type=int, default=3)
    parser.add_argument("--easy-review", type=int, default=10)
    parser.add_argument("--refresh-interval", type=int, default=5)
    parser.add_argument("--afss-conf", type=float, default=0.25)
    parser.add_argument("--afss-save-refresh-json", action="store_true")
    parser.add_argument(
        "--afss-stats-sample-ratio",
        type=float,
        default=1.0,
        help="Kept for compatibility; ignored by the patched Ultralytics AFSS trainer.",
    )

    parser.add_argument("--mg-min-epochs", "--v3-min-epochs", dest="v3_min_epochs", type=int, default=10)
    parser.add_argument(
        "--mg-fit-min-points",
        "--v3-fit-min-points",
        dest="v3_fit_min_points",
        type=int,
        default=5,
        help=(
            "Minimum number of cumulative validation points required before fitting "
            "the MG-AFSS maturity curve. This is not a sliding window."
        ),
    )
    parser.add_argument(
        "--mg-fit-window",
        "--v3-fit-window",
        dest="v3_fit_window",
        type=int,
        default=None,
        help=(
            "Deprecated alias for --mg-fit-min-points. Kept for old commands; "
            "MG-AFSS now fits the cumulative validation history."
        ),
    )
    parser.add_argument("--mg-trigger-policy", "--v3-trigger-policy", dest="v3_trigger_policy", default="fix2")
    parser.add_argument("--mg-phase2", "--v3-phase2", dest="v3_phase2", type=float, default=0.70)
    parser.add_argument("--mg-phase3", "--v3-phase3", dest="v3_phase3", type=float, default=0.85)
    parser.add_argument("--mg-plateau-thr", "--v3-plateau-thr", dest="v3_plateau_thr", type=float, default=0.70)
    parser.add_argument("--mg-discount-window", "--v3-discount-window", dest="v3_discount_window", type=int, default=20)
    parser.add_argument("--mg-safe-min-classes", "--v3-safe-min-classes", dest="v3_safe_min_classes", type=int, default=2)

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict-time", action="store_true")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    if args.v3_fit_window is not None:
        args.v3_fit_min_points = args.v3_fit_window
    return args


def resolve_user_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def validate_run_id(run_id: str) -> None:
    if not RUN_ID_RE.match(run_id):
        raise ValueError("run-id may only contain letters, numbers, underscore, dash, and dot.")


def load_data_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "nc" not in data:
        names = data.get("names")
        if isinstance(names, dict):
            data["nc"] = len(names)
        elif isinstance(names, list):
            data["nc"] = len(names)
    return data


def ensure_output_available(run_dir: Path, overwrite: bool, dry_run: bool) -> None:
    if run_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. Use a new --run-id or pass --overwrite."
        )
    if dry_run:
        return
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    if run_dir.exists() and overwrite:
        # Keep deletion out of this script. Let Ultralytics write into the explicit directory.
        return


def resolve_model_source(args: argparse.Namespace) -> str:
    if args.model is not None:
        model_path = resolve_user_path(args.model)
        if model_path.exists():
            return str(model_path)
        return str(args.model)

    suffix = "-obb" if args.task == "obb" else ""
    stem = f"yolo11{args.model_size}{suffix}"

    if args.train_mode == "scratch":
        return f"{stem}.yaml"

    candidates = [
        get_weights_root() / "pretrained" / f"{stem}.pt",
        get_weights_root() / f"{stem}.pt",
        get_ultralytics_path() / f"{stem}.pt",
        PROJECT_ROOT / f"{stem}.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    if args.allow_download:
        return f"{stem}.pt"
    raise FileNotFoundError(
        f"Pretrained weight for {stem} not found under weights/. "
        "Pass --model or --allow-download if this is intentional."
    )


def get_metric(row: dict[str, str], candidates: tuple[str, ...]) -> float | None:
    lowered = {k.strip().lower(): k for k in row}
    for candidate in candidates:
        key = lowered.get(candidate.lower())
        if key is not None and row.get(key, "") != "":
            try:
                return float(row[key])
            except ValueError:
                return None
    if candidates is METRIC_MAP50_CANDIDATES:
        for key, value in row.items():
            low = key.lower()
            if "map50" in low and "50-95" not in low and value != "":
                try:
                    return float(value)
                except ValueError:
                    return None
    else:
        for key, value in row.items():
            if "map50-95" in key.lower() and value != "":
                try:
                    return float(value)
                except ValueError:
                    return None
    return None


def read_results_summary(results_csv: Path) -> dict[str, Any]:
    if not results_csv.exists():
        return {"results_status": "missing"}
    with results_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"results_status": "empty", "epoch_count": 0}

    times = []
    for row in rows:
        try:
            times.append(float(row.get("time", "")))
        except ValueError:
            pass

    epoch_times = []
    if times:
        epoch_times = [times[0]]
        epoch_times.extend(times[i] - times[i - 1] for i in range(1, len(times)))
        epoch_times = [x for x in epoch_times if x >= 0]

    map50_vals = [get_metric(row, METRIC_MAP50_CANDIDATES) for row in rows]
    map50_vals = [v for v in map50_vals if v is not None]
    map5095_vals = [get_metric(row, METRIC_MAP5095_CANDIDATES) for row in rows]
    map5095_vals = [v for v in map5095_vals if v is not None]

    best_epoch = None
    if map50_vals:
        best_idx = max(range(len(map50_vals)), key=lambda i: map50_vals[i])
        best_epoch = best_idx + 1

    return {
        "results_status": "ok",
        "epoch_count": len(rows),
        "results_time_s": times[-1] if times else None,
        "avg_epoch_time_s": round(sum(epoch_times) / len(epoch_times), 6) if epoch_times else None,
        "final_map50": map50_vals[-1] if map50_vals else None,
        "final_map5095": map5095_vals[-1] if map5095_vals else None,
        "best_map50": max(map50_vals) if map50_vals else None,
        "best_map5095": max(map5095_vals) if map5095_vals else None,
        "best_epoch": best_epoch,
        # Legacy-friendly aliases for collect_results_from_summary.py.
        "mAP50": map50_vals[-1] if map50_vals else None,
        "mAP50-95": map5095_vals[-1] if map5095_vals else None,
    }


def run_command_text() -> str:
    return " ".join([shlex_quote(sys.executable), *[shlex_quote(x) for x in sys.argv]])


def shlex_quote(value: str) -> str:
    if re.match(r"^[A-Za-z0-9_./:\\=-]+$", value):
        return value
    return '"' + value.replace('"', '\\"') + '"'


def git_snapshot(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def collect_device_info(device: str) -> dict[str, Any]:
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "device": device,
        "gpu_name": "",
        "cuda": "",
        "torch": "",
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda or ""
        if torch.cuda.is_available():
            if str(device).lower() != "cpu":
                idx = int(str(device).split(",")[0]) if str(device).split(",")[0].isdigit() else 0
                info["gpu_name"] = torch.cuda.get_device_name(idx)
            else:
                info["gpu_name"] = "CUDA available but run requested CPU"
        else:
            info["gpu_name"] = "CUDA unavailable"
    except Exception as exc:
        info["torch_error"] = str(exc)
    return info


def strict_time_check(device: str) -> dict[str, Any]:
    result = {"strict_time_checked": False, "nvidia_smi": ""}
    if str(device).lower() == "cpu":
        result["strict_time_checked"] = True
        result["nvidia_smi"] = "cpu run"
        return result
    nvidia = shutil.which("nvidia-smi")
    if not nvidia:
        result["nvidia_smi"] = "nvidia-smi not found"
        return result
    try:
        proc = subprocess.run([nvidia], text=True, capture_output=True, check=False)
        result["strict_time_checked"] = True
        result["nvidia_smi"] = proc.stdout[-4000:]
    except Exception as exc:
        result["nvidia_smi"] = f"nvidia-smi error: {exc}"
    return result


def configure_import_paths() -> None:
    project_fork = str(get_ultralytics_path())
    if project_fork not in sys.path:
        sys.path.insert(0, project_fork)
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    if os.name == "nt":
        # Windows-only compatibility guard for local dry-runs/smoke tests.
        # Some PyTorch/NumPy/SciPy/OpenCV stacks load Intel/OpenMP runtimes
        # through more than one package and can fail with "libiomp5md.dll
        # already initialized". This flag allows duplicate runtime loading so
        # the script can proceed on such Windows environments. It is not part
        # of AFSS/MG-AFSS logic and does not prove that a formal training server is
        # clean; if a formal run needs this workaround, record that explicitly.
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def import_yolo():
    configure_import_paths()
    saved_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        from ultralytics import YOLO
        from ultralytics.utils import SETTINGS

        try:
            SETTINGS["wandb"] = False
        except Exception:
            pass
        patch_ultralytics_weight_lookup()

        return YOLO
    finally:
        sys.argv = saved_argv


def patch_ultralytics_weight_lookup() -> None:
    """Let Ultralytics AMP/model checks find project-local weights before downloading."""
    try:
        from ultralytics.utils import downloads
    except Exception:
        return
    if getattr(downloads.attempt_download_asset, "_afss_patched", False):
        return

    original_attempt_download_asset = downloads.attempt_download_asset

    def attempt_download_asset_local_first(file, *args, **kwargs):  # noqa: ANN001
        name = Path(str(file)).name
        for candidate in (get_weights_root() / "pretrained" / name, get_weights_root() / name):
            if candidate.exists():
                return str(candidate.resolve())
        return original_attempt_download_asset(file, *args, **kwargs)

    attempt_download_asset_local_first._afss_patched = True  # type: ignore[attr-defined]
    downloads.attempt_download_asset = attempt_download_asset_local_first


def get_reference_afss_trainer_cls(task: str):
    """Return the explicit AFSS trainer class from the trusted reference fork."""
    configure_import_paths()
    if task == "obb":
        from ultralytics.models.yolo.obb.afss_train import AFSSOBBTrainer

        return AFSSOBBTrainer
    from ultralytics.models.yolo.detect.afss_train import AFSSDetectionTrainer

    return AFSSDetectionTrainer


class V3Phase:
    SILENT = 1
    PROBING = 2
    ACTIVE = 3


def make_v3_trainer_cls(args: argparse.Namespace, nc: int):
    configure_import_paths()
    from ultralytics.utils import DEFAULT_CFG

    BaseTrainer = get_reference_afss_trainer_cls(args.task)

    phase2_thr = args.v3_phase2
    phase3_thr = args.v3_phase3
    min_epochs = args.v3_min_epochs
    fit_min_points = max(2, args.v3_fit_min_points)
    plateau_thr = args.v3_plateau_thr
    discount_window = max(1, args.v3_discount_window)
    safe_guard = is_mg_afss_safe_method(args.method)
    safe_min_classes = args.v3_safe_min_classes

    class V3OnlineTrainer(BaseTrainer):
        """Trainer subclass that activates AFSS from validation-curve maturity."""

        def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):  # noqa: ANN001
            overrides = {} if overrides is None else dict(overrides)
            overrides["afss"] = True
            overrides["afss_warmup_epochs"] = 999999
            super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
            self._v3_phase = V3Phase.SILENT
            self._v3_log: list[dict[str, Any]] = []
            self._v3_trigger_epoch: int | None = None
            self._v3_phase2_epoch: int | None = None
            self._v3_tau_hat: float | None = None
            self._v3_a_hat: float | None = None
            self._v3_mu_hat: float = 0.0
            self._v3_history: deque[tuple[int, float]] = deque(maxlen=200)
            self.add_callback("on_fit_epoch_end", self._v3_on_fit_epoch_end)

        @staticmethod
        def _exp_model(e, a, tau):  # noqa: ANN001
            import numpy as np

            return a * (1 - np.exp(-np.clip(e, 1e-6, None) / tau))

        def _current_map50(self) -> float | None:
            metrics = getattr(self, "metrics", None) or {}
            for key in ("metrics/mAP50(B)", "metrics/mAP50", "metrics/mAP50-95(B)"):
                value = metrics.get(key)
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return None
            csv_path = Path(getattr(self, "csv", ""))
            if csv_path.exists():
                try:
                    stats = read_results_summary(csv_path)
                    value = stats.get("final_map50")
                    return float(value) if value is not None else None
                except Exception:
                    return None
            return None

        def _fit_curve(self) -> tuple[float | None, float | None]:
            if len(self._v3_history) < fit_min_points:
                return None, None
            try:
                import numpy as np
                from scipy.optimize import curve_fit

                # Important: this fit estimates a global maturity curve,
                # mAP(e) = A * (1 - exp(-e / tau)), and the resulting tau is
                # interpreted against the absolute epoch in _estimate_maturity().
                # Therefore the fit must stay anchored to the early rise of the
                # curve. Using only the latest few validation points turns this
                # into a local trend fit: a short flat/noisy segment can make A
                # too small, pass the plateau check too early, and trigger AFSS
                # long before the model is actually mature. The archived MG-AFSS and
                # protocol-verification scripts used cumulative/prefix histories;
                # keep that behavior here. The CLI value is only the minimum
                # number of points required before fitting starts.
                history = list(self._v3_history)
                epochs = np.array([e for e, _ in history], dtype=float)
                maps = np.array([m for _, m in history], dtype=float)
                p0 = [max(float(maps[-1]) * 1.2, 1e-3), max(float(len(epochs)) * 0.5, 1.0)]
                popt, _ = curve_fit(
                    self._exp_model,
                    epochs,
                    maps,
                    p0=p0,
                    maxfev=10000,
                    bounds=([0.0, 0.5], [5.0, 500.0]),
                )
                return float(popt[0]), float(popt[1])
            except Exception:
                return None, None

        def _estimate_maturity(self, epoch: int, map50: float) -> float:
            self._v3_history.append((epoch, map50))
            a_hat, tau_hat = self._fit_curve()
            if tau_hat is None or tau_hat <= 0:
                return 0.0
            self._v3_a_hat = a_hat
            self._v3_tau_hat = tau_hat
            mu_raw = 1.0 - math.exp(-epoch / tau_hat)
            if a_hat and a_hat > 0 and map50 / a_hat > plateau_thr:
                discount = 1.0
            else:
                discount = min(1.0, len(self._v3_history) / discount_window)
            self._v3_mu_hat = mu_raw * discount
            return self._v3_mu_hat

        def _v3_on_fit_epoch_end(self, trainer) -> None:  # noqa: ANN001
            if trainer is not self:
                return
            epoch = int(getattr(self, "epoch", 0)) + 1
            map50 = self._current_map50()
            if map50 is None:
                return
            mu = self._estimate_maturity(epoch, map50)
            entry = {
                "epoch": epoch,
                "map50": map50,
                "mu_hat": round(mu, 6),
                "tau_hat": round(self._v3_tau_hat, 6) if self._v3_tau_hat is not None else None,
                "a_hat": round(self._v3_a_hat, 6) if self._v3_a_hat is not None else None,
                "phase": self._v3_phase,
                "policy": args.v3_trigger_policy,
                "fit_mode": "cumulative",
                "fit_points": len(self._v3_history),
            }
            self._v3_log.append(entry)

            if epoch < min_epochs or self._v3_tau_hat is None:
                return

            if self._v3_phase == V3Phase.SILENT and mu >= phase2_thr:
                self._v3_phase = V3Phase.PROBING
                self._v3_phase2_epoch = epoch

            if self._v3_phase == V3Phase.PROBING:
                if safe_guard and nc < safe_min_classes:
                    return
                if mu >= phase3_thr:
                    self._v3_phase = V3Phase.ACTIVE
                    self._v3_trigger_epoch = epoch
                    self.args.afss_warmup_epochs = epoch
                    self.afss_enabled = True
                    self.args.afss = True
                    self._v3_log.append(
                        {
                            "epoch": epoch,
                            "event": "activate_afss_next_epoch",
                            "warmup_epochs": epoch,
                            "tau_hat": round(self._v3_tau_hat, 6),
                            "w_star": round(1.9 * self._v3_tau_hat, 6),
                        }
                    )

    return V3OnlineTrainer


def build_train_kwargs(args: argparse.Namespace, model_source: str, data_yaml: Path, output_root: Path) -> dict[str, Any]:
    close_mosaic = max(0, int(args.epochs * 0.12))
    kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "pretrained": args.train_mode == "pretrained",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "single_cls": args.single_cls,
        "amp": args.amp,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "cos_lr": True,
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "close_mosaic": close_mosaic,
        "val": True,
        "plots": args.plots,
        "save": not args.no_save,
        "save_period": args.save_period,
        "project": str(output_root),
        "name": args.run_id,
        "exist_ok": args.overwrite,
        "patience": args.patience,
        "fraction": args.fraction,
        "cache": args.cache,
    }

    if args.method == "afss" or is_mg_afss_method(args.method):
        warmup = args.warmup
        if warmup is None:
            warmup = 999999 if is_mg_afss_method(args.method) else 0
        afss_thresholds = {
            "detect": [args.hard_thr, args.easy_thr],
            "obb": [args.hard_thr, args.easy_thr],
        }
        kwargs.update(
            {
                "afss": True,
                "afss_warmup_epochs": warmup,
                "afss_update_interval": args.refresh_interval,
                "afss_easy_ratio": args.easy_ratio,
                "afss_moderate_ratio": args.moderate_ratio,
                "afss_easy_forced_gap": args.easy_review,
                "afss_moderate_forced_gap": args.moderate_cover,
                "afss_conf": args.afss_conf,
                "afss_thresholds": afss_thresholds,
                "afss_save_refresh_json": args.afss_save_refresh_json,
            }
        )
    else:
        kwargs["afss"] = False
    return kwargs


def summarize_afss(trainer, recorder: RuntimeRecorder, run_dir: Path) -> dict[str, Any]:  # noqa: ANN001
    state = getattr(trainer, "afss_state_manager", None)
    ref_state = getattr(trainer, "afss_state", None)
    summary: dict[str, Any] = {
        "active_sizes": recorder.active_sizes,
        "active_ratios": recorder.active_ratios,
        "difficulty_by_epoch": recorder.difficulty_by_epoch,
        "epoch_wall_time_s": recorder.epoch_wall_time_s,
    }
    if state is not None:
        dist = state.get_difficulty_distribution()
        summary.update(
            {
                "num_images": getattr(state, "num_images", None),
                "warmup_epochs": getattr(state, "warmup_epochs", None),
                "last_update_epoch": getattr(state, "last_update_epoch", None),
                "difficulty_distribution": dist,
            }
        )
        states = getattr(state, "states", [])
        if states:
            si_values = [float(getattr(s, "S_i", 0.0)) for s in states if getattr(s, "_evaluated", False)]
            if si_values:
                summary.update(
                    {
                        "si_mean": round(sum(si_values) / len(si_values), 6),
                        "si_min": round(min(si_values), 6),
                        "si_max": round(max(si_values), 6),
                        "si_evaluated": len(si_values),
                    }
                )
    elif isinstance(ref_state, dict):
        states = list(ref_state.values())
        last_updates = [int(getattr(s, "last_eval_epoch", -1)) for s in states]
        summary.update(
            {
                "num_images": len(ref_state),
                "warmup_epochs": getattr(getattr(trainer, "args", None), "afss_warmup_epochs", None),
                "last_update_epoch": max(last_updates) if last_updates else None,
                "difficulty_distribution": {},
            }
        )
        if hasattr(trainer, "_get_afss_level_counts"):
            try:
                summary["difficulty_distribution"] = trainer._get_afss_level_counts()
            except Exception:
                pass
        scores = [float(getattr(s, "task_score", 0.0)) for s in states if getattr(s, "last_eval_epoch", -1) >= 0]
        if scores:
            summary.update(
                {
                    "si_mean": round(sum(scores) / len(scores), 6),
                    "si_min": round(min(scores), 6),
                    "si_max": round(max(scores), 6),
                    "si_evaluated": len(scores),
                }
            )

    if summary:
        with (run_dir / "afss_stats.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, ensure_ascii=False, indent=2)


def to_jsonable(value):  # noqa: ANN001
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_base_summary(
    args: argparse.Namespace,
    data_yaml: Path,
    data_cfg: dict[str, Any],
    model_source: str,
    output_root: Path,
    run_dir: Path,
    status: str,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict[str, Any]:
    pretrained = args.train_mode == "pretrained"
    if args.method == "off":
        afss_warmup = None
    elif args.warmup is not None:
        afss_warmup = args.warmup
    elif is_mg_afss_method(args.method):
        afss_warmup = 999999
    else:
        afss_warmup = 0
    return {
        "tag": args.run_id,
        "run_id": args.run_id,
        "status": status,
        "dataset": args.dataset,
        "data_yaml": str(data_yaml),
        "task": args.task,
        "model_size": args.model_size,
        "model_source": model_source,
        "train_mode": args.train_mode,
        "method": public_method_name(args.method),
        "method_alias_used": args.method if public_method_name(args.method) != args.method else "",
        "afss_on": args.method == "afss" or is_mg_afss_method(args.method),
        "pretrained": pretrained,
        "v3_policy": args.v3_trigger_policy if is_mg_afss_method(args.method) else "",
        "v3_fit_mode": "cumulative" if is_mg_afss_method(args.method) else "",
        "v3_fit_min_points": args.v3_fit_min_points if is_mg_afss_method(args.method) else None,
        "mg_afss_policy": args.v3_trigger_policy if is_mg_afss_method(args.method) else "",
        "mg_afss_fit_mode": "cumulative" if is_mg_afss_method(args.method) else "",
        "mg_afss_fit_min_points": args.v3_fit_min_points if is_mg_afss_method(args.method) else None,
        "epochs": args.epochs,
        "seed": args.seed,
        "warmup": afss_warmup,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "single_cls": args.single_cls,
        "workers": args.workers,
        "fraction": args.fraction,
        "device": args.device,
        "nc": data_cfg.get("nc"),
        "output_root": str(output_root),
        "run_dir": str(run_dir),
        "ultralytics_runtime_path": str(get_ultralytics_path()),
        "ultralytics_project_path": str(get_ultralytics_path()),
        "reference_fork_path": str(get_reference_ultralytics_path()),
        "git_commit_or_snapshot": git_snapshot(get_ultralytics_path()),
        "command": run_command_text(),
        "start_time": start_time or "",
        "end_time": end_time or "",
        "notes": args.notes,
    }


def dry_run_report(args: argparse.Namespace, data_yaml: Path, data_cfg: dict[str, Any], model_source: str, run_dir: Path):
    report = {
        "dry_run": True,
        "project_root": str(PROJECT_ROOT),
        "ultralytics_runtime_path": str(get_ultralytics_path()),
        "ultralytics_path": str(get_ultralytics_path()),
        "reference_ultralytics_path": str(get_reference_ultralytics_path()),
        "data_yaml": str(data_yaml),
        "dataset_nc": data_cfg.get("nc"),
        "model_source": model_source,
        "output_run_dir": str(run_dir),
        "method": public_method_name(args.method),
        "train_mode": args.train_mode,
        "epochs": args.epochs,
        "seed": args.seed,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "single_cls": args.single_cls,
        "would_overwrite": run_dir.exists(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> int:
    try:
        args = parse_args()
        validate_run_id(args.run_id)

        data_yaml = resolve_user_path(args.data)
        output_root = resolve_user_path(args.output_root)
        run_dir = output_root / args.run_id

        if not data_yaml.exists():
            raise FileNotFoundError(f"Data yaml not found: {data_yaml}")
        data_cfg = load_data_yaml(data_yaml)
        model_source = resolve_model_source(args)
        ensure_output_available(run_dir, args.overwrite, args.dry_run)

        if args.dry_run:
            dry_run_report(args, data_yaml, data_cfg, model_source, run_dir)
            return 0
    except Exception as exc:
        print(f"Preflight failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    start_time = datetime.now().isoformat(timespec="seconds")
    t0 = time.time()
    recorder = RuntimeRecorder()
    status = "finished"
    error_info: dict[str, Any] = {}
    actual_run_dir = run_dir
    trainer_obj = None

    if args.strict_time:
        strict_info = strict_time_check(args.device)
    else:
        strict_info = {}

    try:
        YOLO = import_yolo()
        model = YOLO(model_source, task=args.task)
        model.add_callback("on_train_epoch_start", recorder.on_train_epoch_start)
        model.add_callback("on_train_epoch_end", recorder.on_train_epoch_end)

        train_kwargs = build_train_kwargs(args, model_source, data_yaml, output_root)
        trainer_cls = None
        if args.method == "afss":
            trainer_cls = get_reference_afss_trainer_cls(args.task)
        elif is_mg_afss_method(args.method):
            trainer_cls = make_v3_trainer_cls(args, int(data_cfg.get("nc") or 0))

        metrics = model.train(trainer=trainer_cls, **train_kwargs)
        trainer_obj = getattr(model, "trainer", None)
        if trainer_obj is not None:
            actual_run_dir = Path(trainer_obj.save_dir)
    except Exception as exc:
        status = "failed"
        error_info = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        if not actual_run_dir.exists():
            actual_run_dir.mkdir(parents=True, exist_ok=True)
    finally:
        end_time = datetime.now().isoformat(timespec="seconds")
        elapsed = time.time() - t0
        summary = build_base_summary(
            args=args,
            data_yaml=data_yaml,
            data_cfg=data_cfg,
            model_source=model_source,
            output_root=output_root,
            run_dir=actual_run_dir,
            status=status,
            start_time=start_time,
            end_time=end_time,
        )
        summary.update(collect_device_info(args.device))
        summary.update(strict_info)
        summary.update(
            {
                "total_wall_time_s": round(elapsed, 6),
                "total_h": round(elapsed / 3600, 6),
                "results_csv": str(actual_run_dir / "results.csv"),
            }
        )
        summary.update(read_results_summary(actual_run_dir / "results.csv"))

        afss_summary = {}
        if trainer_obj is not None and (args.method == "afss" or is_mg_afss_method(args.method)):
            afss_summary = summarize_afss(trainer_obj, recorder, actual_run_dir)
            if afss_summary.get("active_ratios"):
                ratios = list(afss_summary["active_ratios"].values())
                summary["active_ratio_mean"] = round(sum(ratios) / len(ratios), 6)
            if afss_summary.get("active_sizes") and afss_summary.get("num_images"):
                sizes = list(afss_summary["active_sizes"].values())
                total = int(afss_summary["num_images"]) * len(sizes)
                used = sum(int(x) for x in sizes)
                summary["used_images_total"] = used
                summary["skipped_images_total"] = total - used
            for key in ("si_mean", "si_min", "si_max", "si_evaluated"):
                if key in afss_summary:
                    summary[key] = afss_summary[key]

        if trainer_obj is not None and is_mg_afss_method(args.method):
            v3_payload = {
                "log": getattr(trainer_obj, "_v3_log", []),
                "trigger_epoch": getattr(trainer_obj, "_v3_trigger_epoch", None),
                "phase2_epoch": getattr(trainer_obj, "_v3_phase2_epoch", None),
                "policy": args.v3_trigger_policy,
                "fit_mode": "cumulative",
                "fit_min_points": args.v3_fit_min_points,
                "safe_guard": is_mg_afss_safe_method(args.method),
                "nc": data_cfg.get("nc"),
            }
            write_json(actual_run_dir / "v3_log.json", v3_payload)
            summary["v3_trigger_epoch"] = v3_payload["trigger_epoch"]
            summary["phase2_epoch"] = v3_payload["phase2_epoch"]

        if error_info:
            summary.update(error_info)
        summary.setdefault("afss_update_time_s", None)
        summary.setdefault("sampling_decision_time_s", None)

        write_json(actual_run_dir / "summary.json", summary)
        write_json(actual_run_dir / "config.json", vars(args))
        (actual_run_dir / "command.txt").write_text(summary["command"] + "\n", encoding="utf-8")

    if status != "finished":
        print(f"Run failed. Summary written to: {actual_run_dir / 'summary.json'}")
        return 1
    print(f"Run finished. Summary written to: {actual_run_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
