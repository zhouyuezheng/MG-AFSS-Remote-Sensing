"""Compile experiment results from per-run summary.json and results.csv.

This script intentionally ignores legacy _all_results.csv files because several
of them have header/row mismatches. It uses a fixed output schema so OFF, AFSS,
V3, random, and staged runs can live in one stable table.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean

RELEASE_ROOT = Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

from mg_afss.paths import get_experiments_root


SCHEMA = [
    "experiment_family",
    "run_group",
    "tag",
    "relative_run_dir",
    "summary_path",
    "results_csv",
    "afss_on",
    "pretrained",
    "epochs",
    "seed",
    "warmup",
    "update_interval",
    "moderate_ratio",
    "easy_ratio",
    "ratio",
    "d_protocol",
    "nc",
    "v3_trigger_epoch",
    "phase2_epoch",
    "stage1",
    "stage2",
    "total_h",
    "results_time_h",
    "epoch_count",
    "epoch_avg_s",
    "mAP50",
    "mAP50_95",
    "best_mAP50",
    "best_mAP50_95",
    "si_mean",
    "si_median",
    "si_std",
    "si_hard",
    "si_moderate",
    "si_easy",
    "active_sizes_count",
    "active_avg",
    "active_min",
    "active_max",
    "source_status",
    "extra_json",
]

SUMMARY_KEY_MAP = {
    "mAP50-95": "mAP50_95",
    "update_interval": "update_interval",
    "afss_update_interval": "update_interval",
    "moderate_ratio": "moderate_ratio",
    "afss_moderate_ratio": "moderate_ratio",
    "easy_ratio": "easy_ratio",
    "afss_easy_ratio": "easy_ratio",
    "v3_trigger_epoch": "v3_trigger_epoch",
    "phase2_epoch": "phase2_epoch",
    "stage1_end": "stage1",
    "stage2_end": "stage2",
}


def parse_number(value):
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return ""
        return value
    text = str(value).strip()
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return value
    if number.is_integer():
        return int(number)
    return number


def parse_float(value):
    parsed = parse_number(value)
    return float(parsed) if isinstance(parsed, (int, float)) and not isinstance(parsed, bool) else None


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_results_csv(path: Path) -> dict:
    if not path.exists():
        return {"source_status": "missing_results_csv"}

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = [
                {k.strip(): v for k, v in row.items() if k is not None}
                for row in reader
            ]
    except Exception as exc:
        return {"source_status": f"results_csv_error:{type(exc).__name__}"}

    if not rows:
        return {"source_status": "empty_results_csv", "epoch_count": 0}

    columns = list(rows[0].keys())
    time_values = [parse_float(row.get("time")) for row in rows]
    time_values = [x for x in time_values if x is not None]
    epoch_times = []
    if time_values:
        epoch_times = [time_values[0]]
        epoch_times.extend(time_values[i] - time_values[i - 1] for i in range(1, len(time_values)))
        epoch_times = [x for x in epoch_times if x >= 0]

    m50_col = find_metric_column(columns, "mAP50")
    m5095_col = find_metric_column(columns, "mAP50-95")

    result = {
        "source_status": "ok",
        "epoch_count": len(rows),
        "results_time_h": round(time_values[-1] / 3600, 4) if time_values else "",
        "epoch_avg_s": round(mean(epoch_times), 3) if epoch_times else "",
    }
    if m50_col:
        vals = numeric_column(rows, m50_col)
        if vals:
            result["mAP50"] = vals[-1]
            result["best_mAP50"] = max(vals)
    if m5095_col:
        vals = numeric_column(rows, m5095_col)
        if vals:
            result["mAP50_95"] = vals[-1]
            result["best_mAP50_95"] = max(vals)
    return result


def find_metric_column(columns: list[str], metric: str) -> str | None:
    lowered = [(c, c.lower()) for c in columns]
    if metric == "mAP50":
        for original, low in lowered:
            if "map50" in low and "50-95" not in low:
                return original
    if metric == "mAP50-95":
        for original, low in lowered:
            if "map50-95" in low:
                return original
    return None


def numeric_column(rows: list[dict], column: str) -> list[float]:
    vals = []
    for row in rows:
        value = parse_float(row.get(column))
        if value is not None:
            vals.append(value)
    return vals


def active_size_stats(active_sizes) -> dict:
    if not isinstance(active_sizes, dict):
        return {}
    vals = []
    for value in active_sizes.values():
        parsed = parse_float(value)
        if parsed is not None:
            vals.append(parsed)
    if not vals:
        return {}
    return {
        "active_sizes_count": len(vals),
        "active_avg": round(mean(vals), 3),
        "active_min": min(vals),
        "active_max": max(vals),
    }


def infer_family_and_group(run_dir: Path, experiments_root: Path) -> tuple[str, str, str]:
    rel = run_dir.relative_to(experiments_root)
    parts = rel.parts
    family = parts[0] if parts else ""
    group = "/".join(parts[:-1]) if len(parts) > 1 else family
    return family, group, rel.as_posix()


def build_row(summary_path: Path, experiments_root: Path) -> dict:
    run_dir = summary_path.parent
    results_csv = run_dir / "results.csv"
    summary = load_summary(summary_path)
    family, group, rel_dir = infer_family_and_group(run_dir, experiments_root)

    row = {key: "" for key in SCHEMA}
    row.update(
        {
            "experiment_family": family,
            "run_group": group,
            "tag": summary.get("tag", run_dir.name),
            "relative_run_dir": rel_dir,
            "summary_path": summary_path.relative_to(experiments_root).as_posix(),
            "results_csv": results_csv.relative_to(experiments_root).as_posix() if results_csv.exists() else "",
        }
    )

    for key, value in summary.items():
        out_key = SUMMARY_KEY_MAP.get(key, key)
        if out_key in row and out_key != "extra_json":
            row[out_key] = value

    csv_stats = read_results_csv(results_csv)
    for key, value in csv_stats.items():
        if key in row and row.get(key, "") == "":
            row[key] = value
        elif key == "source_status":
            row[key] = value

    for key, value in active_size_stats(summary.get("active_sizes")).items():
        row[key] = value

    extras = {}
    for key, value in summary.items():
        out_key = SUMMARY_KEY_MAP.get(key, key)
        if out_key not in SCHEMA:
            extras[key] = value
    row["extra_json"] = json.dumps(extras, ensure_ascii=False, sort_keys=True) if extras else ""

    for key, value in list(row.items()):
        if isinstance(value, (dict, list)):
            row[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            row[key] = parse_number(value)
    return row


def should_skip(summary_path: Path, include_test: bool) -> bool:
    if include_test:
        return False
    return any("TEST" in part.upper() for part in summary_path.parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile stable experiment results from summary.json files.")
    parser.add_argument("--experiments-root", type=Path, default=get_experiments_root())
    parser.add_argument("--out", type=Path, default=None,
                        help="Output CSV path. Default: experiments/compiled_results_from_summary.csv")
    parser.add_argument("--include-test", action="store_true",
                        help="Include directories whose path contains TEST.")
    args = parser.parse_args()

    experiments_root = args.experiments_root.resolve()
    out_path = (args.out or experiments_root / "compiled_results_from_summary.csv").resolve()

    summaries = sorted(experiments_root.rglob("summary.json"))
    rows = []
    for summary_path in summaries:
        if should_skip(summary_path, args.include_test):
            continue
        try:
            rows.append(build_row(summary_path.resolve(), experiments_root))
        except Exception as exc:
            family, group, rel_dir = infer_family_and_group(summary_path.parent.resolve(), experiments_root)
            rows.append({
                **{key: "" for key in SCHEMA},
                "experiment_family": family,
                "run_group": group,
                "relative_run_dir": rel_dir,
                "summary_path": summary_path.relative_to(experiments_root).as_posix(),
                "source_status": f"summary_error:{type(exc).__name__}",
                "extra_json": json.dumps({"error": str(exc)}, ensure_ascii=False),
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCHEMA)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Compiled {len(rows)} runs from {experiments_root}")
    print(f"Output: {out_path}")
    status_counts = {}
    for row in rows:
        status = row.get("source_status") or "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
