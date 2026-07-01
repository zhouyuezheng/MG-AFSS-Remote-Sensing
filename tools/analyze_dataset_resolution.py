#!/usr/bin/env python3
"""Analyze image resolution and object sizes before choosing YOLO imgsz.

The script reads a YOLO-style dataset YAML, scans image dimensions and labels,
then estimates how object sizes would look after square letterbox resizing for
candidate imgsz values such as 640, 768, and 1024.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import yaml
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze original image/object sizes for YOLO HBB/OBB datasets."
    )
    parser.add_argument("--data", required=True, type=Path, help="Dataset YAML path.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to scan.")
    parser.add_argument(
        "--imgsz-candidates",
        nargs="+",
        type=int,
        default=[640, 768, 1024],
        help="Candidate YOLO square input sizes to evaluate.",
    )
    parser.add_argument("--preferred-imgsz", type=int, default=1024)
    parser.add_argument("--max-images", type=int, default=0, help="0 means no limit.")
    parser.add_argument(
        "--no-scaleup",
        action="store_true",
        help="Use scale=min(1, imgsz/max_side). Ultralytics training normally allows scaleup.",
    )
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_base(data_yaml: Path, cfg: dict[str, Any]) -> Path:
    root = cfg.get("path") or data_yaml.parent
    root_path = Path(root)
    if not root_path.is_absolute():
        root_path = data_yaml.parent / root_path
    return root_path.resolve()


def iter_images_from_entry(base: Path, entry: Any) -> list[Path]:
    if entry is None:
        return []
    entries = entry if isinstance(entry, list) else [entry]
    images: list[Path] = []
    for item in entries:
        p = Path(str(item))
        if not p.is_absolute():
            p = base / p
        p = p.resolve()
        if p.is_file() and p.suffix.lower() == ".txt":
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                img = Path(line)
                if not img.is_absolute():
                    img = base / img
                images.append(img.resolve())
        elif p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            images.append(p)
        elif p.is_dir():
            images.extend(sorted(x for x in p.rglob("*") if x.suffix.lower() in IMAGE_EXTS))
    return images


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.parent.parent / "labels" / image_path.with_suffix(".txt").name


def polygon_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def parse_label_line(line: str, img_w: int, img_h: int) -> dict[str, float] | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        cls_id = int(float(parts[0]))
        values = [float(x) for x in parts[1:]]
    except ValueError:
        return None

    if len(values) == 4:
        _, _, bw, bh = values
        if max(abs(bw), abs(bh)) <= 2.0:
            box_w = abs(bw) * img_w
            box_h = abs(bh) * img_h
        else:
            box_w = abs(bw)
            box_h = abs(bh)
        area = box_w * box_h
        return {
            "cls": cls_id,
            "format": "hbb",
            "box_w": box_w,
            "box_h": box_h,
            "area": area,
        }

    if len(values) >= 8:
        coords = values[:8]
        normalized = max(abs(x) for x in coords) <= 2.0
        points: list[tuple[float, float]] = []
        for i in range(0, 8, 2):
            x = coords[i] * img_w if normalized else coords[i]
            y = coords[i + 1] * img_h if normalized else coords[i + 1]
            points.append((x, y))
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        area = polygon_area(points)
        return {
            "cls": cls_id,
            "format": "obb",
            "box_w": max(xs) - min(xs),
            "box_h": max(ys) - min(ys),
            "area": area,
        }
    return None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(values[lo])
    return float(values[lo] * (hi - pos) + values[hi] * (pos - lo))


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": round(mean(values), 4),
        "median": round(median(values), 4),
        "min": round(min(values), 4),
        "p10": round(percentile(values, 0.10), 4),
        "p25": round(percentile(values, 0.25), 4),
        "p75": round(percentile(values, 0.75), 4),
        "p90": round(percentile(values, 0.90), 4),
        "p95": round(percentile(values, 0.95), 4),
        "max": round(max(values), 4),
    }


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def scan_dataset(args: argparse.Namespace) -> dict[str, Any]:
    data_yaml = args.data.resolve()
    cfg = read_yaml(data_yaml)
    base = resolve_base(data_yaml, cfg)
    names = cfg.get("names") or {}
    if isinstance(names, list):
        names_map = {i: name for i, name in enumerate(names)}
    else:
        names_map = {int(k): v for k, v in names.items()} if isinstance(names, dict) else {}

    image_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    missing_labels = 0
    unreadable_images = 0

    for split in args.splits:
        images = iter_images_from_entry(base, cfg.get(split))
        if args.max_images and args.max_images > 0:
            images = images[: args.max_images]
        for image_path in images:
            try:
                with Image.open(image_path) as im:
                    img_w, img_h = im.size
            except Exception:
                unreadable_images += 1
                continue
            image_rows.append(
                {
                    "split": split,
                    "path": image_path,
                    "name": image_path.name,
                    "w": img_w,
                    "h": img_h,
                    "area": img_w * img_h,
                    "long": max(img_w, img_h),
                    "short": min(img_w, img_h),
                }
            )
            label_path = label_path_for_image(image_path)
            if not label_path.exists():
                missing_labels += 1
                continue
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parsed = parse_label_line(line, img_w, img_h)
                if parsed is None:
                    continue
                area = max(float(parsed["area"]), 0.0)
                obj = {
                    "split": split,
                    "image": image_path.name,
                    "cls": int(parsed["cls"]),
                    "format": parsed["format"],
                    "img_w": img_w,
                    "img_h": img_h,
                    "img_long": max(img_w, img_h),
                    "box_w": float(parsed["box_w"]),
                    "box_h": float(parsed["box_h"]),
                    "area": area,
                    "sqrt_area": math.sqrt(area) if area > 0 else 0.0,
                    "relative_area": area / (img_w * img_h) if img_w and img_h else 0.0,
                }
                object_rows.append(obj)

    return build_report(args, data_yaml, cfg, base, names_map, image_rows, object_rows, missing_labels, unreadable_images)


def summarize_images(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    min_area = min(rows, key=lambda x: x["area"])
    max_area = max(rows, key=lambda x: x["area"])
    min_long = min(rows, key=lambda x: x["long"])
    max_long = max(rows, key=lambda x: x["long"])
    dims = Counter((r["w"], r["h"]) for r in rows)
    return {
        "count": len(rows),
        "width": summarize_values([r["w"] for r in rows]),
        "height": summarize_values([r["h"] for r in rows]),
        "long_side": summarize_values([r["long"] for r in rows]),
        "short_side": summarize_values([r["short"] for r in rows]),
        "area": summarize_values([r["area"] for r in rows]),
        "min_area_image": min_area,
        "max_area_image": max_area,
        "min_long_side_image": min_long,
        "max_long_side_image": max_long,
        "unique_dimensions": len(dims),
        "shape_counts": {
            "landscape": sum(1 for r in rows if r["w"] > r["h"]),
            "portrait": sum(1 for r in rows if r["w"] < r["h"]),
            "square": sum(1 for r in rows if r["w"] == r["h"]),
        },
        "top_dimensions": [
            {"w": w, "h": h, "count": count} for (w, h), count in dims.most_common(10)
        ],
    }


def summarize_objects(rows: list[dict[str, Any]], names_map: dict[int, str]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    per_class: dict[str, Any] = {}
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["cls"])].append(row)
    for cls_id, items in sorted(grouped.items()):
        per_class[str(cls_id)] = {
            "name": names_map.get(cls_id, f"class_{cls_id}"),
            "count": len(items),
            "sqrt_area": summarize_values([x["sqrt_area"] for x in items]),
            "relative_area": summarize_values([x["relative_area"] for x in items]),
        }
    return {
        "count": len(rows),
        "format_counts": dict(Counter(str(r["format"]) for r in rows)),
        "class_counts": {
            str(k): {
                "name": names_map.get(k, f"class_{k}"),
                "count": v,
            }
            for k, v in sorted(Counter(int(r["cls"]) for r in rows).items())
        },
        "sqrt_area": summarize_values([r["sqrt_area"] for r in rows]),
        "box_w": summarize_values([r["box_w"] for r in rows]),
        "box_h": summarize_values([r["box_h"] for r in rows]),
        "relative_area": summarize_values([r["relative_area"] for r in rows]),
        "size_buckets_original_sqrt_area": {
            "<16": sum(1 for r in rows if r["sqrt_area"] < 16),
            "16-32": sum(1 for r in rows if 16 <= r["sqrt_area"] < 32),
            "32-96": sum(1 for r in rows if 32 <= r["sqrt_area"] < 96),
            ">=96": sum(1 for r in rows if r["sqrt_area"] >= 96),
        },
        "per_class": per_class,
    }


def summarize_candidate(
    imgsz: int,
    image_rows: list[dict[str, Any]],
    object_rows: list[dict[str, Any]],
    scaleup: bool,
) -> dict[str, Any]:
    image_scales = []
    for row in image_rows:
        scale = imgsz / row["long"] if row["long"] else 1.0
        if not scaleup:
            scale = min(1.0, scale)
        image_scales.append(scale)

    projected = []
    for row in object_rows:
        scale = imgsz / row["img_long"] if row["img_long"] else 1.0
        if not scaleup:
            scale = min(1.0, scale)
        projected.append(row["sqrt_area"] * scale)

    total_img = max(1, len(image_scales))
    total_obj = max(1, len(projected))
    return {
        "imgsz": imgsz,
        "scaleup": scaleup,
        "image_scale": summarize_values(image_scales),
        "image_scale_counts": {
            "downsample": sum(1 for s in image_scales if s < 0.995),
            "near_original": sum(1 for s in image_scales if 0.995 <= s <= 1.005),
            "upsample": sum(1 for s in image_scales if s > 1.005),
        },
        "image_scale_ratio": {
            "downsample_pct": round(sum(1 for s in image_scales if s < 0.995) / total_img * 100, 2),
            "upsample_pct": round(sum(1 for s in image_scales if s > 1.005) / total_img * 100, 2),
        },
        "projected_sqrt_area": summarize_values(projected),
        "projected_size_buckets_sqrt_area": {
            "<16": sum(1 for x in projected if x < 16),
            "16-32": sum(1 for x in projected if 16 <= x < 32),
            "32-96": sum(1 for x in projected if 32 <= x < 96),
            ">=96": sum(1 for x in projected if x >= 96),
        },
        "projected_size_ratio": {
            "<16_pct": round(sum(1 for x in projected if x < 16) / total_obj * 100, 2),
            "<32_pct": round(sum(1 for x in projected if x < 32) / total_obj * 100, 2),
            ">=96_pct": round(sum(1 for x in projected if x >= 96) / total_obj * 100, 2),
        },
    }


def make_recommendation(
    preferred: int,
    candidates: dict[str, Any],
    image_summary: dict[str, Any],
    object_summary: dict[str, Any],
) -> dict[str, Any]:
    preferred_key = str(preferred)
    preferred_stats = candidates.get(preferred_key) or next(iter(candidates.values()), {})
    long_side = image_summary.get("long_side", {})
    p75_long = long_side.get("p75") or 0
    p95_long = long_side.get("p95") or 0
    obj_median = object_summary.get("sqrt_area", {}).get("median") or 0
    rec = {
        "default_imgsz": preferred,
        "decision": "use_default_after_audit",
        "reasons": [
            "Next-stage protocol defaults to 1024 to reduce avoidable downsampling in remote-sensing images.",
            "Keep standard training, fixed AFSS, and MG-AFSS within the same imgsz/batch/device/seed block.",
        ],
        "cautions": [],
    }

    if p75_long <= 700 and obj_median >= 64:
        rec["cautions"].append(
            "Images and objects are already relatively small/large enough; 640 or 768 may be sufficient if speed is critical."
        )
    if p95_long > preferred * 1.5:
        rec["cautions"].append(
            "Many images are still much larger than the preferred imgsz; consider tiling instead of whole-image resizing."
        )
    if preferred_stats.get("projected_size_ratio", {}).get("<16_pct", 0) > 20:
        rec["cautions"].append(
            "A large share of objects remains below 16 px after resizing; consider larger imgsz, tiling, or small-object-specific settings."
        )
    if preferred_stats.get("image_scale_ratio", {}).get("upsample_pct", 0) > 70:
        rec["cautions"].append(
            "Most images would be upsampled at the preferred imgsz; verify that the accuracy gain justifies extra compute."
        )
    return rec


def build_report(
    args: argparse.Namespace,
    data_yaml: Path,
    cfg: dict[str, Any],
    base: Path,
    names_map: dict[int, str],
    image_rows: list[dict[str, Any]],
    object_rows: list[dict[str, Any]],
    missing_labels: int,
    unreadable_images: int,
) -> dict[str, Any]:
    image_summary = summarize_images(image_rows)
    object_summary = summarize_objects(object_rows, names_map)
    scaleup = not args.no_scaleup
    candidates = {
        str(imgsz): summarize_candidate(imgsz, image_rows, object_rows, scaleup)
        for imgsz in sorted(set(args.imgsz_candidates))
    }
    split_summary = {}
    for split in args.splits:
        split_images = [r for r in image_rows if r["split"] == split]
        split_objects = [r for r in object_rows if r["split"] == split]
        split_summary[split] = {
            "images": summarize_images(split_images),
            "objects": summarize_objects(split_objects, names_map),
        }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_yaml": str(data_yaml),
        "dataset_root": str(base),
        "names": names_map,
        "splits": args.splits,
        "preferred_imgsz": args.preferred_imgsz,
        "scaleup": scaleup,
        "warnings": {
            "missing_labels": missing_labels,
            "unreadable_images": unreadable_images,
        },
        "overall": {
            "images": image_summary,
            "objects": object_summary,
            "candidate_imgsz": candidates,
            "recommendation": make_recommendation(
                args.preferred_imgsz, candidates, image_summary, object_summary
            ),
        },
        "by_split": split_summary,
    }


def fmt_num(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_report(report: dict[str, Any]) -> str:
    images = report["overall"]["images"]
    objects = report["overall"]["objects"]
    rec = report["overall"]["recommendation"]
    lines = [
        "# Dataset Resolution and Object Size Audit",
        "",
        f"Generated at: {report['generated_at']}",
        f"Data yaml: `{report['data_yaml']}`",
        f"Dataset root: `{report['dataset_root']}`",
        "",
        "## Summary",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Images | {images.get('count', 0)} |",
        f"| Objects | {objects.get('count', 0)} |",
        f"| Mean width x height | {fmt_num(images.get('width', {}).get('mean'))} x {fmt_num(images.get('height', {}).get('mean'))} |",
        f"| Median width x height | {fmt_num(images.get('width', {}).get('median'))} x {fmt_num(images.get('height', {}).get('median'))} |",
        f"| Long side median / p95 / max | {fmt_num(images.get('long_side', {}).get('median'))} / {fmt_num(images.get('long_side', {}).get('p95'))} / {fmt_num(images.get('long_side', {}).get('max'))} |",
        f"| Object sqrt(area) median / p25 / p75 | {fmt_num(objects.get('sqrt_area', {}).get('median'))} / {fmt_num(objects.get('sqrt_area', {}).get('p25'))} / {fmt_num(objects.get('sqrt_area', {}).get('p75'))} |",
        "",
        "## Candidate imgsz",
        "",
        "| imgsz | downsample images | upsample images | projected object median sqrt(area) | projected <16 px | projected <32 px |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in report["overall"]["candidate_imgsz"].items():
        ratio = item["image_scale_ratio"]
        proj = item["projected_sqrt_area"]
        size_ratio = item["projected_size_ratio"]
        lines.append(
            f"| {key} | {fmt_num(ratio.get('downsample_pct'), 2)}% | "
            f"{fmt_num(ratio.get('upsample_pct'), 2)}% | "
            f"{fmt_num(proj.get('median'))} | "
            f"{fmt_num(size_ratio.get('<16_pct'), 2)}% | "
            f"{fmt_num(size_ratio.get('<32_pct'), 2)}% |"
        )

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"Default imgsz: **{rec['default_imgsz']}**",
            "",
            "Reasons:",
        ]
    )
    lines.extend(f"- {x}" for x in rec.get("reasons", []))
    if rec.get("cautions"):
        lines.append("")
        lines.append("Cautions:")
        lines.extend(f"- {x}" for x in rec["cautions"])

    if objects.get("class_counts"):
        lines.extend(["", "## Class Counts", "", "| Class | Name | Objects |", "|---:|---|---:|"])
        for cid, item in objects["class_counts"].items():
            lines.append(f"| {cid} | {item['name']} | {item['count']} |")

    warnings = report.get("warnings", {})
    lines.extend(
        [
            "",
            "## Scan Warnings",
            "",
            f"- Missing labels: {warnings.get('missing_labels', 0)}",
            f"- Unreadable images: {warnings.get('unreadable_images', 0)}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    report = scan_dataset(args)
    json_text = json.dumps(to_jsonable(report), ensure_ascii=False, indent=2)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json_text + "\n", encoding="utf-8")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(markdown_report(report), encoding="utf-8")

    images = report["overall"]["images"]
    objects = report["overall"]["objects"]
    print(
        "Scanned "
        f"{images.get('count', 0)} images and {objects.get('count', 0)} objects. "
        f"Recommended default imgsz: {report['overall']['recommendation']['default_imgsz']}"
    )
    if args.out_json:
        print(f"JSON: {args.out_json}")
    if args.out_md:
        print(f"Markdown: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
