"""Generate local data.local.yaml files with workspace-correct dataset paths.

The original copied data.yaml files may contain absolute paths from old devices.
By default this script writes a sibling data.local.yaml and leaves the original
data.yaml untouched.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

RELEASE_ROOT = Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

from mg_afss.paths import get_data_root


def as_posix(path: Path) -> str:
    return path.resolve().as_posix()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def split_exists(dataset_dir: Path, split_value) -> str:
    if not split_value:
        return ""
    if isinstance(split_value, list):
        states = [split_exists(dataset_dir, item) for item in split_value]
        return "ok" if all(s == "ok" for s in states) else "missing"
    split_path = Path(str(split_value))
    if not split_path.is_absolute():
        split_path = dataset_dir / split_path
    return "ok" if split_path.exists() else "missing"


def prepare_one(data_yaml: Path, output_name: str, in_place: bool, dry_run: bool) -> dict:
    data_yaml = data_yaml.resolve()
    dataset_dir = data_yaml.parent
    data = load_yaml(data_yaml)
    old_path = data.get("path", "")
    data["path"] = as_posix(dataset_dir)

    out_path = data_yaml if in_place else data_yaml.with_name(output_name)
    rendered = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    changed = True
    if out_path.exists():
        changed = out_path.read_text(encoding="utf-8") != rendered

    if not dry_run and changed:
        out_path.write_text(rendered, encoding="utf-8")

    return {
        "source": data_yaml,
        "output": out_path,
        "old_path": old_path,
        "new_path": data["path"],
        "changed": changed,
        "train": split_exists(dataset_dir, data.get("train")),
        "val": split_exists(dataset_dir, data.get("val")),
        "test": split_exists(dataset_dir, data.get("test")),
    }


def find_data_yamls(dataset_root: Path) -> list[Path]:
    return sorted(p for p in dataset_root.rglob("data.yaml") if p.is_file())


def resolve_targets(args) -> list[Path]:
    dataset_root = get_data_root()
    targets = []
    if args.all:
        targets.extend(find_data_yamls(dataset_root))
    for name in args.dataset or []:
        targets.append(dataset_root / name / "data.yaml")
    for path in args.data_yaml or []:
        targets.append(path if path.is_absolute() else Path.cwd() / path)
    unique = []
    seen = set()
    for path in targets:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description="Create local data yaml files for this workspace.")
    parser.add_argument("--all", action="store_true", help="Process every dataset/**/data.yaml file.")
    parser.add_argument("--dataset", action="append", help="Dataset name under dataset/, repeatable.")
    parser.add_argument("--data-yaml", action="append", type=Path, help="Explicit data.yaml path, repeatable.")
    parser.add_argument("--output-name", default="data.local.yaml")
    parser.add_argument("--in-place", action="store_true", help="Rewrite data.yaml instead of creating data.local.yaml.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    targets = resolve_targets(args)
    if not targets:
        parser.error("Provide --all, --dataset NAME, or --data-yaml PATH.")

    ok = 0
    missing = 0
    for target in targets:
        if not target.exists():
            missing += 1
            print(f"[MISS] {target}")
            continue
        try:
            info = prepare_one(target, args.output_name, args.in_place, args.dry_run)
        except Exception as exc:
            missing += 1
            print(f"[FAIL] {target}: {type(exc).__name__}: {exc}")
            continue
        ok += 1
        marker = "DRY" if args.dry_run else ("WRITE" if info["changed"] else "OK")
        checks = ", ".join(f"{k}={v}" for k, v in [("train", info["train"]), ("val", info["val"]), ("test", info["test"])] if v)
        print(f"[{marker}] {info['output']}  path: {info['old_path']} -> {info['new_path']}  {checks}")

    print(f"Processed: ok={ok}, missing_or_failed={missing}")
    return 0 if missing == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
