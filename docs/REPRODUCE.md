# Reproduction Guide

This guide describes the intended workflow for reproducing MG-AFSS experiments.

## 1. Patch Ultralytics

Apply the files under `ultralytics_patch/` to a compatible Ultralytics checkout:

```text
ultralytics_patch/ultralytics/afss/                         -> ultralytics/afss/
ultralytics_patch/ultralytics/models/yolo/detect/afss_*.py  -> ultralytics/models/yolo/detect/
ultralytics_patch/ultralytics/models/yolo/obb/afss_*.py     -> ultralytics/models/yolo/obb/
ultralytics_patch/config_patch.md                           -> add options to ultralytics/cfg/default.yaml
```

Then set:

```bash
export ULTRALYTICS_PATH=/path/to/patched/ultralytics-repo
```

On Windows PowerShell:

```powershell
$env:ULTRALYTICS_PATH="<path-to-patched-ultralytics-repo>"
```

## 2. Check Environment

```bash
python tools/check_env.py
```

The manuscript experiments used PyTorch 2.5.1+cu121 and CUDA 12.1. Use a recent PyTorch version that provides `torch.nn.functional.scaled_dot_product_attention`.

After applying the patch, the focused tests can be run from this release directory by putting the patched Ultralytics runtime on `PYTHONPATH`:

```bash
export PYTHONPATH=$ULTRALYTICS_PATH:$PYTHONPATH
python -m pytest tests
```

## 3. Prepare Datasets

Prepare Ultralytics-compatible dataset yaml files. See `docs/DATASETS.md`.

## 4. Run Matched Training Blocks

Run standard training, fixed AFSS, and MG-AFSS under matched conditions:

- Same dataset protocol.
- Same model size.
- Same initialization condition.
- Same epoch budget.
- Same image size.
- Same batch size.
- Same seed.
- Same runtime and device.

Use:

```bash
python -m mg_afss.run_mg_afss_train --method off ...
python -m mg_afss.run_mg_afss_train --method afss --warmup 20 ...
python -m mg_afss.run_mg_afss_train --method mg_afss --mg-fit-min-points 5 ...
```

## 5. Collect Results

Each run writes:

| File | Role |
|---|---|
| `summary.json` | Run metadata and best/final metric summary |
| `results.csv` | Per-epoch Ultralytics metrics |
| `afss_stats.json` | AFSS active-ratio and difficulty-state records |
| `v3_log.json` | MG-AFSS maturity-gate log; field name retained for compatibility |
| `config.json` | CLI argument record |
| `command.txt` | Reconstructed command line |

Compile summaries with:

```bash
python tools/collect_results_from_summary.py
```

## 6. Generate Example Paper Figure

The release candidate includes processed aggregate data for the introductory accuracy-vs-cost figure:

```bash
python paper_artifacts/figure_scripts/generate_accuracy_vs_cost.py
```

The script reads `paper_artifacts/processed_results/fig_acc_vs_cost_data.csv` and writes figure files into `paper_artifacts/figure_scripts/`.

## Notes

- The public package uses `mg_afss` as the method name. Some generated JSON fields retain `v3_*` names because the development experiments used that historical internal label.
- Do not compare wall-clock times across different devices or different runtime conditions.
- For manuscript-style comparisons, prefer checkpoint-best metrics matched to the same standard-training baseline.
