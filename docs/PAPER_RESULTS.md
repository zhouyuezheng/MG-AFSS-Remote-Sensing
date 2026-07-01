# Processed Paper Results

This release candidate includes a small set of processed numerical summaries used for paper figures. These files contain aggregate metrics, not raw datasets or trained checkpoints.

## Included Files

| File | Purpose |
|---|---|
| `paper_artifacts/processed_results/fig_acc_vs_cost_data.csv` | Data for the introductory accuracy-vs-cost figure |
| `paper_artifacts/processed_results/fig_cumulative_fit_delayed_seed42_data.csv` | Cumulative-fit example for a later-trigger NWPU run |
| `paper_artifacts/processed_results/fig_cumulative_fit_early_pt80_seed42_data.csv` | Cumulative-fit example for an earlier-trigger pretrained NWPU run |
| `paper_artifacts/processed_results/fig_stability_diagnostic_data.csv` | Processed transition probabilities for the difficulty-state diagnostic |

## Figure Script

`paper_artifacts/figure_scripts/generate_accuracy_vs_cost.py` can regenerate the introductory accuracy-vs-cost figure from `fig_acc_vs_cost_data.csv`.

The qualitative detection figure is not included because it depends on local third-party dataset images and trained checkpoints that are not redistributed in this code release candidate.

## Interpreting Method Labels

The manuscript uses:

- `Standard training`
- `AFSS`
- `MG-AFSS`

Historical internal run IDs may contain `V3fix2_cumul_m40`. In the paper and this release candidate, that internal label corresponds to MG-AFSS with cumulative validation-history fitting and moderate-image sampling ratio 0.40.
