"""
Generate the introductory accuracy-vs-cost figure for the submission package.

The figure is intentionally schematic: it shows the relative-time position of
standard training, AFSS, and MG-AFSS within matched experimental blocks without
repeating exact table values inside the plot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


PAPER_ARTIFACTS_DIR = Path(__file__).resolve().parents[1]
FIG_DIR = Path(__file__).resolve().parent
DATA_CSV = PAPER_ARTIFACTS_DIR / "processed_results" / "fig_acc_vs_cost_data.csv"

FONT_FAMILY = ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]
TOKENS = {
    "surface": "#FFFFFF",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#60687A",
    "grid": "#E2E5EC",
    "axis": "#D1D6E0",
}
NEUTRAL = {"base": "#B8BEC8", "dark": "#4B515C"}
BLUE = {"base": "#77A1F7", "dark": "#2E5EB8"}
ORANGE = {"base": "#F39A6B", "dark": "#B85C32"}

METHOD_STYLE = {
    "Standard training": {"color": NEUTRAL["base"], "edge": NEUTRAL["dark"], "marker": "s"},
    "AFSS": {"color": ORANGE["base"], "edge": ORANGE["dark"], "marker": "o"},
    "MG-AFSS": {"color": BLUE["base"], "edge": BLUE["dark"], "marker": "^"},
}

E1_ROWS = [
    {"block_short": "E1 NWPU scratch", "method": "Standard training", "map50": 0.8161, "map5095": 0.4893, "time_ratio": 1.000},
    {"block_short": "E1 NWPU scratch", "method": "AFSS", "map50": 0.7878, "map5095": 0.4577, "time_ratio": 0.690},
    {"block_short": "E1 NWPU scratch", "method": "MG-AFSS", "map50": 0.8142, "map5095": 0.4896, "time_ratio": 0.867},
]

NWPU200_ROWS = [
    {"block_short": "E3 NWPU scratch 200ep", "method": "Standard training", "map50": 0.8947, "map5095": 0.5773, "time_ratio": 1.000},
    {"block_short": "E3 NWPU scratch 200ep", "method": "AFSS", "map50": 0.8977, "map5095": 0.5631, "time_ratio": 0.531},
    {"block_short": "E3 NWPU scratch 200ep", "method": "MG-AFSS", "map50": 0.8997, "map5095": 0.5762, "time_ratio": 0.820},
]


def use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.75,
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
        },
    )


def build_blocks() -> pd.DataFrame:
    return pd.read_csv(DATA_CSV).sort_values(["block_short", "method"]).reset_index(drop=True)


def _method_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=METHOD_STYLE[m]["marker"],
            color="none",
            markerfacecolor=METHOD_STYLE[m]["color"],
            markeredgecolor=METHOD_STYLE[m]["edge"],
            markeredgewidth=1.2,
            markersize=8.5,
            label=m,
        )
        for m in METHOD_STYLE
    ]


def _panel_limits(values: pd.Series, *, low_pad: float = 0.18, high_pad: float = 0.18) -> tuple[float, float]:
    vmin, vmax = float(values.min()), float(values.max())
    span = vmax - vmin
    if span < 1e-9:
        span = abs(vmax) * 0.05 or 0.05
    return vmin - span * low_pad, vmax + span * high_pad


def draw_relative_cost(df: pd.DataFrame) -> None:
    blocks = ["E1 NWPU scratch", "E3 NWPU pretrained", "E3 NWPU scratch 200ep"]
    titles = {
        "E1 NWPU scratch": "(a) NWPU scratch, 80 epochs",
        "E3 NWPU pretrained": "(b) NWPU pretrained, 80 epochs",
        "E3 NWPU scratch 200ep": "(c) NWPU scratch, 200 epochs",
    }
    metrics = [
        ("map50", "Best mAP50"),
        ("map5095", "Best mAP50-95"),
    ]
    # Match the canvas to the manuscript text width and keep the six-panel
    # figure tall enough for legible tick labels after LaTeX scaling.
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.35), sharex=True)

    x_min = max(0.25, df["time_ratio"].min() - 0.12)
    x_max = min(1.08, df["time_ratio"].max() + 0.05)

    for row_idx, (metric, ylabel) in enumerate(metrics):
        for col_idx, blk in enumerate(blocks):
            ax = axes[row_idx, col_idx]
            sub = df[df["block_short"] == blk].copy()
            ax.axvline(1.0, color=TOKENS["muted"], linestyle=":", linewidth=1.0, zorder=1)

            for method, style in METHOD_STYLE.items():
                row = sub[sub["method"] == method]
                if row.empty:
                    continue
                ax.scatter(
                    row["time_ratio"],
                    row[metric],
                    s=54,
                    marker=style["marker"],
                    facecolor=style["color"],
                    edgecolor=style["edge"],
                    linewidth=1.0,
                    zorder=5,
                )

            y_min, y_max = _panel_limits(sub[metric], low_pad=0.24, high_pad=0.22)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            if row_idx == 0:
                ax.set_title(titles[blk], fontsize=8.5, color=TOKENS["ink"], pad=5)
            ax.xaxis.set_major_locator(mticker.FixedLocator([0.50, 0.75, 1.00]))
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
            ax.tick_params(axis="both", colors=TOKENS["muted"], labelsize=7.1, pad=1.5)
            ax.grid(axis="y", linestyle="-", color=TOKENS["grid"])
            ax.grid(axis="x", visible=False)
            sns.despine(ax=ax)

        axes[row_idx, 0].set_ylabel(ylabel, fontsize=8.2)

    fig.legend(
        handles=_method_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        frameon=False,
        ncol=3,
        fontsize=8.1,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    fig.supxlabel("Time ratio relative to standard training", y=0.055, fontsize=8.4, color=TOKENS["ink"])
    fig.subplots_adjust(left=0.077, right=0.992, bottom=0.14, top=0.875, wspace=0.27, hspace=0.38)

    # Keep both filenames current so older references cannot silently pick up a stale image.
    for name in ("fig_acc_vs_cost_ratio", "fig_acc_vs_cost_abs"):
        fig.savefig(FIG_DIR / f"{name}.png", dpi=600, bbox_inches="tight")
        fig.savefig(FIG_DIR / f"{name}.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    use_chart_theme()
    df = build_blocks()
    print(df[["block_short", "method", "map50", "map5095", "time_ratio"]].to_string(index=False))
    draw_relative_cost(df)
    print("Saved to", FIG_DIR)


if __name__ == "__main__":
    main()
