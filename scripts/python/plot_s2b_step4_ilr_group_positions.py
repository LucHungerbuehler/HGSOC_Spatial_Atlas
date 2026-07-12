"""
Plot paired SNAI1-ac HIGH and LOW positions on established ILR balances.

This is a companion to threshold_ilr_primary_balance_high_vs_low_cohens_d_heatmap.
The heatmap shows the standardized HIGH-minus-LOW contrast. This script shows
where each group sits on the balance axis, so the sign of the ILR coordinate
can be interpreted directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


BASE_OUT = Path(
    r"D:\HGSOC_Spatial_Atlas\05_analysis_ready"
    r"\S2b_CellType_Composition_Correlation\step4_threshold_high_low_diagnostics"
)
INFILE_NAME = "threshold_high_vs_low_cohens_d_per_sample.csv"

PRIMARY_BALANCES = ["b1", "b2", "b3", "b4", "b9"]
BALANCE_LABELS = {
    "b1": "b1 Malignant vs TME",
    "b2": "b2 Stromal vs immune",
    "b3": "b3 CAF vs endothelial",
    "b4": "b4 Myeloid vs lymphoid",
    "b9": "b9 T/NK vs B/plasma",
}

NEGATIVE_SIDE = {
    "b1": "TME",
    "b2": "Immune",
    "b3": "Endothelial",
    "b4": "Lymphoid",
    "b9": "B/plasma",
}
POSITIVE_SIDE = {
    "b1": "Malignant",
    "b2": "Stromal",
    "b3": "CAF",
    "b4": "Myeloid",
    "b9": "T/NK",
}

PALETTE = {
    "HIGH SNAI1-ac": "#CC6F47",
    "LOW SNAI1-ac": "#5477C4",
}
INK = "#1F2430"
MUTED = "#6F768A"
GRID = "#E6E8F0"
AXIS = "#D7DBE7"


def holm_adjust(p_values: pd.Series) -> pd.Series:
    """Holm-Bonferroni correction without adding a statsmodels dependency."""
    out = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().sort_values()
    m = len(valid)
    if m == 0:
        return out
    adjusted = []
    running_max = 0.0
    for rank, (idx, p) in enumerate(valid.items(), start=1):
        val = min((m - rank + 1) * p, 1.0)
        running_max = max(running_max, val)
        adjusted.append((idx, running_max))
    for idx, val in adjusted:
        out.loc[idx] = val
    return out


def star(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def summarize_positions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    long_rows = []
    raw_rows = []
    for balance in PRIMARY_BALANCES:
        subset = df[df["variable"].eq(balance)].copy()
        for threshold, g in subset.groupby("malignant_threshold", sort=True):
            high = pd.to_numeric(g["high_mean"], errors="coerce")
            low = pd.to_numeric(g["low_mean"], errors="coerce")
            diff = high - low
            keep = high.notna() & low.notna()
            high = high[keep]
            low = low[keep]
            diff = diff[keep]

            if len(diff) > 0 and not np.allclose(diff.to_numpy(), 0):
                try:
                    wilcoxon_p = float(stats.wilcoxon(high, low, zero_method="wilcox").pvalue)
                except ValueError:
                    wilcoxon_p = np.nan
            else:
                wilcoxon_p = np.nan

            rows.append(
                {
                    "balance": balance,
                    "balance_label": BALANCE_LABELS[balance],
                    "malignant_threshold": float(threshold),
                    "n_samples": int(len(diff)),
                    "high_median": float(high.median()) if len(high) else np.nan,
                    "high_q25": float(high.quantile(0.25)) if len(high) else np.nan,
                    "high_q75": float(high.quantile(0.75)) if len(high) else np.nan,
                    "low_median": float(low.median()) if len(low) else np.nan,
                    "low_q25": float(low.quantile(0.25)) if len(low) else np.nan,
                    "low_q75": float(low.quantile(0.75)) if len(low) else np.nan,
                    "median_high_minus_low": float(diff.median()) if len(diff) else np.nan,
                    "diff_q25": float(diff.quantile(0.25)) if len(diff) else np.nan,
                    "diff_q75": float(diff.quantile(0.75)) if len(diff) else np.nan,
                    "direction_consistency_high_gt_low": float((diff > 0).mean()) if len(diff) else np.nan,
                    "high_fraction_above_zero": float((high > 0).mean()) if len(high) else np.nan,
                    "low_fraction_above_zero": float((low > 0).mean()) if len(low) else np.nan,
                    "paired_wilcoxon_p": wilcoxon_p,
                }
            )

            for label, values in [("HIGH SNAI1-ac", high), ("LOW SNAI1-ac", low)]:
                long_rows.append(
                    {
                        "balance": balance,
                        "balance_label": BALANCE_LABELS[balance],
                        "malignant_threshold": float(threshold),
                        "group": label,
                        "median": float(values.median()) if len(values) else np.nan,
                        "q25": float(values.quantile(0.25)) if len(values) else np.nan,
                        "q75": float(values.quantile(0.75)) if len(values) else np.nan,
                        "n_samples": int(len(values)),
                    }
                )

            for _, sample_row in g.loc[keep].iterrows():
                raw_rows.append(
                    {
                        "dataset": sample_row["dataset"],
                        "sample": sample_row["sample"],
                        "sample_label": sample_row["sample_label"],
                        "balance": balance,
                        "balance_label": BALANCE_LABELS[balance],
                        "malignant_threshold": float(threshold),
                        "low_mean": float(sample_row["low_mean"]),
                        "high_mean": float(sample_row["high_mean"]),
                        "high_minus_low": float(sample_row["high_mean"] - sample_row["low_mean"]),
                    }
                )

    summary = pd.DataFrame(rows)
    summary["paired_wilcoxon_p_holm"] = holm_adjust(summary["paired_wilcoxon_p"])
    summary["paired_wilcoxon_sig_holm"] = summary["paired_wilcoxon_p_holm"].map(star)
    positions = pd.DataFrame(long_rows)
    raw = pd.DataFrame(raw_rows)
    return summary, positions, raw


def threshold_ticks(df: pd.DataFrame) -> list[int]:
    return sorted((df["malignant_threshold"].dropna().astype(float) * 100).round().astype(int).unique().tolist())


def plot_positions(summary: pd.DataFrame, positions: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(8.2, 11.0), sharex=True)
    fig.patch.set_facecolor("#FCFCFD")

    for ax, balance in zip(axes, PRIMARY_BALANCES):
        ax.set_facecolor("#FFFFFF")
        plot_df = positions[positions["balance"].eq(balance)].sort_values("malignant_threshold")
        stat_df = summary[summary["balance"].eq(balance)].sort_values("malignant_threshold")

        for group, group_df in plot_df.groupby("group", sort=False):
            group_df = group_df.sort_values("malignant_threshold")
            x = group_df["malignant_threshold"].to_numpy(dtype=float) * 100
            y = group_df["median"].to_numpy(dtype=float)
            y1 = group_df["q25"].to_numpy(dtype=float)
            y2 = group_df["q75"].to_numpy(dtype=float)
            ax.fill_between(x, y1, y2, color=PALETTE[group], alpha=0.12, linewidth=0)
            ax.plot(
                x,
                y,
                color=PALETTE[group],
                marker="o",
                markersize=4.5,
                linewidth=2.0,
                label=group,
            )

        ax.axhline(0, color=INK, linewidth=0.9, alpha=0.55)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.grid(axis="x", color=GRID, linewidth=0.4, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(AXIS)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_ylabel("ILR balance", color=INK, fontsize=9)

        y_min, y_max = ax.get_ylim()
        span = y_max - y_min
        ax.text(
            1.012,
            0.90,
            POSITIVE_SIDE[balance],
            ha="left",
            va="top",
            color=MUTED,
            fontsize=8.5,
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.text(
            1.012,
            0.10,
            NEGATIVE_SIDE[balance],
            ha="left",
            va="bottom",
            color=MUTED,
            fontsize=8.5,
            transform=ax.transAxes,
            clip_on=False,
        )

        for _, row in stat_df.iterrows():
            sig = row["paired_wilcoxon_sig_holm"]
            if sig:
                ax.text(
                    row["malignant_threshold"] * 100,
                    y_max - 0.08 * span,
                    sig,
                    ha="center",
                    va="top",
                    color=INK,
                    fontsize=9,
                    fontweight="bold",
                )

        ax.set_title(BALANCE_LABELS[balance], loc="left", fontsize=11, color=INK, pad=5)

    axes[-1].set_xlabel("Minimum malignant fraction in analysed spots (%)", color=INK, fontsize=10)
    axes[-1].set_xticks(threshold_ticks(positions))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.982),
        ncol=2,
        frameon=False,
        fontsize=9.5,
    )
    fig.subplots_adjust(left=0.12, right=0.87, top=0.91, bottom=0.06, hspace=0.52)

    fig.savefig(out_dir / "threshold_ilr_primary_balance_high_low_positions.png", dpi=300)
    fig.savefig(out_dir / "threshold_ilr_primary_balance_high_low_positions.pdf")
    plt.close(fig)


def plot_box_positions(summary: pd.DataFrame, raw: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(8.4, 12.0), sharex=True)
    fig.patch.set_facecolor("#FCFCFD")

    offsets = {"LOW SNAI1-ac": -1.6, "HIGH SNAI1-ac": 1.6}
    width = 2.35

    for ax, balance in zip(axes, PRIMARY_BALANCES):
        ax.set_facecolor("#FFFFFF")
        balance_raw = raw[raw["balance"].eq(balance)].copy()
        balance_stats = summary[summary["balance"].eq(balance)].copy()

        for threshold in sorted(balance_raw["malignant_threshold"].unique()):
            g = balance_raw[balance_raw["malignant_threshold"].eq(threshold)]
            x_base = threshold * 100

            low = g["low_mean"].to_numpy(dtype=float)
            high = g["high_mean"].to_numpy(dtype=float)
            low_x = x_base + offsets["LOW SNAI1-ac"]
            high_x = x_base + offsets["HIGH SNAI1-ac"]

            for lval, hval in zip(low, high):
                ax.plot([low_x, high_x], [lval, hval], color="#C5CAD3", linewidth=0.65, alpha=0.55, zorder=1)

            bp = ax.boxplot(
                [low, high],
                positions=[low_x, high_x],
                widths=width,
                patch_artist=True,
                manage_ticks=False,
                showfliers=False,
                medianprops={"color": INK, "linewidth": 1.25},
                whiskerprops={"color": "#7A828F", "linewidth": 0.9},
                capprops={"color": "#7A828F", "linewidth": 0.9},
                boxprops={"edgecolor": "#7A828F", "linewidth": 0.9},
            )
            bp["boxes"][0].set_facecolor(PALETTE["LOW SNAI1-ac"])
            bp["boxes"][0].set_alpha(0.30)
            bp["boxes"][1].set_facecolor(PALETTE["HIGH SNAI1-ac"])
            bp["boxes"][1].set_alpha(0.30)

            rng = np.random.default_rng(int(threshold * 1000) + PRIMARY_BALANCES.index(balance))
            low_jitter = rng.normal(0, 0.18, size=len(low))
            high_jitter = rng.normal(0, 0.18, size=len(high))
            ax.scatter(
                np.full(len(low), low_x) + low_jitter,
                low,
                s=10,
                color=PALETTE["LOW SNAI1-ac"],
                alpha=0.65,
                linewidths=0,
                zorder=3,
            )
            ax.scatter(
                np.full(len(high), high_x) + high_jitter,
                high,
                s=10,
                color=PALETTE["HIGH SNAI1-ac"],
                alpha=0.65,
                linewidths=0,
                zorder=3,
            )

        ax.axhline(0, color=INK, linewidth=0.9, alpha=0.55)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(AXIS)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_ylabel("ILR balance", color=INK, fontsize=9)

        y_min, y_max = ax.get_ylim()
        span = y_max - y_min
        ax.text(
            1.012,
            0.90,
            POSITIVE_SIDE[balance],
            ha="left",
            va="top",
            color=MUTED,
            fontsize=8.5,
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.text(
            1.012,
            0.10,
            NEGATIVE_SIDE[balance],
            ha="left",
            va="bottom",
            color=MUTED,
            fontsize=8.5,
            transform=ax.transAxes,
            clip_on=False,
        )

        for _, row in balance_stats.iterrows():
            sig = row["paired_wilcoxon_sig_holm"]
            if sig:
                ax.text(
                    row["malignant_threshold"] * 100,
                    y_max - 0.08 * span,
                    sig,
                    ha="center",
                    va="top",
                    color=INK,
                    fontsize=9,
                    fontweight="bold",
                )

        ax.set_title(BALANCE_LABELS[balance], loc="left", fontsize=11, color=INK, pad=5)
        ticks = threshold_ticks(raw)
        ax.set_xlim(min(ticks) - 4, max(ticks) + 4)

    axes[-1].set_xlabel("Minimum malignant fraction in analysed spots (%)", color=INK, fontsize=10)
    axes[-1].set_xticks(threshold_ticks(raw))

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=8, markerfacecolor=PALETTE["LOW SNAI1-ac"], alpha=0.45, markeredgecolor="#7A828F"),
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=8, markerfacecolor=PALETTE["HIGH SNAI1-ac"], alpha=0.45, markeredgecolor="#7A828F"),
        plt.Line2D([0], [0], color="#C5CAD3", linewidth=1.0),
    ]
    labels = ["LOW SNAI1-ac", "HIGH SNAI1-ac", "Same sample pair"]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.982),
        ncol=3,
        frameon=False,
        fontsize=9.5,
    )
    fig.subplots_adjust(left=0.12, right=0.87, top=0.91, bottom=0.06, hspace=0.54)

    fig.savefig(out_dir / "threshold_ilr_primary_balance_high_low_positions_boxplot.png", dpi=300)
    fig.savefig(out_dir / "threshold_ilr_primary_balance_high_low_positions_boxplot.pdf")
    plt.close(fig)


def main(output_subdir: str | None = None) -> None:
    out_dir = BASE_OUT / output_subdir if output_subdir else BASE_OUT
    infile = out_dir / INFILE_NAME
    df = pd.read_csv(infile)
    df = df[df["variable_type"].eq("ilr_balance") & df["variable"].isin(PRIMARY_BALANCES)].copy()
    df["malignant_threshold"] = pd.to_numeric(df["malignant_threshold"], errors="coerce")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary, positions, raw = summarize_positions(df)
    summary.to_csv(out_dir / "threshold_ilr_primary_balance_high_low_positions_stats.csv", index=False)
    positions.to_csv(out_dir / "threshold_ilr_primary_balance_high_low_positions_summary.csv", index=False)
    raw.to_csv(out_dir / "threshold_ilr_primary_balance_high_low_positions_per_sample.csv", index=False)
    plot_positions(summary, positions, out_dir)
    plot_box_positions(summary, raw, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-subdir",
        default=None,
        help="Optional subfolder under the step4 threshold diagnostics directory containing the input table and receiving outputs.",
    )
    args = parser.parse_args()
    main(output_subdir=args.output_subdir)
