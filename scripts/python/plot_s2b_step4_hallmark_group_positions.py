"""
Plot grouped Hallmark HIGH/LOW positions across malignant-fraction thresholds.

Inputs are the already-generated threshold HIGH-vs-LOW contrast table from
Section 2.4 step 4 diagnostics. Sample is the unit of replication: for each
sample, threshold, and Hallmark, SNAI1-ac HIGH and LOW group means are paired.
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

HALLMARK_GROUPS = {
    "proliferation_biosynthesis": [
        "HALLMARK_MYC_TARGETS_V1_score",
        "HALLMARK_E2F_TARGETS_score",
        "HALLMARK_G2M_CHECKPOINT_score",
        "HALLMARK_MTORC1_SIGNALING_score",
        "HALLMARK_DNA_REPAIR_score",
    ],
    "metabolism_stress": [
        "HALLMARK_OXIDATIVE_PHOSPHORYLATION_score",
        "HALLMARK_GLYCOLYSIS_score",
        "HALLMARK_HYPOXIA_score",
        "HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY_score",
        "HALLMARK_PI3K_AKT_MTOR_SIGNALING_score",
    ],
    "emt_inflammatory_signaling": [
        "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION_score",
        "HALLMARK_TGF_BETA_SIGNALING_score",
        "HALLMARK_TNFA_SIGNALING_VIA_NFKB_score",
        "HALLMARK_IL6_JAK_STAT3_SIGNALING_score",
        "HALLMARK_KRAS_SIGNALING_UP_score",
    ],
    "epithelial_differentiation_context": [
        "HALLMARK_APICAL_JUNCTION_score",
        "HALLMARK_ESTROGEN_RESPONSE_EARLY_score",
        "HALLMARK_ESTROGEN_RESPONSE_LATE_score",
        "HALLMARK_NOTCH_SIGNALING_score",
        "HALLMARK_ADIPOGENESIS_score",
    ],
}

REMAINING_HALLMARK_GROUPS = {
    "immune_interferon": [
        "HALLMARK_ALLOGRAFT_REJECTION_score",
        "HALLMARK_COMPLEMENT_score",
        "HALLMARK_INFLAMMATORY_RESPONSE_score",
        "HALLMARK_INTERFERON_ALPHA_RESPONSE_score",
        "HALLMARK_INTERFERON_GAMMA_RESPONSE_score",
    ],
    "cytokine_stress_fate": [
        "HALLMARK_IL2_STAT5_SIGNALING_score",
        "HALLMARK_KRAS_SIGNALING_DN_score",
        "HALLMARK_P53_PATHWAY_score",
        "HALLMARK_APOPTOSIS_score",
        "HALLMARK_PROTEIN_SECRETION_score",
    ],
    "vascular_stromal_development": [
        "HALLMARK_ANGIOGENESIS_score",
        "HALLMARK_COAGULATION_score",
        "HALLMARK_MYOGENESIS_score",
        "HALLMARK_HEDGEHOG_SIGNALING_score",
        "HALLMARK_WNT_BETA_CATENIN_SIGNALING_score",
    ],
    "lipid_xenobiotic_metabolism": [
        "HALLMARK_CHOLESTEROL_HOMEOSTASIS_score",
        "HALLMARK_FATTY_ACID_METABOLISM_score",
        "HALLMARK_BILE_ACID_METABOLISM_score",
        "HALLMARK_PEROXISOME_score",
        "HALLMARK_XENOBIOTIC_METABOLISM_score",
    ],
    "cellular_stress_structure": [
        "HALLMARK_HEME_METABOLISM_score",
        "HALLMARK_UNFOLDED_PROTEIN_RESPONSE_score",
        "HALLMARK_UV_RESPONSE_UP_score",
        "HALLMARK_UV_RESPONSE_DN_score",
        "HALLMARK_MITOTIC_SPINDLE_score",
    ],
    "remaining_context": [
        "HALLMARK_ANDROGEN_RESPONSE_score",
        "HALLMARK_APICAL_SURFACE_score",
        "HALLMARK_MYC_TARGETS_V2_score",
        "HALLMARK_PANCREAS_BETA_CELLS_score",
        "HALLMARK_SPERMATOGENESIS_score",
    ],
}

STRICT_CONSISTENT_HALLMARK_GROUPS = {
    "strict7_consistent": [
        "HALLMARK_ANDROGEN_RESPONSE_score",
        "HALLMARK_ESTROGEN_RESPONSE_LATE_score",
        "HALLMARK_HEDGEHOG_SIGNALING_score",
        "HALLMARK_ADIPOGENESIS_score",
        "HALLMARK_IL2_STAT5_SIGNALING_score",
        "HALLMARK_NOTCH_SIGNALING_score",
        "HALLMARK_OXIDATIVE_PHOSPHORYLATION_score",
    ],
}

GROUP_TITLES = {
    "proliferation_biosynthesis": "Proliferation and biosynthesis",
    "metabolism_stress": "Metabolism and stress",
    "emt_inflammatory_signaling": "EMT and inflammatory signaling",
    "epithelial_differentiation_context": "Epithelial and differentiation context",
    "immune_interferon": "Immune and interferon",
    "cytokine_stress_fate": "Cytokine, stress, and cell fate",
    "vascular_stromal_development": "Vascular, stromal, and development",
    "lipid_xenobiotic_metabolism": "Lipid and xenobiotic metabolism",
    "cellular_stress_structure": "Cellular stress and structure",
    "remaining_context": "Remaining context pathways",
    "strict7_consistent": "Strictly consistent Hallmark contrasts",
}

LABELS = {
    "HALLMARK_MYC_TARGETS_V1_score": "MYC targets V1",
    "HALLMARK_E2F_TARGETS_score": "E2F targets",
    "HALLMARK_G2M_CHECKPOINT_score": "G2M checkpoint",
    "HALLMARK_MTORC1_SIGNALING_score": "mTORC1 signaling",
    "HALLMARK_DNA_REPAIR_score": "DNA repair",
    "HALLMARK_OXIDATIVE_PHOSPHORYLATION_score": "Oxidative phosphorylation",
    "HALLMARK_GLYCOLYSIS_score": "Glycolysis",
    "HALLMARK_HYPOXIA_score": "Hypoxia",
    "HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY_score": "Reactive oxygen species",
    "HALLMARK_PI3K_AKT_MTOR_SIGNALING_score": "PI3K/AKT/mTOR signaling",
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION_score": "EMT",
    "HALLMARK_TGF_BETA_SIGNALING_score": "TGF-beta signaling",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB_score": "TNF-alpha via NF-kB",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING_score": "IL6/JAK/STAT3 signaling",
    "HALLMARK_KRAS_SIGNALING_UP_score": "KRAS signaling up",
    "HALLMARK_APICAL_JUNCTION_score": "Apical junction",
    "HALLMARK_ESTROGEN_RESPONSE_EARLY_score": "Estrogen response early",
    "HALLMARK_ESTROGEN_RESPONSE_LATE_score": "Estrogen response late",
    "HALLMARK_NOTCH_SIGNALING_score": "Notch signaling",
    "HALLMARK_ADIPOGENESIS_score": "Adipogenesis",
    "HALLMARK_ALLOGRAFT_REJECTION_score": "Allograft rejection",
    "HALLMARK_COMPLEMENT_score": "Complement",
    "HALLMARK_INFLAMMATORY_RESPONSE_score": "Inflammatory response",
    "HALLMARK_INTERFERON_ALPHA_RESPONSE_score": "Interferon alpha response",
    "HALLMARK_INTERFERON_GAMMA_RESPONSE_score": "Interferon gamma response",
    "HALLMARK_IL2_STAT5_SIGNALING_score": "IL2 STAT5 signaling",
    "HALLMARK_KRAS_SIGNALING_DN_score": "KRAS signaling DN",
    "HALLMARK_P53_PATHWAY_score": "P53 pathway",
    "HALLMARK_APOPTOSIS_score": "Apoptosis",
    "HALLMARK_PROTEIN_SECRETION_score": "Protein secretion",
    "HALLMARK_ANGIOGENESIS_score": "Angiogenesis",
    "HALLMARK_COAGULATION_score": "Coagulation",
    "HALLMARK_MYOGENESIS_score": "Myogenesis",
    "HALLMARK_HEDGEHOG_SIGNALING_score": "Hedgehog signaling",
    "HALLMARK_WNT_BETA_CATENIN_SIGNALING_score": "WNT beta catenin signaling",
    "HALLMARK_CHOLESTEROL_HOMEOSTASIS_score": "Cholesterol homeostasis",
    "HALLMARK_FATTY_ACID_METABOLISM_score": "Fatty acid metabolism",
    "HALLMARK_BILE_ACID_METABOLISM_score": "Bile acid metabolism",
    "HALLMARK_PEROXISOME_score": "Peroxisome",
    "HALLMARK_XENOBIOTIC_METABOLISM_score": "Xenobiotic metabolism",
    "HALLMARK_HEME_METABOLISM_score": "Heme metabolism",
    "HALLMARK_UNFOLDED_PROTEIN_RESPONSE_score": "Unfolded protein response",
    "HALLMARK_UV_RESPONSE_UP_score": "UV response up",
    "HALLMARK_UV_RESPONSE_DN_score": "UV response DN",
    "HALLMARK_MITOTIC_SPINDLE_score": "Mitotic spindle",
    "HALLMARK_ANDROGEN_RESPONSE_score": "Androgen response",
    "HALLMARK_APICAL_SURFACE_score": "Apical surface",
    "HALLMARK_MYC_TARGETS_V2_score": "MYC targets V2",
    "HALLMARK_PANCREAS_BETA_CELLS_score": "Pancreas beta cells",
    "HALLMARK_SPERMATOGENESIS_score": "Spermatogenesis",
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
    out = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().sort_values()
    m = len(valid)
    running_max = 0.0
    for rank, (idx, p) in enumerate(valid.items(), start=1):
        val = min((m - rank + 1) * p, 1.0)
        running_max = max(running_max, val)
        out.loc[idx] = running_max
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


def summarize(df: pd.DataFrame, hallmark_groups: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    long_rows = []
    raw_rows = []
    for group_name, variables in hallmark_groups.items():
        for variable in variables:
            subset = df[df["variable"].eq(variable)].copy()
            for threshold, g in subset.groupby("malignant_threshold", sort=True):
                high = pd.to_numeric(g["high_mean"], errors="coerce")
                low = pd.to_numeric(g["low_mean"], errors="coerce")
                keep = high.notna() & low.notna()
                high = high[keep]
                low = low[keep]
                diff = high - low

                if len(diff) > 0 and not np.allclose(diff.to_numpy(), 0):
                    try:
                        wilcoxon_p = float(stats.wilcoxon(high, low, zero_method="wilcox").pvalue)
                    except ValueError:
                        wilcoxon_p = np.nan
                else:
                    wilcoxon_p = np.nan

                rows.append(
                    {
                        "figure_group": group_name,
                        "figure_group_title": GROUP_TITLES[group_name],
                        "variable": variable,
                        "pathway_label": LABELS[variable],
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
                        "paired_wilcoxon_p": wilcoxon_p,
                    }
                )

                for label, values in [("HIGH SNAI1-ac", high), ("LOW SNAI1-ac", low)]:
                    long_rows.append(
                        {
                            "figure_group": group_name,
                            "figure_group_title": GROUP_TITLES[group_name],
                            "variable": variable,
                            "pathway_label": LABELS[variable],
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
                            "figure_group": group_name,
                            "figure_group_title": GROUP_TITLES[group_name],
                            "variable": variable,
                            "pathway_label": LABELS[variable],
                            "malignant_threshold": float(threshold),
                            "low_mean": float(sample_row["low_mean"]),
                            "high_mean": float(sample_row["high_mean"]),
                            "high_minus_low": float(sample_row["high_mean"] - sample_row["low_mean"]),
                        }
                    )

    summary = pd.DataFrame(rows)
    summary["paired_wilcoxon_p_holm_all_displayed"] = holm_adjust(summary["paired_wilcoxon_p"])
    summary["paired_wilcoxon_sig_holm_all_displayed"] = summary["paired_wilcoxon_p_holm_all_displayed"].map(star)
    for group_name, idx in summary.groupby("figure_group").groups.items():
        group_adj = holm_adjust(summary.loc[idx, "paired_wilcoxon_p"])
        summary.loc[idx, "paired_wilcoxon_p_holm_within_figure"] = group_adj
        summary.loc[idx, "paired_wilcoxon_sig_holm_within_figure"] = group_adj.map(star)
    return summary, pd.DataFrame(long_rows), pd.DataFrame(raw_rows)


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("#FFFFFF")
    ax.axhline(0, color=INK, linewidth=0.9, alpha=0.55)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", color=GRID, linewidth=0.4, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_ylabel("Hallmark score", color=INK, fontsize=9)


def threshold_ticks(df: pd.DataFrame) -> list[int]:
    return sorted((df["malignant_threshold"].dropna().astype(float) * 100).round().astype(int).unique().tolist())


def plot_line_group(
    group_name: str,
    positions: pd.DataFrame,
    summary: pd.DataFrame,
    out_dir: Path,
    hallmark_groups: dict[str, list[str]],
    png_only: bool = False,
) -> None:
    variables = hallmark_groups[group_name]
    fig_height = max(9.6, 1.9 * len(variables))
    fig, axes = plt.subplots(len(variables), 1, figsize=(8.2, fig_height), sharex=True)
    fig.patch.set_facecolor("#FCFCFD")
    if len(variables) == 1:
        axes = [axes]

    for ax, variable in zip(axes, variables):
        plot_df = positions[positions["variable"].eq(variable)].sort_values("malignant_threshold")
        stat_df = summary[summary["variable"].eq(variable)].sort_values("malignant_threshold")
        style_axis(ax)

        for group, group_df in plot_df.groupby("group", sort=False):
            group_df = group_df.sort_values("malignant_threshold")
            x = group_df["malignant_threshold"].to_numpy(dtype=float) * 100
            y = group_df["median"].to_numpy(dtype=float)
            y1 = group_df["q25"].to_numpy(dtype=float)
            y2 = group_df["q75"].to_numpy(dtype=float)
            ax.fill_between(x, y1, y2, color=PALETTE[group], alpha=0.12, linewidth=0)
            ax.plot(x, y, color=PALETTE[group], marker="o", markersize=4.2, linewidth=1.9, label=group)

        y_min, y_max = ax.get_ylim()
        span = y_max - y_min
        for _, row in stat_df.iterrows():
            sig = row["paired_wilcoxon_sig_holm_within_figure"]
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
        ax.set_title(LABELS[variable], loc="left", fontsize=11, color=INK, pad=5)

    axes[-1].set_xlabel("Minimum malignant fraction in analysed spots (%)", color=INK, fontsize=10)
    axes[-1].set_xticks(threshold_ticks(positions))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.982), ncol=2, frameon=False, fontsize=9.5)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.91, bottom=0.07, hspace=0.54)

    fig.savefig(out_dir / f"threshold_hallmark_{group_name}_high_low_positions.png", dpi=300)
    if not png_only:
        fig.savefig(out_dir / f"threshold_hallmark_{group_name}_high_low_positions.pdf")
    plt.close(fig)


def plot_box_group(
    group_name: str,
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    out_dir: Path,
    hallmark_groups: dict[str, list[str]],
    png_only: bool = False,
) -> None:
    variables = hallmark_groups[group_name]
    fig_height = max(10.4, 2.05 * len(variables))
    fig, axes = plt.subplots(len(variables), 1, figsize=(8.4, fig_height), sharex=True)
    fig.patch.set_facecolor("#FCFCFD")
    if len(variables) == 1:
        axes = [axes]

    offsets = {"LOW SNAI1-ac": -1.6, "HIGH SNAI1-ac": 1.6}
    width = 2.35

    for ax, variable in zip(axes, variables):
        var_raw = raw[raw["variable"].eq(variable)].copy()
        var_stats = summary[summary["variable"].eq(variable)].copy()
        style_axis(ax)

        for threshold in sorted(var_raw["malignant_threshold"].unique()):
            g = var_raw[var_raw["malignant_threshold"].eq(threshold)]
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

            rng = np.random.default_rng(int(threshold * 1000) + variables.index(variable))
            ax.scatter(np.full(len(low), low_x) + rng.normal(0, 0.18, len(low)), low, s=10, color=PALETTE["LOW SNAI1-ac"], alpha=0.65, linewidths=0, zorder=3)
            ax.scatter(np.full(len(high), high_x) + rng.normal(0, 0.18, len(high)), high, s=10, color=PALETTE["HIGH SNAI1-ac"], alpha=0.65, linewidths=0, zorder=3)

        y_min, y_max = ax.get_ylim()
        span = y_max - y_min
        for _, row in var_stats.iterrows():
            sig = row["paired_wilcoxon_sig_holm_within_figure"]
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

        ax.set_title(LABELS[variable], loc="left", fontsize=11, color=INK, pad=5)
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
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.982), ncol=3, frameon=False, fontsize=9.5)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.91, bottom=0.065, hspace=0.54)

    fig.savefig(out_dir / f"threshold_hallmark_{group_name}_high_low_positions_boxplot.png", dpi=300)
    if not png_only:
        fig.savefig(out_dir / f"threshold_hallmark_{group_name}_high_low_positions_boxplot.pdf")
    plt.close(fig)


def main(
    input_subdir: str | None = None,
    output_subdir: str | None = None,
    pathway_set: str = "default",
    png_only: bool = False,
    skip_tables: bool = False,
    plot_kind: str = "both",
) -> None:
    in_dir = BASE_OUT / input_subdir if input_subdir else BASE_OUT
    out_dir = BASE_OUT / output_subdir if output_subdir else in_dir
    infile = in_dir / INFILE_NAME
    df = pd.read_csv(infile)
    df = df[df["variable_type"].eq("hallmark")].copy()
    df["malignant_threshold"] = pd.to_numeric(df["malignant_threshold"], errors="coerce")
    if pathway_set == "default":
        hallmark_groups = HALLMARK_GROUPS
    elif pathway_set == "remaining":
        hallmark_groups = REMAINING_HALLMARK_GROUPS
    elif pathway_set == "strict7":
        hallmark_groups = STRICT_CONSISTENT_HALLMARK_GROUPS
    else:
        raise ValueError(f"Unknown pathway set: {pathway_set}")
    wanted = {v for values in hallmark_groups.values() for v in values}
    missing = sorted(wanted - set(df["variable"].unique()))
    if missing:
        raise RuntimeError(f"Missing expected Hallmark rows: {missing}")
    df = df[df["variable"].isin(wanted)].copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    summary, positions, raw = summarize(df, hallmark_groups)
    if not skip_tables:
        summary.to_csv(out_dir / "threshold_hallmark_group_high_low_positions_stats.csv", index=False)
        positions.to_csv(out_dir / "threshold_hallmark_group_high_low_positions_summary.csv", index=False)
        raw.to_csv(out_dir / "threshold_hallmark_group_high_low_positions_per_sample.csv", index=False)

    for group_name in hallmark_groups:
        if plot_kind in {"both", "line"}:
            plot_line_group(group_name, positions, summary, out_dir, hallmark_groups, png_only=png_only)
        if plot_kind in {"both", "box"}:
            plot_box_group(group_name, raw, summary, out_dir, hallmark_groups, png_only=png_only)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-subdir",
        default=None,
        help="Optional subfolder under the step4 threshold diagnostics directory receiving outputs.",
    )
    parser.add_argument(
        "--input-subdir",
        default=None,
        help="Optional subfolder under the step4 threshold diagnostics directory containing the input table.",
    )
    parser.add_argument(
        "--pathway-set",
        choices=["default", "remaining", "strict7"],
        default="default",
        help="Hallmark pathway set to plot.",
    )
    parser.add_argument(
        "--png-only",
        action="store_true",
        help="Write PNG files only and skip PDF exports.",
    )
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        help="Do not write summary CSV sidecars.",
    )
    parser.add_argument(
        "--plot-kind",
        choices=["both", "line", "box"],
        default="both",
        help="Which figure type to write.",
    )
    args = parser.parse_args()
    main(
        input_subdir=args.input_subdir if args.input_subdir is not None else args.output_subdir,
        output_subdir=args.output_subdir,
        pathway_set=args.pathway_set,
        png_only=args.png_only,
        skip_tables=args.skip_tables,
        plot_kind=args.plot_kind,
    )
