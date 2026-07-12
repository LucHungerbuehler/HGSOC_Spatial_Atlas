"""Plot cohort-level cross-validated R2 for cNMF ridge model comparisons."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BRANCH = Path(
    r"D:\HGSOC_Spatial_Atlas\05_analysis_ready\S3_cNMF_Tumor_Programs"
) / "snai1ac_program_decomposition_unsmoothed_uncorrected_v1"

PERFORMANCE = BRANCH / "02_per_sample_usage_models" / "tables" / "per_sample_model_performance.csv"
FIGURE_DIR = BRANCH / "07_report_examples" / "figures"
TABLE_DIR = BRANCH / "07_report_examples" / "tables"
USAGE_NORM_ONLY = TABLE_DIR / "cohort_usage_norm_only_model_performance.csv"

RAW_MODEL_ORDER = [
    ("spatial_malignant", "Spatial plus malignant", "#7A828F", "o"),
    ("usage_raw_only", "Usage only", "#0F8B8D", "s"),
    ("spatial_malignant_usage_raw", "Full model", "#9F4F63", "D"),
]

ROW_NORM_MODEL_ORDER = [
    ("spatial_malignant", "Spatial plus malignant", "#7A828F", "o"),
    ("usage_norm_only", "Usage only, row-normalized", "#0F8B8D", "s"),
    ("spatial_malignant_usage_norm", "Full model, row-normalized usage", "#9F4F63", "D"),
]


def sample_display_name(sample_label: str) -> str:
    if sample_label.startswith("denisenko_2022__"):
        return sample_label.replace("denisenko_2022__", "")
    if sample_label.startswith("yamamoto_2025__"):
        return sample_label.replace("yamamoto_2025__", "")
    if sample_label.startswith("ju_2024__"):
        return sample_label.replace("ju_2024__", "")
    return sample_label


def plot_model_comparison(
    performance: pd.DataFrame,
    model_order: list[tuple[str, str, str, str]],
    full_model: str,
    out_name: str,
    source_name: str,
) -> tuple[Path, Path]:
    keep_models = [model for model, _, _, _ in model_order]
    plot_df = performance.loc[performance["model"].isin(keep_models)].copy()
    plot_df["cv_r2"] = pd.to_numeric(plot_df["cv_r2"], errors="raise")

    wide = plot_df.pivot_table(index="sample_label", columns="model", values="cv_r2")
    wide = wide.dropna(subset=keep_models)
    wide["delta_full_vs_baseline"] = (
        wide[full_model] - wide["spatial_malignant"]
    )
    wide["included_in_plot"] = wide.index.to_series().ne("denisenko_2022__SP8")
    wide = wide.sort_values(full_model, ascending=True)
    wide["sample_display"] = [sample_display_name(s) for s in wide.index]
    source_path = TABLE_DIR / source_name
    wide.to_csv(source_path)
    plot_wide = wide.loc[wide["included_in_plot"]].copy()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.facecolor": "#FCFCFD",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#D7DBE7",
            "axes.labelcolor": "#1F2430",
            "xtick.color": "#1F2430",
            "ytick.color": "#1F2430",
        }
    )

    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    y = np.arange(len(plot_wide))

    for yi, (_, row) in zip(y, plot_wide.iterrows()):
        ax.plot(
            [row["spatial_malignant"], row[full_model]],
            [yi, yi],
            color="#C5CAD3",
            linewidth=1.0,
            zorder=1,
        )

    offset_values = np.linspace(-0.18, 0.18, num=len(model_order))
    offsets = {model: offset for (model, _, _, _), offset in zip(model_order, offset_values)}
    for model, label, color, marker in model_order:
        ax.scatter(
            plot_wide[model],
            y + offsets[model],
            s=34,
            marker=marker,
            color=color,
            edgecolor="#1F2430",
            linewidth=0.35,
            label=label,
            zorder=3,
        )

    ax.axvline(0, color="#1F2430", linestyle="--", linewidth=0.9, alpha=0.75)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_wide["sample_display"])
    ax.set_xlabel("Cross-validated $R^2$")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#E6E8F0", linewidth=0.7)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=3,
        frameon=False,
        borderaxespad=0,
    )

    x_min = min(float(plot_wide[keep_models].min().min()), -0.05)
    x_max = max(float(plot_wide[keep_models].max().max()), 0.5)
    pad = (x_max - x_min) * 0.05
    ax.set_xlim(x_min - pad, x_max + pad)

    fig.tight_layout()
    out = FIGURE_DIR / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out, source_path


def plot_raw_vs_row_normalized_composite(performance: pd.DataFrame) -> tuple[Path, Path]:
    models = [
        "spatial_malignant",
        "usage_raw_only",
        "usage_norm_only",
        "spatial_malignant_usage_raw",
        "spatial_malignant_usage_norm",
    ]
    plot_df = performance.loc[performance["model"].isin(models)].copy()
    plot_df["cv_r2"] = pd.to_numeric(plot_df["cv_r2"], errors="raise")

    wide = plot_df.pivot_table(index="sample_label", columns="model", values="cv_r2")
    wide = wide.dropna(subset=models)
    wide["raw_delta_full_vs_baseline"] = (
        wide["spatial_malignant_usage_raw"] - wide["spatial_malignant"]
    )
    wide["row_norm_delta_full_vs_baseline"] = (
        wide["spatial_malignant_usage_norm"] - wide["spatial_malignant"]
    )
    wide["included_in_plot"] = wide.index.to_series().ne("denisenko_2022__SP8")
    wide = wide.sort_values("spatial_malignant_usage_raw", ascending=True)
    wide["sample_display"] = [sample_display_name(s) for s in wide.index]

    source_path = TABLE_DIR / "cohort_cv_r2_raw_vs_row_normalized_composite_plot_source.csv"
    wide.to_csv(source_path)
    plot_wide = wide.loc[wide["included_in_plot"]].copy()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.3,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.facecolor": "#FCFCFD",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#D7DBE7",
            "axes.labelcolor": "#1F2430",
            "xtick.color": "#1F2430",
            "ytick.color": "#1F2430",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 7.0), sharey=True)
    y = np.arange(len(plot_wide))

    panel_specs = [
        {
            "ax": axes[0],
            "panel_label": "Raw usage",
            "model_order": RAW_MODEL_ORDER,
            "full_model": "spatial_malignant_usage_raw",
        },
        {
            "ax": axes[1],
            "panel_label": "Row-normalized usage",
            "model_order": ROW_NORM_MODEL_ORDER,
            "full_model": "spatial_malignant_usage_norm",
        },
    ]

    x_min = min(float(plot_wide[models].min().min()), -0.05)
    x_max = max(float(plot_wide[models].max().max()), 0.5)
    pad = (x_max - x_min) * 0.05

    for spec in panel_specs:
        ax = spec["ax"]
        full_model = spec["full_model"]
        model_order = spec["model_order"]

        for yi, (_, row) in zip(y, plot_wide.iterrows()):
            ax.plot(
                [row["spatial_malignant"], row[full_model]],
                [yi, yi],
                color="#C5CAD3",
                linewidth=1.0,
                zorder=1,
            )

        offset_values = np.linspace(-0.16, 0.16, num=len(model_order))
        offsets = {model: offset for (model, _, _, _), offset in zip(model_order, offset_values)}
        for model, label, color, marker in model_order:
            ax.scatter(
                plot_wide[model],
                y + offsets[model],
                s=34,
                marker=marker,
                color=color,
                edgecolor="#1F2430",
                linewidth=0.35,
                label=label if ax is axes[0] else "_nolegend_",
                zorder=3,
            )

        ax.axvline(0, color="#1F2430", linestyle="--", linewidth=0.9, alpha=0.75)
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_xlabel("Cross-validated $R^2$")
        ax.grid(axis="x", color="#E6E8F0", linewidth=0.7)
        ax.grid(axis="y", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            0.0,
            1.015,
            spec["panel_label"],
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="#1F2430",
            fontweight="semibold",
        )

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(plot_wide["sample_display"])
    axes[0].set_ylabel("")
    axes[1].tick_params(axis="y", left=False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=3,
        frameon=False,
        borderaxespad=0,
    )

    fig.tight_layout(w_pad=2.0, rect=(0, 0, 1, 0.985))
    out = FIGURE_DIR / "cohort_cv_r2_raw_vs_row_normalized_composite.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out, source_path


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    performance = pd.read_csv(PERFORMANCE)
    if USAGE_NORM_ONLY.exists():
        performance = pd.concat([performance, pd.read_csv(USAGE_NORM_ONLY)], ignore_index=True)
    outputs = [
        plot_model_comparison(
            performance,
            RAW_MODEL_ORDER,
            "spatial_malignant_usage_raw",
            "cohort_cv_r2_model_comparison.png",
            "cohort_cv_r2_model_comparison_plot_source.csv",
        ),
        plot_model_comparison(
            performance,
            ROW_NORM_MODEL_ORDER,
            "spatial_malignant_usage_norm",
            "cohort_cv_r2_model_comparison_row_normalized_usage.png",
            "cohort_cv_r2_model_comparison_row_normalized_usage_plot_source.csv",
        ),
        plot_raw_vs_row_normalized_composite(performance),
    ]
    for out, source in outputs:
        print(out)
        print(source)


if __name__ == "__main__":
    main()
