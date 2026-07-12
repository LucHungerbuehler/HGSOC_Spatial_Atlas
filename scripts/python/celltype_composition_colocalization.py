# celltype_composition_colocalization.py
"""
Block 2: broad SpaCET cell-type composition co-localization with SNAI1-ac.

This script derives a report-ready 23-sample cohort version from the existing
per-sample correlation outputs. The original cross_sample compile contains
legacy Stur and 10X samples; this output is aligned to the downstream cohort
used for Hallmark, tumor-only, and C-SIDE analyses.

Input:
    D:/HGSOC_Spatial_Atlas/05_analysis_ready/cross_sample/compiled/
    D:/HGSOC_Spatial_Atlas/03_metadata/clinical_annotations/clinical annotations.xlsx

Output:
    D:/HGSOC_Spatial_Atlas/05_analysis_ready/S2b_CellType_Composition_Correlation/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch


BASE_DIR = Path(r"D:\HGSOC_Spatial_Atlas")
COMPILED_DIR = BASE_DIR / "05_analysis_ready" / "cross_sample" / "compiled"
OUTPUT_DIR = BASE_DIR / "05_analysis_ready" / "S2b_CellType_Composition_Correlation"
CLINICAL_PATH = (
    BASE_DIR
    / "03_metadata"
    / "clinical_annotations"
    / "clinical annotations.xlsx"
)

COHORT_23 = {
    "visium/denisenko_2022": ["SP1", "SP2", "SP3", "SP4", "SP5", "SP6", "SP7", "SP8"],
    "visium/yamamoto_2025": [
        "Pt1-1",
        "Pt1-2",
        "Pt1-3",
        "Pt1-4",
        "Pt2-1",
        "Pt2-2",
        "Pt2-3",
        "Pt2-4",
    ],
    "visium/ju_2024": [
        "CPS_OV19_LtOV1",
        "CPS_OV1RtOV3",
        "CPS_OV20RtOV4",
        "CPS_OV24RTOV4",
        "CPS_OV34RtOV1",
        "CPS_OV5LtOV4",
        "CPS_OV71_1",
    ],
}

DATASET_TO_CLINICAL = {
    "denisenko_2022": "denisenko",
    "yamamoto_2025": "yamamoto",
    "ju_2024": "ju",
}

DATASET_COLORS = {
    "denisenko_2022": "#4E79A7",
    "yamamoto_2025": "#F28E2B",
    "ju_2024": "#E15759",
}

SITE_COLORS = {
    "primary tumor": "#4C78A8",
    "secondary site": "#F58518",
    "unknown": "#B9B9B9",
}

BRCA_COLORS = {
    "BRCAmut": "#B279A2",
    "BRCAwt": "#59A14F",
    "unknown": "#B9B9B9",
}


def dataset_short(dataset: str) -> str:
    return str(dataset).replace("\\", "/").split("/")[-1]


def sample_label(dataset: str, sample: str) -> str:
    return f"{dataset_short(dataset)}__{sample}"


def clinical_sample_key(sample: str) -> str:
    return str(sample).split()[0].strip()


def site_group(value: object) -> str:
    text = str(value).lower()
    if "secondary" in text:
        return "secondary site"
    if "primary" in text:
        return "primary tumor"
    return "unknown"


def brca_group(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).lower()
    if "mut" in text:
        return "BRCAmut"
    if "negative" in text or "wt" in text or "wild" in text:
        return "BRCAwt"
    return "unknown"


def load_clinical_annotations() -> pd.DataFrame:
    clinical = pd.read_excel(CLINICAL_PATH)
    clinical = clinical.rename(columns=lambda c: str(c).strip())
    clinical["clinical_dataset"] = clinical["dataset"].astype(str).str.strip().str.lower()
    clinical["sample_key"] = clinical["sample"].map(clinical_sample_key)
    clinical["site_group"] = clinical["sections source"].map(site_group)
    clinical["brca_group"] = clinical["BRCA1/2"].map(brca_group)
    return clinical


def add_metadata(df: pd.DataFrame, clinical: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dataset_short"] = out["dataset"].map(dataset_short)
    out["sample_label"] = [
        sample_label(dataset, sample) for dataset, sample in zip(out["dataset"], out["sample"])
    ]
    meta_rows = []
    for row in out[["dataset", "dataset_short", "sample"]].drop_duplicates().itertuples(index=False):
        clinical_dataset = DATASET_TO_CLINICAL.get(row.dataset_short, row.dataset_short)
        match = clinical[
            (clinical["clinical_dataset"] == clinical_dataset)
            & (clinical["sample_key"] == row.sample)
        ]
        if len(match) == 0:
            site = "unknown"
            brca = "unknown"
        else:
            site = str(match.iloc[0]["site_group"])
            brca = str(match.iloc[0]["brca_group"])
        meta_rows.append(
            {
                "dataset": row.dataset,
                "sample": row.sample,
                "dataset_short": row.dataset_short,
                "sample_label": sample_label(row.dataset, row.sample),
                "site_group": site,
                "brca_group": brca,
            }
        )
    sample_meta = pd.DataFrame(meta_rows)
    out = out.merge(
        sample_meta[
            ["dataset", "sample", "site_group", "brca_group"]
        ],
        on=["dataset", "sample"],
        how="left",
    )
    return out


def filter_cohort(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(False, index=df.index)
    for dataset, samples in COHORT_23.items():
        mask |= (df["dataset"] == dataset) & (df["sample"].isin(samples))
    return df.loc[mask].copy()


def summarize(df: pd.DataFrame, value_col: str, metric_name: str) -> pd.DataFrame:
    rows = []
    for cell_type, group in df.groupby("cell_type"):
        values = group[value_col].dropna().astype(float)
        if len(values) == 0:
            continue
        sd = values.std(ddof=1)
        se = sd / np.sqrt(len(values)) if len(values) > 1 else np.nan
        pos_fraction = float((values > 0).mean())
        rows.append(
            {
                "cell_type": cell_type,
                f"{metric_name}_mean": float(values.mean()),
                f"{metric_name}_median": float(values.median()),
                f"{metric_name}_sd": float(sd),
                f"{metric_name}_ci_lower": float(values.mean() - 1.96 * se),
                f"{metric_name}_ci_upper": float(values.mean() + 1.96 * se),
                "n_samples": int(len(values)),
                f"{metric_name}_direction_consistency": float(
                    100 * max(pos_fraction, 1 - pos_fraction)
                ),
                f"{metric_name}_positive_fraction": float(100 * pos_fraction),
            }
        )
    return pd.DataFrame(rows)


def order_samples(sample_meta: pd.DataFrame) -> list[str]:
    brca_order = {"BRCAmut": 0, "BRCAwt": 1, "unknown": 2}
    site_order = {"secondary site": 0, "primary tumor": 1, "unknown": 2}
    dataset_order = {"denisenko_2022": 0, "yamamoto_2025": 1, "ju_2024": 2}
    return (
        sample_meta.drop_duplicates("sample_label")
        .assign(
            brca_sort=lambda x: x["brca_group"].map(brca_order).fillna(99),
            site_sort=lambda x: x["site_group"].map(site_order).fillna(99),
            dataset_sort=lambda x: x["dataset_short"].map(dataset_order).fillna(99),
        )
        .sort_values(["brca_sort", "site_sort", "dataset_sort", "sample_label"])
        ["sample_label"]
        .tolist()
    )


def colors_to_rgb(values: pd.Series, color_map: dict[str, str]) -> np.ndarray:
    return np.array(
        [matplotlib.colors.to_rgb(color_map.get(str(value), "#B9B9B9")) for value in values]
    ).reshape(1, -1, 3)


def plot_raw_heatmap(raw: pd.DataFrame, summary: pd.DataFrame) -> None:
    pivot = raw.pivot(index="cell_type", columns="sample_label", values="spearman_r")
    row_order = summary.sort_values("raw_mean", ascending=False)["cell_type"].tolist()
    pivot = pivot.loc[[ct for ct in row_order if ct in pivot.index]]
    sample_meta = raw[
        ["sample_label", "dataset_short", "site_group", "brca_group"]
    ].drop_duplicates().set_index("sample_label")
    col_order = order_samples(sample_meta.reset_index())
    pivot = pivot[col_order]
    sample_meta = sample_meta.loc[col_order]

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(
        nrows=4,
        ncols=3,
        width_ratios=[0.12, 8.8, 2.4],
        height_ratios=[0.18, 0.18, 0.18, 4.5],
        wspace=0.05,
        hspace=0.04,
    )
    ax_dataset = fig.add_subplot(gs[0, 1])
    ax_site = fig.add_subplot(gs[1, 1])
    ax_brca = fig.add_subplot(gs[2, 1])
    ax_heat = fig.add_subplot(gs[3, 1])
    ax_leg = fig.add_subplot(gs[3, 2])
    ax_leg.axis("off")

    for ax, rgb, label in [
        (ax_dataset, colors_to_rgb(sample_meta["dataset_short"], DATASET_COLORS), "Dataset"),
        (ax_site, colors_to_rgb(sample_meta["site_group"], SITE_COLORS), "Site"),
        (ax_brca, colors_to_rgb(sample_meta["brca_group"], BRCA_COLORS), "BRCA"),
    ]:
        ax.imshow(rgb, aspect="auto", interpolation="nearest")
        ax.set_yticks([0])
        ax.set_yticklabels([label], fontsize=10)
        ax.set_xticks([])
        ax.tick_params(axis="y", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    im = ax_heat.imshow(
        pivot.to_numpy(dtype=float),
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-0.5,
        vmax=0.5,
    )
    ax_heat.set_xticks(np.arange(pivot.shape[1]))
    ax_heat.set_xticklabels(
        [c.replace("__", "\n") for c in pivot.columns],
        fontsize=8,
        rotation=90,
    )
    ax_heat.set_yticks(np.arange(pivot.shape[0]))
    ax_heat.set_yticklabels(pivot.index, fontsize=10)
    ax_heat.tick_params(length=0)
    ax_heat.set_xlabel("Sample", fontsize=11)

    cax = ax_leg.inset_axes([0.08, 0.68, 0.10, 0.25])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Spearman r", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    legend_y = 0.52
    for title, colors in [
        ("Dataset", DATASET_COLORS),
        ("Site", SITE_COLORS),
        ("BRCA", BRCA_COLORS),
    ]:
        handles = [Patch(facecolor=color, label=label) for label, color in colors.items()]
        leg = ax_leg.legend(
            handles=handles,
            title=title,
            loc="upper left",
            bbox_to_anchor=(0.05, legend_y),
            fontsize=9,
            title_fontsize=10,
            frameon=False,
            borderaxespad=0,
            handlelength=1.4,
            handletextpad=0.5,
            labelspacing=0.35,
        )
        ax_leg.add_artist(leg)
        legend_y -= 0.22

    fig.savefig(OUTPUT_DIR / "celltype_raw_spearman_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_raw_vs_partial(summary: pd.DataFrame) -> None:
    plot_df = summary.sort_values("raw_mean").copy()
    y = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.hlines(y, plot_df["partial_mean"], plot_df["raw_mean"], color="#BBBBBB", linewidth=1.5)
    ax.scatter(plot_df["raw_mean"], y, label="Raw Spearman", color="#4E79A7", s=45)
    ax.scatter(
        plot_df["partial_mean"],
        y,
        label="Partial Spearman\n(controlling malignant)",
        color="#E15759",
        s=45,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["cell_type"], fontsize=10)
    ax.set_xlabel("Correlation with SNAI1-ac", fontsize=11)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "celltype_raw_vs_malignant_adjusted_dotplot.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clinical = load_clinical_annotations()

    raw = filter_cohort(pd.read_csv(COMPILED_DIR / "all_correlations_raw.csv"))
    partial = filter_cohort(pd.read_csv(COMPILED_DIR / "all_correlations_partial.csv"))
    comparison = filter_cohort(pd.read_csv(COMPILED_DIR / "all_correlations_comparison.csv"))

    raw = add_metadata(raw, clinical)
    partial = add_metadata(partial, clinical)
    comparison = add_metadata(comparison, clinical)

    raw.to_csv(OUTPUT_DIR / "celltype_raw_correlations_23sample.csv", index=False)
    partial.to_csv(OUTPUT_DIR / "celltype_partial_correlations_23sample.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "celltype_raw_vs_partial_23sample.csv", index=False)

    raw_summary = summarize(raw, "spearman_r", "raw")
    partial_summary = summarize(partial, "partial_r", "partial")
    summary = raw_summary.merge(partial_summary, on=["cell_type", "n_samples"], how="outer")
    summary["raw_abs_mean"] = summary["raw_mean"].abs()
    summary = summary.sort_values("raw_mean", ascending=False)
    summary.to_csv(OUTPUT_DIR / "celltype_correlations_summary_23sample.csv", index=False)

    plot_raw_heatmap(raw, summary)
    plot_raw_vs_partial(summary)

    print(f"Saved outputs to {OUTPUT_DIR}")
    print(f"Samples: {raw[['dataset', 'sample']].drop_duplicates().shape[0]}")
    print(
        summary[
            [
                "cell_type",
                "raw_mean",
                "raw_median",
                "raw_direction_consistency",
                "partial_mean",
                "partial_median",
                "partial_direction_consistency",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
