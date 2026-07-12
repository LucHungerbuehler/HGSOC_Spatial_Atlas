# hallmark_correlation_analysis.py
"""
S2a: Cross-sample co-localization of Hallmark pathway scores with SNAI1-ac.

This script uses the same 23-sample cohort used by the downstream tumor-only
Hallmark analyses: Denisenko (8), Yamamoto (8), and Ju (7), excluding the Ju
sample that is missing from the downstream analysis-ready h5ad set.

For each sample it computes both Spearman and Pearson correlations between the
continuous SNAI1-ac score and each Hallmark score. The per-pathway summary is
sample-level first: mean/median correlations and direction consistency. Nominal
Stouffer signed-Z values are reported as evidence summaries, but the thesis
interpretation should emphasize effect sizes and direction consistency because
Visium spots are spatially autocorrelated and are not independent tests.

Input:
    D:/HGSOC_Spatial_Atlas/05_analysis_ready/visium/<dataset>/<sample>/<sample>.h5ad
    D:/HGSOC_Spatial_Atlas/03_metadata/clinical_annotations/clinical annotations.xlsx

Output:
    D:/HGSOC_Spatial_Atlas/05_analysis_ready/S2a_Hallmark_Correlation/
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy import stats


BASE_DIR = Path(r"D:\HGSOC_Spatial_Atlas")
VISIUM_DIR = BASE_DIR / "05_analysis_ready" / "visium"
OUTPUT_DIR = BASE_DIR / "05_analysis_ready" / "S2a_Hallmark_Correlation"
CLINICAL_PATH = (
    BASE_DIR
    / "03_metadata"
    / "clinical_annotations"
    / "clinical annotations.xlsx"
)

SNAI1_COL = "SNAI1-ac_score"
HALLMARK_PREFIX = "HALLMARK_"
HALLMARK_SUFFIX = "_score"

COHORT_23 = {
    "denisenko_2022": ["SP1", "SP2", "SP3", "SP4", "SP5", "SP6", "SP7", "SP8"],
    "yamamoto_2025": [
        "Pt1-1",
        "Pt1-2",
        "Pt1-3",
        "Pt1-4",
        "Pt2-1",
        "Pt2-2",
        "Pt2-3",
        "Pt2-4",
    ],
    "ju_2024": [
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

DATASET_COLORS = {
    "denisenko_2022": "#4E79A7",
    "yamamoto_2025": "#F28E2B",
    "ju_2024": "#E15759",
}


def clean_pathway_name(score_col: str) -> str:
    return score_col.replace(HALLMARK_PREFIX, "").replace(HALLMARK_SUFFIX, "")


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


def h5ad_path_for(dataset: str, sample: str) -> Path:
    return VISIUM_DIR / dataset / sample / f"{sample}.h5ad"


def build_manifest(clinical: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, samples in COHORT_23.items():
        clinical_dataset = DATASET_TO_CLINICAL[dataset]
        for sample in samples:
            h5ad_path = h5ad_path_for(dataset, sample)
            clin = clinical[
                (clinical["clinical_dataset"] == clinical_dataset)
                & (clinical["sample_key"] == sample)
            ]
            if len(clin) == 1:
                clin_row = clin.iloc[0]
                site = clin_row["site_group"]
                brca = clin_row["brca_group"]
                section_source = clin_row["sections source"]
                brca_raw = clin_row["BRCA1/2"]
            else:
                site = "unknown"
                brca = "unknown"
                section_source = np.nan
                brca_raw = np.nan

            rows.append(
                {
                    "dataset": dataset,
                    "sample": sample,
                    "sample_label": f"{dataset}__{sample}",
                    "h5ad_path": str(h5ad_path),
                    "h5ad_exists": h5ad_path.exists(),
                    "clinical_rows_matched": len(clin),
                    "section_source": section_source,
                    "site_group": site,
                    "brca_raw": brca_raw,
                    "brca_group": brca,
                }
            )
    return pd.DataFrame(rows)


def finite_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def safe_correlations(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x, y = finite_pair(x, y)
    n = len(x)
    out = {
        "n_spots": n,
        "spearman_r": np.nan,
        "spearman_p": np.nan,
        "pearson_r": np.nan,
        "pearson_p": np.nan,
    }
    if n < 4 or np.std(x) == 0 or np.std(y) == 0:
        return out

    spearman_r, spearman_p = stats.spearmanr(x, y)
    pearson_r, pearson_p = stats.pearsonr(x, y)
    out.update(
        {
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
        }
    )
    return out


def collect_correlations(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, sample_row in manifest.iterrows():
        if not sample_row["h5ad_exists"]:
            print(f"Missing h5ad, skipping: {sample_row['sample_label']}")
            continue

        print(f"Loading {sample_row['sample_label']}...")
        adata = ad.read_h5ad(sample_row["h5ad_path"], backed="r")
        obs = adata.obs

        if SNAI1_COL not in obs.columns:
            print(f"  WARNING: {SNAI1_COL} missing, skipping")
            continue

        hallmark_cols = sorted(
            c
            for c in obs.columns
            if c.startswith(HALLMARK_PREFIX) and c.endswith(HALLMARK_SUFFIX)
        )
        if len(hallmark_cols) == 0:
            print("  WARNING: no Hallmark scores found, skipping")
            continue

        snai1 = obs[SNAI1_COL].to_numpy(dtype=float)
        for col in hallmark_cols:
            metrics = safe_correlations(snai1, obs[col].to_numpy(dtype=float))
            if not np.isfinite(metrics["spearman_r"]):
                continue
            rows.append(
                {
                    "dataset": sample_row["dataset"],
                    "sample": sample_row["sample"],
                    "sample_label": sample_row["sample_label"],
                    "site_group": sample_row["site_group"],
                    "brca_group": sample_row["brca_group"],
                    "pathway": clean_pathway_name(col),
                    **metrics,
                }
            )
        adata.file.close()
        print(f"  {len(hallmark_cols)} pathways")

    return pd.DataFrame(rows)


def stouffer_signed_z(r_values: pd.Series, p_values: pd.Series) -> tuple[float, float]:
    valid = np.isfinite(r_values.to_numpy(dtype=float)) & np.isfinite(
        p_values.to_numpy(dtype=float)
    )
    if valid.sum() == 0:
        return np.nan, np.nan
    r = r_values.to_numpy(dtype=float)[valid]
    p = p_values.to_numpy(dtype=float)[valid]
    p = np.clip(p, np.finfo(float).tiny, 1.0)
    z = stats.norm.isf(p / 2.0) * np.sign(r)
    z_combined = np.sum(z) / np.sqrt(len(z))
    p_combined = 2.0 * stats.norm.sf(abs(z_combined))
    return float(z_combined), float(p_combined)


def summarize_metric(sub: pd.DataFrame, metric: str) -> dict[str, float]:
    r = sub[f"{metric}_r"].astype(float)
    p = sub[f"{metric}_p"].astype(float)
    mean_r = float(r.mean())
    median_r = float(r.median())
    pos_fraction = float((r > 0).mean())
    direction_consistency = max(pos_fraction, 1.0 - pos_fraction)
    dominant_direction = "positive" if pos_fraction >= 0.5 else "negative"
    stouffer_z, stouffer_p = stouffer_signed_z(r, p)
    return {
        f"{metric}_mean_r": mean_r,
        f"{metric}_median_r": median_r,
        f"{metric}_sd_r": float(r.std(ddof=1)),
        f"{metric}_min_r": float(r.min()),
        f"{metric}_max_r": float(r.max()),
        f"{metric}_positive_fraction": pos_fraction,
        f"{metric}_direction_consistency": float(direction_consistency),
        f"{metric}_dominant_direction": dominant_direction,
        f"{metric}_stouffer_z": stouffer_z,
        f"{metric}_stouffer_p_nominal": stouffer_p,
    }


def compute_meta_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pathway, sub in df.groupby("pathway", sort=False):
        row = {
            "pathway": pathway,
            "n_samples": int(sub["sample_label"].nunique()),
            "mean_n_spots": float(sub["n_spots"].mean()),
        }
        row.update(summarize_metric(sub, "spearman"))
        row.update(summarize_metric(sub, "pearson"))
        rows.append(row)

    meta = pd.DataFrame(rows)
    meta = meta.sort_values(
        ["spearman_direction_consistency", "spearman_mean_r"],
        ascending=[False, False],
    ).reset_index(drop=True)
    return meta


def make_overview_heatmap(df: pd.DataFrame, meta: pd.DataFrame) -> None:
    pivot = df.pivot(index="pathway", columns="sample_label", values="spearman_r")

    row_order = meta.sort_values(
        ["spearman_direction_consistency", "spearman_mean_r"],
        ascending=[False, False],
    )["pathway"].tolist()
    pivot = pivot.loc[[p for p in row_order if p in pivot.index]]

    sample_annot = (
        df[["sample_label", "dataset", "site_group", "brca_group"]]
        .drop_duplicates()
        .set_index("sample_label")
        .loc[pivot.columns]
    )

    brca_order = {"BRCAmut": 0, "BRCAwt": 1, "unknown": 2}
    site_order = {"secondary site": 0, "primary tumor": 1, "unknown": 2}
    dataset_order = {"denisenko_2022": 0, "yamamoto_2025": 1, "ju_2024": 2}
    col_order = (
        sample_annot.assign(
            brca_sort=lambda x: x["brca_group"].map(brca_order).fillna(99),
            site_sort=lambda x: x["site_group"].map(site_order).fillna(99),
            dataset_sort=lambda x: x["dataset"].map(dataset_order).fillna(99),
        )
        .sort_values(["brca_sort", "site_sort", "dataset_sort", "sample_label"])
        .index.tolist()
    )
    pivot = pivot[col_order]
    sample_annot = sample_annot.loc[col_order]

    corr_cmap = plt.get_cmap("RdBu_r")

    def colors_to_rgb(values: pd.Series, color_map: dict[str, str]) -> np.ndarray:
        return np.array(
            [
                matplotlib.colors.to_rgb(color_map.get(str(value), "#B9B9B9"))
                for value in values
            ]
        ).reshape(1, -1, 3)

    dataset_rgb = colors_to_rgb(sample_annot["dataset"], DATASET_COLORS)
    site_rgb = colors_to_rgb(sample_annot["site_group"], SITE_COLORS)
    brca_rgb = colors_to_rgb(sample_annot["brca_group"], BRCA_COLORS)

    fig = plt.figure(figsize=(15, 16))
    gs = fig.add_gridspec(
        nrows=4,
        ncols=3,
        width_ratios=[0.18, 9.8, 2.7],
        height_ratios=[0.18, 0.18, 0.18, 9.6],
        wspace=0.05,
        hspace=0.03,
    )

    ax_dataset = fig.add_subplot(gs[0, 1])
    ax_site = fig.add_subplot(gs[1, 1])
    ax_brca = fig.add_subplot(gs[2, 1])
    ax_heat = fig.add_subplot(gs[3, 1])
    ax_leg = fig.add_subplot(gs[3, 2])
    ax_leg.axis("off")

    for ax, rgb, label in [
        (ax_dataset, dataset_rgb, "Dataset"),
        (ax_site, site_rgb, "Site"),
        (ax_brca, brca_rgb, "BRCA"),
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
        cmap=corr_cmap,
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
    ax_heat.set_yticklabels(pivot.index, fontsize=9)
    ax_heat.tick_params(length=0)
    ax_heat.set_xlabel("Sample", fontsize=11)
    ax_heat.set_ylabel("")

    cax1 = ax_leg.inset_axes([0.08, 0.72, 0.10, 0.22])
    cb1 = fig.colorbar(im, cax=cax1)
    cb1.set_label("Spearman r", fontsize=10)
    cb1.ax.tick_params(labelsize=9)

    legend_y = 0.58
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
        legend_y -= 0.19

    fig.savefig(OUTPUT_DIR / "s2a_overview_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Saved s2a_overview_heatmap.png")


def make_ranked_summary_plot(meta: pd.DataFrame) -> None:
    plot_df = meta.sort_values("spearman_mean_r").copy()
    y = np.arange(len(plot_df))
    colors = plot_df["spearman_direction_consistency"]

    fig, ax = plt.subplots(figsize=(8, 11))
    sc = ax.scatter(
        plot_df["spearman_mean_r"],
        y,
        c=colors,
        cmap="viridis",
        vmin=0.5,
        vmax=1.0,
        s=45,
        edgecolor="black",
        linewidth=0.2,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["pathway"], fontsize=7)
    ax.set_xlabel("Mean Spearman r with SNAI1-ac")
    ax.set_title("Hallmark co-localization effect sizes")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Directional consistency")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "s2a_ranked_effects.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Saved s2a_ranked_effects.png")


def make_forest_plot(df: pd.DataFrame, meta: pd.DataFrame, pathway: str) -> None:
    sub = df[df["pathway"] == pathway].copy()
    sub = sub.sort_values(["dataset", "sample"])
    meta_row = meta[meta["pathway"] == pathway].iloc[0]

    fig, ax = plt.subplots(figsize=(8, max(4, len(sub) * 0.32 + 1.5)))
    site_palette = {"primary tumor": "#4C78A8", "secondary site": "#F58518", "unknown": "#B9B9B9"}
    for i, (_, row) in enumerate(sub.iterrows()):
        ax.plot(
            row["spearman_r"],
            i,
            "o",
            color=site_palette.get(row["site_group"], "#B9B9B9"),
            markersize=6,
        )

    y_meta = len(sub) + 1
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(len(sub) + 0.5, color="black", linewidth=0.6)
    ax.plot(meta_row["spearman_mean_r"], y_meta, "D", color="black", markersize=8)

    labels = [
        f"{row['dataset']}/{row['sample']} ({row['site_group']}, {row['brca_group']})"
        for _, row in sub.iterrows()
    ]
    labels.extend(["", f"Mean (k={meta_row['n_samples']})"])
    ax.set_yticks(list(range(len(sub))) + [len(sub), y_meta])
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Spearman r with SNAI1-ac")
    ax.set_title(
        f"{pathway}\nmean r={meta_row['spearman_mean_r']:.3f}, "
        f"median r={meta_row['spearman_median_r']:.3f}, "
        f"direction consistency={meta_row['spearman_direction_consistency']:.0%}"
    )
    ax.set_xlim(-1, 1)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"forest_{pathway}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("S2a Hallmark co-localization: 23-sample cohort")
    print("=" * 80)

    clinical = load_clinical_annotations()
    manifest = build_manifest(clinical)
    manifest.to_csv(OUTPUT_DIR / "s2a_cohort_manifest.csv", index=False)
    if not manifest["h5ad_exists"].all():
        missing = manifest.loc[~manifest["h5ad_exists"], ["dataset", "sample", "h5ad_path"]]
        raise FileNotFoundError(f"Missing cohort h5ad files:\n{missing.to_string(index=False)}")
    if not (manifest["clinical_rows_matched"] == 1).all():
        print("WARNING: some samples did not map uniquely to clinical annotations")
        print(
            manifest.loc[
                manifest["clinical_rows_matched"] != 1,
                ["dataset", "sample", "clinical_rows_matched"],
            ].to_string(index=False)
        )

    df = collect_correlations(manifest)
    if df.empty:
        raise RuntimeError("No Hallmark correlations were collected.")
    df.to_csv(OUTPUT_DIR / "s2a_per_sample_correlations.csv", index=False)
    print("Saved s2a_per_sample_correlations.csv")

    meta = compute_meta_summary(df)
    meta.to_csv(OUTPUT_DIR / "s2a_meta_analysis.csv", index=False)
    print("Saved s2a_meta_analysis.csv")

    print("\nTop pathways by Spearman mean r:")
    print(
        meta.sort_values("spearman_mean_r", ascending=False)
        .head(10)[
            [
                "pathway",
                "spearman_mean_r",
                "spearman_median_r",
                "spearman_direction_consistency",
                "pearson_mean_r",
            ]
        ]
        .to_string(index=False)
    )
    print("\nBottom pathways by Spearman mean r:")
    print(
        meta.sort_values("spearman_mean_r")
        .head(10)[
            [
                "pathway",
                "spearman_mean_r",
                "spearman_median_r",
                "spearman_direction_consistency",
                "pearson_mean_r",
            ]
        ]
        .to_string(index=False)
    )

    make_overview_heatmap(df, meta)
    make_ranked_summary_plot(meta)

    forest_pathways = set()
    forest_pathways.update(meta.sort_values("spearman_mean_r", ascending=False).head(5)["pathway"])
    forest_pathways.update(meta.sort_values("spearman_mean_r").head(5)["pathway"])
    forest_pathways.update(
        [
            "EPITHELIAL_MESENCHYMAL_TRANSITION",
            "HYPOXIA",
            "OXIDATIVE_PHOSPHORYLATION",
            "KRAS_SIGNALING_UP",
        ]
    )
    for pathway in sorted(p for p in forest_pathways if p in set(meta["pathway"])):
        make_forest_plot(df, meta, pathway)
    print(f"Saved {len(forest_pathways)} forest plots")


if __name__ == "__main__":
    main()
