"""
Section 2.4 Step 4 threshold diagnostics.

Purpose
-------
Answer the user's corrected diagnostic questions without changing the
established ILR geometry:

1. Continuous question: within malignant-fraction thresholds, which features
   are monotonically associated with SNAI1-ac?  -> Spearman r.
2. State-contrast question: within malignant-fraction thresholds, how do
   SNAI1-ac-high and SNAI1-ac-low spots differ? -> Cohen's d.

Composition contrasts use the established ILR tree/basis from
S2b_CellType_Composition_Correlation/ilr_first_pass. Hallmark contrasts are
computed from the numeric h5ad obs columns. This script is diagnostic, not a
polished manuscript figure generator.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
from skbio.stats.composition import closure, ilr, multi_replace

import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(r"D:\HGSOC_Spatial_Atlas")
READY = ROOT / "05_analysis_ready"
H5AD_ROOT = ROOT / "02_processed_data" / "visium"
ILR_DIR = READY / "S2b_CellType_Composition_Correlation" / "ilr_first_pass"
SPOT_TABLE_ROOT = READY / "20260424_definition3b_definition4_raw_geneNMF" / "02_definition3b_mixture_programme_niches"
OUT = READY / "S2b_CellType_Composition_Correlation" / "step4_threshold_high_low_diagnostics"

THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90]
MIN_GROUP_N = 15

EXACT_23 = [
    ("denisenko_2022", "SP1"),
    ("denisenko_2022", "SP2"),
    ("denisenko_2022", "SP3"),
    ("denisenko_2022", "SP4"),
    ("denisenko_2022", "SP5"),
    ("denisenko_2022", "SP6"),
    ("denisenko_2022", "SP7"),
    ("denisenko_2022", "SP8"),
    ("ju_2024", "CPS_OV19_LtOV1"),
    ("ju_2024", "CPS_OV1RtOV3"),
    ("ju_2024", "CPS_OV20RtOV4"),
    ("ju_2024", "CPS_OV24RTOV4"),
    ("ju_2024", "CPS_OV34RtOV1"),
    ("ju_2024", "CPS_OV5LtOV4"),
    ("ju_2024", "CPS_OV71_1"),
    ("yamamoto_2025", "Pt1-1"),
    ("yamamoto_2025", "Pt1-2"),
    ("yamamoto_2025", "Pt1-3"),
    ("yamamoto_2025", "Pt1-4"),
    ("yamamoto_2025", "Pt2-1"),
    ("yamamoto_2025", "Pt2-2"),
    ("yamamoto_2025", "Pt2-3"),
    ("yamamoto_2025", "Pt2-4"),
]

FOCUS_HALLMARKS = [
    "HALLMARK_OXIDATIVE_PHOSPHORYLATION_score",
    "HALLMARK_GLYCOLYSIS_score",
    "HALLMARK_HYPOXIA_score",
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION_score",
    "HALLMARK_TGF_BETA_SIGNALING_score",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB_score",
    "HALLMARK_MYC_TARGETS_V1_score",
    "HALLMARK_E2F_TARGETS_score",
    "HALLMARK_G2M_CHECKPOINT_score",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING_score",
    "HALLMARK_KRAS_SIGNALING_UP_score",
    "HALLMARK_APICAL_JUNCTION_score",
]

FOCUS_BALANCES = ["b1", "b2", "b3", "b4", "b9"]
BALANCE_LABELS = {
    "b1": "b1 Malignant vs TME",
    "b2": "b2 Stromal vs immune",
    "b3": "b3 CAF vs endothelial",
    "b4": "b4 Myeloid vs lymphoid",
    "b5": "b5 Mac/DC/pDC vs mast/neutrophil",
    "b6": "b6 Macrophage vs DC",
    "b7": "b7 cDC vs pDC",
    "b8": "b8 Mast vs neutrophil",
    "b9": "b9 T/NK vs B/plasma",
    "b10": "b10 T vs NK",
    "b11": "b11 CD4 vs CD8",
    "b12": "b12 B vs plasma",
}

QC_COLS = ["Malignant", "total_counts", "n_genes_by_counts", "pct_counts_mt"]


def cohen_d(high: pd.Series, low: pd.Series) -> float:
    x = pd.to_numeric(high, errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(low, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) < 2 or len(y) < 2:
        return np.nan
    pooled = np.sqrt(
        ((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1))
        / (len(x) + len(y) - 2)
    )
    if pooled == 0:
        return np.nan
    return float((np.mean(x) - np.mean(y)) / pooled)


def load_ilr_spec() -> tuple[list[str], list[str], np.ndarray, float, list[str]]:
    manifest = json.loads((ILR_DIR / "run_manifest.json").read_text())
    lineages = manifest["lineages"]
    primary = manifest["primary_balances"]
    base_delta = float(manifest["base_delta"])
    basis_df = pd.read_csv(ILR_DIR / "ilr_basis_matrix.csv", index_col=0)
    basis = basis_df.loc[:, lineages].to_numpy(dtype=float)
    balance_names = basis_df.index.tolist()
    tree = json.loads((ILR_DIR / "balance_tree.json").read_text())
    realised = json.loads((ILR_DIR / "realised_partition.json").read_text())
    if tree != realised:
        raise RuntimeError("Established balance_tree.json differs from realised_partition.json")
    if primary != FOCUS_BALANCES:
        raise RuntimeError(f"Unexpected primary balances in manifest: {primary}")
    return lineages, balance_names, basis, base_delta, primary


def h5ad_path(dataset: str, sample: str) -> Path:
    return H5AD_ROOT / dataset / f"{sample}.h5ad"


def spot_table_path(dataset: str, sample: str) -> Path:
    return SPOT_TABLE_ROOT / f"{dataset}__{sample}" / "tables" / "spot_level_table.csv"


def load_sample(dataset: str, sample: str, lineages: list[str]) -> pd.DataFrame:
    h5 = h5ad_path(dataset, sample)
    spot_path = spot_table_path(dataset, sample)
    if not h5.exists():
        raise FileNotFoundError(h5)
    if not spot_path.exists():
        raise FileNotFoundError(spot_path)

    a = ad.read_h5ad(h5, backed="r")
    obs = a.obs.copy()
    a.file.close()
    obs = obs.reset_index(names="spot_id")

    hallmark_cols = [c for c in obs.columns if c.startswith("HALLMARK_") and c.endswith("_score")]
    obs_cols = ["spot_id", "SNAI1-ac_score"] + hallmark_cols + [c for c in QC_COLS if c in obs.columns and c != "Malignant"]
    obs = obs[obs_cols].copy()

    spot_cols = ["spot_id", "Malignant"] + lineages
    spot = pd.read_csv(spot_path, usecols=lambda c: c in set(spot_cols))

    df = obs.merge(spot, on="spot_id", how="inner")
    df["dataset"] = dataset
    df["sample"] = sample
    df["sample_label"] = f"{dataset}__{sample}"
    return df


def compute_balances(df: pd.DataFrame, lineages: list[str], balance_names: list[str], basis: np.ndarray) -> pd.DataFrame:
    x = df[lineages].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    closed = closure(x.to_numpy(dtype=float))
    replaced = multi_replace(closed)
    balances = ilr(replaced, basis=basis)
    return pd.DataFrame(balances, index=df.index, columns=balance_names)


def summarize() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lineages, balance_names, basis, _base_delta, _primary = load_ilr_spec()
    contrast_rows = []
    spearman_rows = []
    availability_rows = []

    for dataset, sample in EXACT_23:
        df = load_sample(dataset, sample, lineages)
        balances = compute_balances(df, lineages, balance_names, basis)
        df = pd.concat([df, balances], axis=1)
        hallmark_cols = [c for c in df.columns if c.startswith("HALLMARK_") and c.endswith("_score")]

        variables = (
            [(c, "hallmark") for c in hallmark_cols]
            + [(c, "ilr_balance") for c in balance_names]
            + [(c, "covariate") for c in QC_COLS if c in df.columns]
        )

        for threshold in THRESHOLDS:
            sub = df[pd.to_numeric(df["Malignant"], errors="coerce") >= threshold].copy()
            availability_rows.append(
                {
                    "dataset": dataset,
                    "sample": sample,
                    "sample_label": f"{dataset}__{sample}",
                    "malignant_threshold": threshold,
                    "n_spots": len(sub),
                    "n_exact_1_or_more": int((pd.to_numeric(df["Malignant"], errors="coerce") >= 0.999).sum()),
                    "eligible_for_quartile_contrast": len(sub) >= 2 * MIN_GROUP_N,
                }
            )
            if len(sub) < 2 * MIN_GROUP_N:
                continue

            low_cut, high_cut = sub["SNAI1-ac_score"].quantile([0.25, 0.75])
            low = sub[sub["SNAI1-ac_score"] <= low_cut]
            high = sub[sub["SNAI1-ac_score"] >= high_cut]
            if len(low) < MIN_GROUP_N or len(high) < MIN_GROUP_N:
                continue

            for variable, variable_type in variables:
                x = pd.to_numeric(sub[variable], errors="coerce")
                y = pd.to_numeric(sub["SNAI1-ac_score"], errors="coerce")
                ok = x.notna() & y.notna()
                if ok.sum() >= 3 and x[ok].nunique() > 1 and y[ok].nunique() > 1:
                    rho, pval = stats.spearmanr(x[ok], y[ok])
                else:
                    rho, pval = np.nan, np.nan

                spearman_rows.append(
                    {
                        "dataset": dataset,
                        "sample": sample,
                        "sample_label": f"{dataset}__{sample}",
                        "malignant_threshold": threshold,
                        "variable": variable,
                        "variable_type": variable_type,
                        "n_spots": len(sub),
                        "spearman_r": rho,
                        "spearman_p": pval,
                    }
                )

                contrast_rows.append(
                    {
                        "dataset": dataset,
                        "sample": sample,
                        "sample_label": f"{dataset}__{sample}",
                        "malignant_threshold": threshold,
                        "variable": variable,
                        "variable_type": variable_type,
                        "n_spots": len(sub),
                        "n_low": len(low),
                        "n_high": len(high),
                        "low_snai1ac_threshold": low_cut,
                        "high_snai1ac_threshold": high_cut,
                        "low_mean": pd.to_numeric(low[variable], errors="coerce").mean(),
                        "high_mean": pd.to_numeric(high[variable], errors="coerce").mean(),
                        "high_minus_low": pd.to_numeric(high[variable], errors="coerce").mean()
                        - pd.to_numeric(low[variable], errors="coerce").mean(),
                        "cohens_d": cohen_d(high[variable], low[variable]),
                    }
                )

    return pd.DataFrame(contrast_rows), pd.DataFrame(spearman_rows), pd.DataFrame(availability_rows)


def meta_summary(df: pd.DataFrame, value: str) -> pd.DataFrame:
    return (
        df.groupby(["malignant_threshold", "variable_type", "variable"], dropna=False)[value]
        .agg(
            n_samples="count",
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            mean="mean",
            positive_fraction=lambda x: (x > 0).mean(),
            negative_fraction=lambda x: (x < 0).mean(),
        )
        .reset_index()
    )


def hallmark_label(col: str) -> str:
    return (
        col.replace("HALLMARK_", "")
        .replace("_score", "")
        .replace("_", " ")
        .replace("OXIDATIVE PHOSPHORYLATION", "OXPHOS")
        .replace("EPITHELIAL MESENCHYMAL TRANSITION", "EMT")
        .replace("TNFA SIGNALING VIA NFKB", "TNF-a/NF-kB")
        .replace("IL6 JAK STAT3 SIGNALING", "IL6/JAK/STAT3")
        .replace("KRAS SIGNALING UP", "KRAS up")
        .replace("MYC TARGETS V1", "MYC targets")
        .replace("E2F TARGETS", "E2F targets")
        .title()
        .replace("OXPHOS", "OXPHOS")
        .replace("Emt", "EMT")
        .replace("Tnf-A/Nf-Kb", "TNF-a/NF-kB")
        .replace("Il6/Jak/Stat3", "IL6/JAK/STAT3")
        .replace("Kras Up", "KRAS up")
        .replace("Myc", "MYC")
        .replace("E2F", "E2F")
    )


def plot_heatmap(summary: pd.DataFrame, variable_type: str, variables: list[str], labels: dict[str, str] | None, out_name: str, title: str) -> None:
    sub = summary[(summary["variable_type"] == variable_type) & (summary["variable"].isin(variables))].copy()
    if labels is None:
        sub["label"] = sub["variable"].map(hallmark_label)
    else:
        sub["label"] = sub["variable"].map(labels)
    pivot = sub.pivot(index="label", columns="malignant_threshold", values="median")
    fig_h = max(3.0, 0.35 * len(pivot) + 1.4)
    fig, ax = plt.subplots(figsize=(7.2, fig_h))
    sns.heatmap(pivot, center=0, cmap="vlag", linewidths=0.5, linecolor="white", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Malignant fraction threshold")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(OUT / f"{out_name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{out_name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_availability(availability: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sns.boxplot(data=availability, x="malignant_threshold", y="n_spots", color="#D9D9D9", ax=ax, fliersize=2)
    sns.stripplot(data=availability, x="malignant_threshold", y="n_spots", color="0.25", size=3, jitter=0.16, ax=ax)
    ax.axhline(2 * MIN_GROUP_N, color="#B2182B", linestyle="--", linewidth=1)
    ax.set_title("Spots retained per sample after malignant-fraction thresholding")
    ax.set_xlabel("Malignant fraction threshold")
    ax.set_ylabel("Spots retained")
    fig.tight_layout()
    fig.savefig(OUT / "threshold_sample_availability.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "threshold_sample_availability.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)

    contrasts, spearman, availability = summarize()
    contrast_summary = meta_summary(contrasts, "cohens_d")
    spearman_summary = meta_summary(spearman, "spearman_r")

    contrasts.to_csv(OUT / "threshold_high_vs_low_cohens_d_per_sample.csv", index=False)
    spearman.to_csv(OUT / "threshold_continuous_spearman_per_sample.csv", index=False)
    availability.to_csv(OUT / "threshold_sample_availability.csv", index=False)
    contrast_summary.to_csv(OUT / "threshold_high_vs_low_cohens_d_summary.csv", index=False)
    spearman_summary.to_csv(OUT / "threshold_continuous_spearman_summary.csv", index=False)

    plot_heatmap(
        contrast_summary,
        "hallmark",
        FOCUS_HALLMARKS,
        labels=None,
        out_name="threshold_hallmark_high_vs_low_cohens_d_heatmap",
        title="Hallmark score differences: high vs low SNAI1-ac within malignant thresholds",
    )
    plot_heatmap(
        contrast_summary,
        "ilr_balance",
        FOCUS_BALANCES,
        labels=BALANCE_LABELS,
        out_name="threshold_ilr_primary_balance_high_vs_low_cohens_d_heatmap",
        title="Established ILR balance differences: high vs low SNAI1-ac within malignant thresholds",
    )
    plot_heatmap(
        contrast_summary,
        "covariate",
        QC_COLS,
        labels={c: c for c in QC_COLS},
        out_name="threshold_covariate_high_vs_low_cohens_d_heatmap",
        title="Covariate differences: high vs low SNAI1-ac within malignant thresholds",
    )
    plot_availability(availability)
    print(f"Wrote corrected Step 4 threshold diagnostics to: {OUT}")


if __name__ == "__main__":
    main()
