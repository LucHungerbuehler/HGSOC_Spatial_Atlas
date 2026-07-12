#!/usr/bin/env python
"""Compare k=5 SpaGCN domains with similar malignant fractions.

This analysis asks whether domains that are compositionally similar along the
malignant-fraction axis differ in SNAI1-ac score distribution and broad EnrichR
annotation themes.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VISIUM_DIR = Path(r"D:\HGSOC_Spatial_Atlas\05_analysis_ready\visium")
DEFAULT_ANNOTATION_DIR = Path(
    r"D:\HGSOC_Spatial_Atlas\05_analysis_ready\cross_sample\spagcn_k5_domain_annotations"
)
DEFAULT_OUTPUT_DIR = Path(
    r"D:\HGSOC_Spatial_Atlas\05_analysis_ready\cross_sample\spagcn_k5_matched_malignancy"
)

COHORT = {
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
        "CPS_OV1RtOV3",
        "CPS_OV5LtOV4",
        "CPS_OV19_LtOV1",
        "CPS_OV20RtOV4",
        "CPS_OV24RTOV4",
        "CPS_OV34RtOV1",
        "CPS_OV71_1",
    ],
}

THEME_PATTERNS = {
    "ECM_stromal": (
        r"extracellular matrix|collagen|focal adhesion|epithelial mesenchymal|"
        r"myogenesis|myofibroblast|fibroblast|stellate|mesenchymal|smoothmuscle"
    ),
    "Translation_ribosome": (
        r"translation|ribosom|peptide chain|macromolecule biosynthetic|"
        r"gene expression|myc targets"
    ),
    "Immune_inflammatory": (
        r"immune|interferon|allograft|antigen|b cell|t cell|complement|"
        r"cytokine|inflammatory|rejection|macrophage|nk|neutrophil"
    ),
    "Vascular_coag_hypoxia": (
        r"angiogenesis|hypoxia|coagulation|endothelial|blood vessel|"
        r"platelet|hemostasis"
    ),
    "Epithelial_tumor": (
        r"epithelial|ovary|ovarian|cancer cell|ductal|luminal|"
        r"cholangiocyte|estrogen|apical"
    ),
    "Metabolic_stress": (
        r"oxidative phosphorylation|mtorc1|glycolysis|hypoxia|peroxisome|"
        r"xenobiotic|fatty acid|cholesterol|unfolded protein|reactive oxygen"
    ),
}

THEME_COLORS = {
    "ECM_stromal": "#8c564b",
    "Translation_ribosome": "#9467bd",
    "Immune_inflammatory": "#1f77b4",
    "Vascular_coag_hypoxia": "#d62728",
    "Epithelial_tumor": "#2ca02c",
    "Metabolic_stress": "#ff7f0e",
    "No_recurrent_theme": "#7f7f7f",
    "Multi_theme": "#4d4d4d",
}

ANNOTATION_LIBRARIES = [
    "MSigDB_Hallmark_2020",
    "GO_Biological_Process_2023",
    "Reactome_2022",
    "KEGG_2021_Human",
    "WikiPathway_2023_Human",
    "PanglaoDB_Augmented_2021",
    "Human_Gene_Atlas",
]


def decode_value(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def read_categorical(obs: h5py.Group, key: str) -> np.ndarray:
    group = obs[key]
    categories = [decode_value(x) for x in group["categories"][:]]
    codes = group["codes"][:]
    return np.array(
        [categories[int(code)] if int(code) >= 0 else None for code in codes],
        dtype=object,
    )


def summarize_spot_distributions(dataset: str, sample: str, k: int) -> list[dict]:
    h5ad_path = VISIUM_DIR / dataset / sample / f"{sample}.h5ad"
    domain_key = f"spagcn_{k}_refined"

    rows = []
    with h5py.File(h5ad_path, "r") as handle:
        obs = handle["obs"]
        domains = read_categorical(obs, domain_key)
        lisa = read_categorical(obs, "LISA_category")
        score = obs["SNAI1-ac_score"][:].astype(float)

        for domain in sorted(set(domains), key=lambda x: int(x)):
            mask = domains == domain
            vals = score[mask]
            lisa_vals = lisa[mask]
            row = {
                "dataset": dataset,
                "sample": sample,
                "k": k,
                "domain": str(domain),
                "spot_n": int(mask.sum()),
                "snai1ac_mean": float(np.mean(vals)),
                "snai1ac_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                "snai1ac_median": float(np.median(vals)),
                "snai1ac_q10": float(np.quantile(vals, 0.10)),
                "snai1ac_q25": float(np.quantile(vals, 0.25)),
                "snai1ac_q75": float(np.quantile(vals, 0.75)),
                "snai1ac_q90": float(np.quantile(vals, 0.90)),
                "snai1ac_iqr": float(np.quantile(vals, 0.75) - np.quantile(vals, 0.25)),
                "lisa_hh_n": int(np.sum(lisa_vals == "High-High")),
                "lisa_ll_n": int(np.sum(lisa_vals == "Low-Low")),
                "lisa_not_sig_n": int(np.sum(lisa_vals == "Not significant")),
            }
            row["lisa_hh_frac"] = row["lisa_hh_n"] / row["spot_n"]
            row["lisa_ll_frac"] = row["lisa_ll_n"] / row["spot_n"]
            rows.append(row)

    return rows


def compact_terms(values: pd.Series, max_terms: int = 3) -> str:
    terms = [str(v) for v in values.dropna().astype(str)]
    return "; ".join(terms[:max_terms])


def build_domain_annotation_table(
    annotation_dir: Path,
    distribution_rows: list[dict],
    padj_cutoff: float,
) -> pd.DataFrame:
    domain_summary = pd.read_csv(annotation_dir / "spagcn5_domain_marker_summary.csv")
    top = pd.read_csv(annotation_dir / "spagcn5_enrichr_top5_by_library.csv")
    top["adjusted_p_value"] = pd.to_numeric(top["adjusted_p_value"], errors="coerce")
    top = top[
        (top["adjusted_p_value"] < padj_cutoff)
        & (top["library"].isin(ANNOTATION_LIBRARIES))
    ].copy()

    comp_cols = [
        col
        for col in domain_summary.columns
        if col.startswith("spacet_") and col != "spacet_Unidentifiable"
    ]
    domain_summary["dominant_spacet"] = (
        domain_summary[comp_cols]
        .astype(float)
        .idxmax(axis=1)
        .str.replace("spacet_", "", regex=False)
    )

    keys = ["dataset", "sample", "domain"]
    domain_summary["domain"] = domain_summary["domain"].astype(str)
    distribution = pd.DataFrame(distribution_rows)
    distribution["domain"] = distribution["domain"].astype(str)

    annotated = domain_summary.merge(distribution, on=["dataset", "sample", "k", "domain"], how="left")

    for library in ANNOTATION_LIBRARIES:
        terms = (
            top.loc[top["library"] == library]
            .sort_values(keys + ["rank"])
            .groupby(keys)["term"]
            .apply(compact_terms)
            .reset_index(name=f"top_terms_{library}")
        )
        terms["domain"] = terms["domain"].astype(str)
        annotated = annotated.merge(terms, on=keys, how="left")

    term_columns = [col for col in annotated.columns if col.startswith("top_terms_")]
    annotated["annotation_text"] = annotated[term_columns].fillna("").agg(" | ".join, axis=1)
    for theme, pattern in THEME_PATTERNS.items():
        annotated[f"theme_{theme}"] = annotated["annotation_text"].str.contains(
            pattern, flags=re.IGNORECASE, regex=True
        )

    theme_cols = [f"theme_{theme}" for theme in THEME_PATTERNS]
    annotated["theme_n"] = annotated[theme_cols].sum(axis=1)

    def label_themes(row: pd.Series) -> str:
        labels = [theme for theme in THEME_PATTERNS if bool(row[f"theme_{theme}"])]
        return ";".join(labels) if labels else "No_recurrent_theme"

    annotated["theme_label"] = annotated.apply(label_themes, axis=1)
    return annotated


def signed_delta(row_a: pd.Series, row_b: pd.Series, column: str) -> float:
    return float(row_a[column]) - float(row_b[column])


def build_pair_table(annotated: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    pairs = []
    value_cols = [
        "spacet_Malignant",
        "spacet_CAF",
        "spacet_Macrophage",
        "spacet_Endothelial",
        "spacet_Unidentifiable",
        "snai1ac_mean",
        "snai1ac_median",
        "snai1ac_q90",
        "snai1ac_iqr",
        "lisa_hh_frac",
        "lisa_ll_frac",
    ]

    for (dataset, sample), sample_df in annotated.groupby(["dataset", "sample"], sort=False):
        sample_domains = list(sample_df.sort_values("domain").iterrows())
        for (_, a), (_, b) in itertools.combinations(sample_domains, 2):
            snai_delta = signed_delta(a, b, "snai1ac_mean")
            high, low = (a, b) if snai_delta >= 0 else (b, a)
            pair = {
                "dataset": dataset,
                "sample": sample,
                "domain_a": a["domain"],
                "domain_b": b["domain"],
                "domain_high_snai": high["domain"],
                "domain_low_snai": low["domain"],
                "malignant_abs_delta": abs(signed_delta(a, b, "spacet_Malignant")),
                "snai1ac_mean_abs_delta": abs(snai_delta),
                "snai1ac_median_abs_delta": abs(signed_delta(a, b, "snai1ac_median")),
                "snai1ac_q90_abs_delta": abs(signed_delta(a, b, "snai1ac_q90")),
                "hh_frac_abs_delta": abs(signed_delta(a, b, "lisa_hh_frac")),
                "ll_frac_abs_delta": abs(signed_delta(a, b, "lisa_ll_frac")),
                "high_theme_label": high["theme_label"],
                "low_theme_label": low["theme_label"],
                "high_dominant_spacet": high["dominant_spacet"],
                "low_dominant_spacet": low["dominant_spacet"],
                "high_top_hallmark": high.get("top_terms_MSigDB_Hallmark_2020", ""),
                "low_top_hallmark": low.get("top_terms_MSigDB_Hallmark_2020", ""),
                "high_top_go": high.get("top_terms_GO_Biological_Process_2023", ""),
                "low_top_go": low.get("top_terms_GO_Biological_Process_2023", ""),
                "high_primary_marker_n": int(high["primary_marker_n"]),
                "low_primary_marker_n": int(low["primary_marker_n"]),
            }
            for threshold in thresholds:
                pair[f"matched_malignancy_le_{threshold:g}"] = (
                    pair["malignant_abs_delta"] <= threshold
                )
            for col in value_cols:
                pair[f"high_{col}"] = high[col]
                pair[f"low_{col}"] = low[col]
                pair[f"high_minus_low_{col}"] = signed_delta(high, low, col)
            pairs.append(pair)

    return pd.DataFrame(pairs)


def build_nearest_pairs(pair_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    seen = set()
    for (dataset, sample), sample_pairs in pair_table.groupby(["dataset", "sample"], sort=False):
        domains = sorted(set(sample_pairs["domain_a"]) | set(sample_pairs["domain_b"]), key=int)
        for domain in domains:
            candidates = sample_pairs[
                (sample_pairs["domain_a"] == domain) | (sample_pairs["domain_b"] == domain)
            ].sort_values(["malignant_abs_delta", "snai1ac_mean_abs_delta"], ascending=[True, False])
            if candidates.empty:
                continue
            row = candidates.iloc[0].copy()
            pair_key = tuple(sorted([str(row["domain_a"]), str(row["domain_b"])]))
            key = (dataset, sample, pair_key)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_pairs(pair_table: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        subset = pair_table.loc[pair_table[f"matched_malignancy_le_{threshold:g}"]].copy()
        if subset.empty:
            continue
        rows.append(
            {
                "matched_malignancy_threshold": threshold,
                "pair_n": len(subset),
                "sample_n": subset[["dataset", "sample"]].drop_duplicates().shape[0],
                "median_malignant_abs_delta": subset["malignant_abs_delta"].median(),
                "median_snai1ac_mean_abs_delta": subset["snai1ac_mean_abs_delta"].median(),
                "median_snai1ac_median_abs_delta": subset["snai1ac_median_abs_delta"].median(),
                "median_snai1ac_q90_abs_delta": subset["snai1ac_q90_abs_delta"].median(),
                "median_hh_frac_abs_delta": subset["hh_frac_abs_delta"].median(),
                "median_ll_frac_abs_delta": subset["ll_frac_abs_delta"].median(),
                "pairs_with_mean_delta_gt_0_25": int((subset["snai1ac_mean_abs_delta"] > 0.25).sum()),
                "pairs_with_mean_delta_gt_0_5": int((subset["snai1ac_mean_abs_delta"] > 0.5).sum()),
                "pairs_with_hh_frac_delta_gt_0_1": int((subset["hh_frac_abs_delta"] > 0.10).sum()),
            }
        )
    return pd.DataFrame(rows)


def theme_direction_summary(pair_table: pd.DataFrame, threshold: float) -> pd.DataFrame:
    subset = pair_table.loc[pair_table[f"matched_malignancy_le_{threshold:g}"]].copy()
    rows = []
    for side in ["high", "low"]:
        expanded = []
        for labels in subset[f"{side}_theme_label"].fillna("No_recurrent_theme"):
            expanded.extend(labels.split(";"))
        counts = pd.Series(expanded).value_counts()
        for theme, count in counts.items():
            rows.append(
                {
                    "matched_malignancy_threshold": threshold,
                    "snai1ac_side": side,
                    "theme": theme,
                    "count": int(count),
                }
            )
    return pd.DataFrame(rows)


def primary_theme(theme_label: str) -> str:
    if not isinstance(theme_label, str) or not theme_label:
        return "No_recurrent_theme"
    themes = [theme for theme in theme_label.split(";") if theme]
    if not themes:
        return "No_recurrent_theme"
    if len(themes) > 2:
        return "Multi_theme"
    return themes[0]


def create_matched_pair_figures(pair_table: pd.DataFrame, output_dir: Path) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    strict_col = "matched_malignancy_le_0.05"
    loose_col = "matched_malignancy_le_0.1"

    strict = pair_table.loc[pair_table[strict_col]].copy()
    if not strict.empty:
        strict["high_primary_theme"] = strict["high_theme_label"].apply(primary_theme)
        strict = strict.sort_values("snai1ac_mean_abs_delta", ascending=True)
        strict["pair_label"] = (
            strict["dataset"].str.replace("visium/", "", regex=False)
            + " / "
            + strict["sample"].astype(str)
            + "  d"
            + strict["domain_low_snai"].astype(str)
            + " -> d"
            + strict["domain_high_snai"].astype(str)
        )

        height = max(7, 0.32 * len(strict) + 1.8)
        fig, ax = plt.subplots(figsize=(9.5, height))
        y = np.arange(len(strict))
        for i, (_, row) in enumerate(strict.iterrows()):
            color = THEME_COLORS.get(row["high_primary_theme"], THEME_COLORS["No_recurrent_theme"])
            ax.plot(
                [row["low_snai1ac_mean"], row["high_snai1ac_mean"]],
                [i, i],
                color=color,
                linewidth=2.2,
                alpha=0.75,
            )
            ax.scatter(row["low_snai1ac_mean"], i, color="#d9d9d9", edgecolor="#595959", s=30, zorder=3)
            ax.scatter(row["high_snai1ac_mean"], i, color=color, edgecolor="black", s=38, zorder=4)

        ax.axvline(0, color="#8c8c8c", linewidth=1, linestyle="--")
        ax.set_yticks(y)
        ax.set_yticklabels(strict["pair_label"], fontsize=7)
        ax.set_xlabel("Domain mean SNAI1-ac score")
        ax.set_title("k=5 SpaGCN domains matched for malignant fraction (|delta malignant| <= 0.05)")
        ax.grid(axis="x", color="#e6e6e6", linewidth=0.7)

        legend_handles = []
        for theme in sorted(strict["high_primary_theme"].unique()):
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    label=theme.replace("_", " "),
                    markerfacecolor=THEME_COLORS.get(theme, THEME_COLORS["No_recurrent_theme"]),
                    markeredgecolor="black",
                    markersize=6,
                )
            )
        ax.legend(handles=legend_handles, title="Higher-SNAI1-ac domain theme", loc="lower right", fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / "matched_malignancy_slopeplot_strict.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    matched = pair_table.loc[pair_table[loose_col]].copy()
    if not matched.empty:
        matched["match_band"] = np.where(
            matched[strict_col],
            "<=0.05",
            "0.05-0.10",
        )
        colors = {"<=0.05": "#2f6fbb", "0.05-0.10": "#9ecae1"}

        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        for band, group in matched.groupby("match_band"):
            ax.scatter(
                group["malignant_abs_delta"],
                group["snai1ac_mean_abs_delta"],
                s=28 + 140 * group["hh_frac_abs_delta"],
                color=colors[band],
                edgecolor="black",
                linewidth=0.4,
                alpha=0.85,
                label=band,
            )
        ax.axvline(0.05, color="#595959", linestyle="--", linewidth=1)
        ax.axhline(0.25, color="#b2182b", linestyle="--", linewidth=1)
        ax.axhline(0.50, color="#b2182b", linestyle=":", linewidth=1)
        ax.set_xlabel("Absolute difference in malignant fraction")
        ax.set_ylabel("Absolute difference in domain mean SNAI1-ac score")
        ax.set_title("SNAI1-ac domain differences after malignant-fraction matching")
        ax.set_xlim(-0.005, 0.105)
        ax.grid(color="#e6e6e6", linewidth=0.7)
        ax.legend(title="Match band", frameon=True)
        fig.tight_layout()
        fig.savefig(fig_dir / "matched_malignancy_delta_scatter.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--padj-cutoff", type=float, default=0.05)
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.05, 0.10])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    distribution_rows = []
    for dataset, samples in COHORT.items():
        for sample in samples:
            print(f"Summarizing {dataset}/{sample}")
            distribution_rows.extend(summarize_spot_distributions(dataset, sample, args.k))

    annotated = build_domain_annotation_table(
        args.annotation_dir,
        distribution_rows,
        args.padj_cutoff,
    )
    pair_table = build_pair_table(annotated, args.thresholds)
    nearest_pairs = build_nearest_pairs(pair_table)
    pair_summary = summarize_pairs(pair_table, args.thresholds)
    theme_summary = pd.concat(
        [theme_direction_summary(pair_table, threshold) for threshold in args.thresholds],
        ignore_index=True,
    )

    annotated.to_csv(args.output_dir / "spagcn5_domain_annotation_distribution_summary.csv", index=False)
    pair_table.to_csv(args.output_dir / "spagcn5_within_sample_domain_pairs.csv", index=False)
    nearest_pairs.to_csv(args.output_dir / "spagcn5_nearest_malignancy_neighbor_pairs.csv", index=False)
    pair_summary.to_csv(args.output_dir / "spagcn5_matched_pair_summary.csv", index=False)
    theme_summary.to_csv(args.output_dir / "spagcn5_matched_pair_theme_direction_summary.csv", index=False)
    create_matched_pair_figures(pair_table, args.output_dir)

    manifest = {
        "k": args.k,
        "cohort": COHORT,
        "annotation_dir": str(args.annotation_dir),
        "padj_cutoff": args.padj_cutoff,
        "malignancy_match_thresholds": args.thresholds,
        "theme_patterns": THEME_PATTERNS,
        "annotation_libraries": ANNOTATION_LIBRARIES,
        "outputs": [
            "spagcn5_domain_annotation_distribution_summary.csv",
            "spagcn5_within_sample_domain_pairs.csv",
            "spagcn5_nearest_malignancy_neighbor_pairs.csv",
            "spagcn5_matched_pair_summary.csv",
            "spagcn5_matched_pair_theme_direction_summary.csv",
            "figures/matched_malignancy_slopeplot_strict.png",
            "figures/matched_malignancy_delta_scatter.png",
        ],
    }
    (args.output_dir / "spagcn5_matched_malignancy_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nDone")
    print(f"Output: {args.output_dir}")
    print(pair_summary.to_string(index=False))


if __name__ == "__main__":
    main()
