"""Build report assets for Results section 2.5 C-SIDE evidence ladder.

This script only uses saved C-SIDE/S2e audit outputs. It does not rerun C-SIDE.
Outputs are written to the S2e audit branch under 09_section25_report_assets.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(r"D:\HGSOC_Spatial_Atlas")
S2E = ROOT / "05_analysis_ready" / "S2e_CSIDE_CellTypeSpecific_DE_Audit"
OUT = S2E / "09_section25_report_assets"
FIG = OUT / "figures"
TAB = OUT / "tables"

MAIN_CELL_TYPES = ["Epithelial", "Fibroblast", "Macrophage", "CAF", "Endothelial"]
DATASET_ORDER = ["denisenko_2022", "ju_2024", "yamamoto_2025"]
CELLTYPE_COLORS = {
    "Epithelial": "#5477C4",
    "Fibroblast": "#71B436",
    "Macrophage": "#CC6F47",
    "CAF": "#BD569B",
    "Endothelial": "#386411",
}
DATASET_COLORS = {
    "denisenko_2022": "#5477C4",
    "ju_2024": "#CC6F47",
    "yamamoto_2025": "#71B436",
}
TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def bh_sig(series: pd.Series, alpha: float = 0.05) -> pd.Series:
    return pd.to_numeric(series, errors="coerce") < alpha


def clean_pathway_name(pathway: str) -> str:
    text = str(pathway)
    for prefix in ("HALLMARK_", "KEGG_"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    text = text.replace("_", " ").title()
    replacements = {
        "Dna": "DNA",
        "Rna": "RNA",
        "Nfkb": "NF-kB",
        "Kras": "KRAS",
        "Myc": "MYC",
        "E2F": "E2F",
        "Il2": "IL2",
        "Il6": "IL6",
        "Jak": "JAK",
        "Stat3": "STAT3",
        "Tnfa": "TNFA",
        "Uv": "UV",
        "Mtorc1": "MTORC1",
        "Oxidative Phosphorylation": "Oxidative Phosphorylation",
        "Ecm": "ECM",
        "Iga": "IgA",
        "Dn": "DN",
        "Up": "UP",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def to_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    text = df.to_latex(index=False, escape=False, longtable=False)
    text = text.replace("\\begin{tabular}", f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\begin{{tabular}}")
    path.write_text(text, encoding="utf-8")


def style_axis(ax) -> None:
    ax.set_facecolor(TOKENS["panel"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])
    ax.tick_params(colors=TOKENS["muted"], labelsize=8)
    ax.xaxis.label.set_color(TOKENS["ink"])
    ax.yaxis.label.set_color(TOKENS["ink"])
    ax.title.set_color(TOKENS["ink"])


def pearson_safe(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return np.nan
    if x[mask].std() == 0 or y[mask].std() == 0:
        return np.nan
    return float(x[mask].corr(y[mask]))


def build_coverage_table() -> pd.DataFrame:
    audit = read_csv(S2E / "01_signed_ranking_audit" / "signed_z_reconstruction_audit_by_sample_celltype.csv")
    rows = []
    for ct in MAIN_CELL_TYPES:
        sub = audit[audit["cell_type"] == ct]
        rows.append(
            {
                "cell_type": ct,
                "testable_samples": int(sub.shape[0]),
                "gene_level_fits": int(sub["n_rows"].sum()),
                "converged_fits": int(sub["n_converged"].sum()),
            }
        )
    return pd.DataFrame(rows)


def build_pathway_narrowing_table() -> pd.DataFrame:
    old_h = read_csv(ROOT / "scRNA_reference" / "rctd_outputs" / "gsea_results" / "gsea_summary_table.csv")
    old_h = old_h.rename(columns={"Cell Type": "cell_type", "padj": "padj"})
    if "padj" in old_h.columns:
        old_h = old_h[bh_sig(old_h["padj"])]
    old_h_counts = old_h.groupby("cell_type").size().reindex(MAIN_CELL_TYPES, fill_value=0)

    old_k = read_csv(ROOT / "scRNA_reference" / "rctd_outputs" / "gsea_results" / "kegg" / "gsea_kegg_significant.csv")
    if "padj" in old_k.columns:
        old_k = old_k[bh_sig(old_k["padj"])]
    old_k_counts = old_k.groupby("cell_type").size().reindex(MAIN_CELL_TYPES, fill_value=0)

    meanz_ng_h = read_csv(
        S2E / "02b_paper_style_meanZ_gene_set_tests_no_logfc_gate" / "hallmark" / "hallmark_meanZ_permutation_results.csv"
    )
    meanz_ng_k = read_csv(
        S2E / "02b_paper_style_meanZ_gene_set_tests_no_logfc_gate" / "kegg" / "kegg_meanZ_permutation_results.csv"
    )
    meanz_ng_h_counts = meanz_ng_h[bh_sig(meanz_ng_h["q_perm_bh"])].groupby("cell_type").size().reindex(MAIN_CELL_TYPES, fill_value=0)
    meanz_ng_k_counts = meanz_ng_k[bh_sig(meanz_ng_k["q_perm_bh"])].groupby("cell_type").size().reindex(MAIN_CELL_TYPES, fill_value=0)

    sample_ng_h = read_csv(
        S2E / "04b_sample_dataset_heterogeneity_no_logfc_gate" / "hallmark" / "hallmark_pathway_sample_dataset_heterogeneity_summary.csv"
    )
    sample_ng_k = read_csv(
        S2E / "04b_sample_dataset_heterogeneity_no_logfc_gate" / "kegg" / "kegg_pathway_sample_dataset_heterogeneity_summary.csv"
    )
    sample_ng_h_counts = sample_ng_h[bh_sig(sample_ng_h["wilcoxon_q_bh"])].groupby("cell_type").size().reindex(MAIN_CELL_TYPES, fill_value=0)
    sample_ng_k_counts = sample_ng_k[bh_sig(sample_ng_k["wilcoxon_q_bh"])].groupby("cell_type").size().reindex(MAIN_CELL_TYPES, fill_value=0)

    meanz_g_h = read_csv(S2E / "02_paper_style_meanZ_gene_set_tests" / "hallmark" / "hallmark_meanZ_permutation_results.csv")
    meanz_g_k = read_csv(S2E / "02_paper_style_meanZ_gene_set_tests" / "kegg" / "kegg_meanZ_permutation_results.csv")
    meanz_g_h_counts = meanz_g_h[bh_sig(meanz_g_h["q_perm_bh"])].groupby("cell_type").size().reindex(MAIN_CELL_TYPES, fill_value=0)
    meanz_g_k_counts = meanz_g_k[bh_sig(meanz_g_k["q_perm_bh"])].groupby("cell_type").size().reindex(MAIN_CELL_TYPES, fill_value=0)

    sample_g_h = read_csv(S2E / "04_sample_dataset_heterogeneity" / "hallmark" / "hallmark_pathway_sample_dataset_heterogeneity_summary.csv")
    sample_g_k = read_csv(S2E / "04_sample_dataset_heterogeneity" / "kegg" / "kegg_pathway_sample_dataset_heterogeneity_summary.csv")
    sample_g_h_counts = sample_g_h[bh_sig(sample_g_h["wilcoxon_q_bh"])].groupby("cell_type").size().reindex(MAIN_CELL_TYPES, fill_value=0)
    sample_g_k_counts = sample_g_k[bh_sig(sample_g_k["wilcoxon_q_bh"])].groupby("cell_type").size().reindex(MAIN_CELL_TYPES, fill_value=0)

    return pd.DataFrame(
        {
            "cell_type": MAIN_CELL_TYPES,
            "fgsea_hallmark_q_lt_0_05": old_h_counts.values,
            "fgsea_kegg_q_lt_0_05": old_k_counts.values,
            "meanZ_no_logfc_hallmark_q_lt_0_05": meanz_ng_h_counts.values,
            "meanZ_no_logfc_kegg_q_lt_0_05": meanz_ng_k_counts.values,
            "sample_no_logfc_hallmark_q_lt_0_05": sample_ng_h_counts.values,
            "sample_no_logfc_kegg_q_lt_0_05": sample_ng_k_counts.values,
            "meanZ_abs_mean_logfc_ge_1_hallmark_q_lt_0_05": meanz_g_h_counts.values,
            "meanZ_abs_mean_logfc_ge_1_kegg_q_lt_0_05": meanz_g_k_counts.values,
            "sample_abs_logfc_ge_1_hallmark_q_lt_0_05": sample_g_h_counts.values,
            "sample_abs_logfc_ge_1_kegg_q_lt_0_05": sample_g_k_counts.values,
        }
    )


def build_setlevel_summary() -> pd.DataFrame:
    df = read_csv(S2E / "07_report_ready_packaging" / "tables" / "setlevel_permutation_report_table.csv")
    df = df[df["cell_type"].isin(MAIN_CELL_TYPES)].copy()
    df["threshold_T"] = pd.to_numeric(df["threshold_T"])
    df["obs_over_null_mean"] = pd.to_numeric(df["obs_over_null_mean"])
    df["bh_q"] = pd.to_numeric(df["bh_q"])
    df["significant"] = df["bh_q"] < 0.05
    return df


def build_gene_sign_summary() -> pd.DataFrame:
    df = read_csv(S2E / "08_per_gene_direction_sign_test" / "per_gene_direction_sign_test_summary_by_celltype.csv")
    return df[df["cell_type"].isin(MAIN_CELL_TYPES)].copy()


def build_robust_core_summary() -> pd.DataFrame:
    df = read_csv(S2E / "07_report_ready_packaging" / "tables" / "cside_robust_core_73_summary_by_celltype.csv")
    return df[df["cell_type"].isin(MAIN_CELL_TYPES)].copy()


def build_meanz_hits_table(no_logfc_gate: bool = True) -> pd.DataFrame:
    base = (
        S2E / "02b_paper_style_meanZ_gene_set_tests_no_logfc_gate"
        if no_logfc_gate
        else S2E / "02_paper_style_meanZ_gene_set_tests"
    )
    frames = []
    for collection, rel in [
        ("Hallmark", Path("hallmark") / "hallmark_meanZ_permutation_results.csv"),
        ("KEGG", Path("kegg") / "kegg_meanZ_permutation_results.csv"),
    ]:
        df = read_csv(base / rel)
        df = df[bh_sig(df["q_perm_bh"])].copy()
        df["collection"] = collection
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out[out["cell_type"].isin(MAIN_CELL_TYPES)].copy()
    out = out[
        ["collection", "cell_type", "pathway", "n_genes", "mean_z", "direction", "p_perm_two_sided", "q_perm_bh"]
    ].sort_values(["collection", "cell_type", "q_perm_bh", "pathway"])
    out["pathway_label"] = out["pathway"].map(clean_pathway_name)
    return out


def build_sample_level_hits_table(no_logfc_gate: bool = True) -> pd.DataFrame:
    base = (
        S2E / "04b_sample_dataset_heterogeneity_no_logfc_gate"
        if no_logfc_gate
        else S2E / "04_sample_dataset_heterogeneity"
    )
    frames = []
    for collection, rel in [
        ("Hallmark", Path("hallmark") / "hallmark_pathway_sample_dataset_heterogeneity_summary.csv"),
        ("KEGG", Path("kegg") / "kegg_pathway_sample_dataset_heterogeneity_summary.csv"),
    ]:
        df = read_csv(base / rel)
        df = df[bh_sig(df["wilcoxon_q_bh"])].copy()
        df["collection"] = collection
        frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        return pd.DataFrame(
            columns=[
                "collection",
                "cell_type",
                "pathway",
                "pathway_label",
                "n_samples",
                "n_datasets",
                "mean_sample_mean_z",
                "n_positive_samples",
                "n_negative_samples",
                "sample_direction_consistency",
                "wilcoxon_p_vs_zero",
                "wilcoxon_q_bh",
            ]
        )
    out = out[out["cell_type"].isin(MAIN_CELL_TYPES)].copy()
    out["pathway_label"] = out["pathway"].map(clean_pathway_name)
    return out[
        [
            "collection",
            "cell_type",
            "pathway",
            "pathway_label",
            "n_samples",
            "n_datasets",
            "mean_sample_mean_z",
            "n_positive_samples",
            "n_negative_samples",
            "sample_direction_consistency",
            "wilcoxon_p_vs_zero",
            "wilcoxon_q_bh",
        ]
    ].sort_values(["collection", "cell_type", "wilcoxon_q_bh", "pathway"])


def build_celltype_inclusion_audit() -> pd.DataFrame:
    signed = read_csv(S2E / "01_signed_ranking_audit" / "cside_2cov_signed_gene_level_all_samples.csv")
    rows = []
    for ct, sub in signed.groupby("cell_type", sort=False):
        n_samples = int(sub["sample_label"].nunique())
        n_datasets = int(sub["dataset"].nunique())
        role = "main section" if ct in MAIN_CELL_TYPES else "excluded from main section"
        if ct in MAIN_CELL_TYPES:
            reason = ">=13 testable samples across all three datasets; used in the main evidence ladder."
        elif n_samples < 3:
            reason = "Too few testable samples for cross-sample C-SIDE evidence."
        else:
            reason = "Lower support than the main five cell types; retained as audit context only."
        rows.append(
            {
                "cell_type": ct,
                "testable_samples": n_samples,
                "datasets": n_datasets,
                "gene_level_fits": int(sub.shape[0]),
                "report_role": role,
                "reason": reason,
            }
        )
    out = pd.DataFrame(rows)
    out["main_section_order"] = out["cell_type"].map({ct: i for i, ct in enumerate(MAIN_CELL_TYPES)}).fillna(99)
    out = out.sort_values(["report_role", "main_section_order", "testable_samples", "gene_level_fits"],
                          ascending=[False, True, False, False])
    return out.drop(columns=["main_section_order"])


def build_model_setup_by_sample() -> pd.DataFrame:
    manifest = read_csv(S2E / "00_manifest" / "cside_sample_manifest.csv")
    rows = []
    for _, row in manifest.iterrows():
        dataset = row["dataset"]
        sample = row["sample_id_on_disk"]
        metadata_path = ROOT / "scRNA_reference" / "rctd_inputs" / dataset / sample / "metadata.csv"
        if not metadata_path.exists():
            rows.append(
                {
                    "dataset": dataset,
                    "sample_id_on_disk": sample,
                    "sample_label": row["sample_label"],
                    "metadata_path": str(metadata_path),
                    "status": "missing_metadata",
                }
            )
            continue
        meta = pd.read_csv(metadata_path, index_col=0)
        required = ["SNAI1-ac_score", "Malignant"]
        complete = meta.dropna(subset=[c for c in required if c in meta.columns]).copy()
        if not all(c in meta.columns for c in required) or complete.empty:
            rows.append(
                {
                    "dataset": dataset,
                    "sample_id_on_disk": sample,
                    "sample_label": row["sample_label"],
                    "metadata_path": str(metadata_path),
                    "status": "missing_or_empty_required_covariates",
                }
            )
            continue
        snai = pd.to_numeric(complete["SNAI1-ac_score"], errors="coerce")
        mal = pd.to_numeric(complete["Malignant"], errors="coerce")
        rows.append(
            {
                "dataset": dataset,
                "sample_id_on_disk": sample,
                "sample_label": row["sample_label"],
                "metadata_path": str(metadata_path),
                "status": "ok",
                "n_spots_with_model_covariates": int((snai.notna() & mal.notna()).sum()),
                "snai1ac_min": float(snai.min()),
                "snai1ac_median": float(snai.median()),
                "snai1ac_max": float(snai.max()),
                "malignant_min": float(mal.min()),
                "malignant_median": float(mal.median()),
                "malignant_max": float(mal.max()),
                "pearson_snai1ac_vs_malignant": pearson_safe(snai, mal),
            }
        )
    return pd.DataFrame(rows)


def representative_pairs() -> pd.DataFrame:
    pairs = read_csv(S2E / "07_report_ready_packaging" / "tables" / "cside_robust_core_73_gene_celltype_associations.csv")
    pairs = pairs[pairs["cell_type"].isin(MAIN_CELL_TYPES)].copy()
    pairs["pair_label"] = pairs["cell_type"].str.replace("Epithelial", "Epi", regex=False) + ":" + pairs["gene"]
    order = {"Epithelial": 0, "Endothelial": 1, "Fibroblast": 2, "Macrophage": 3, "CAF": 4}
    pairs["order"] = pairs["cell_type"].map(order)
    pairs["random_q"] = pd.to_numeric(pairs["random_q"], errors="coerce")
    pairs = pairs.sort_values(["order", "random_q", "gene"]).drop(columns=["order"])
    return pairs


def write_selection_template(candidates: pd.DataFrame) -> Path:
    """Write a full candidate table for user selection without overwriting edits."""
    path = TAB / "section25_cside_gene_selection_for_figures.csv"
    if path.exists():
        return path
    template = candidates.copy()
    template.insert(0, "selected_for_figure", False)
    keep = [
        "selected_for_figure",
        "cell_type",
        "gene",
        "direction",
        "n_samples",
        "n_datasets",
        "sign_consistency",
        "random_q",
        "i2",
        "is_snai1ac_signature_gene",
        "pair_label",
    ]
    template = template[[c for c in keep if c in template.columns]]
    template.to_csv(path, index=False)
    return path


def remove_unapproved_selected_gene_assets() -> None:
    stale_names = [
        "section25_cside_celltype_specificity_matrix.csv",
        "section25_cside_sample_signed_z_profiles.csv",
        "section25_cside_representative_robust_pairs.csv",
    ]
    stale_figs = [
        "section25_cside_celltype_specificity_matrix.png",
        "section25_cside_celltype_specificity_matrix.pdf",
        "section25_cside_sample_signed_z_profiles.png",
        "section25_cside_sample_signed_z_profiles.pdf",
    ]
    for name in stale_names:
        path = TAB / name
        if path.exists():
            path.unlink()
    for name in stale_figs:
        path = FIG / name
        if path.exists():
            path.unlink()


def remove_unapproved_python_figures() -> None:
    """Remove draft Python figures so they are not mistaken for C-SIDE-style plots."""
    stale_figs = [
        "section25_cside_evidence_ladder_composite.png",
        "section25_cside_evidence_ladder_composite.pdf",
        "section25_cside_model_setup_covariates.png",
        "section25_cside_model_setup_covariates.pdf",
        "section25_cside_celltype_specificity_matrix.png",
        "section25_cside_celltype_specificity_matrix.pdf",
        "section25_cside_sample_signed_z_profiles.png",
        "section25_cside_sample_signed_z_profiles.pdf",
    ]
    for name in stale_figs:
        path = FIG / name
        if path.exists():
            path.unlink()


def build_specificity_matrix(pairs: pd.DataFrame) -> pd.DataFrame:
    meta = read_csv(S2E / "01_signed_ranking_audit" / "meta_gene_effects_signed_stouffer_iv_random_effects.csv")
    genes = pairs["gene"].drop_duplicates().tolist()
    sub = meta[meta["gene"].isin(genes) & meta["cell_type"].isin(MAIN_CELL_TYPES)].copy()
    core_pairs = set(zip(pairs["cell_type"], pairs["gene"]))
    sub["is_selected_robust_pair"] = list(zip(sub["cell_type"], sub["gene"]))
    sub["is_selected_robust_pair"] = sub["is_selected_robust_pair"].isin(core_pairs)
    gene_order = {gene: i for i, gene in enumerate(genes)}
    sub["gene_order"] = sub["gene"].map(gene_order)
    return sub.sort_values(["gene_order", "cell_type"]).drop(columns=["gene_order"])


def build_sample_profiles(pairs: pd.DataFrame) -> pd.DataFrame:
    signed = read_csv(S2E / "01_signed_ranking_audit" / "cside_2cov_signed_gene_level_all_samples.csv")
    frames = []
    for _, pair in pairs.iterrows():
        sub = signed[(signed["cell_type"] == pair["cell_type"]) & (signed["gene"] == pair["gene"])].copy()
        if sub.empty:
            continue
        sub["pair_label"] = pair["pair_label"]
        sub["robust_direction"] = pair["direction"]
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["dataset"] = pd.Categorical(out["dataset"], categories=DATASET_ORDER, ordered=True)
    return out.sort_values(["pair_label", "dataset", "sample_label"])


def make_model_setup_figure(model_setup: pd.DataFrame) -> None:
    ok = model_setup[model_setup["status"] == "ok"].copy()
    ok["dataset"] = pd.Categorical(ok["dataset"], categories=DATASET_ORDER, ordered=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.patch.set_facecolor(TOKENS["surface"])

    ax = axes[0]
    ax.set_axis_off()
    ax.set_title("A. C-SIDE Model 2 setup", loc="left", fontsize=11, fontweight="bold", color=TOKENS["ink"])
    boxes = [
        (0.04, 0.68, 0.25, 0.16, "Spot-level\nVisium counts"),
        (0.04, 0.38, 0.25, 0.16, "RCTD cell-type\nweights"),
        (0.39, 0.68, 0.25, 0.16, "SNAI1-ac score\nmin-max scaled"),
        (0.39, 0.38, 0.25, 0.16, "Malignant fraction\nmin-max scaled"),
        (0.73, 0.52, 0.23, 0.19, "C-SIDE per\ncell type"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor="#F4F5F7", edgecolor="#C5CAD3", linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, color=TOKENS["ink"])
    arrows = [
        ((0.29, 0.76), (0.39, 0.76)),
        ((0.29, 0.46), (0.73, 0.56)),
        ((0.64, 0.76), (0.73, 0.62)),
        ((0.64, 0.46), (0.73, 0.57)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color="#7A828F", lw=1.4))
    ax.text(
        0.04,
        0.14,
        "Model tested: expression ~ intercept + SNAI1-ac + malignant fraction\n"
        "Report statistic: the SNAI1-ac coefficient within each cell type and sample",
        ha="left",
        va="center",
        fontsize=9,
        color=TOKENS["muted"],
    )

    ax = axes[1]
    style_axis(ax)
    rng = np.random.default_rng(42)
    for i, dataset in enumerate(DATASET_ORDER):
        sub = ok[ok["dataset"] == dataset]
        jitter = rng.normal(0, 0.035, len(sub))
        ax.scatter(
            np.full(len(sub), i) + jitter,
            sub["pearson_snai1ac_vs_malignant"],
            s=np.clip(sub["n_spots_with_model_covariates"] / 18, 24, 120),
            color=DATASET_COLORS[dataset],
            edgecolor="#464C55",
            linewidth=0.6,
            alpha=0.85,
            label=dataset,
        )
    ax.axhline(0, color="#7A828F", linewidth=1, linestyle="--")
    ax.set_xticks(range(len(DATASET_ORDER)))
    ax.set_xticklabels([d.replace("_", "\n") for d in DATASET_ORDER])
    ax.set_ylabel("Pearson r")
    ax.set_title("B. SNAI1-ac and malignant fraction are sample-coupled", loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="y", color=TOKENS["grid"], linewidth=0.8)
    ax.text(
        0.02,
        -0.22,
        "Point size scales with spots carrying both model covariates.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color=TOKENS["muted"],
    )
    fig.suptitle("C-SIDE model setup and covariate context", fontsize=13, y=1.02, color=TOKENS["ink"])
    fig.tight_layout()
    fig.savefig(FIG / "section25_cside_model_setup_covariates.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "section25_cside_model_setup_covariates.pdf", bbox_inches="tight")
    plt.close(fig)


def make_specificity_figure(matrix: pd.DataFrame, pairs: pd.DataFrame) -> None:
    genes = pairs["gene"].drop_duplicates().tolist()
    pivot_z = matrix.pivot_table(index="cell_type", columns="gene", values="stouffer_z", aggfunc="first").reindex(MAIN_CELL_TYPES)[genes]
    pivot_n = matrix.pivot_table(index="cell_type", columns="gene", values="n_samples", aggfunc="first").reindex(MAIN_CELL_TYPES)[genes]
    robust = matrix.pivot_table(index="cell_type", columns="gene", values="is_selected_robust_pair", aggfunc="max")
    robust = robust.reindex(MAIN_CELL_TYPES)[genes].astype(object)
    robust[pd.isna(robust)] = False
    robust = robust.astype(bool)

    cmap = LinearSegmentedColormap.from_list("signed_z", ["#2E4780", "#FFFFFF", "#CC6F47"])
    values = pivot_z.to_numpy(dtype=float)
    vmax = float(np.nanpercentile(np.abs(values), 95))
    vmax = max(vmax, 3.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    fig.patch.set_facecolor(TOKENS["surface"])
    ax.set_facecolor(TOKENS["panel"])
    x_positions = np.arange(len(genes))
    y_positions = np.arange(len(MAIN_CELL_TYPES))
    for y, ct in enumerate(MAIN_CELL_TYPES):
        for x, gene in enumerate(genes):
            z = pivot_z.loc[ct, gene]
            n = pivot_n.loc[ct, gene]
            if pd.isna(z):
                ax.scatter(x, y, s=18, facecolor="#E2E5EA", edgecolor="#C5CAD3", linewidth=0.6)
                continue
            size = 18 + 5.2 * float(n)
            edge = TOKENS["ink"] if bool(robust.loc[ct, gene]) else "#C5CAD3"
            lw = 1.5 if bool(robust.loc[ct, gene]) else 0.5
            ax.scatter(x, y, s=size, color=cmap(norm(float(z))), edgecolor=edge, linewidth=lw)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(genes, rotation=45, ha="right")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(MAIN_CELL_TYPES)
    ax.invert_yaxis()
    ax.set_title("Representative robust-core genes retain cell-type structure", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("Representative robust-core genes")
    ax.set_ylabel("Cell type")
    ax.grid(color=TOKENS["grid"], linewidth=0.6, axis="both")
    style_axis(ax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Signed Stouffer Z", color=TOKENS["ink"])
    cbar.ax.tick_params(colors=TOKENS["muted"], labelsize=8)
    for size, label in [(5, "5"), (15, "15"), (23, "23")]:
        ax.scatter([], [], s=18 + 5.2 * size, color="#E2E5EA", edgecolor="#7A828F", label=f"{label} samples")
    ax.scatter([], [], s=80, color="#FFFFFF", edgecolor=TOKENS["ink"], linewidth=1.5, label="selected robust pair")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.07, 1.0))
    fig.tight_layout()
    fig.savefig(FIG / "section25_cside_celltype_specificity_matrix.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "section25_cside_celltype_specificity_matrix.pdf", bbox_inches="tight")
    plt.close(fig)


def make_sample_profile_figure(profiles: pd.DataFrame, pairs: pd.DataFrame) -> None:
    if profiles.empty:
        return
    order = pairs["pair_label"].tolist()
    profiles = profiles.copy()
    profiles["pair_label"] = pd.Categorical(profiles["pair_label"], categories=order, ordered=True)
    fig, ax = plt.subplots(figsize=(12, 5.8))
    fig.patch.set_facecolor(TOKENS["surface"])
    style_axis(ax)
    rng = np.random.default_rng(123)
    x_map = {label: i for i, label in enumerate(order)}
    for dataset in DATASET_ORDER:
        sub = profiles[profiles["dataset"] == dataset]
        xs = sub["pair_label"].map(x_map).astype(float).to_numpy()
        xs = xs + rng.normal(0, 0.08, len(xs))
        ax.scatter(
            xs,
            pd.to_numeric(sub["signed_z"], errors="coerce"),
            s=30,
            color=DATASET_COLORS[dataset],
            edgecolor="#464C55",
            linewidth=0.4,
            alpha=0.82,
            label=dataset,
        )
    ax.axhline(0, color="#464C55", linewidth=1, linestyle="--")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_ylabel("Sample-level signed Z")
    ax.set_xlabel("")
    ax.set_title("Sample-level C-SIDE directions behind selected meta signals", loc="left", fontsize=12, fontweight="bold")
    ax.grid(axis="y", color=TOKENS["grid"], linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3)
    fig.text(
        0.065,
        0.035,
        "Each point is one sample in which the gene-cell-type pair was testable; positive values mean higher expression along the SNAI1-ac coefficient in Model 2.",
        ha="left",
        va="top",
        fontsize=8.5,
        color=TOKENS["muted"],
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "section25_cside_sample_signed_z_profiles.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "section25_cside_sample_signed_z_profiles.pdf", bbox_inches="tight")
    plt.close(fig)


def make_pathway_ladder_heatmap(pathway: pd.DataFrame) -> None:
    columns = [
        ("fgsea_hallmark_q_lt_0_05", "fgSEA\nHallmark"),
        ("fgsea_kegg_q_lt_0_05", "fgSEA\nKEGG"),
        ("meanZ_no_logfc_hallmark_q_lt_0_05", "Mean-Z\nHallmark"),
        ("meanZ_no_logfc_kegg_q_lt_0_05", "Mean-Z\nKEGG"),
        ("sample_no_logfc_hallmark_q_lt_0_05", "Sample\nHallmark"),
        ("sample_no_logfc_kegg_q_lt_0_05", "Sample\nKEGG"),
        ("meanZ_abs_mean_logfc_ge_1_hallmark_q_lt_0_05", "Mean-Z\nH gated"),
        ("meanZ_abs_mean_logfc_ge_1_kegg_q_lt_0_05", "Mean-Z\nK gated"),
        ("sample_abs_logfc_ge_1_hallmark_q_lt_0_05", "Sample\nH gated"),
        ("sample_abs_logfc_ge_1_kegg_q_lt_0_05", "Sample\nK gated"),
    ]
    matrix = pathway.set_index("cell_type").reindex(MAIN_CELL_TYPES)[[c for c, _ in columns]].fillna(0).astype(float)

    fig, ax = plt.subplots(figsize=(13.2, 4.6))
    fig.patch.set_facecolor(TOKENS["surface"])
    ax.set_facecolor(TOKENS["panel"])
    im = ax.imshow(matrix.to_numpy(), cmap="Blues", vmin=0, vmax=max(1, float(np.nanmax(matrix.to_numpy()))))

    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels([label for _, label in columns], fontsize=8)
    ax.set_yticks(np.arange(len(MAIN_CELL_TYPES)))
    ax.set_yticklabels(MAIN_CELL_TYPES, fontsize=9)
    ax.tick_params(length=0, colors=TOKENS["muted"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MAIN_CELL_TYPES), 1), minor=True)
    ax.grid(which="minor", color=TOKENS["panel"], linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for x in [1.5, 3.5, 5.5, 7.5]:
        ax.axvline(x, color=TOKENS["axis"], linewidth=1.2)

    values = matrix.to_numpy()
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            val = int(values[y, x])
            color = TOKENS["ink"] if val <= np.nanmax(values) * 0.55 else "#FFFFFF"
            ax.text(x, y, str(val), ha="center", va="center", fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.set_label("Significant pathways (BH q < 0.05)", color=TOKENS["ink"], fontsize=8)
    cbar.ax.tick_params(colors=TOKENS["muted"], labelsize=8)
    fig.text(
        0.08,
        0.965,
        "Pathway evidence across increasingly strict C-SIDE pathway summaries",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.08,
        0.91,
        "Counts are shown separately for Hallmark and KEGG; no-gate tests use all genes passing pathway overlap, while effect-size-conditioned tests apply logFC gates.",
        ha="left",
        va="top",
        fontsize=8.5,
        color=TOKENS["muted"],
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(FIG / "section25_cside_pathway_ladder_counts_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "section25_cside_pathway_ladder_counts_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def make_meanz_hits_dotplot(meanz_hits: pd.DataFrame, collection: str) -> None:
    sub = meanz_hits[meanz_hits["collection"] == collection].copy()
    if sub.empty:
        return
    sub["mean_z"] = pd.to_numeric(sub["mean_z"], errors="coerce")
    sub["n_genes"] = pd.to_numeric(sub["n_genes"], errors="coerce")
    sub["q_perm_bh"] = pd.to_numeric(sub["q_perm_bh"], errors="coerce")
    sub["cell_type"] = pd.Categorical(sub["cell_type"], categories=MAIN_CELL_TYPES, ordered=True)
    sub = sub.sort_values(["cell_type", "mean_z"], ascending=[True, True]).reset_index(drop=True)
    sub["label"] = sub["cell_type"].astype(str) + ": " + sub["pathway_label"]

    height = max(5.5, 0.28 * len(sub) + 1.8)
    fig, ax = plt.subplots(figsize=(10.8, height))
    fig.patch.set_facecolor(TOKENS["surface"])
    style_axis(ax)
    y = np.arange(len(sub))
    sizes = 24 + np.sqrt(sub["n_genes"].fillna(0).to_numpy()) * 11
    colors = [CELLTYPE_COLORS.get(ct, "#7A828F") for ct in sub["cell_type"].astype(str)]
    ax.scatter(sub["mean_z"], y, s=sizes, color=colors, edgecolor="#464C55", linewidth=0.55, alpha=0.9)
    ax.axvline(0, color="#464C55", linewidth=1, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(sub["label"], fontsize=7.5)
    ax.set_xlabel("Mean signed Stouffer Z")
    ax.set_ylabel("")
    ax.grid(axis="x", color=TOKENS["grid"], linewidth=0.8)
    fig.text(
        0.08,
        0.985,
        f"{collection} pathways passing no-gate mean-Z permutation",
        ha="left",
        va="top",
        fontsize=12.5,
        fontweight="bold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.08,
        0.945,
        "Each point is a cell-type--pathway combination with BH q < 0.05; point size scales with pathway overlap.",
        ha="left",
        va="top",
        fontsize=8.5,
        color=TOKENS["muted"],
    )
    for ct in [ct for ct in MAIN_CELL_TYPES if ct in set(sub["cell_type"].astype(str))]:
        ax.scatter([], [], s=70, color=CELLTYPE_COLORS[ct], edgecolor="#464C55", linewidth=0.55, label=ct)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower right")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    stem = collection.lower()
    fig.savefig(FIG / f"section25_cside_meanz_no_logfc_{stem}_hits_dotplot.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"section25_cside_meanz_no_logfc_{stem}_hits_dotplot.pdf", bbox_inches="tight")
    plt.close(fig)


def make_epithelial_sample_pathway_heatmap(sample_hits: pd.DataFrame) -> None:
    hits = sample_hits[(sample_hits["collection"] == "Hallmark") & (sample_hits["cell_type"] == "Epithelial")].copy()
    if hits.empty:
        return
    hit_order = hits.sort_values("wilcoxon_q_bh")["pathway"].tolist()
    sample = read_csv(
        S2E
        / "04b_sample_dataset_heterogeneity_no_logfc_gate"
        / "hallmark"
        / "hallmark_sample_level_pathway_meanZ.csv"
    )
    sample = sample[(sample["cell_type"] == "Epithelial") & (sample["pathway"].isin(hit_order))].copy()
    sample["dataset"] = pd.Categorical(sample["dataset"], categories=DATASET_ORDER, ordered=True)
    sample = sample.sort_values(["dataset", "sample_id_on_disk", "pathway"])
    sample_order = sample[["dataset", "sample_id_on_disk", "sample_label"]].drop_duplicates().sort_values(
        ["dataset", "sample_id_on_disk"]
    )
    pivot = sample.pivot_table(index="pathway", columns="sample_label", values="sample_mean_z", aggfunc="first")
    pivot = pivot.reindex(hit_order)[sample_order["sample_label"].tolist()]
    row_labels = [clean_pathway_name(p) for p in pivot.index]
    col_labels = sample_order["sample_id_on_disk"].tolist()

    values = pivot.to_numpy(dtype=float)
    vmax = max(0.75, float(np.nanpercentile(np.abs(values), 95)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list("sample_signed_z", ["#2E4780", "#FFFFFF", "#CC6F47"])

    fig = plt.figure(figsize=(13.5, 5.7), facecolor=TOKENS["surface"])
    gs = fig.add_gridspec(2, 1, height_ratios=[0.18, 1], hspace=0.03)
    ax_top = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1])
    ax.set_facecolor(TOKENS["panel"])
    ax_top.set_facecolor(TOKENS["panel"])

    dataset_codes = np.array([[DATASET_ORDER.index(str(d)) for d in sample_order["dataset"]]])
    dataset_cmap = LinearSegmentedColormap.from_list("dataset_strip", [DATASET_COLORS[d] for d in DATASET_ORDER], N=len(DATASET_ORDER))
    ax_top.imshow(dataset_codes, aspect="auto", cmap=dataset_cmap, vmin=0, vmax=len(DATASET_ORDER) - 1)
    ax_top.set_yticks([])
    ax_top.set_xticks([])
    for spine in ax_top.spines.values():
        spine.set_visible(False)

    im = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=7)
    ax.tick_params(length=0, colors=TOKENS["muted"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color=TOKENS["grid"], linewidth=0.55)
    ax.tick_params(which="minor", bottom=False, left=False)
    for boundary in np.where(sample_order["dataset"].astype(str).to_numpy()[1:] != sample_order["dataset"].astype(str).to_numpy()[:-1])[0]:
        ax.axvline(boundary + 0.5, color="#464C55", linewidth=1.2)
        ax_top.axvline(boundary + 0.5, color="#464C55", linewidth=1.2)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.set_label("Sample-level pathway mean signed Z", color=TOKENS["ink"], fontsize=8)
    cbar.ax.tick_params(colors=TOKENS["muted"], labelsize=8)
    for dataset in DATASET_ORDER:
        ax_top.scatter([], [], s=55, color=DATASET_COLORS[dataset], label=dataset)
    ax_top.legend(frameon=False, fontsize=8, ncol=3, loc="center left", bbox_to_anchor=(0, 1.1))

    fig.text(
        0.07,
        0.985,
        "Epithelial Hallmark pathways with sample-level directional reproducibility",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.07,
        0.94,
        "Rows are no-logFC-gate Hallmark pathways passing Wilcoxon BH q < 0.05 across samples; columns are epithelial C-SIDE samples grouped by dataset.",
        ha="left",
        va="top",
        fontsize=8.5,
        color=TOKENS["muted"],
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / "section25_cside_epithelial_hallmark_sample_no_logfc_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "section25_cside_epithelial_hallmark_sample_no_logfc_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def make_composite_figure(coverage: pd.DataFrame, pathway: pd.DataFrame, setlevel: pd.DataFrame, core: pd.DataFrame) -> None:
    colors = {
        "Epithelial": "#2f6f9f",
        "Fibroblast": "#6a994e",
        "Macrophage": "#bc6c25",
        "CAF": "#8a5a83",
        "Endothelial": "#4d908e",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    ax = axes[0, 0]
    x = np.arange(len(MAIN_CELL_TYPES))
    ax.bar(x, coverage["testable_samples"], color=[colors[c] for c in coverage["cell_type"]])
    ax.set_xticks(x)
    ax.set_xticklabels(MAIN_CELL_TYPES, rotation=35, ha="right")
    ax.set_ylim(0, 24)
    ax.set_ylabel("Samples")
    ax.set_title("A. C-SIDE cell-type coverage")
    for i, row in coverage.iterrows():
        ax.text(i, row["testable_samples"] + 0.6, f"{row['testable_samples']}\n{row['gene_level_fits']:,} fits",
                ha="center", va="bottom", fontsize=8)

    ax = axes[0, 1]
    width = 0.22
    ax.bar(x - width, pathway["old_hallmark_fgsea_q_lt_0_05"], width, label="Old Hallmark fgSEA", color="#b08968")
    ax.bar(x, pathway["meanZ_hallmark_q_lt_0_05"], width, label="Mean-Z Hallmark", color="#457b9d")
    ax.bar(x + width, pathway["meanZ_kegg_q_lt_0_05"], width, label="Mean-Z KEGG", color="#2a9d8f")
    ax.set_xticks(x)
    ax.set_xticklabels(MAIN_CELL_TYPES, rotation=35, ha="right")
    ax.set_ylabel("Significant pathways")
    ax.set_title("B. Pathway evidence narrows under stricter tests")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    for ct in MAIN_CELL_TYPES:
        sub = setlevel[setlevel["cell_type"] == ct].sort_values("threshold_T")
        marker = "o" if sub["significant"].any() else "x"
        ax.plot(sub["threshold_T"], sub["obs_over_null_mean"], marker=marker, label=ct, color=colors[ct], linewidth=2)
        for _, row in sub.iterrows():
            if row["significant"]:
                ax.text(row["threshold_T"], row["obs_over_null_mean"] + 0.035, "*", ha="center", va="bottom", color=colors[ct])
    ax.axhline(1, color="#777777", linewidth=1, linestyle="--")
    ax.set_xticks([0.70, 0.80, 0.90])
    ax.set_xlabel("Directional consistency threshold")
    ax.set_ylabel("Observed / null mean")
    ax.set_title("C. Set-level directional reproducibility")
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[1, 1]
    core = core.set_index("cell_type").reindex(MAIN_CELL_TYPES).reset_index()
    ax.bar(x, pd.to_numeric(core["robust_core_n"]), color=[colors[c] for c in core["cell_type"]])
    ax.set_xticks(x)
    ax.set_xticklabels(MAIN_CELL_TYPES, rotation=35, ha="right")
    ax.set_ylabel("Gene-cell-type associations")
    ax.set_title("D. Robust core is epithelial-dominant")
    for i, row in core.iterrows():
        ax.text(i, float(row["robust_core_n"]) + 1, str(int(row["robust_core_n"])), ha="center", va="bottom", fontsize=9)

    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#e8e8e8", linewidth=0.8)

    fig.suptitle("C-SIDE evidence ladder for SNAI1-ac-associated cell-type-specific expression", fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig(FIG / "section25_cside_evidence_ladder_composite.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "section25_cside_evidence_ladder_composite.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    coverage = build_coverage_table()
    pathway = build_pathway_narrowing_table()
    setlevel = build_setlevel_summary()
    sign_summary = build_gene_sign_summary()
    core = build_robust_core_summary()
    meanz_hits = build_meanz_hits_table(no_logfc_gate=True)
    meanz_gated_hits = build_meanz_hits_table(no_logfc_gate=False)
    sample_hits = build_sample_level_hits_table(no_logfc_gate=True)
    sample_gated_hits = build_sample_level_hits_table(no_logfc_gate=False)
    inclusion_audit = build_celltype_inclusion_audit()
    model_setup = build_model_setup_by_sample()
    pairs = representative_pairs()
    candidate_table_path = TAB / "section25_cside_robust_core_all_candidates.csv"
    selection_template_path = write_selection_template(pairs)

    coverage.to_csv(TAB / "section25_cside_celltype_coverage.csv", index=False)
    pathway.to_csv(TAB / "section25_cside_pathway_narrowing_counts.csv", index=False)
    setlevel.to_csv(TAB / "section25_cside_setlevel_summary.csv", index=False)
    sign_summary.to_csv(TAB / "section25_cside_per_gene_sign_test_summary.csv", index=False)
    core.to_csv(TAB / "section25_cside_robust_core_summary.csv", index=False)
    meanz_hits.to_csv(TAB / "section25_cside_meanZ_significant_pathways.csv", index=False)
    meanz_hits.to_csv(TAB / "section25_cside_meanZ_no_logfc_significant_pathways.csv", index=False)
    meanz_gated_hits.to_csv(TAB / "section25_cside_meanZ_effect_size_conditioned_significant_pathways.csv", index=False)
    sample_hits.to_csv(TAB / "section25_cside_sample_level_no_logfc_significant_pathways.csv", index=False)
    sample_gated_hits.to_csv(TAB / "section25_cside_sample_level_effect_size_conditioned_significant_pathways.csv", index=False)
    inclusion_audit.to_csv(TAB / "section25_cside_celltype_inclusion_audit.csv", index=False)
    model_setup.to_csv(TAB / "section25_cside_model_setup_by_sample.csv", index=False)
    pairs.to_csv(candidate_table_path, index=False)

    to_latex_table(
        coverage,
        TAB / "section25_cside_celltype_coverage.tex",
        "C-SIDE sample and gene-level fit coverage for the five main cell types.",
        "tab:cside_celltype_coverage",
    )
    pathway_latex = pathway.rename(
        columns={
            "cell_type": "Cell type",
            "fgsea_hallmark_q_lt_0_05": "fgSEA H",
            "fgsea_kegg_q_lt_0_05": "fgSEA K",
            "meanZ_no_logfc_hallmark_q_lt_0_05": "Mean-Z H",
            "meanZ_no_logfc_kegg_q_lt_0_05": "Mean-Z K",
            "sample_no_logfc_hallmark_q_lt_0_05": "Sample H",
            "sample_no_logfc_kegg_q_lt_0_05": "Sample K",
            "meanZ_abs_mean_logfc_ge_1_hallmark_q_lt_0_05": "Mean-Z H gated",
            "meanZ_abs_mean_logfc_ge_1_kegg_q_lt_0_05": "Mean-Z K gated",
            "sample_abs_logfc_ge_1_hallmark_q_lt_0_05": "Sample H gated",
            "sample_abs_logfc_ge_1_kegg_q_lt_0_05": "Sample K gated",
        }
    )
    to_latex_table(
        pathway_latex,
        TAB / "section25_cside_pathway_narrowing_counts.tex",
        "Pathway-level C-SIDE evidence across ranked-list, no-gate, and effect-size-conditioned pathway tests.",
        "tab:cside_pathway_narrowing",
    )
    meanz_latex = meanz_hits[["collection", "cell_type", "pathway_label", "n_genes", "mean_z", "direction", "q_perm_bh"]].copy()
    meanz_latex["mean_z"] = pd.to_numeric(meanz_latex["mean_z"], errors="coerce").map(lambda x: f"{x:.3f}")
    meanz_latex["q_perm_bh"] = pd.to_numeric(meanz_latex["q_perm_bh"], errors="coerce").map(lambda x: f"{x:.3g}")
    meanz_latex = meanz_latex.rename(
        columns={
            "collection": "Gene set",
            "cell_type": "Cell type",
            "pathway_label": "Pathway",
            "n_genes": "Genes",
            "mean_z": "Mean Z",
            "direction": "Direction",
            "q_perm_bh": "BH q",
        }
    )
    to_latex_table(
        meanz_latex,
        TAB / "section25_cside_meanZ_significant_pathways.tex",
        "Cell-type--pathway combinations passing the no-logFC-gate mean-Z permutation test.",
        "tab:cside_meanZ_hits",
    )
    gated_latex = meanz_gated_hits[["collection", "cell_type", "pathway_label", "n_genes", "mean_z", "direction", "q_perm_bh"]].copy()
    gated_latex["mean_z"] = pd.to_numeric(gated_latex["mean_z"], errors="coerce").map(lambda x: f"{x:.3f}")
    gated_latex["q_perm_bh"] = pd.to_numeric(gated_latex["q_perm_bh"], errors="coerce").map(lambda x: f"{x:.3g}")
    gated_latex = gated_latex.rename(
        columns={
            "collection": "Gene set",
            "cell_type": "Cell type",
            "pathway_label": "Pathway",
            "n_genes": "Genes",
            "mean_z": "Mean Z",
            "direction": "Direction",
            "q_perm_bh": "BH q",
        }
    )
    to_latex_table(
        gated_latex,
        TAB / "section25_cside_meanZ_effect_size_conditioned_significant_pathways.tex",
        "Cell-type--pathway combinations passing the effect-size-conditioned mean-Z permutation test.",
        "tab:cside_meanZ_effect_size_conditioned_hits",
    )
    sample_latex = sample_hits[
        [
            "collection",
            "cell_type",
            "pathway_label",
            "n_samples",
            "mean_sample_mean_z",
            "n_positive_samples",
            "n_negative_samples",
            "sample_direction_consistency",
            "wilcoxon_q_bh",
        ]
    ].copy()
    sample_latex["mean_sample_mean_z"] = pd.to_numeric(sample_latex["mean_sample_mean_z"], errors="coerce").map(lambda x: f"{x:.3f}")
    sample_latex["sample_direction_consistency"] = pd.to_numeric(
        sample_latex["sample_direction_consistency"], errors="coerce"
    ).map(lambda x: f"{x:.3f}")
    sample_latex["wilcoxon_q_bh"] = pd.to_numeric(sample_latex["wilcoxon_q_bh"], errors="coerce").map(lambda x: f"{x:.3g}")
    sample_latex = sample_latex.rename(
        columns={
            "collection": "Gene set",
            "cell_type": "Cell type",
            "pathway_label": "Pathway",
            "n_samples": "Samples",
            "mean_sample_mean_z": "Mean sample Z",
            "n_positive_samples": "Positive samples",
            "n_negative_samples": "Negative samples",
            "sample_direction_consistency": "Direction consistency",
            "wilcoxon_q_bh": "BH q",
        }
    )
    to_latex_table(
        sample_latex,
        TAB / "section25_cside_sample_level_no_logfc_significant_pathways.tex",
        "Pathways passing no-logFC-gate sample-level directional reproducibility testing.",
        "tab:cside_sample_level_no_logfc_hits",
    )
    to_latex_table(
        sign_summary,
        TAB / "section25_cside_per_gene_sign_test_summary.tex",
        "Per-gene directional sign-test summary by cell type.",
        "tab:cside_per_gene_sign_test",
    )
    inclusion_latex = inclusion_audit[["cell_type", "testable_samples", "datasets", "gene_level_fits", "report_role"]].copy()
    to_latex_table(
        inclusion_latex,
        TAB / "section25_cside_celltype_inclusion_audit.tex",
        "Cell-type inclusion audit for the C-SIDE section.",
        "tab:cside_celltype_inclusion_audit",
    )

    remove_unapproved_selected_gene_assets()
    remove_unapproved_python_figures()

    make_pathway_ladder_heatmap(pathway)
    make_meanz_hits_dotplot(meanz_hits, "Hallmark")
    make_meanz_hits_dotplot(meanz_hits, "KEGG")
    make_epithelial_sample_pathway_heatmap(sample_hits)

    manifest = {
        "output_dir": str(OUT),
        "figures": sorted(str(p) for p in FIG.glob("section25_cside_*")),
        "figure_status": "Python report figures generated from saved C-SIDE S2e audit outputs.",
        "candidate_table": str(candidate_table_path),
        "selection_template": str(selection_template_path),
        "tables": sorted(str(p) for p in TAB.glob("section25_*")),
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
