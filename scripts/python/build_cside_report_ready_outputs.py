#!/usr/bin/env python
"""
Build report-ready C-SIDE S2e packaging outputs.

This script does not rerun C-SIDE, gene-set tests, QC sensitivity, or
permutations. It consumes saved S2e audit outputs and creates compact
report/supplement tables plus a figure for the set-level permutation result.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"D:\HGSOC_Spatial_Atlas\05_analysis_ready\S2e_CSIDE_CellTypeSpecific_DE_Audit")
OUT = ROOT / "07_report_ready_packaging"
FIG_DIR = OUT / "figures"
TAB_DIR = OUT / "tables"

META = ROOT / "01_signed_ranking_audit" / "meta_gene_effects_signed_stouffer_iv_random_effects.csv"
PERM = ROOT / "06_setlevel_permutation" / "setlevel_reproducibility_permutation.csv"
PERM_NULL = ROOT / "06_setlevel_permutation" / "setlevel_reproducibility_null_counts_wide.csv"
QC_CORR = ROOT / "05_qc_sensitivity" / "qc_adjusted_cside_comparison" / "qc_vs_original_meta_gene_effect_correlations_by_celltype.csv"
HALL_MEANZ = ROOT / "02_paper_style_meanZ_gene_set_tests" / "hallmark" / "hallmark_meanZ_permutation_results.csv"
KEGG_MEANZ = ROOT / "02_paper_style_meanZ_gene_set_tests" / "kegg" / "kegg_meanZ_permutation_results.csv"
QC_HALL = ROOT / "05_qc_sensitivity" / "qc_adjusted_gene_set_tests" / "hallmark" / "qc_adjusted_hallmark_meanZ_vs_original.csv"
QC_KEGG = ROOT / "05_qc_sensitivity" / "qc_adjusted_gene_set_tests" / "kegg" / "qc_adjusted_kegg_meanZ_vs_original.csv"
SIGNATURE = Path(r"D:\HGSOC_Spatial_Atlas\05_analysis_ready\Signature\snai1_acetylation_signature_full.csv")

MAIN = ["Epithelial", "Fibroblast", "Macrophage", "CAF", "Endothelial"]
ORDER = ["Epithelial", "CAF", "Endothelial", "Macrophage", "Fibroblast", "pooled"]
COLORS = {
    "Epithelial": "#386cb0",
    "CAF": "#bf5b17",
    "Endothelial": "#7fc97f",
    "Macrophage": "#beaed4",
    "Fibroblast": "#f0027f",
    "pooled": "#303030",
}


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)


def fmt_float(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return ""
    if abs(x) < 0.001 and x != 0:
        return f"{x:.2e}"
    return f"{x:.{digits}f}"


def latex_escape(text: object) -> str:
    if pd.isna(text):
        return ""
    s = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def write_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{" + "l" * len(df.columns) + "}",
        r"\hline",
        " & ".join(latex_escape(c) for c in df.columns) + r" \\",
        r"\hline",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(latex_escape(v) for v in row.tolist()) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_permutation_figure() -> pd.DataFrame:
    perm = pd.read_csv(PERM)
    perm["obs_over_null_mean"] = perm["observed_count"] / perm["null_mean"]
    perm.to_csv(TAB_DIR / "setlevel_permutation_report_table.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True)
    axes = axes.ravel()
    x = np.array([0.70, 0.80, 0.90])

    for ax, ct in zip(axes, ORDER):
        sub = perm[perm["cell_type"] == ct].set_index("threshold_T").loc[x].reset_index()
        color = COLORS[ct]
        ax.fill_between(
            sub["threshold_T"],
            sub["null_p05"],
            sub["null_p95"],
            color=color,
            alpha=0.18,
            linewidth=0,
            label="Null 5-95%",
        )
        ax.plot(
            sub["threshold_T"],
            sub["null_mean"],
            color=color,
            linestyle="--",
            linewidth=1.8,
            label="Null mean",
        )
        ax.plot(
            sub["threshold_T"],
            sub["observed_count"],
            color=color,
            marker="o",
            linewidth=2.5,
            label="Observed",
        )
        for _, row in sub.iterrows():
            label = f"q={row['bh_q']:.3g}"
            ax.annotate(
                label,
                (row["threshold_T"], row["observed_count"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=7.5,
                color="#303030",
            )
        ax.set_title(ct, fontsize=11, fontweight="bold")
        ax.set_xlim(0.67, 0.93)
        ax.set_xticks(x)
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Directional consistency threshold")
        ax.set_ylabel("Passing gene-cell-type associations")

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIG_DIR / "cside_setlevel_permutation_observed_vs_null.png", dpi=300)
    plt.close(fig)
    return perm


def build_robust_core_tables() -> pd.DataFrame:
    meta = pd.read_csv(META)
    sig = pd.read_csv(SIGNATURE, sep=";")
    sig_genes = set(sig["Gene"].dropna().astype(str))

    core = meta[
        meta["cell_type"].isin(MAIN)
        & (meta["random_q"] < 0.05)
        & (meta["n_samples"] >= 5)
        & (meta["n_datasets"] == 3)
        & (meta["sign_consistency"] >= 0.8)
    ].copy()
    core["direction"] = np.where(core["n_positive"] >= core["n_negative"], "positive", "negative")
    core["is_snai1ac_signature_gene"] = core["gene"].astype(str).isin(sig_genes)
    core["majority_samples"] = core[["n_positive", "n_negative"]].max(axis=1)
    core["minority_samples"] = core[["n_positive", "n_negative"]].min(axis=1)

    cols = [
        "cell_type",
        "gene",
        "direction",
        "n_samples",
        "n_datasets",
        "majority_samples",
        "minority_samples",
        "sign_consistency",
        "mean_signed_z",
        "stouffer_z",
        "stouffer_q",
        "random_z",
        "random_q",
        "i2",
        "is_snai1ac_signature_gene",
        "datasets",
    ]
    core = core[cols].sort_values(["cell_type", "random_q", "gene"])
    core.to_csv(TAB_DIR / "cside_robust_core_73_gene_celltype_associations.csv", index=False)

    summary = (
        core.groupby("cell_type")
        .agg(
            robust_core_n=("gene", "size"),
            positive_n=("direction", lambda s: int((s == "positive").sum())),
            negative_n=("direction", lambda s: int((s == "negative").sum())),
            median_sign_consistency=("sign_consistency", "median"),
            median_i2=("i2", "median"),
            snai1ac_signature_overlap_n=("is_snai1ac_signature_gene", "sum"),
        )
        .reindex(MAIN, fill_value=0)
        .reset_index()
    )
    pooled = pd.DataFrame(
        [
            {
                "cell_type": "pooled",
                "robust_core_n": len(core),
                "positive_n": int((core["direction"] == "positive").sum()),
                "negative_n": int((core["direction"] == "negative").sum()),
                "median_sign_consistency": core["sign_consistency"].median(),
                "median_i2": core["i2"].median(),
                "snai1ac_signature_overlap_n": int(core["is_snai1ac_signature_gene"].sum()),
            }
        ]
    )
    summary = pd.concat([summary, pooled], ignore_index=True)
    summary.to_csv(TAB_DIR / "cside_robust_core_73_summary_by_celltype.csv", index=False)

    latex_summary = summary.copy()
    for col in ["median_sign_consistency", "median_i2"]:
        latex_summary[col] = latex_summary[col].map(lambda x: fmt_float(float(x), 3) if x != "" else "")
    write_latex_table(
        latex_summary,
        TAB_DIR / "cside_robust_core_73_summary_by_celltype.tex",
        "Robust C-SIDE core defined as five main cell types, random-effects q < 0.05, tested in at least five samples, present in all three datasets, and sign consistency >= 0.8.",
        "tab:cside_robust_core_summary",
    )

    # A compact display table for main text/supplement previews.
    display = core.copy()
    display["abs_stouffer_z"] = display["stouffer_z"].abs()
    display = display.sort_values(["cell_type", "abs_stouffer_z"], ascending=[True, False])
    top_display = (
        display.groupby("cell_type", group_keys=False)
        .head(8)
        .loc[
            :,
            [
                "cell_type",
                "gene",
                "direction",
                "n_samples",
                "sign_consistency",
                "random_q",
                "i2",
                "is_snai1ac_signature_gene",
            ],
        ]
    )
    top_display.to_csv(TAB_DIR / "cside_robust_core_top_examples_for_display.csv", index=False)
    latex_display = top_display.copy()
    for col in ["sign_consistency", "random_q", "i2"]:
        latex_display[col] = latex_display[col].map(lambda x: fmt_float(float(x), 3))
    write_latex_table(
        latex_display,
        TAB_DIR / "cside_robust_core_top_examples_for_display.tex",
        "Representative robust-core C-SIDE gene-cell-type associations ranked within each cell type by absolute Stouffer Z.",
        "tab:cside_robust_core_examples",
    )
    return core


def build_method_hierarchy_table() -> pd.DataFrame:
    hall = pd.read_csv(HALL_MEANZ)
    kegg = pd.read_csv(KEGG_MEANZ)
    qc = pd.read_csv(QC_CORR)
    qc_hall = pd.read_csv(QC_HALL)
    qc_kegg = pd.read_csv(QC_KEGG)
    perm = pd.read_csv(PERM)

    for df in [qc_hall, qc_kegg]:
        for col in ["original_significant_q05", "qc_significant_q05"]:
            df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)

    rows = [
        {
            "evidence_layer": "Older fgSEA on signed Stouffer rankings",
            "question_answered": "Do ranked gene lists show pathway enrichment?",
            "primary_result": "Pathway-rich output, especially epithelial Hallmark/KEGG signals in the older report.",
            "strength": "Useful exploration and leading-edge inventory.",
            "limitation": "Superseded for inference by stricter mean-Z/permutation checks; sensitive to ranking structure and shared genes.",
            "report_role": "Demote to exploratory/supplementary context.",
        },
        {
            "evidence_layer": "Paper-style mean-Z gene-set permutation",
            "question_answered": "Do large-effect C-SIDE genes in a set have non-random average signed Z?",
            "primary_result": f"Hallmark q<0.05: {(hall['q_perm_bh'] < 0.05).sum()}; KEGG q<0.05: {(kegg['q_perm_bh'] < 0.05).sum()}. No epithelial Hallmark/KEGG pathway passed this stricter test.",
            "strength": "Closer to C-SIDE paper-style gene-set testing; avoids pathway overinterpretation.",
            "limitation": "Filters to |logFC| >= 1, so it is stricter and can miss diffuse small effects.",
            "report_role": "Use to state that pathway-level C-SIDE signal is mostly null/fragile.",
        },
        {
            "evidence_layer": "Sample/dataset heterogeneity summaries",
            "question_answered": "Are pathway directions stable across samples/datasets?",
            "primary_result": "No Hallmark or KEGG pathway survived sample-level Wilcoxon BH correction across samples.",
            "strength": "Directly addresses cross-sample reproducibility.",
            "limitation": "Per-sample gene-set coverage varies, especially for sparse cell types/pathways.",
            "report_role": "Use as caution against pathway-level claims.",
        },
        {
            "evidence_layer": "QC-adjusted C-SIDE sensitivity",
            "question_answered": "Do gene-level directions persist after adding total counts, detected genes, and mt fraction when available?",
            "primary_result": f"Gene-level original-vs-QC Stouffer-Z correlations by cell type range {qc['pearson_stouffer_z_original_vs_qc'].min():.2f}-{qc['pearson_stouffer_z_original_vs_qc'].max():.2f}; direction agreement range {qc['direction_agreement_fraction'].min():.2f}-{qc['direction_agreement_fraction'].max():.2f}. QC-adjusted significant pathways: Hallmark {(qc_hall['qc_significant_q05']).sum()}, KEGG {(qc_kegg['qc_significant_q05']).sum()}.",
            "strength": "Shows gene-level direction is partly preserved.",
            "limitation": "QC covariates are strongly collinear with malignant fraction/tumor content, so attenuation is ambiguous.",
            "report_role": "Use as sensitivity, not as a binary pass/fail test.",
        },
        {
            "evidence_layer": "Random-effects robust core",
            "question_answered": "Which scale-sensitive candidate gene-cell-type associations remain directionally reproducible across samples and datasets?",
            "primary_result": "73 robust-core associations: 70 epithelial, 1 fibroblast, 1 macrophage, 1 endothelial; 4 overlap the 109-gene SNAI1-ac signature.",
            "strength": "Identifies a compact epithelial-dominant reproducible candidate set.",
            "limitation": "Still depends on random_q, which uses scale-sensitive logFC/SE.",
            "report_role": "Use as candidate core, with scale caveat.",
        },
        {
            "evidence_layer": "Set-level sample-flip sign permutation",
            "question_answered": "Are there more directionally reproducible gene-cell-type associations than expected under sample-level sign exchangeability?",
            "primary_result": "Pooled and epithelial counts exceed null across T=0.70, 0.80, and 0.90 after BH correction; endothelial significant at T=0.80 and 0.90; CAF at T=0.70 and 0.80.",
            "strength": "Scale-invariant direction-only test preserving sample-level gene-gene correlation.",
            "limitation": "Does not test effect magnitude or differential expression.",
            "report_role": "Use as statistical anchor for set-level directional reproducibility.",
        },
    ]
    hierarchy = pd.DataFrame(rows)
    hierarchy.to_csv(TAB_DIR / "cside_method_evidence_hierarchy.csv", index=False)
    write_latex_table(
        hierarchy,
        TAB_DIR / "cside_method_evidence_hierarchy.tex",
        "Hierarchy of C-SIDE evidence layers used to separate exploratory pathway signals from reproducible gene-level directionality.",
        "tab:cside_method_hierarchy",
    )
    return hierarchy


def main() -> None:
    ensure_dirs()
    perm = build_permutation_figure()
    core = build_robust_core_tables()
    hierarchy = build_method_hierarchy_table()

    shutil.copy2(Path(__file__), OUT / "build_cside_report_ready_outputs.py")
    manifest = {
        "script": str(Path(__file__)),
        "output_root": str(OUT),
        "inputs": {
            "meta": str(META),
            "permutation": str(PERM),
            "permutation_null": str(PERM_NULL),
            "qc_correlations": str(QC_CORR),
            "hallmark_meanZ": str(HALL_MEANZ),
            "kegg_meanZ": str(KEGG_MEANZ),
            "signature": str(SIGNATURE),
        },
        "outputs": {
            "figure": str(FIG_DIR / "cside_setlevel_permutation_observed_vs_null.png"),
            "robust_core_full": str(TAB_DIR / "cside_robust_core_73_gene_celltype_associations.csv"),
            "robust_core_summary": str(TAB_DIR / "cside_robust_core_73_summary_by_celltype.csv"),
            "method_hierarchy": str(TAB_DIR / "cside_method_evidence_hierarchy.csv"),
        },
        "n_permutation_rows": int(len(perm)),
        "n_robust_core": int(len(core)),
        "n_method_rows": int(len(hierarchy)),
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Built C-SIDE report-ready outputs")
    print(f"  output_root: {OUT}")
    print(f"  figure: {FIG_DIR / 'cside_setlevel_permutation_observed_vs_null.png'}")
    print(f"  robust_core_full: {TAB_DIR / 'cside_robust_core_73_gene_celltype_associations.csv'}")
    print(f"  robust_core_summary: {TAB_DIR / 'cside_robust_core_73_summary_by_celltype.csv'}")
    print(f"  method_hierarchy: {TAB_DIR / 'cside_method_evidence_hierarchy.csv'}")
    print("Robust core split:")
    print(pd.read_csv(TAB_DIR / "cside_robust_core_73_summary_by_celltype.csv").to_string(index=False))


if __name__ == "__main__":
    main()
