"""
Signature robustness audit for the SNAI1-ac Visium projection.

This script evaluates whether the projected SNAI1-ac signal is robust across
alternative signature formulations and whether those formulations preserve
non-random spatial structure in Visium.

Implemented blocks:
  1. Alternative weight generation
  2. Detection audit
  3. Re-scoring of h5ad files with robustness score columns
  4. Signature comparison and biology summaries
  5. Spatial autocorrelation / local structure / neighborhood smoothness
  6. Matched null comparison
  7. Top-gene dominance and leave-out audit
  8. Summary plots

Usage:
    python signature_robustness_audit.py --all
    python signature_robustness_audit.py visium/denisenko_2022
    python signature_robustness_audit.py visium/denisenko_2022 --step score
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr, skew

sc = None
em = None
ENRICHMAP_AVAILABLE = None


def require_scanpy():
    global sc
    if sc is not None:
        return sc
    print("Importing scanpy...", flush=True)
    import scanpy as _sc
    sc = _sc
    print("Imported scanpy.", flush=True)
    return sc


def require_enrichmap():
    global em, ENRICHMAP_AVAILABLE
    if em is not None:
        return em
    print("Importing enrichmap...", flush=True)
    try:
        import enrichmap as _em
    except ImportError as exc:
        ENRICHMAP_AVAILABLE = False
        raise ImportError("enrichmap is required for this step.") from exc
    em = _em
    ENRICHMAP_AVAILABLE = True
    print("Imported enrichmap.", flush=True)
    return em

try:
    import libpysal
    from esda.moran import Moran, Moran_Local
    SPATIAL_AVAILABLE = True
except ImportError:
    SPATIAL_AVAILABLE = False

from analysis_utils import cohens_d, bimodality_coefficient


# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR = Path(r"D:\HGSOC_Spatial_Atlas")
PROCESSED_DATA_DIR = BASE_DIR / "02_processed_data"
ANALYSIS_DIR = BASE_DIR / "05_analysis_ready"
SIGNATURE_DIR = ANALYSIS_DIR / "Signature"

ROBUSTNESS_DIR = SIGNATURE_DIR / "robustness"
WEIGHTS_DIR = ROBUSTNESS_DIR / "weights"
TABLES_DIR = ROBUSTNESS_DIR / "tables"
FIGURES_DIR = ROBUSTNESS_DIR / "figures"
LOGS_DIR = ROBUSTNESS_DIR / "logs"
CACHE_DIR = ROBUSTNESS_DIR / "h5ad_cache"

EXCEL_FILE = SIGNATURE_DIR / "tt_PEO4-SNAI1-2R_Analysis.xlsx"

GENE_COL = "Gene"
GENETYPE_COL = "GeneType"
ENSEMBL_COL = "Ensembl"
FC_COL = "PEO4-2R_lg2fc (SNAI1-SNAI1)"
PPEE_COL = "PEO4-2R_SNAI1vsSNAI1_PPEE"

PADJ_THRESHOLD = 0.05
LOG2FC_THRESHOLD = 1.0
CAP_THRESHOLD = 3.0
TOP_N = 100
MORAN_K = 6
NULL_ITERATIONS = 50
ENRICHMAP_NULL_ITERATIONS = 100
RANDOM_SEED = 42

KNOWN_DATASETS = [
    "visium/yamamoto_2025",
    "visium/ju_2024",
    "visium/denisenko_2022",
    "visium/stur_2021",
    "visium/10X_ov_standard",
]

CELLTYPE_COLS = [
    "Malignant", "CAF", "Macrophage", "Endothelial", "Fibroblast"
]

SIGNATURE_SPECS = [
    {
        "signature_id": "full_thresholded",
        "score_key": "SNAI1_ac_full_thresholded",
        "selection_rule": "all biotypes, padj < 0.05, |log2FC| > 1",
    },
    {
        "signature_id": "pc_thresholded",
        "score_key": "SNAI1_ac_pc_thresholded",
        "selection_rule": "protein-coding only, padj < 0.05, |log2FC| > 1",
    },
    {
        "signature_id": "pc_top100",
        "score_key": "SNAI1_ac_pc_top100",
        "selection_rule": "protein-coding only, padj < 0.05, top 100 by |log2FC|",
    },
    {
        "signature_id": "pc_up",
        "score_key": "SNAI1_ac_pc_up",
        "selection_rule": "protein-coding only, padj < 0.05, acetylation-activated only",
    },
    {
        "signature_id": "pc_down",
        "score_key": "SNAI1_ac_pc_down",
        "selection_rule": "protein-coding only, padj < 0.05, acetylation-suppressed only",
    },
]

STEP_ORDER = [
    "weights", "detection", "score", "compare", "spatial", "null",
    "enrichmap_null_benchmark", "enrichmap_null", "enrichmap_null_sensitivity",
    "enrichmap_unsmoothed_sensitivity", "topgenes", "volcano", "plots",
    "cleanup_current", "signature_robustness", "current_robustness",
    "finish_current_robustness", "all",
]


# =============================================================================
# HELPERS
# =============================================================================

def ensure_dirs() -> None:
    for path in (ROBUSTNESS_DIR, WEIGHTS_DIR, TABLES_DIR, FIGURES_DIR, LOGS_DIR, CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)


CURRENT_ROBUSTNESS_OUTPUTS = [
    TABLES_DIR / "signature_source_diagnostics.csv",
    TABLES_DIR / "signature_definition_summary.csv",
    TABLES_DIR / "signature_detection_summary_per_sample.csv",
    TABLES_DIR / "signature_detection_summary_overall.csv",
    TABLES_DIR / "scoring_status.csv",
    TABLES_DIR / "signature_comparison_spotwise_correlations.csv",
    TABLES_DIR / "signature_comparison_samplewise_summary.csv",
    TABLES_DIR / "signature_vs_biology_summary.csv",
    TABLES_DIR / "spatial_autocorrelation_per_sample.csv",
    TABLES_DIR / "spatial_autocorrelation_summary.csv",
    TABLES_DIR / "neighborhood_smoothness_per_sample.csv",
    TABLES_DIR / "local_spatial_structure_per_sample.csv",
    TABLES_DIR / "null_signature_comparison.csv",
    TABLES_DIR / "null_signature_summary.csv",
    TABLES_DIR / "top_gene_contribution_per_sample.csv",
    TABLES_DIR / "leave_top_genes_out_summary.csv",
    TABLES_DIR / "enrichmap_null_comparison.csv",
    TABLES_DIR / "enrichmap_null_summary_per_sample.csv",
    TABLES_DIR / "enrichmap_null_summary_cohort.csv",
    TABLES_DIR / "enrichmap_null_all_detected_uniform_no_stur_comparison.csv",
    TABLES_DIR / "enrichmap_null_all_detected_uniform_no_stur_summary_per_sample.csv",
    TABLES_DIR / "enrichmap_null_all_detected_uniform_no_stur_summary_cohort.csv",
    TABLES_DIR / "enrichmap_null_pc_matched_signed_no_stur_comparison.csv",
    TABLES_DIR / "enrichmap_null_pc_matched_signed_no_stur_summary_per_sample.csv",
    TABLES_DIR / "enrichmap_null_pc_matched_signed_no_stur_summary_cohort.csv",
    TABLES_DIR / "enrichmap_unsmoothed_null_all_detected_uniform_no_stur_comparison.csv",
    TABLES_DIR / "enrichmap_unsmoothed_null_all_detected_uniform_no_stur_summary_per_sample.csv",
    TABLES_DIR / "enrichmap_unsmoothed_null_all_detected_uniform_no_stur_summary_cohort.csv",
    TABLES_DIR / "enrichmap_unsmoothed_null_pc_matched_signed_no_stur_comparison.csv",
    TABLES_DIR / "enrichmap_unsmoothed_null_pc_matched_signed_no_stur_summary_per_sample.csv",
    TABLES_DIR / "enrichmap_unsmoothed_null_pc_matched_signed_no_stur_summary_cohort.csv",
    TABLES_DIR / "snai1_2r_vs_snai1_volcano_source_with_signature109.csv",
    FIGURES_DIR / "signature_detection_heatmap.png",
    FIGURES_DIR / "signature_detection_boxplot.png",
    FIGURES_DIR / "signature_correlation_heatmap.png",
    FIGURES_DIR / "moransI_by_signature.png",
    FIGURES_DIR / "neighborhood_smoothness_by_signature.png",
    FIGURES_DIR / "real_vs_null_moransI.png",
    FIGURES_DIR / "top_gene_contribution_curve.png",
    FIGURES_DIR / "drop_top_genes_stability.png",
    FIGURES_DIR / "enrichmap_matched_null_moransI_composite.png",
    FIGURES_DIR / "enrichmap_matched_null_A_real_vs_null_per_sample.png",
    FIGURES_DIR / "enrichmap_matched_null_B_real_minus_null_delta.png",
    FIGURES_DIR / "enrichmap_matched_null_C_real_null_percentile.png",
    FIGURES_DIR / "enrichmap_matched_null_D_hallmark_context.png",
    FIGURES_DIR / "snai1_2r_vs_snai1_volcano_signature109.png",
    FIGURES_DIR / "snai1_2r_vs_snai1_volcano_signature109.pdf",
]

NO_STUR_SENSITIVITY_OUTPUTS = [
    TABLES_DIR / "enrichmap_null_all_detected_uniform_no_stur_comparison.csv",
    TABLES_DIR / "enrichmap_null_all_detected_uniform_no_stur_summary_per_sample.csv",
    TABLES_DIR / "enrichmap_null_all_detected_uniform_no_stur_summary_cohort.csv",
    TABLES_DIR / "enrichmap_null_pc_matched_signed_no_stur_comparison.csv",
    TABLES_DIR / "enrichmap_null_pc_matched_signed_no_stur_summary_per_sample.csv",
    TABLES_DIR / "enrichmap_null_pc_matched_signed_no_stur_summary_cohort.csv",
    TABLES_DIR / "enrichmap_unsmoothed_null_all_detected_uniform_no_stur_comparison.csv",
    TABLES_DIR / "enrichmap_unsmoothed_null_all_detected_uniform_no_stur_summary_per_sample.csv",
    TABLES_DIR / "enrichmap_unsmoothed_null_all_detected_uniform_no_stur_summary_cohort.csv",
    TABLES_DIR / "enrichmap_unsmoothed_null_pc_matched_signed_no_stur_comparison.csv",
    TABLES_DIR / "enrichmap_unsmoothed_null_pc_matched_signed_no_stur_summary_per_sample.csv",
    TABLES_DIR / "enrichmap_unsmoothed_null_pc_matched_signed_no_stur_summary_cohort.csv",
]

VOLCANO_OUTPUTS = [
    TABLES_DIR / "snai1_2r_vs_snai1_volcano_source_with_signature109.csv",
    FIGURES_DIR / "snai1_2r_vs_snai1_volcano_signature109.png",
    FIGURES_DIR / "snai1_2r_vs_snai1_volcano_signature109.pdf",
]


def outputs_exist(paths: List[Path]) -> bool:
    return all(path.exists() and path.stat().st_size > 0 for path in paths)


def cleanup_current_outputs(remove_stur_cache: bool = True) -> None:
    """Remove stale report-facing robustness outputs before a current rerun."""
    ensure_dirs()
    root = ROBUSTNESS_DIR.resolve()

    for path in CURRENT_ROBUSTNESS_OUTPUTS:
        if not path.exists():
            continue
        resolved = path.resolve()
        if root not in resolved.parents and resolved != root:
            raise RuntimeError(f"Refusing to delete outside robustness dir: {resolved}")
        try:
            path.unlink()
            print(f"Removed stale output: {path}", flush=True)
        except PermissionError:
            raise PermissionError(
                f"Cannot overwrite stale output because it is locked: {path}. "
                "Close the file or stop the process holding it, then rerun."
            )

    for weight_path in WEIGHTS_DIR.glob("snai1_ac_*.json"):
        resolved = weight_path.resolve()
        if root not in resolved.parents:
            raise RuntimeError(f"Refusing to delete outside robustness dir: {resolved}")
        weight_path.unlink()
        print(f"Removed stale output: {weight_path}", flush=True)

    if remove_stur_cache:
        stur_cache = CACHE_DIR / "visium" / "stur_2021"
        if stur_cache.exists():
            resolved = stur_cache.resolve()
            if root not in resolved.parents:
                raise RuntimeError(f"Refusing to delete outside robustness dir: {resolved}")
            shutil.rmtree(stur_cache)
            print(f"Removed stale Stur cache: {stur_cache}", flush=True)


def bh_correct(df: pd.DataFrame, ppee_col: str) -> pd.Series:
    """Benjamini-Hochberg correction on PPEE values with NaN-safe coercion."""
    pvals = pd.to_numeric(df[ppee_col], errors="coerce")
    padj = pd.Series(np.nan, index=df.index, dtype=float)

    valid = pvals.dropna().sort_values()
    if valid.empty:
        return padj

    n = len(valid)
    ranks = np.arange(1, n + 1, dtype=float)
    adjusted = (valid.to_numpy(dtype=float) * n / ranks).clip(max=1.0)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    padj.loc[valid.index] = adjusted
    return padj


def _normalize_gene_symbols(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip()
    out = out.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    return out


def _normalize_gene_types(series: pd.Series) -> pd.Series:
    return (
        series.fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("-", "_", regex=False)
        .str.replace(" ", "_", regex=False)
    )


def _deduplicate_genes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one row per gene symbol, preferring the lowest padj and then largest |FC|.
    """
    ranked = df.copy()
    ranked["_abs_fc"] = ranked[FC_COL].abs()
    ranked = ranked.sort_values(
        ["padj", "_abs_fc", PPEE_COL],
        ascending=[True, False, True],
        na_position="last",
    )
    ranked = ranked.drop_duplicates(subset=[GENE_COL], keep="first")
    return ranked.drop(columns="_abs_fc")


def _write_source_diagnostics(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    diagnostics = [
        {"metric": "n_rows_raw", "value": int(len(df_raw))},
        {"metric": "n_rows_clean", "value": int(len(df_clean))},
        {"metric": "n_missing_gene", "value": int(df_raw[GENE_COL].isna().sum())},
        {"metric": "n_missing_fc", "value": int(pd.to_numeric(df_raw[FC_COL], errors="coerce").isna().sum())},
        {"metric": "n_missing_ppee", "value": int(pd.to_numeric(df_raw[PPEE_COL], errors="coerce").isna().sum())},
        {"metric": "n_unique_genes_clean", "value": int(df_clean[GENE_COL].nunique())},
        {"metric": "n_fdr_lt_0_05", "value": int((df_clean["padj"] < PADJ_THRESHOLD).sum())},
        {
            "metric": "n_fdr_lt_0_05_abs_fc_gt_1",
            "value": int(((df_clean["padj"] < PADJ_THRESHOLD) & (df_clean[FC_COL].abs() > LOG2FC_THRESHOLD)).sum()),
        },
        {
            "metric": "n_pc_fdr_lt_0_05_abs_fc_gt_1",
            "value": int(
                (
                    (df_clean["padj"] < PADJ_THRESHOLD)
                    & (df_clean[FC_COL].abs() > LOG2FC_THRESHOLD)
                    & (df_clean[GENETYPE_COL] == "protein_coding")
                ).sum()
            ),
        },
    ]
    pd.DataFrame(diagnostics).to_csv(TABLES_DIR / "signature_source_diagnostics.csv", index=False)


def load_signature_source(write_diagnostics: bool = True) -> pd.DataFrame:
    """
    Load the Excel source directly.
    Uses GeneType from the Excel file as the authoritative source.
    """
    if not EXCEL_FILE.exists():
        raise FileNotFoundError(f"Missing Excel file: {EXCEL_FILE}")

    df = pd.read_excel(EXCEL_FILE, sheet_name="Cmpr")
    df_raw = df.copy()

    required = [GENE_COL, GENETYPE_COL, FC_COL, PPEE_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in Excel: {missing}")

    df[GENE_COL] = _normalize_gene_symbols(df[GENE_COL])
    df[GENETYPE_COL] = _normalize_gene_types(df[GENETYPE_COL])
    df[FC_COL] = pd.to_numeric(df[FC_COL], errors="coerce")
    df[PPEE_COL] = pd.to_numeric(df[PPEE_COL], errors="coerce")

    df = df.dropna(subset=[GENE_COL, FC_COL, PPEE_COL]).copy()
    df["padj"] = bh_correct(df, PPEE_COL)
    df = _deduplicate_genes(df)
    if write_diagnostics:
        _write_source_diagnostics(df_raw, df)

    return df


def make_weights(series: pd.Series, flip_sign: bool = True) -> Dict[str, float]:
    """
    Convert signed log2FC values into capped/scaled weights.

    For the acetylation contrast, negative FC means activated by acetylation,
    so we flip sign so that positive weight = higher SNAI1-ac program activity.
    """
    capped = series.clip(lower=-CAP_THRESHOLD, upper=CAP_THRESHOLD)
    weights = {}
    for gene, value in capped.items():
        weight = float(value) / CAP_THRESHOLD
        if flip_sign:
            weight = -weight
        weights[gene] = round(weight, 6)
    return weights


def build_signature_variants(df: pd.DataFrame, write_sidecars: bool = True) -> Dict[str, Dict[str, object]]:
    """
    Build all robustness signature variants.
    """
    df = df.copy()
    df_fdr = df[df["padj"] < PADJ_THRESHOLD].copy()

    protein_coding = df_fdr[df_fdr[GENETYPE_COL] == "protein_coding"].copy()

    full_thresholded = df_fdr[df_fdr[FC_COL].abs() > LOG2FC_THRESHOLD].copy()
    pc_thresholded = protein_coding[protein_coding[FC_COL].abs() > LOG2FC_THRESHOLD].copy()

    pc_ranked = protein_coding.reindex(
        protein_coding[FC_COL].abs().sort_values(ascending=False).index
    )
    pc_top100 = pc_ranked.head(min(TOP_N, len(pc_ranked))).copy()

    pc_up = protein_coding[protein_coding[FC_COL] < 0].copy()
    pc_down = protein_coding[protein_coding[FC_COL] > 0].copy()

    variants = {
        "full_thresholded": full_thresholded,
        "pc_thresholded": pc_thresholded,
        "pc_top100": pc_top100,
        "pc_up": pc_up,
        "pc_down": pc_down,
    }

    out = {}
    rows = []

    for spec in SIGNATURE_SPECS:
        sig_id = spec["signature_id"]
        subset = variants[sig_id].copy()
        weights = make_weights(subset.set_index(GENE_COL)[FC_COL], flip_sign=True)

        out[sig_id] = {
            "df": subset,
            "weights": weights,
            "score_key": spec["score_key"],
            "score_col": f"{spec['score_key']}_score",
            "selection_rule": spec["selection_rule"],
        }

        rows.append({
            "signature_id": sig_id,
            "n_genes": len(subset),
            "n_protein_coding": int((subset[GENETYPE_COL] == "protein_coding").sum()) if len(subset) else 0,
            "n_noncoding": int((subset[GENETYPE_COL] != "protein_coding").sum()) if len(subset) else 0,
            "n_up": int((subset[FC_COL] < 0).sum()),
            "n_down": int((subset[FC_COL] > 0).sum()),
            "selection_rule": spec["selection_rule"],
            "weight_min": min(weights.values()) if weights else np.nan,
            "weight_max": max(weights.values()) if weights else np.nan,
        })

        if write_sidecars:
            with open(WEIGHTS_DIR / f"snai1_ac_{sig_id}.json", "w") as f:
                json.dump(weights, f, indent=2)

    summary = pd.DataFrame(rows)
    if write_sidecars:
        summary.to_csv(TABLES_DIR / "signature_definition_summary.csv", index=False)

    if summary["n_genes"].max() == 0:
        raise RuntimeError(
            "No genes passed robustness signature filters. "
            f"Check {TABLES_DIR / 'signature_source_diagnostics.csv'} for parsing diagnostics."
        )

    return out


def find_h5ad_samples(dataset_id: str) -> List[Dict[str, Path]]:
    dataset_path = PROCESSED_DATA_DIR / dataset_id
    if not dataset_path.exists():
        return []
    return [{"dataset": dataset_id, "sample": p.stem, "path": p} for p in sorted(dataset_path.glob("*.h5ad"))]


def iter_samples(dataset_filter: Optional[str]) -> List[Dict[str, Path]]:
    datasets = [dataset_filter] if dataset_filter else KNOWN_DATASETS
    samples = []
    for dataset in datasets:
        samples.extend(find_h5ad_samples(dataset))
    return samples


def exclude_samples(samples: List[Dict[str, Path]], excluded_datasets: Optional[List[str]]) -> List[Dict[str, Path]]:
    if not excluded_datasets:
        return samples
    return [
        sample
        for sample in samples
        if not any(excluded in sample["dataset"] for excluded in excluded_datasets)
    ]


def safe_read_h5ad_backed(path: Path):
    require_scanpy()
    return sc.read_h5ad(path, backed="r")


def _dataset_to_path(dataset_id: str) -> Path:
    return Path(*dataset_id.split("/"))


def cache_h5ad_path(sample_info: Dict[str, Path]) -> Path:
    return CACHE_DIR / _dataset_to_path(sample_info["dataset"]) / f"{sample_info['sample']}.h5ad"


def ensure_cache_parent(sample_info: Dict[str, Path]) -> Path:
    cache_path = cache_h5ad_path(sample_info)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    return cache_path


def cached_sample_exists(sample_info: Dict[str, Path]) -> bool:
    return cache_h5ad_path(sample_info).exists()


def get_cached_read_path(sample_info: Dict[str, Path]) -> Optional[Path]:
    cache_path = cache_h5ad_path(sample_info)
    if cache_path.exists():
        return cache_path
    return None


def _safe_corr(a: pd.Series, b: pd.Series, method: str = "spearman") -> float:
    mask = ~(pd.isna(a) | pd.isna(b))
    if mask.sum() < 3:
        return np.nan
    if method == "spearman":
        return float(spearmanr(a[mask], b[mask]).statistic)
    return float(pearsonr(a[mask], b[mask])[0])


def _make_knn_weights(coords: np.ndarray):
    w = libpysal.weights.KNN.from_array(coords, k=MORAN_K)
    w.transform = "r"
    return w


def _neighbor_self_corr(scores: np.ndarray, w) -> float:
    self_scores = []
    neighbor_means = []
    for idx in range(len(scores)):
        neighbors = w.neighbors.get(idx, [])
        if len(neighbors) == 0:
            continue
        self_scores.append(scores[idx])
        neighbor_means.append(scores[neighbors].mean())
    if len(self_scores) < 3:
        return np.nan
    return float(spearmanr(self_scores, neighbor_means).statistic)


def _score_to_null_expression(adata, genes: List[str]) -> np.ndarray:
    """
    Lightweight null score based on mean expression of matched random genes.
    This is intentionally simple for robustness benchmarking, not a replacement
    for the main EnrichMap score.
    """
    if len(genes) == 0:
        return np.full(adata.n_obs, np.nan)
    x = adata[:, genes].X
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x.mean(axis=1)).ravel()


def _score_to_null_expression_indices(x, gene_indices: np.ndarray) -> np.ndarray:
    """Mean-expression null score using pre-resolved gene indices."""
    if len(gene_indices) == 0:
        return np.full(x.shape[0], np.nan)
    subset = x[:, gene_indices]
    return np.asarray(subset.mean(axis=1)).ravel()


# =============================================================================
# STEP 1: DETECTION
# =============================================================================

def run_detection(signatures: Dict[str, Dict[str, object]], samples: List[Dict[str, Path]]) -> None:
    rows = []

    for sample_info in samples:
        adata = safe_read_h5ad_backed(sample_info["path"])
        genes_present = set(adata.var_names)

        for sig_id, sig in signatures.items():
            subset = sig["df"]
            detected_mask = subset[GENE_COL].isin(genes_present)
            detected = subset[detected_mask]
            missing = subset[~detected_mask]

            rows.append({
                "dataset": sample_info["dataset"],
                "sample": sample_info["sample"],
                "signature_id": sig_id,
                "n_signature_genes": len(subset),
                "n_detected": int(detected_mask.sum()),
                "pct_detected": float(detected_mask.mean()) if len(subset) else np.nan,
                "n_up_detected": int((detected[FC_COL] < 0).sum()),
                "n_down_detected": int((detected[FC_COL] > 0).sum()),
                "mean_abs_weight_detected": float(np.mean(np.abs(detected[FC_COL].clip(-CAP_THRESHOLD, CAP_THRESHOLD) / CAP_THRESHOLD))) if len(detected) else np.nan,
                "mean_abs_weight_missing": float(np.mean(np.abs(missing[FC_COL].clip(-CAP_THRESHOLD, CAP_THRESHOLD) / CAP_THRESHOLD))) if len(missing) else np.nan,
            })

        adata.file.close()

    per_sample = pd.DataFrame(rows)
    per_sample.to_csv(TABLES_DIR / "signature_detection_summary_per_sample.csv", index=False)

    overall = per_sample.groupby("signature_id", as_index=False).agg(
        mean_n_detected=("n_detected", "mean"),
        mean_pct_detected=("pct_detected", "mean"),
        sd_pct_detected=("pct_detected", "std"),
        min_pct_detected=("pct_detected", "min"),
        max_pct_detected=("pct_detected", "max"),
    )
    overall.to_csv(TABLES_DIR / "signature_detection_summary_overall.csv", index=False)


# =============================================================================
# STEP 2: SCORING
# =============================================================================

def run_scoring(signatures: Dict[str, Dict[str, object]], samples: List[Dict[str, Path]]) -> None:
    require_scanpy()
    require_enrichmap()

    rows = []

    for sample_info in samples:
        adata = sc.read_h5ad(sample_info["path"])
        cache_path = ensure_cache_parent(sample_info)

        for sig_id, sig in signatures.items():
            weights = sig["weights"]
            score_key = sig["score_key"]
            score_col = sig["score_col"]

            weights_filtered = {g: w for g, w in weights.items() if g in adata.var_names}

            if not weights_filtered:
                rows.append({
                    "dataset": sample_info["dataset"],
                    "sample": sample_info["sample"],
                    "signature_id": sig_id,
                    "cache_path": str(cache_path),
                    "success": False,
                    "n_genes_scored": 0,
                    "score_min": np.nan,
                    "score_max": np.nan,
                    "score_mean": np.nan,
                    "score_sd": np.nan,
                })
                continue

            em.tl.score(
                adata=adata,
                gene_set=list(weights_filtered.keys()),
                gene_weights={score_key: weights_filtered},
                score_key=score_key,
                smoothing=True,
                correct_spatial_covariates=True
            )

            if score_col not in adata.obs.columns:
                raise RuntimeError(f"Expected score column '{score_col}' was not created for {sig_id}")

            scores = adata.obs[score_col]

            rows.append({
                "dataset": sample_info["dataset"],
                "sample": sample_info["sample"],
                "signature_id": sig_id,
                "cache_path": str(cache_path),
                "success": True,
                "n_genes_scored": len(weights_filtered),
                "score_min": float(scores.min()),
                "score_max": float(scores.max()),
                "score_mean": float(scores.mean()),
                "score_sd": float(scores.std()),
            })

        adata.write_h5ad(cache_path)

    pd.DataFrame(rows).to_csv(TABLES_DIR / "scoring_status.csv", index=False)


# =============================================================================
# STEP 3: COMPARISON
# =============================================================================

def run_comparison(signatures: Dict[str, Dict[str, object]], samples: List[Dict[str, Path]]) -> None:
    corr_rows = []
    sample_rows = []
    biology_rows = []

    compare_cols = ["SNAI1-ac_score"] + [sig["score_col"] for sig in signatures.values()]

    for sample_info in samples:
        read_path = get_cached_read_path(sample_info)
        if read_path is None:
            continue
        adata = safe_read_h5ad_backed(read_path)
        obs = adata.obs.copy()
        available_cols = [c for c in compare_cols if c in obs.columns]

        for col in available_cols:
            bc, _ = bimodality_coefficient(obs[col].values)
            sample_rows.append({
                "dataset": sample_info["dataset"],
                "sample": sample_info["sample"],
                "signature_id": col,
                "mean_score": float(obs[col].mean()),
                "sd_score": float(obs[col].std()),
                "skewness": float(skew(obs[col].values)),
                "bimodality_coefficient": float(bc),
            })

        for i, col_a in enumerate(available_cols):
            for col_b in available_cols[i + 1:]:
                corr_rows.append({
                    "dataset": sample_info["dataset"],
                    "sample": sample_info["sample"],
                    "sig_a": col_a,
                    "sig_b": col_b,
                    "spearman_r": _safe_corr(obs[col_a], obs[col_b], "spearman"),
                    "pearson_r": _safe_corr(obs[col_a], obs[col_b], "pearson"),
                })

        for col in available_cols:
            row = {
                "dataset": sample_info["dataset"],
                "sample": sample_info["sample"],
                "signature_id": col,
            }

            for celltype in CELLTYPE_COLS:
                row[f"corr_{celltype}"] = _safe_corr(obs[col], obs[celltype], "spearman") if celltype in obs.columns else np.nan

            for pair_name, a_label, b_label in [
                ("tumor_vs_stroma", "Tumor", "Stroma"),
                ("tumor_vs_interface", "Tumor", "Interface"),
            ]:
                if "interface" in obs.columns:
                    a_vals = obs.loc[obs["interface"] == a_label, col].values
                    b_vals = obs.loc[obs["interface"] == b_label, col].values
                    row[f"cohensd_{pair_name}"] = float(cohens_d(a_vals, b_vals)) if len(a_vals) and len(b_vals) else np.nan
                else:
                    row[f"cohensd_{pair_name}"] = np.nan

            biology_rows.append(row)

        adata.file.close()

    pd.DataFrame(corr_rows).to_csv(TABLES_DIR / "signature_comparison_spotwise_correlations.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(TABLES_DIR / "signature_comparison_samplewise_summary.csv", index=False)
    pd.DataFrame(biology_rows).to_csv(TABLES_DIR / "signature_vs_biology_summary.csv", index=False)


# =============================================================================
# STEP 4: SPATIAL
# =============================================================================

def run_spatial(signatures: Dict[str, Dict[str, object]], samples: List[Dict[str, Path]]) -> None:
    if not SPATIAL_AVAILABLE:
        raise ImportError("libpysal and esda are required for the spatial step.")

    autocorr_rows = []
    smooth_rows = []
    local_rows = []

    score_cols = ["SNAI1-ac_score"] + [sig["score_col"] for sig in signatures.values()]

    for sample_info in samples:
        read_path = get_cached_read_path(sample_info)
        if read_path is None:
            continue
        adata = safe_read_h5ad_backed(read_path)

        if "spatial" not in adata.obsm:
            adata.file.close()
            continue

        coords = adata.obsm["spatial"]
        w = _make_knn_weights(coords)

        for col in score_cols:
            if col not in adata.obs.columns:
                continue

            scores = adata.obs[col].to_numpy()

            moran = Moran(scores, w)
            autocorr_rows.append({
                "dataset": sample_info["dataset"],
                "sample": sample_info["sample"],
                "signature_id": col,
                "morans_I": float(moran.I),
                "morans_p": float(moran.p_sim),
                "n_spots": int(len(scores)),
            })

            smooth_rows.append({
                "dataset": sample_info["dataset"],
                "sample": sample_info["sample"],
                "signature_id": col,
                "neighbor_self_corr": _neighbor_self_corr(scores, w),
                "neighbor_mean_diff": float(np.mean(np.abs(scores - np.median(scores)))),
            })

            local = Moran_Local(scores, w, permutations=99)
            sig_mask = local.p_sim < 0.05
            q = local.q

            local_rows.append({
                "dataset": sample_info["dataset"],
                "sample": sample_info["sample"],
                "signature_id": col,
                "pct_HH": float(np.mean(sig_mask & (q == 1))),
                "pct_LL": float(np.mean(sig_mask & (q == 3))),
                "pct_LH": float(np.mean(sig_mask & (q == 2))),
                "pct_HL": float(np.mean(sig_mask & (q == 4))),
                "pct_not_significant": float(np.mean(~sig_mask)),
            })

        adata.file.close()

    per_sample = pd.DataFrame(autocorr_rows)
    per_sample.to_csv(TABLES_DIR / "spatial_autocorrelation_per_sample.csv", index=False)

    summary = per_sample.groupby("signature_id", as_index=False).agg(
        mean_morans_I=("morans_I", "mean"),
        sd_morans_I=("morans_I", "std"),
        median_morans_I=("morans_I", "median"),
        pct_positive=("morans_I", lambda x: float(np.mean(x > 0))),
        pct_significant=("morans_p", lambda x: float(np.mean(x < 0.05))),
    )
    summary.to_csv(TABLES_DIR / "spatial_autocorrelation_summary.csv", index=False)

    pd.DataFrame(smooth_rows).to_csv(TABLES_DIR / "neighborhood_smoothness_per_sample.csv", index=False)
    pd.DataFrame(local_rows).to_csv(TABLES_DIR / "local_spatial_structure_per_sample.csv", index=False)


# =============================================================================
# STEP 5: NULL MODEL
# =============================================================================

def run_null_model(signatures: Dict[str, Dict[str, object]], samples: List[Dict[str, Path]]) -> None:
    """
    Compare real signatures to matched null signatures.

    Matching:
    - same gene count
    - same biotype universe: protein-coding only for pc signatures,
      all available genes for full signature
    - match the number of genes actually detected in the sample
    - null score uses mean expression over random matched genes
    """
    if not SPATIAL_AVAILABLE:
        raise ImportError("libpysal and esda are required for the null step.")
    require_scanpy()

    rng = np.random.default_rng(RANDOM_SEED)
    source = load_signature_source()

    full_pool_genes = source[GENE_COL].dropna().unique().tolist()
    pc_pool_genes = source.loc[source[GENETYPE_COL] == "protein_coding", GENE_COL].dropna().unique().tolist()

    rows = []

    for sample_info in samples:
        read_path = get_cached_read_path(sample_info)
        if read_path is None:
            continue
        print(f"Null model: {sample_info['dataset']} / {sample_info['sample']}", flush=True)
        adata = sc.read_h5ad(read_path)

        if "spatial" not in adata.obsm:
            continue

        coords = adata.obsm["spatial"]
        w = _make_knn_weights(coords)

        genes_present = set(adata.var_names)
        gene_to_idx = {str(g): i for i, g in enumerate(adata.var_names)}
        x = adata.X

        for sig_id, sig in signatures.items():
            real_col = sig["score_col"]
            if real_col not in adata.obs.columns:
                continue

            real_scores = adata.obs[real_col].to_numpy()
            real_moran = float(Moran(real_scores, w).I)
            real_neighbor = _neighbor_self_corr(real_scores, w)

            real_genes = [g for g in sig["weights"].keys() if g in genes_present]
            n_genes = len(real_genes)
            if n_genes < 5:
                continue

            if sig_id == "full_thresholded":
                pool = [g for g in full_pool_genes if g in genes_present and g not in real_genes]
            else:
                pool = [g for g in pc_pool_genes if g in genes_present and g not in real_genes]

            if len(pool) < max(10, n_genes):
                continue

            null_morans = []
            null_neighbors = []
            pool_indices = np.array([gene_to_idx[g] for g in pool], dtype=int)

            for _ in range(NULL_ITERATIONS):
                chosen_indices = rng.choice(pool_indices, size=n_genes, replace=False)
                null_expr = _score_to_null_expression_indices(x, chosen_indices)

                if np.isnan(null_expr).all() or np.std(null_expr) == 0:
                    continue

                null_morans.append(float(Moran(null_expr, w).I))
                null_neighbors.append(_neighbor_self_corr(null_expr, w))

            if len(null_morans) == 0:
                continue

            rows.append({
                "dataset": sample_info["dataset"],
                "sample": sample_info["sample"],
                "signature_id": sig_id,
                "n_real_genes_detected": n_genes,
                "null_pool_size": len(pool),
                "real_morans_I": real_moran,
                "null_mean_morans_I": float(np.mean(null_morans)),
                "null_sd_morans_I": float(np.std(null_morans)),
                "real_minus_null_morans_I": float(real_moran - np.mean(null_morans)),
                "null_empirical_p_morans": float((np.sum(np.array(null_morans) >= real_moran) + 1) / (len(null_morans) + 1)),
                "real_neighbor_corr": real_neighbor,
                "null_mean_neighbor_corr": float(np.nanmean(null_neighbors)),
                "real_minus_null_neighbor_corr": float(real_neighbor - np.nanmean(null_neighbors)),
                "null_empirical_p_neighbor": float((np.sum(np.array(null_neighbors) >= real_neighbor) + 1) / (len(null_neighbors) + 1)),
            })

        del adata

    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "null_signature_comparison.csv", index=False)

    if len(df):
        summary = df.groupby("signature_id", as_index=False).agg(
            mean_real_morans_I=("real_morans_I", "mean"),
            mean_null_morans_I=("null_mean_morans_I", "mean"),
            mean_delta_morans_I=("real_minus_null_morans_I", "mean"),
            pct_samples_real_gt_null_95=("null_empirical_p_morans", lambda x: float(np.mean(x < 0.05))),
            mean_real_neighbor_corr=("real_neighbor_corr", "mean"),
            mean_null_neighbor_corr=("null_mean_neighbor_corr", "mean"),
        )
    else:
        summary = pd.DataFrame()

    summary.to_csv(TABLES_DIR / "null_signature_summary.csv", index=False)


# =============================================================================
# STEP 5B: ENRICHMAP-MATCHED NULL MODEL
# =============================================================================

def _report_sample_pairs() -> set[tuple[str, str]]:
    """
    Use the same cohort as the report-facing spatial distribution table.
    This intentionally excludes scored-but-not-compiled samples.
    """
    path = ANALYSIS_DIR / "cross_sample" / "compiled" / "all_distribution_stats.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing report cohort table: {path}")
    df = pd.read_csv(path, usecols=["dataset", "sample"])
    return set(zip(df["dataset"].astype(str), df["sample"].astype(str)))


def _report_samples(samples: List[Dict[str, Path]]) -> List[Dict[str, Path]]:
    pairs = _report_sample_pairs()
    return [
        sample for sample in samples
        if (str(sample["dataset"]), str(sample["sample"])) in pairs
    ]


def _load_production_weights() -> Dict[str, float]:
    path = SIGNATURE_DIR / "snai1_ac_weights.json"
    with open(path, "r", encoding="utf-8") as handle:
        return {str(g): float(w) for g, w in json.load(handle).items()}


def _protein_coding_gene_universe() -> set[str]:
    source = load_signature_source()
    pc = source.loc[source[GENETYPE_COL] == "protein_coding", GENE_COL].dropna()
    return set(pc.astype(str))


def _remove_obs_columns(adata, columns: List[str]) -> None:
    for col in columns:
        if col in adata.obs:
            del adata.obs[col]


def run_enrichmap_matched_null(
    signatures: Dict[str, Dict[str, object]],
    samples: List[Dict[str, Path]],
    n_iterations: int = ENRICHMAP_NULL_ITERATIONS,
    benchmark: bool = False,
    background_universe: str = "detected_protein_coding",
    weighting: str = "uniform",
    output_prefix: str = "enrichmap_null",
    exclude_datasets: Optional[List[str]] = None,
    smoothing: bool = True,
    correct_spatial_covariates: bool = True,
) -> None:
    """
    Compare the production SNAI1-ac score to matched random gene sets scored
    through the same EnrichMap pipeline.

    Design:
    - same samples as all_distribution_stats.csv
    - same number of detected genes as the production SNAI1-ac signature
    - random genes drawn from the requested detected background universe
    - uniform or production-matched signed weights for random signatures
    - configurable EnrichMap smoothing/covariate correction settings
    """
    if background_universe not in {"detected_protein_coding", "all_detected"}:
        raise ValueError(f"Unsupported background_universe: {background_universe}")
    if weighting not in {"uniform", "matched_signed"}:
        raise ValueError(f"Unsupported weighting: {weighting}")

    require_scanpy()
    require_enrichmap()
    if not SPATIAL_AVAILABLE:
        raise ImportError("libpysal and esda are required for enrichmap_null.")

    rng = np.random.default_rng(RANDOM_SEED)
    production_weights = _load_production_weights()
    pc_universe = _protein_coding_gene_universe()
    report_samples = _report_samples(samples)
    if exclude_datasets:
        report_samples = [
            s for s in report_samples
            if not any(excluded in s["dataset"] for excluded in exclude_datasets)
        ]

    if benchmark:
        report_samples = [s for s in report_samples if s["dataset"] == "visium/denisenko_2022" and s["sample"] == "SP4"]
        n_iterations = min(n_iterations, 5)

    long_rows = []
    summary_rows = []

    for sample_info in report_samples:
        print(
            f"{output_prefix}: {sample_info['dataset']} / {sample_info['sample']}",
            flush=True,
        )
        adata = sc.read_h5ad(sample_info["path"])

        if "spatial" not in adata.obsm or "SNAI1-ac_score" not in adata.obs:
            print("  skipping: missing spatial coordinates or SNAI1-ac_score")
            continue

        present_genes = set(map(str, adata.var_names))
        real_genes = [g for g in production_weights if g in present_genes]
        n_genes = len(real_genes)
        if n_genes < 5:
            print(f"  skipping: only {n_genes} production genes detected")
            continue

        coords = adata.obsm["spatial"]
        w = _make_knn_weights(coords)
        real_score_col_to_remove = None
        if smoothing:
            real_scores = adata.obs["SNAI1-ac_score"].to_numpy()
        else:
            real_score_key = f"{output_prefix}_real"
            real_score_col = f"{real_score_key}_score"
            em.tl.score(
                adata=adata,
                gene_set=real_genes,
                gene_weights={real_score_key: {g: production_weights[g] for g in real_genes}},
                score_key=real_score_key,
                smoothing=False,
                correct_spatial_covariates=correct_spatial_covariates,
            )
            if real_score_col not in adata.obs:
                raise RuntimeError(f"Expected unsmoothed real score column missing: {real_score_col}")
            real_scores = adata.obs[real_score_col].to_numpy()
            real_score_col_to_remove = real_score_col
        real_moran = float(Moran(real_scores, w).I)
        real_neighbor = _neighbor_self_corr(real_scores, w)

        if background_universe == "detected_protein_coding":
            pool = sorted((present_genes & pc_universe) - set(real_genes))
        else:
            pool = sorted(present_genes - set(real_genes))
        if len(pool) < n_genes:
            print(f"  skipping: null pool too small ({len(pool)} < {n_genes})")
            continue

        real_weight_values = np.asarray([production_weights[g] for g in real_genes], dtype=float)
        null_morans = []
        null_neighbors = []
        for i in range(n_iterations):
            chosen = rng.choice(pool, size=n_genes, replace=False).tolist()
            score_key = f"{output_prefix}_{i:03d}"
            score_col = f"{score_key}_score"
            if weighting == "uniform":
                weights = {gene: 1.0 for gene in chosen}
            else:
                shuffled_weights = rng.permutation(real_weight_values)
                weights = {gene: float(weight) for gene, weight in zip(chosen, shuffled_weights)}

            import time
            t0 = time.perf_counter()
            em.tl.score(
                adata=adata,
                gene_set=chosen,
                gene_weights={score_key: weights},
                score_key=score_key,
                smoothing=smoothing,
                correct_spatial_covariates=correct_spatial_covariates,
            )
            elapsed = time.perf_counter() - t0

            if score_col not in adata.obs:
                raise RuntimeError(f"Expected null score column missing: {score_col}")

            null_scores = adata.obs[score_col].to_numpy()
            if np.isnan(null_scores).all() or np.nanstd(null_scores) == 0:
                continue
            null_moran = float(Moran(null_scores, w).I)
            null_neighbor = _neighbor_self_corr(null_scores, w)
            null_morans.append(null_moran)
            null_neighbors.append(null_neighbor)
            print(f"  null {i + 1}/{n_iterations}: {elapsed:.1f}s, Moran's I={null_moran:.3f}", flush=True)

            long_rows.append({
                "dataset": sample_info["dataset"],
                "sample": sample_info["sample"],
                "null_id": i,
                "background_universe": background_universe,
                "weighting": weighting,
                "smoothing": smoothing,
                "correct_spatial_covariates": correct_spatial_covariates,
                "n_real_genes_detected": n_genes,
                "null_pool_size": len(pool),
                "real_morans_I": real_moran,
                "null_morans_I": null_moran,
                "real_minus_null_morans_I": real_moran - null_moran,
                "real_neighbor_corr": real_neighbor,
                "null_neighbor_corr": null_neighbor,
                "real_minus_null_neighbor_corr": real_neighbor - null_neighbor,
                "score_seconds": elapsed,
            })

            _remove_obs_columns(adata, [score_col])
            if "gene_contributions" in adata.uns and score_key in adata.uns["gene_contributions"]:
                del adata.uns["gene_contributions"][score_key]

        if real_score_col_to_remove:
            _remove_obs_columns(adata, [real_score_col_to_remove])
            if "gene_contributions" in adata.uns and f"{output_prefix}_real" in adata.uns["gene_contributions"]:
                del adata.uns["gene_contributions"][f"{output_prefix}_real"]

        if len(null_morans) == 0:
            print("  skipping: no usable null scores", flush=True)
            del adata
            continue

        null_morans_arr = np.asarray(null_morans)
        null_neighbors_arr = np.asarray(null_neighbors)
        empirical_p = float((np.sum(null_morans_arr >= real_moran) + 1) / (len(null_morans_arr) + 1))
        neighbor_p = float((np.sum(null_neighbors_arr >= real_neighbor) + 1) / (len(null_neighbors_arr) + 1))

        summary_rows.append({
            "dataset": sample_info["dataset"],
            "sample": sample_info["sample"],
            "background_universe": background_universe,
            "weighting": weighting,
            "smoothing": smoothing,
            "correct_spatial_covariates": correct_spatial_covariates,
            "n_null_iterations": len(null_morans_arr),
            "n_real_genes_detected": n_genes,
            "null_pool_size": len(pool),
            "real_morans_I": real_moran,
            "null_mean_morans_I": float(np.mean(null_morans_arr)),
            "null_sd_morans_I": float(np.std(null_morans_arr, ddof=1)),
            "null_q95_morans_I": float(np.quantile(null_morans_arr, 0.95)),
            "real_minus_null_mean_morans_I": float(real_moran - np.mean(null_morans_arr)),
            "real_moran_percentile": float(np.mean(null_morans_arr < real_moran)),
            "null_empirical_p_morans": empirical_p,
            "real_neighbor_corr": real_neighbor,
            "null_mean_neighbor_corr": float(np.nanmean(null_neighbors_arr)),
            "null_q95_neighbor_corr": float(np.nanquantile(null_neighbors_arr, 0.95)),
            "real_minus_null_mean_neighbor_corr": float(real_neighbor - np.nanmean(null_neighbors_arr)),
            "real_neighbor_percentile": float(np.mean(null_neighbors_arr < real_neighbor)),
            "null_empirical_p_neighbor": neighbor_p,
        })

        del adata

    long_df = pd.DataFrame(long_rows)
    sample_df = pd.DataFrame(summary_rows)
    long_df.to_csv(TABLES_DIR / f"{output_prefix}_comparison.csv", index=False)
    sample_df.to_csv(TABLES_DIR / f"{output_prefix}_summary_per_sample.csv", index=False)

    if len(sample_df):
        cohort = pd.DataFrame([{
            "background_universe": background_universe,
            "weighting": weighting,
            "smoothing": smoothing,
            "correct_spatial_covariates": correct_spatial_covariates,
            "n_samples": int(len(sample_df)),
            "n_null_iterations_per_sample": int(sample_df["n_null_iterations"].min()),
            "mean_real_morans_I": float(sample_df["real_morans_I"].mean()),
            "mean_null_morans_I": float(sample_df["null_mean_morans_I"].mean()),
            "mean_delta_morans_I": float(sample_df["real_minus_null_mean_morans_I"].mean()),
            "median_delta_morans_I": float(sample_df["real_minus_null_mean_morans_I"].median()),
            "pct_samples_real_gt_null_95": float(np.mean(sample_df["null_empirical_p_morans"] < 0.05)),
            "median_real_moran_percentile": float(sample_df["real_moran_percentile"].median()),
            "mean_real_neighbor_corr": float(sample_df["real_neighbor_corr"].mean()),
            "mean_null_neighbor_corr": float(sample_df["null_mean_neighbor_corr"].mean()),
            "mean_delta_neighbor_corr": float(sample_df["real_minus_null_mean_neighbor_corr"].mean()),
            "pct_samples_neighbor_gt_null_95": float(np.mean(sample_df["null_empirical_p_neighbor"] < 0.05)),
        }])
    else:
        cohort = pd.DataFrame()
    cohort.to_csv(TABLES_DIR / f"{output_prefix}_summary_cohort.csv", index=False)


def run_enrichmap_null_sensitivity(
    signatures: Dict[str, Dict[str, object]],
    samples: List[Dict[str, Path]],
    n_iterations: int = ENRICHMAP_NULL_ITERATIONS,
) -> None:
    """Run sensitivity nulls excluding low-quality Stur sections."""
    sensitivity_specs = [
        {
            "background_universe": "all_detected",
            "weighting": "uniform",
            "output_prefix": "enrichmap_null_all_detected_uniform_no_stur",
        },
        {
            "background_universe": "detected_protein_coding",
            "weighting": "matched_signed",
            "output_prefix": "enrichmap_null_pc_matched_signed_no_stur",
        },
    ]
    for spec in sensitivity_specs:
        run_enrichmap_matched_null(
            signatures,
            samples,
            n_iterations=n_iterations,
            exclude_datasets=["stur_2021"],
            **spec,
        )


def run_enrichmap_unsmoothed_sensitivity(
    signatures: Dict[str, Dict[str, object]],
    samples: List[Dict[str, Path]],
    n_iterations: int = ENRICHMAP_NULL_ITERATIONS,
) -> None:
    """Test whether the production signature is spatial before neighborhood smoothing."""
    sensitivity_specs = [
        {
            "background_universe": "all_detected",
            "weighting": "uniform",
            "output_prefix": "enrichmap_unsmoothed_null_all_detected_uniform_no_stur",
        },
        {
            "background_universe": "detected_protein_coding",
            "weighting": "matched_signed",
            "output_prefix": "enrichmap_unsmoothed_null_pc_matched_signed_no_stur",
        },
    ]
    for spec in sensitivity_specs:
        run_enrichmap_matched_null(
            signatures,
            samples,
            n_iterations=n_iterations,
            exclude_datasets=["stur_2021"],
            smoothing=False,
            correct_spatial_covariates=True,
            **spec,
        )


def _load_enrichmap_null_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    long_path = TABLES_DIR / "enrichmap_null_comparison.csv"
    sample_path = TABLES_DIR / "enrichmap_null_summary_per_sample.csv"
    hallmark_path = ANALYSIS_DIR / "cross_sample" / "compiled" / "all_hallmark_morans_I.csv"
    if not long_path.exists() or not sample_path.exists():
        raise FileNotFoundError("Run --step enrichmap_null before plotting EnrichMap null figures.")
    long_df = pd.read_csv(long_path)
    sample_df = pd.read_csv(sample_path)
    hallmark_df = pd.read_csv(hallmark_path) if hallmark_path.exists() else pd.DataFrame()
    return long_df, sample_df, hallmark_df


def _sorted_enrichmap_null_samples(sample_df: pd.DataFrame) -> List[str]:
    ordered = sample_df.sort_values("real_morans_I", ascending=True)
    return [f"{row.dataset} / {row.sample}" for row in ordered.itertuples()]


def _save_enrichmap_null_panel_a(ax, long_df: pd.DataFrame, sample_df: pd.DataFrame) -> None:
    long_df = long_df.copy()
    sample_df = sample_df.copy()
    long_df["sample_label"] = long_df["dataset"].astype(str) + " / " + long_df["sample"].astype(str)
    sample_df["sample_label"] = sample_df["dataset"].astype(str) + " / " + sample_df["sample"].astype(str)
    order = _sorted_enrichmap_null_samples(sample_df)

    sns.boxplot(data=long_df, y="sample_label", x="null_morans_I", order=order, ax=ax, color="#d9d9d9", fliersize=0)
    sns.scatterplot(data=sample_df, y="sample_label", x="real_morans_I", ax=ax, color="#b2182b", s=28, zorder=3, legend=False)
    ax.set_xlabel("Moran's I")
    ax.set_ylabel("")
    ax.set_title("A. Real SNAI1-ac vs EnrichMap nulls")


def _save_enrichmap_null_panel_b(ax, sample_df: pd.DataFrame) -> None:
    df = sample_df.copy()
    df["sample_label"] = df["dataset"].astype(str) + " / " + df["sample"].astype(str)
    df = df.sort_values("real_minus_null_mean_morans_I", ascending=True)
    colors = np.where(df["real_minus_null_mean_morans_I"] >= 0, "#b2182b", "#2166ac")
    ax.barh(df["sample_label"], df["real_minus_null_mean_morans_I"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Real - mean null Moran's I")
    ax.set_ylabel("")
    ax.set_title("B. Spatial structure above null")


def _save_enrichmap_null_panel_c(ax, sample_df: pd.DataFrame) -> None:
    df = sample_df.copy()
    df["sample_label"] = df["dataset"].astype(str) + " / " + df["sample"].astype(str)
    df = df.sort_values("real_moran_percentile", ascending=True)
    ax.barh(df["sample_label"], df["real_moran_percentile"], color="#762a83")
    ax.axvline(0.95, color="black", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Percentile of real score vs null")
    ax.set_ylabel("")
    ax.set_title("C. Empirical null percentile")


def _save_enrichmap_null_panel_d(ax, sample_df: pd.DataFrame, hallmark_df: pd.DataFrame) -> None:
    real = sample_df[["dataset", "sample", "real_morans_I"]].copy()
    real["group"] = "SNAI1-ac"
    real = real.rename(columns={"real_morans_I": "morans_I"})

    if len(hallmark_df):
        pairs = set(zip(sample_df["dataset"].astype(str), sample_df["sample"].astype(str)))
        hallmark = hallmark_df[
            hallmark_df.apply(lambda row: (str(row["dataset"]), str(row["sample"])) in pairs, axis=1)
        ][["dataset", "sample", "pathway", "morans_I"]].copy()
        hallmark["group"] = "Hallmark"
        plot_df = pd.concat([hallmark[["morans_I", "group"]], real[["morans_I", "group"]]], ignore_index=True)
    else:
        plot_df = real[["morans_I", "group"]]

    sns.boxplot(data=plot_df, x="group", y="morans_I", ax=ax, color="#d9d9d9", fliersize=0)
    sns.stripplot(data=plot_df, x="group", y="morans_I", ax=ax, color="#333333", size=2, alpha=0.25)
    ax.set_xlabel("")
    ax.set_ylabel("Moran's I")
    ax.set_title("D. Hallmark context")


def run_enrichmap_null_plots() -> None:
    sns.set_theme(style="whitegrid")
    long_df, sample_df, hallmark_df = _load_enrichmap_null_tables()

    fig, axes = plt.subplots(2, 2, figsize=(15, 14))
    _save_enrichmap_null_panel_a(axes[0, 0], long_df, sample_df)
    _save_enrichmap_null_panel_b(axes[0, 1], sample_df)
    _save_enrichmap_null_panel_c(axes[1, 0], sample_df)
    _save_enrichmap_null_panel_d(axes[1, 1], sample_df, hallmark_df)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "enrichmap_matched_null_moransI_composite.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    panel_specs = [
        ("A_real_vs_null_per_sample", _save_enrichmap_null_panel_a, (long_df, sample_df)),
        ("B_real_minus_null_delta", _save_enrichmap_null_panel_b, (sample_df,)),
        ("C_real_null_percentile", _save_enrichmap_null_panel_c, (sample_df,)),
        ("D_hallmark_context", _save_enrichmap_null_panel_d, (sample_df, hallmark_df)),
    ]
    for stem, func, args in panel_specs:
        height = 9 if stem.startswith(("A", "B", "C")) else 5
        fig, ax = plt.subplots(figsize=(8, height))
        func(ax, *args)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"enrichmap_matched_null_{stem}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


# =============================================================================
# STEP 6: TOP-GENE AUDIT
# =============================================================================

def run_topgene_audit(signatures: Dict[str, Dict[str, object]], samples: List[Dict[str, Path]]) -> None:
    """
    Audit how much the score is dominated by top |log2FC| genes and what happens
    when those genes are removed from the scoring set.
    """
    require_scanpy()
    require_enrichmap()
    if not SPATIAL_AVAILABLE:
        raise ImportError("Spatial packages are required for top-gene audit.")

    contrib_rows = []
    leaveout_rows = []

    for sample_info in samples:
        read_path = get_cached_read_path(sample_info)
        if read_path is None:
            continue
        adata = sc.read_h5ad(read_path)

        if "spatial" not in adata.obsm:
            continue

        coords = adata.obsm["spatial"]
        w = _make_knn_weights(coords)

        for sig_id, sig in signatures.items():
            subset = sig["df"].copy()
            if len(subset) == 0:
                continue

            subset["abs_fc"] = subset[FC_COL].abs()
            ranked = subset.sort_values("abs_fc", ascending=False)
            total_abs = ranked["abs_fc"].sum()

            if total_abs == 0:
                continue

            contrib_rows.append({
                "dataset": sample_info["dataset"],
                "sample": sample_info["sample"],
                "signature_id": sig_id,
                "top_1_frac": float(ranked["abs_fc"].head(1).sum() / total_abs),
                "top_5_frac": float(ranked["abs_fc"].head(5).sum() / total_abs),
                "top_10_frac": float(ranked["abs_fc"].head(10).sum() / total_abs),
                "top_20_frac": float(ranked["abs_fc"].head(20).sum() / total_abs),
            })

            real_col = sig["score_col"]
            if real_col not in adata.obs.columns:
                continue

            original = adata.obs[real_col].to_numpy()
            original_moran = Moran(original, w).I

            for drop_n in (1, 5, 10):
                remaining = ranked.iloc[drop_n:].copy()
                if len(remaining) < 5:
                    continue

                weights = make_weights(remaining.set_index(GENE_COL)[FC_COL], flip_sign=True)
                weights_filtered = {g: w for g, w in weights.items() if g in adata.var_names}
                if len(weights_filtered) < 5:
                    continue

                temp = adata.copy()
                temp_key = f"temp_{sig_id}_drop{drop_n}"

                em.tl.score(
                    adata=temp,
                    gene_set=list(weights_filtered.keys()),
                    gene_weights={temp_key: weights_filtered},
                    score_key=temp_key,
                    smoothing=True,
                    correct_spatial_covariates=True
                )

                temp_col = f"{temp_key}_score"
                if temp_col not in temp.obs.columns:
                    continue

                temp_scores = temp.obs[temp_col].to_numpy()
                moran_after = Moran(temp_scores, w).I

                leaveout_rows.append({
                    "dataset": sample_info["dataset"],
                    "sample": sample_info["sample"],
                    "signature_id": sig_id,
                    "drop_top_n": drop_n,
                    "corr_with_original": float(spearmanr(original, temp_scores).statistic),
                    "morans_I_after_drop": float(moran_after),
                    "delta_morans_I": float(moran_after - original_moran),
                })

        # no write-back needed
        del adata

    pd.DataFrame(contrib_rows).to_csv(TABLES_DIR / "top_gene_contribution_per_sample.csv", index=False)
    pd.DataFrame(leaveout_rows).to_csv(TABLES_DIR / "leave_top_genes_out_summary.csv", index=False)


# =============================================================================
# STEP 7: VOLCANO PLOT
# =============================================================================

def run_volcano_plot() -> None:
    """Plot the source SNAI1-2R vs SNAI1 contrast and highlight the 109-gene signature."""
    source = load_signature_source().copy()
    signature_path = SIGNATURE_DIR / "snai1_acetylation_signature_short.csv"
    if not signature_path.exists():
        signature_path = SIGNATURE_DIR / "snai1_acetylation_signature_full.csv"
    selected = pd.read_csv(signature_path, sep=";")
    selected_genes = set(_normalize_gene_symbols(selected[GENE_COL]).dropna())

    source["in_production_109_signature"] = source[GENE_COL].isin(selected_genes)
    source["volcano_group"] = "Other genes"
    source.loc[
        source["in_production_109_signature"] & (source[FC_COL] < 0),
        "volcano_group",
    ] = "109 signature: acetylation-activated"
    source.loc[
        source["in_production_109_signature"] & (source[FC_COL] > 0),
        "volcano_group",
    ] = "109 signature: acetylation-suppressed"

    finite_padj = source["padj"].replace(0, np.nan).dropna()
    min_positive = float(finite_padj.min()) if len(finite_padj) else 1e-300
    source["padj_for_plot"] = source["padj"].replace(0, min_positive / 10).clip(lower=min_positive / 10)
    source["neg_log10_padj"] = -np.log10(source["padj_for_plot"])
    source.to_csv(TABLES_DIR / "snai1_2r_vs_snai1_volcano_source_with_signature109.csv", index=False)

    palette = {
        "Other genes": "#B8B8B8",
        "109 signature: acetylation-activated": "#B2182B",
        "109 signature: acetylation-suppressed": "#2166AC",
    }
    order = [
        "Other genes",
        "109 signature: acetylation-activated",
        "109 signature: acetylation-suppressed",
    ]

    plt.figure(figsize=(7.2, 6.2))
    for group in order:
        subset = source[source["volcano_group"] == group]
        plt.scatter(
            subset[FC_COL],
            subset["neg_log10_padj"],
            s=26 if group != "Other genes" else 8,
            c=palette[group],
            alpha=0.9 if group != "Other genes" else 0.25,
            edgecolors="none",
            label=f"{group} (n={len(subset)})",
            rasterized=group == "Other genes",
        )

    plt.axvline(-LOG2FC_THRESHOLD, color="#555555", linewidth=1, linestyle="--")
    plt.axvline(LOG2FC_THRESHOLD, color="#555555", linewidth=1, linestyle="--")
    plt.axhline(-np.log10(PADJ_THRESHOLD), color="#555555", linewidth=1, linestyle="--")
    plt.xlabel("log2FC, SNAI1-2R vs SNAI1")
    plt.ylabel("-log10 adjusted p-value")
    plt.title("Source bulk contrast for the 109-gene SNAI1-ac signature")
    plt.legend(frameon=False, loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "snai1_2r_vs_snai1_volcano_signature109.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "snai1_2r_vs_snai1_volcano_signature109.pdf", bbox_inches="tight")
    plt.close()


# =============================================================================
# STEP 8: PLOTS
# =============================================================================

def run_plots() -> None:
    sns.set_theme(style="whitegrid")

    det_path = TABLES_DIR / "signature_detection_summary_per_sample.csv"
    if det_path.exists():
        df = pd.read_csv(det_path)

        pivot = df.pivot(index="sample", columns="signature_id", values="pct_detected")
        plt.figure(figsize=(8, max(4, len(pivot) * 0.25)))
        sns.heatmap(pivot, cmap="viridis", vmin=0, vmax=1)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "signature_detection_heatmap.png", dpi=200, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="signature_id", y="pct_detected")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "signature_detection_boxplot.png", dpi=200, bbox_inches="tight")
        plt.close()

    corr_path = TABLES_DIR / "signature_comparison_spotwise_correlations.csv"
    if corr_path.exists():
        df = pd.read_csv(corr_path)

        summary = df.groupby(["sig_a", "sig_b"], as_index=False)["spearman_r"].mean()
        ids = sorted(set(summary["sig_a"]) | set(summary["sig_b"]))

        mat = pd.DataFrame(np.nan, index=ids, columns=ids)
        np.fill_diagonal(mat.values, 1.0)

        for _, row in summary.iterrows():
            mat.loc[row["sig_a"], row["sig_b"]] = row["spearman_r"]
            mat.loc[row["sig_b"], row["sig_a"]] = row["spearman_r"]

        plt.figure(figsize=(8, 6))
        sns.heatmap(mat, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "signature_correlation_heatmap.png", dpi=200, bbox_inches="tight")
        plt.close()

    spatial_path = TABLES_DIR / "spatial_autocorrelation_per_sample.csv"
    if spatial_path.exists():
        df = pd.read_csv(spatial_path)

        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="signature_id", y="morans_I")
        plt.axhline(0, color="black", linestyle="--", linewidth=1)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "moransI_by_signature.png", dpi=200, bbox_inches="tight")
        plt.close()

    smooth_path = TABLES_DIR / "neighborhood_smoothness_per_sample.csv"
    if smooth_path.exists():
        df = pd.read_csv(smooth_path)

        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="signature_id", y="neighbor_self_corr")
        plt.axhline(0, color="black", linestyle="--", linewidth=1)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "neighborhood_smoothness_by_signature.png", dpi=200, bbox_inches="tight")
        plt.close()

    null_path = TABLES_DIR / "null_signature_comparison.csv"
    if null_path.exists():
        df = pd.read_csv(null_path)

        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="signature_id", y="real_minus_null_morans_I")
        plt.axhline(0, color="black", linestyle="--", linewidth=1)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "real_vs_null_moransI.png", dpi=200, bbox_inches="tight")
        plt.close()

    top_path = TABLES_DIR / "top_gene_contribution_per_sample.csv"
    if top_path.exists():
        df = pd.read_csv(top_path)

        melted = df.melt(
            id_vars=["dataset", "sample", "signature_id"],
            value_vars=["top_1_frac", "top_5_frac", "top_10_frac", "top_20_frac"],
            var_name="top_n",
            value_name="fraction",
        )

        plt.figure(figsize=(8, 5))
        sns.lineplot(data=melted, x="top_n", y="fraction", hue="signature_id", estimator="mean", errorbar=None, marker="o")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "top_gene_contribution_curve.png", dpi=200, bbox_inches="tight")
        plt.close()

    leaveout_path = TABLES_DIR / "leave_top_genes_out_summary.csv"
    if leaveout_path.exists():
        df = pd.read_csv(leaveout_path)

        plt.figure(figsize=(8, 5))
        sns.lineplot(data=df, x="drop_top_n", y="corr_with_original", hue="signature_id", estimator="mean", errorbar=None, marker="o")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "drop_top_genes_stability.png", dpi=200, bbox_inches="tight")
        plt.close()

    enrichmap_null_path = TABLES_DIR / "enrichmap_null_comparison.csv"
    if enrichmap_null_path.exists():
        run_enrichmap_null_plots()


# =============================================================================
# ORCHESTRATION
# =============================================================================

def run_steps(
    step: str,
    dataset_filter: Optional[str],
    null_iterations: int,
    excluded_datasets: Optional[List[str]] = None,
) -> None:
    ensure_dirs()
    if step == "cleanup_current":
        cleanup_current_outputs()
        return
    if step == "current_robustness":
        cleanup_current_outputs()
        excluded = list(excluded_datasets or [])
        if "stur_2021" not in excluded:
            excluded.append("stur_2021")
        df = load_signature_source()
        signatures = build_signature_variants(df)
        samples = exclude_samples(iter_samples(dataset_filter), excluded)
        run_detection(signatures, samples)
        run_scoring(signatures, samples)
        run_comparison(signatures, samples)
        run_spatial(signatures, samples)
        run_null_model(signatures, samples)
        run_enrichmap_matched_null(
            signatures,
            samples,
            n_iterations=null_iterations,
            background_universe="detected_protein_coding",
            weighting="uniform",
            output_prefix="enrichmap_null",
            exclude_datasets=excluded,
            smoothing=True,
            correct_spatial_covariates=True,
        )
        run_enrichmap_null_sensitivity(
            signatures,
            samples,
            n_iterations=null_iterations,
        )
        run_enrichmap_unsmoothed_sensitivity(
            signatures,
            samples,
            n_iterations=null_iterations,
        )
        run_topgene_audit(signatures, samples)
        run_volcano_plot()
        run_plots()
        run_enrichmap_null_plots()
        return
    if step == "finish_current_robustness":
        excluded = list(excluded_datasets or [])
        if "stur_2021" not in excluded:
            excluded.append("stur_2021")
        df = load_signature_source(write_diagnostics=False)
        signatures = build_signature_variants(df, write_sidecars=False)
        samples = exclude_samples(iter_samples(dataset_filter), excluded)
        run_detection(signatures, samples)
        run_scoring(signatures, samples)
        run_comparison(signatures, samples)
        run_spatial(signatures, samples)
        run_null_model(signatures, samples)
        run_enrichmap_matched_null(
            signatures,
            samples,
            n_iterations=null_iterations,
            background_universe="detected_protein_coding",
            weighting="uniform",
            output_prefix="enrichmap_null",
            exclude_datasets=excluded,
            smoothing=True,
            correct_spatial_covariates=True,
        )
        run_topgene_audit(signatures, samples)
        run_plots()
        run_enrichmap_null_plots()
        return

    df = load_signature_source()
    signatures = build_signature_variants(df)
    samples = exclude_samples(iter_samples(dataset_filter), excluded_datasets)

    if step == "weights":
        return
    if step == "detection":
        run_detection(signatures, samples)
        return
    if step == "score":
        run_scoring(signatures, samples)
        return
    if step == "compare":
        run_comparison(signatures, samples)
        return
    if step == "spatial":
        run_spatial(signatures, samples)
        return
    if step == "null":
        run_null_model(signatures, samples)
        return
    if step == "enrichmap_null_benchmark":
        run_enrichmap_matched_null(
            signatures,
            samples,
            n_iterations=null_iterations,
            benchmark=True,
        )
        return
    if step == "enrichmap_null":
        run_enrichmap_matched_null(
            signatures,
            samples,
            n_iterations=null_iterations,
        )
        run_enrichmap_null_plots()
        return
    if step == "enrichmap_null_sensitivity":
        run_enrichmap_null_sensitivity(
            signatures,
            samples,
            n_iterations=null_iterations,
        )
        return
    if step == "enrichmap_unsmoothed_sensitivity":
        run_enrichmap_unsmoothed_sensitivity(
            signatures,
            samples,
            n_iterations=null_iterations,
        )
        return
    if step == "topgenes":
        run_topgene_audit(signatures, samples)
        return
    if step == "volcano":
        run_volcano_plot()
        return
    if step == "plots":
        run_plots()
        return
    if step == "signature_robustness":
        run_detection(signatures, samples)
        run_scoring(signatures, samples)
        run_comparison(signatures, samples)
        run_spatial(signatures, samples)
        run_null_model(signatures, samples)
        run_topgene_audit(signatures, samples)
        run_volcano_plot()
        run_plots()
        return
    if step == "all":
        run_detection(signatures, samples)
        run_scoring(signatures, samples)
        run_comparison(signatures, samples)
        run_spatial(signatures, samples)
        run_null_model(signatures, samples)
        run_enrichmap_matched_null(
            signatures,
            samples,
            n_iterations=null_iterations,
        )
        run_topgene_audit(signatures, samples)
        run_plots()
        return

    raise ValueError(f"Unknown step: {step}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SNAI1-ac signature robustness audit.")
    parser.add_argument("dataset", nargs="?", help="Optional dataset id, e.g. visium/denisenko_2022")
    parser.add_argument("--step", choices=STEP_ORDER, default="all")
    parser.add_argument("--all", action="store_true", help="Run all steps across the main cohort.")
    parser.add_argument(
        "--exclude-dataset",
        action="append",
        default=[],
        help="Exclude datasets containing this text, e.g. --exclude-dataset stur_2021.",
    )
    parser.add_argument(
        "--null-iterations",
        type=int,
        default=ENRICHMAP_NULL_ITERATIONS,
        help="Random EnrichMap null iterations per sample. Benchmark mode caps this at 5.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_filter = None if args.all or not args.dataset else args.dataset
    run_steps("all" if args.all else args.step, dataset_filter, args.null_iterations, args.exclude_dataset)


if __name__ == "__main__":
    main()
