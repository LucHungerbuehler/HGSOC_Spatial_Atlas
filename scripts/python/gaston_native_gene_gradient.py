"""
Run GASTON-native gene-gradient analysis after whole-tissue GASTON training.

This script deliberately starts from the existing post-training alignment files:
selected whole-tissue domains, malignant-oriented isodepth, and spot alignment.
It does not retrain GASTON and it does not use score/gene correlations as a
substitute for the package's own piecewise Poisson gene-gradient machinery.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import fisher_exact


DATA_ROOT = Path(r"D:\HGSOC_Spatial_Atlas")
GASTON_ROOT = DATA_ROOT / "05_analysis_ready" / "GASTON_method_v1"
GASTON_SRC = DATA_ROOT / "git_clones" / "GASTON" / "src"
SIGNATURE_ROOT = DATA_ROOT / "05_analysis_ready" / "Signature"
CSIDE_ROOT = DATA_ROOT / "05_analysis_ready" / "S2e_CSIDE_CellTypeSpecific_DE_Audit"

DEFAULT_ALIGNMENT_MANIFEST = GASTON_ROOT / "04_isodepth_score_alignment" / "sample_alignment_manifest.csv"
DEFAULT_OUT_DIR = GASTON_ROOT / "07_gradient_review" / "02_gaston_native_gradient_identity"
DEFAULT_HALLMARK_JSON = SIGNATURE_ROOT / "hallmark_gene_sets.json"
DEFAULT_KEGG_JSON = CSIDE_ROOT / "00_manifest" / "kegg_legacy_gene_sets_from_msigdbr.json"
DEFAULT_ROBUST_CORE = (
    CSIDE_ROOT
    / "07_report_ready_packaging"
    / "tables"
    / "cside_robust_core_73_gene_celltype_associations.csv"
)
DEFAULT_SNAI1_WEIGHTS = SIGNATURE_ROOT / "snai1_ac_weights.json"

DEFAULT_UMI_THRESHOLD = 1000
DEFAULT_EXCLUDE_PREFIX = ("MT-", "RPL", "RPS")
DEFAULT_ISODEPTH_MULT = 0.01
DEFAULT_SLOPE_PVALUE_T = 0.1
DEFAULT_CONT_Q = 0.8
DEFAULT_DISCONT_Q = 0.95
DEFAULT_NUM_BINS = 15
DEFAULT_MIN_SET_SIZE = 5
DEFAULT_MIN_CLASS_SIZE = 3
DEFAULT_MAX_PACKAGE_STYLE_GENES = 8
SNAI1AC_SCORE_VARIANTS = [
    ("snai1ac_em_smooth_corrected", "SNAI1-ac smooth+GAM"),
    ("snai1ac_em_unsmoothed_corrected", "SNAI1-ac unsmoothed+GAM"),
    ("snai1ac_em_unsmoothed_uncorrected", "SNAI1-ac unsmoothed no GAM"),
]


@dataclass(frozen=True)
class SampleResult:
    dataset: str
    sample: str
    layer: str
    feature_method: str
    analysis_tier: str
    status: str
    elapsed_sec: float
    n_spots: int = 0
    n_domains: int = 0
    n_genes_total: int = 0
    n_genes_kept: int = 0
    n_continuous_events: int = 0
    n_continuous_genes: int = 0
    n_discontinuous_events: int = 0
    n_discontinuous_genes: int = 0
    n_identity_classes: int = 0
    n_ora_rows: int = 0
    warnings: str = ""
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GASTON-native piecewise Poisson gene-gradient analysis."
    )
    parser.add_argument("--alignment-manifest", type=Path, default=DEFAULT_ALIGNMENT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--samples", nargs="*", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--feature-method", default=None)
    parser.add_argument("--analysis-tier", default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--counts-layer",
        default="counts",
        help="h5ad layer to use for raw counts. Use 'X' to force adata.X.",
    )
    parser.add_argument("--umi-threshold", type=int, default=DEFAULT_UMI_THRESHOLD)
    parser.add_argument("--isodepth-mult-factor", type=float, default=DEFAULT_ISODEPTH_MULT)
    parser.add_argument("--slope-pvalue-t", type=float, default=DEFAULT_SLOPE_PVALUE_T)
    parser.add_argument("--continuous-q", type=float, default=DEFAULT_CONT_Q)
    parser.add_argument("--discontinuous-q", type=float, default=DEFAULT_DISCONT_Q)
    parser.add_argument("--num-bins", type=int, default=DEFAULT_NUM_BINS)
    parser.add_argument(
        "--max-package-style-genes",
        type=int,
        default=DEFAULT_MAX_PACKAGE_STYLE_GENES,
        help="Maximum class-leader genes per sample for package-native raw/function/pwlinear panels.",
    )
    parser.add_argument("--hallmark-json", type=Path, default=DEFAULT_HALLMARK_JSON)
    parser.add_argument("--kegg-json", type=Path, default=DEFAULT_KEGG_JSON)
    parser.add_argument("--robust-core-csv", type=Path, default=DEFAULT_ROBUST_CORE)
    parser.add_argument("--snai1-weights-json", type=Path, default=DEFAULT_SNAI1_WEIGHTS)
    return parser.parse_args()


def import_runtime_modules() -> Any:
    if str(GASTON_SRC) not in sys.path:
        sys.path.insert(0, str(GASTON_SRC))
    try:
        import anndata as ad
        from gaston import (
            binning_and_plotting,
            cluster_plotting,
            filter_genes,
            segmented_fit,
            spatial_gene_classification,
        )
    except Exception as exc:
        raise RuntimeError(
            "Required modules could not be imported. Run in gaston_env and check the local GASTON clone."
        ) from exc
    return ad, segmented_fit, binning_and_plotting, spatial_gene_classification, filter_genes, cluster_plotting


def ensure_dirs(out_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": out_dir,
        "tables": out_dir / "tables",
        "figures": out_dir / "figures",
        "objects": out_dir / "objects",
        "logs": out_dir / "logs",
        "summaries": out_dir / "summaries",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def stem(row: pd.Series) -> str:
    return "__".join(
        [
            str(row["dataset"]),
            str(row["sample"]),
            str(row["layer"]),
            str(row["feature_method"]),
        ]
    )


def clean_gene(value: Any) -> str:
    return str(value).strip().upper()


def load_gene_sets(path: Path) -> dict[str, set[str]]:
    data = read_json(path)
    return {str(k): {clean_gene(g) for g in v if str(g).strip()} for k, v in data.items()}


def load_gene_annotations(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, float]]:
    robust = pd.read_csv(args.robust_core_csv)
    if "gene" not in robust.columns:
        raise ValueError(f"Robust-core table lacks a gene column: {args.robust_core_csv}")

    robust["gene_upper"] = robust["gene"].map(clean_gene)
    robust_by_gene: dict[str, dict[str, Any]] = {}
    for gene, sub in robust.groupby("gene_upper", sort=True):
        robust_by_gene[gene] = {
            "is_robust_core_gene": True,
            "robust_core_gene": ";".join(sorted(set(sub["gene"].astype(str)))),
            "robust_core_cell_types": ";".join(sorted(set(sub.get("cell_type", pd.Series(dtype=str)).astype(str)))),
            "robust_core_directions": ";".join(sorted(set(sub.get("direction", pd.Series(dtype=str)).astype(str)))),
        }

    weights_raw = read_json(args.snai1_weights_json)
    weights = {clean_gene(gene): float(weight) for gene, weight in weights_raw.items()}
    return robust, robust_by_gene, weights


def bh_adjust(values: np.ndarray) -> np.ndarray:
    pvals = np.asarray(values, dtype=float)
    qvals = np.full(pvals.shape, np.nan, dtype=float)
    mask = np.isfinite(pvals)
    if not mask.any():
        return qvals
    p = pvals[mask]
    order = np.argsort(p)
    ranked = p[order]
    m = float(len(ranked))
    raw = ranked * m / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(raw[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    qvals[mask] = restored
    return qvals


def order_labels_by_isodepth(
    labels: np.ndarray,
    isodepth: np.ndarray,
) -> tuple[np.ndarray, dict[int, int], dict[int, int], pd.DataFrame]:
    """Relabel arbitrary GASTON domain labels into low-to-high isodepth segments."""
    labels = np.asarray(labels, dtype=int)
    isodepth = np.asarray(isodepth, dtype=float)
    rows = []
    for original_label in sorted(int(x) for x in np.unique(labels)):
        values = isodepth[labels == original_label]
        rows.append(
            {
                "original_domain_label": original_label,
                "n_spots": int(len(values)),
                "isodepth_min": float(np.min(values)),
                "isodepth_q05": float(np.quantile(values, 0.05)),
                "isodepth_median": float(np.median(values)),
                "isodepth_mean": float(np.mean(values)),
                "isodepth_q95": float(np.quantile(values, 0.95)),
                "isodepth_max": float(np.max(values)),
            }
        )
    stats = pd.DataFrame(rows).sort_values("isodepth_median").reset_index(drop=True)
    original_to_segment = {
        int(row.original_domain_label): int(i)
        for i, row in enumerate(stats.itertuples(index=False))
    }
    segment_to_original = {segment: original for original, segment in original_to_segment.items()}
    ordered = np.array([original_to_segment[int(x)] for x in labels], dtype=int)
    stats.insert(0, "gradient_segment", np.arange(len(stats), dtype=int))
    return ordered, original_to_segment, segment_to_original, stats


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def read_counts_matrix(adata: Any, spot_ids: pd.Series, counts_layer: str) -> tuple[Any, np.ndarray, str, list[str]]:
    missing = [spot for spot in spot_ids.astype(str) if spot not in adata.obs_names]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{len(missing)} spot IDs from alignment table are absent from h5ad obs_names: {preview}")
    adata_spots = adata[spot_ids.astype(str).tolist(), :].copy()
    if counts_layer == "X":
        matrix = adata_spots.X
        counts_source = "X"
    elif counts_layer in adata_spots.layers:
        matrix = adata_spots.layers[counts_layer]
        counts_source = f"layer:{counts_layer}"
    else:
        matrix = adata_spots.X
        counts_source = "X"
        warnings = [f"requested_counts_layer_missing={counts_layer};used_X"]
        if sparse.issparse(matrix):
            counts = matrix.toarray()
        else:
            counts = np.asarray(matrix)
        counts = np.asarray(counts, dtype=np.float64)
        gene_labels = np.asarray(adata_spots.var_names.astype(str))
        return counts, gene_labels, counts_source, warnings
    if sparse.issparse(matrix):
        counts = matrix.toarray()
    else:
        counts = np.asarray(matrix)
    counts = np.asarray(counts, dtype=np.float64)
    gene_labels = np.asarray(adata_spots.var_names.astype(str))
    warnings: list[str] = []
    if np.any(~np.isfinite(counts)):
        raise ValueError("Counts matrix contains non-finite values.")
    if np.any(counts < 0):
        raise ValueError("Counts matrix contains negative values.")
    if counts.size:
        max_fractional = float(np.max(np.abs(counts - np.rint(counts))))
        if max_fractional > 1e-6:
            warnings.append(f"counts_not_integer_like_max_fractional={max_fractional:.4g}")
    return counts, gene_labels, counts_source, warnings


def build_gene_domain_table(
    row: pd.Series,
    gene_labels_idx: np.ndarray,
    counts_idx: np.ndarray,
    labels: np.ndarray,
    segment_to_original_label: dict[int, int],
    slope_mat: np.ndarray,
    intercept_mat: np.ndarray,
    pv_mat: np.ndarray,
    cont_genes: dict[str, list[int]],
    robust_by_gene: dict[str, dict[str, Any]],
    snai1_weights: dict[str, float],
    continuous_q: float,
) -> pd.DataFrame:
    n_domains = slope_mat.shape[1]
    total_umi = np.sum(counts_idx, axis=0)
    nonzero_all = np.count_nonzero(counts_idx, axis=0)
    rows: list[pd.DataFrame] = []
    slope_thresholds = np.quantile(np.abs(slope_mat), continuous_q, axis=0)
    pv_q_by_domain = np.column_stack([bh_adjust(pv_mat[:, domain]) for domain in range(n_domains)])

    for domain in range(n_domains):
        domain_mask = labels == domain
        nonzero_domain = np.count_nonzero(counts_idx[domain_mask, :], axis=0)
        genes_upper = [clean_gene(gene) for gene in gene_labels_idx]
        table = pd.DataFrame(
            {
                "dataset": row["dataset"],
                "sample": row["sample"],
                "layer": row["layer"],
                "feature_method": row["feature_method"],
                "analysis_tier": row["analysis_tier"],
                "gene": gene_labels_idx,
                "gene_upper": genes_upper,
                "gradient_segment": domain,
                "original_domain_label": segment_to_original_label.get(domain, domain),
                "domain": domain,
                "slope": slope_mat[:, domain],
                "abs_slope": np.abs(slope_mat[:, domain]),
                "intercept": intercept_mat[:, domain],
                "p_value_slope_llr": pv_mat[:, domain],
                "q_value_slope_llr_bh_domain": pv_q_by_domain[:, domain],
                "slope_quantile_threshold": slope_thresholds[domain],
                "is_continuous_q": [
                    bool(str(gene) in cont_genes and domain in cont_genes[str(gene)])
                    for gene in gene_labels_idx
                ],
                "slope_sign": np.where(slope_mat[:, domain] > 0, "positive", np.where(slope_mat[:, domain] < 0, "negative", "zero")),
                "total_umi_aligned_spots": total_umi,
                "nonzero_spots_all": nonzero_all,
                "nonzero_spots_domain": nonzero_domain,
                "n_spots_domain": int(domain_mask.sum()),
            }
        )
        rows.append(table)

    result = pd.concat(rows, ignore_index=True)
    result = annotate_gene_table(result, robust_by_gene, snai1_weights)
    return result


def build_gene_boundary_table(
    row: pd.Series,
    gene_labels_idx: np.ndarray,
    discont_mat: np.ndarray,
    segment_to_original_label: dict[int, int],
    discont_genes: dict[str, list[int]],
    robust_by_gene: dict[str, dict[str, Any]],
    snai1_weights: dict[str, float],
    discontinuous_q: float,
) -> pd.DataFrame:
    n_boundaries = discont_mat.shape[1]
    if n_boundaries == 0:
        return pd.DataFrame()
    thresholds = np.quantile(np.abs(discont_mat), discontinuous_q, axis=0)
    rows: list[pd.DataFrame] = []
    for boundary in range(n_boundaries):
        table = pd.DataFrame(
            {
                "dataset": row["dataset"],
                "sample": row["sample"],
                "layer": row["layer"],
                "feature_method": row["feature_method"],
                "analysis_tier": row["analysis_tier"],
                "gene": gene_labels_idx,
                "gene_upper": [clean_gene(gene) for gene in gene_labels_idx],
                "boundary": boundary,
                "gradient_segment_left": boundary,
                "gradient_segment_right": boundary + 1,
                "original_domain_left": segment_to_original_label.get(boundary, boundary),
                "original_domain_right": segment_to_original_label.get(boundary + 1, boundary + 1),
                "domain_left": boundary,
                "domain_right": boundary + 1,
                "discontinuity": discont_mat[:, boundary],
                "abs_discontinuity": np.abs(discont_mat[:, boundary]),
                "discontinuity_quantile_threshold": thresholds[boundary],
                "is_discontinuous_q": [
                    bool(str(gene) in discont_genes and boundary in discont_genes[str(gene)])
                    for gene in gene_labels_idx
                ],
                "discontinuity_sign": np.where(
                    discont_mat[:, boundary] > 0,
                    "positive",
                    np.where(discont_mat[:, boundary] < 0, "negative", "zero"),
                ),
            }
        )
        rows.append(table)
    result = pd.concat(rows, ignore_index=True)
    result = annotate_gene_table(result, robust_by_gene, snai1_weights)
    return result


def annotate_gene_table(
    table: pd.DataFrame,
    robust_by_gene: dict[str, dict[str, Any]],
    snai1_weights: dict[str, float],
) -> pd.DataFrame:
    table = table.copy()
    table["is_robust_core_gene"] = table["gene_upper"].isin(robust_by_gene)
    table["robust_core_gene"] = table["gene_upper"].map(
        lambda gene: robust_by_gene.get(gene, {}).get("robust_core_gene", "")
    )
    table["robust_core_cell_types"] = table["gene_upper"].map(
        lambda gene: robust_by_gene.get(gene, {}).get("robust_core_cell_types", "")
    )
    table["robust_core_directions"] = table["gene_upper"].map(
        lambda gene: robust_by_gene.get(gene, {}).get("robust_core_directions", "")
    )
    table["is_snai1ac_signature_gene"] = table["gene_upper"].isin(snai1_weights)
    table["snai1ac_signature_weight"] = table["gene_upper"].map(snai1_weights).astype(float)
    table["snai1ac_signature_direction"] = np.where(
        table["snai1ac_signature_weight"] > 0,
        "positive",
        np.where(table["snai1ac_signature_weight"] < 0, "negative", ""),
    )
    return table


def build_gene_pattern_table(domain_table: pd.DataFrame, boundary_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    boundary_lookup = boundary_table if not boundary_table.empty else pd.DataFrame()
    for gene, sub in domain_table.groupby("gene", sort=True):
        cont = sub[sub["is_continuous_q"]].copy()
        disc = (
            boundary_lookup[(boundary_lookup["gene"] == gene) & (boundary_lookup["is_discontinuous_q"])].copy()
            if not boundary_lookup.empty
            else pd.DataFrame()
        )
        first = sub.iloc[0]
        rows.append(
            {
                "dataset": first["dataset"],
                "sample": first["sample"],
                "layer": first["layer"],
                "feature_method": first["feature_method"],
                "analysis_tier": first["analysis_tier"],
                "gene": gene,
                "gene_upper": first["gene_upper"],
                "n_continuous_segments": int(len(cont)),
                "continuous_segments": ";".join(str(int(x)) for x in cont["gradient_segment"].tolist()),
                "continuous_segment_signs": ";".join(
                    f"{int(r.gradient_segment)}:{r.slope_sign}" for r in cont.itertuples(index=False)
                ),
                "max_abs_slope": float(sub["abs_slope"].max()),
                "gradient_segment_of_max_abs_slope": int(sub.loc[sub["abs_slope"].idxmax(), "gradient_segment"]),
                "original_domain_of_max_abs_slope": int(sub.loc[sub["abs_slope"].idxmax(), "original_domain_label"]),
                "n_discontinuous_boundaries": int(len(disc)),
                "discontinuous_boundaries": ";".join(str(int(x)) for x in disc["boundary"].tolist()) if not disc.empty else "",
                "discontinuous_boundary_signs": ";".join(
                    f"{int(r.boundary)}:{r.discontinuity_sign}" for r in disc.itertuples(index=False)
                )
                if not disc.empty
                else "",
                "max_abs_discontinuity": float(boundary_lookup[boundary_lookup["gene"] == gene]["abs_discontinuity"].max())
                if not boundary_lookup.empty
                else 0.0,
                "n_continuous_domains": int(len(cont)),
                "continuous_domains": ";".join(str(int(x)) for x in cont["original_domain_label"].tolist()),
                "continuous_domain_signs": ";".join(
                    f"{int(r.original_domain_label)}:{r.slope_sign}" for r in cont.itertuples(index=False)
                ),
                "is_robust_core_gene": bool(first["is_robust_core_gene"]),
                "robust_core_cell_types": first["robust_core_cell_types"],
                "robust_core_directions": first["robust_core_directions"],
                "is_snai1ac_signature_gene": bool(first["is_snai1ac_signature_gene"]),
                "snai1ac_signature_weight": first["snai1ac_signature_weight"],
                "snai1ac_signature_direction": first["snai1ac_signature_direction"],
            }
        )
    return pd.DataFrame(rows)


def build_continuous_class_table(domain_table: pd.DataFrame) -> pd.DataFrame:
    table = domain_table[domain_table["is_continuous_q"]].copy()
    if table.empty:
        return table
    table["class_type"] = "continuous_slope"
    table["direction"] = table["slope_sign"]
    table["gradient_segment"] = table["gradient_segment"].astype(int)
    table["class_id"] = (
        "continuous_segment_"
        + table["gradient_segment"].astype(str)
        + "_"
        + table["direction"].astype(str)
    )
    table["effect_value"] = table["slope"]
    table["abs_effect_value"] = table["abs_slope"]
    table = table.sort_values(["class_id", "abs_effect_value", "gene"], ascending=[True, False, True])
    table["rank_within_class"] = table.groupby("class_id").cumcount() + 1
    keep = [
        "source_stem",
        "dataset",
        "sample",
        "layer",
        "feature_method",
        "analysis_tier",
        "class_type",
        "class_id",
        "gradient_segment",
        "original_domain_label",
        "domain",
        "direction",
        "rank_within_class",
        "gene",
        "gene_upper",
        "effect_value",
        "abs_effect_value",
        "p_value_slope_llr",
        "q_value_slope_llr_bh_domain",
        "slope_quantile_threshold",
        "total_umi_aligned_spots",
        "nonzero_spots_all",
        "nonzero_spots_domain",
        "n_spots_domain",
        "is_robust_core_gene",
        "robust_core_cell_types",
        "robust_core_directions",
        "is_snai1ac_signature_gene",
        "snai1ac_signature_weight",
        "snai1ac_signature_direction",
    ]
    return table[[col for col in keep if col in table.columns]].reset_index(drop=True)


def build_discontinuous_class_table(boundary_table: pd.DataFrame) -> pd.DataFrame:
    if boundary_table.empty:
        return boundary_table
    table = boundary_table[boundary_table["is_discontinuous_q"]].copy()
    if table.empty:
        return table
    table["class_type"] = "boundary_discontinuity"
    table["direction"] = table["discontinuity_sign"]
    table["class_id"] = (
        "discontinuous_boundary_"
        + table["boundary"].astype(int).astype(str)
        + "_"
        + table["direction"].astype(str)
    )
    table["effect_value"] = table["discontinuity"]
    table["abs_effect_value"] = table["abs_discontinuity"]
    table = table.sort_values(["class_id", "abs_effect_value", "gene"], ascending=[True, False, True])
    table["rank_within_class"] = table.groupby("class_id").cumcount() + 1
    keep = [
        "source_stem",
        "dataset",
        "sample",
        "layer",
        "feature_method",
        "analysis_tier",
        "class_type",
        "class_id",
        "boundary",
        "gradient_segment_left",
        "gradient_segment_right",
        "original_domain_left",
        "original_domain_right",
        "domain_left",
        "domain_right",
        "direction",
        "rank_within_class",
        "gene",
        "gene_upper",
        "effect_value",
        "abs_effect_value",
        "discontinuity_quantile_threshold",
        "is_robust_core_gene",
        "robust_core_cell_types",
        "robust_core_directions",
        "is_snai1ac_signature_gene",
        "snai1ac_signature_weight",
        "snai1ac_signature_direction",
    ]
    return table[[col for col in keep if col in table.columns]].reset_index(drop=True)


def class_gene_sets_from_tables(
    continuous_class_table: pd.DataFrame,
    discontinuous_class_table: pd.DataFrame,
) -> dict[str, set[str]]:
    classes: dict[str, set[str]] = {}
    for table in (continuous_class_table, discontinuous_class_table):
        if table.empty:
            continue
        for class_id, sub in table.groupby("class_id", sort=True):
            classes[str(class_id)] = set(sub["gene_upper"].astype(str))
    return classes


def run_ora(
    classes: dict[str, set[str]],
    gene_sets_by_collection: dict[str, dict[str, set[str]]],
    universe_genes: set[str],
    min_set_size: int = DEFAULT_MIN_SET_SIZE,
    min_class_size: int = DEFAULT_MIN_CLASS_SIZE,
) -> pd.DataFrame:
    rows = []
    universe = set(universe_genes)
    n_universe = len(universe)
    for class_id, genes in classes.items():
        class_genes = set(genes) & universe
        if len(class_genes) < min_class_size:
            continue
        for collection, gene_sets in gene_sets_by_collection.items():
            for pathway, pathway_genes_raw in gene_sets.items():
                pathway_genes = set(pathway_genes_raw) & universe
                if len(pathway_genes) < min_set_size:
                    continue
                overlap = class_genes & pathway_genes
                a = len(overlap)
                b = len(class_genes) - a
                c = len(pathway_genes) - a
                d = n_universe - a - b - c
                if d < 0:
                    continue
                odds, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
                rows.append(
                    {
                        "class_id": class_id,
                        "collection": collection,
                        "pathway": pathway,
                        "p_value": p_value,
                        "odds_ratio": odds if math.isfinite(odds) else np.inf,
                        "n_overlap": a,
                        "n_class_genes": len(class_genes),
                        "n_pathway_genes_in_universe": len(pathway_genes),
                        "n_universe": n_universe,
                        "overlap_genes": ";".join(sorted(overlap)),
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["q_value_bh"] = bh_adjust(result["p_value"].to_numpy())
    return result.sort_values(["q_value_bh", "p_value", "collection", "class_id", "pathway"]).reset_index(drop=True)


def build_identity_class_summary(
    continuous_class_table: pd.DataFrame,
    discontinuous_class_table: pd.DataFrame,
    ora_table: pd.DataFrame,
    top_n_genes: int = 12,
    top_n_terms: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    combined = pd.concat(
        [continuous_class_table, discontinuous_class_table],
        ignore_index=True,
        sort=False,
    )
    if combined.empty:
        return pd.DataFrame()
    for class_id, sub in combined.groupby("class_id", sort=True):
        first = sub.iloc[0]
        top = sub.sort_values(["rank_within_class", "gene"]).head(top_n_genes)
        class_ora = ora_table[ora_table["class_id"] == class_id].copy() if not ora_table.empty else pd.DataFrame()
        row: dict[str, Any] = {
            "source_stem": first["source_stem"],
            "dataset": first["dataset"],
            "sample": first["sample"],
            "layer": first["layer"],
            "feature_method": first["feature_method"],
            "analysis_tier": first["analysis_tier"],
            "class_type": first["class_type"],
            "class_id": class_id,
            "direction": first["direction"],
            "n_genes": int(len(sub)),
            "top_genes": ";".join(top["gene"].astype(str).tolist()),
            "top_effect_values": ";".join(f"{float(v):.4g}" for v in top["effect_value"].tolist()),
            "n_robust_core_genes": int(sub["is_robust_core_gene"].sum()) if "is_robust_core_gene" in sub else 0,
            "robust_core_genes": ";".join(
                top.loc[top.get("is_robust_core_gene", False).astype(bool), "gene"].astype(str).tolist()
            )
            if "is_robust_core_gene" in top
            else "",
            "n_snai1ac_signature_genes": int(sub["is_snai1ac_signature_gene"].sum())
            if "is_snai1ac_signature_gene" in sub
            else 0,
            "snai1ac_signature_genes": ";".join(
                top.loc[top.get("is_snai1ac_signature_gene", False).astype(bool), "gene"].astype(str).tolist()
            )
            if "is_snai1ac_signature_gene" in top
            else "",
        }
        for field in (
            "gradient_segment",
            "original_domain_label",
            "domain",
            "boundary",
            "gradient_segment_left",
            "gradient_segment_right",
            "original_domain_left",
            "original_domain_right",
            "domain_left",
            "domain_right",
        ):
            if field in first.index and pd.notna(first[field]):
                row[field] = int(first[field])
        for collection in ("hallmark", "kegg_legacy"):
            if class_ora.empty:
                row[f"top_{collection}_terms"] = ""
                row[f"top_{collection}_q_values"] = ""
                continue
            terms = class_ora[class_ora["collection"] == collection].sort_values(["q_value_bh", "p_value"]).head(top_n_terms)
            row[f"top_{collection}_terms"] = ";".join(terms["pathway"].astype(str).tolist())
            row[f"top_{collection}_q_values"] = ";".join(f"{float(v):.3g}" for v in terms["q_value_bh"].tolist())
        rows.append(row)
    return pd.DataFrame(rows)


def save_gene_curve_grid(
    path: Path,
    title: str,
    plot_rows: pd.DataFrame,
    pw_fit_dict: dict[str, Any],
    binning_output: dict[str, Any],
) -> None:
    if plot_rows.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    def short_class_id(value: Any) -> str:
        text = str(value)
        text = text.replace("continuous_segment_", "cont S")
        text = text.replace("continuous_domain_", "cont D")
        text = text.replace("_positive", " pos")
        text = text.replace("_negative", " neg")
        text = text.replace("discontinuous_boundary_", "disc B")
        return text

    def row_target(row: Any) -> tuple[set[int], set[int]]:
        segments: set[int] = set()
        boundaries: set[int] = set()
        if hasattr(row, "gradient_segment") and pd.notna(row.gradient_segment):
            segments.add(int(row.gradient_segment))
        if hasattr(row, "gradient_segment_left") and pd.notna(row.gradient_segment_left):
            segments.add(int(row.gradient_segment_left))
        if hasattr(row, "gradient_segment_right") and pd.notna(row.gradient_segment_right):
            segments.add(int(row.gradient_segment_right))
        if hasattr(row, "boundary") and pd.notna(row.boundary):
            boundaries.add(int(row.boundary))
        return segments, boundaries

    gene_labels_idx = np.asarray(binning_output["gene_labels_idx"]).astype(str)
    unique_binned_isodepths = np.asarray(binning_output["unique_binned_isodepths"], dtype=float)
    binned_labels = np.asarray(binning_output["binned_labels"], dtype=int)
    binned_count = np.asarray(binning_output["binned_count"], dtype=float)
    binned_exposure = np.asarray(binning_output["binned_exposure"], dtype=float)
    slope_mat, intercept_mat, _, _ = pw_fit_dict["all_cell_types"]
    n_domains = slope_mat.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_domains, 3)))

    n = min(len(plot_rows), 12)
    ncols = 3
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows), constrained_layout=True)
    axes = np.asarray(axes).ravel()

    boundary_positions = []
    for i in range(len(binned_labels) - 1):
        if binned_labels[i] != binned_labels[i + 1]:
            boundary_positions.append(0.5 * (unique_binned_isodepths[i] + unique_binned_isodepths[i + 1]))

    for ax, row in zip(axes, plot_rows.head(n).itertuples(index=False)):
        gene = str(row.gene)
        gene_idx = np.where(gene_labels_idx == gene)[0]
        if len(gene_idx) == 0:
            ax.axis("off")
            continue
        g = int(gene_idx[0])
        y = np.log1p((binned_count[:, g] / np.maximum(binned_exposure, 1.0)) * 1e6)
        target_segments, target_boundaries = row_target(row)
        for domain in range(n_domains):
            pts = np.where(binned_labels == domain)[0]
            if len(pts) == 0:
                continue
            is_target = not target_segments or domain in target_segments
            spot_sizes = np.clip(np.sqrt(np.asarray(binning_output["binned_number_spots"])[pts]) * 7.0, 16, 68)
            ax.scatter(
                unique_binned_isodepths[pts],
                y[pts],
                s=spot_sizes,
                color=colors[domain],
                alpha=0.78 if is_target else 0.22,
            )
            fit_y = np.log(1e6) + intercept_mat[g, domain] + slope_mat[g, domain] * unique_binned_isodepths[pts]
            ax.plot(
                unique_binned_isodepths[pts],
                fit_y,
                color="black" if is_target else "0.45",
                linewidth=1.7 if is_target else 0.8,
                alpha=0.9 if is_target else 0.3,
            )
        for boundary_idx, xpos in enumerate(boundary_positions):
            is_target_boundary = not target_boundaries or boundary_idx in target_boundaries
            ax.axvline(
                xpos,
                color="0.2" if is_target_boundary else "0.7",
                linewidth=1.2 if is_target_boundary else 0.7,
                linestyle="--",
                alpha=0.75 if is_target_boundary else 0.35,
            )
        subtitle_parts = [short_class_id(row.class_id), gene]
        if hasattr(row, "effect_value"):
            subtitle_parts.append(f"effect={float(row.effect_value):.3g}")
        ax.set_title(" | ".join(subtitle_parts), fontsize=9)
        ax.tick_params(labelsize=8)
    for ax in axes[n:]:
        ax.axis("off")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors[domain],
            markeredgecolor=colors[domain],
            markersize=7,
            label=f"S{domain}",
        )
        for domain in range(n_domains)
    ]
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=13)
    fig.supxlabel("GASTON isodepth, malignant-oriented")
    fig.supylabel("Binned log1p(CPM)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def class_leaders(class_table: pd.DataFrame, rank_max: int = 1) -> pd.DataFrame:
    if class_table.empty:
        return class_table
    leaders = class_table[class_table["rank_within_class"] <= rank_max].copy()
    return leaders.sort_values(["class_id", "rank_within_class", "gene"]).reset_index(drop=True)


def safe_filename(value: Any) -> str:
    text = str(value)
    keep = []
    for char in text:
        keep.append(char if char.isalnum() or char in ("-", "_", ".") else "_")
    collapsed = "".join(keep).strip("_")
    return collapsed[:120] if collapsed else "value"


def load_training_geometry(
    row: pd.Series,
    sample_stem: str,
    alignment: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, Path | None, Path | None, list[str]]:
    warnings: list[str] = []
    result_path = Path(str(row["result_npz"])) if "result_npz" in row and pd.notna(row["result_npz"]) else None
    model_path: Path | None = None
    training_coords: np.ndarray | None = None
    display_coords: np.ndarray | None = None

    if result_path is not None:
        model_path = result_path.parent.parent / "models" / f"{sample_stem}__best_model.pt"
        if result_path.exists():
            result = np.load(result_path, allow_pickle=True)
            if "S_scaled" in result.files:
                training_coords = np.asarray(result["S_scaled"], dtype=float)
            else:
                warnings.append(f"result_npz_missing_S_scaled={result_path}")
        else:
            warnings.append(f"result_npz_missing={result_path}")

    if {"spatial_x", "spatial_y"}.issubset(alignment.columns):
        display_coords = alignment[["spatial_x", "spatial_y"]].astype(float).to_numpy()
    elif training_coords is not None:
        display_coords = training_coords.copy()
        warnings.append("used_training_coords_for_display_because_alignment_spatial_xy_missing")
    else:
        raise ValueError("Could not find spatial_x/spatial_y in alignment table.")

    if training_coords is None:
        training_coords = display_coords.copy()
        warnings.append("used_display_coords_for_training_gradient_because_S_scaled_missing")

    if training_coords.shape[0] != len(alignment) or display_coords.shape[0] != len(alignment):
        raise ValueError(
            "Training/display coordinate row mismatch: "
            f"training={training_coords.shape[0]} display={display_coords.shape[0]} alignment={len(alignment)}"
        )
    if np.any(~np.isfinite(training_coords)) or np.any(~np.isfinite(display_coords)):
        raise ValueError("Training/display coordinates contain non-finite values.")

    return training_coords, display_coords, result_path, model_path, warnings


def load_gaston_model(model_path: Path | None, warnings: list[str]) -> Any | None:
    if model_path is None:
        warnings.append("gaston_model_path_unavailable")
        return None
    if not model_path.exists():
        warnings.append(f"gaston_model_missing={model_path}")
        return None
    try:
        import torch

        try:
            model = torch.load(model_path, map_location="cpu", weights_only=False)
        except TypeError:
            model = torch.load(model_path, map_location="cpu")
        model.eval()
        return model
    except Exception as exc:
        warnings.append(f"gaston_model_load_failed={type(exc).__name__}:{exc}")
        return None


def save_current_figure(path: Path, dpi: int = 180) -> str:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.gcf().savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close("all")
    return str(path)


def orient_image_axis(ax: Any) -> None:
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")


def save_topology_plots(
    dirs: dict[str, Path],
    sample_stem: str,
    training_coords: np.ndarray,
    labels: np.ndarray,
    isodepth: np.ndarray,
    cluster_plotting: Any,
    model_path: Path | None,
    warnings: list[str],
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs: dict[str, str] = {}
    model = load_gaston_model(model_path, warnings)
    point_size = float(np.clip(18000.0 / max(training_coords.shape[0], 1), 4.0, 14.0))
    n_domains = len(np.unique(labels))
    colors = np.array([plt.cm.tab10(i % 10) for i in range(max(n_domains, 1))])

    try:
        use_streamlines = model is not None and training_coords.shape[0] >= 1000
        if model is not None and not use_streamlines:
            warnings.append("streamlines_skipped_fewer_than_1000_spots")
        cluster_plotting.plot_isodepth(
            isodepth,
            training_coords,
            model,
            figsize=(7, 6),
            contours=True,
            contour_levels=8,
            contour_lw=0.8,
            contour_fs=7,
            colorbar=True,
            s=point_size,
            streamlines=use_streamlines,
            streamlines_lw=0.45,
            arrowsize=1.4,
            cmap="coolwarm",
        )
        orient_image_axis(plt.gca())
        outputs["isodepth_contours_streamlines"] = save_current_figure(
            dirs["figures"] / f"{sample_stem}__isodepth_contours_streamlines.png"
        )
    except Exception as exc:
        warnings.append(f"isodepth_contour_streamline_plot_failed={type(exc).__name__}:{exc}")
        plt.close("all")

    try:
        cluster_plotting.plot_clusters(
            labels,
            training_coords,
            figsize=(7, 6),
            colors=colors,
            s=point_size,
            labels=[f"S{i}" for i in range(n_domains)],
            lgd=True,
            show_boundary=True,
            gaston_isodepth=isodepth,
            boundary_lw=2,
            bbox_to_anchor=(1.05, 1),
        )
        orient_image_axis(plt.gca())
        outputs["ordered_domain_boundaries"] = save_current_figure(
            dirs["figures"] / f"{sample_stem}__ordered_domain_boundaries.png"
        )
    except Exception as exc:
        warnings.append(f"ordered_domain_boundary_plot_failed={type(exc).__name__}:{exc}")
        plt.close("all")

    return outputs


def select_package_style_gene_rows(
    continuous_class_table: pd.DataFrame,
    discontinuous_class_table: pd.DataFrame,
    max_genes: int,
) -> pd.DataFrame:
    if max_genes <= 0:
        return pd.DataFrame()
    cont = class_leaders(continuous_class_table, rank_max=1).copy()
    disc = class_leaders(discontinuous_class_table, rank_max=1).copy()
    cont["package_plot_group"] = "continuous"
    disc["package_plot_group"] = "discontinuous"
    cont_n = min(len(cont), int(math.ceil(max_genes / 2)))
    disc_n = min(len(disc), max_genes - cont_n)
    selected = pd.concat([cont.head(cont_n), disc.head(disc_n)], ignore_index=True, sort=False)
    if len(selected) < max_genes:
        already = set(zip(selected.get("class_id", []), selected.get("gene", [])))
        extras = pd.concat([cont, disc], ignore_index=True, sort=False)
        extras = extras[
            ~extras.apply(lambda r: (r.get("class_id"), r.get("gene")) in already, axis=1)
        ]
        selected = pd.concat([selected, extras.head(max_genes - len(selected))], ignore_index=True, sort=False)
    return selected.head(max_genes).reset_index(drop=True)


def save_package_style_gene_panels(
    dirs: dict[str, Path],
    sample_stem: str,
    plot_rows: pd.DataFrame,
    binning_and_plotting: Any,
    pw_fit_dict: dict[str, Any],
    binning_output: dict[str, Any],
    labels: np.ndarray,
    isodepth: np.ndarray,
    counts_mat: np.ndarray,
    gene_labels: np.ndarray,
    coords: np.ndarray,
    warnings: list[str],
) -> pd.DataFrame:
    if plot_rows.empty:
        return pd.DataFrame()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = dirs["figures"] / "package_style_gene_panels" / sample_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    n_domains = len(np.unique(labels))
    colors = np.array([plt.cm.tab10(i % 10) for i in range(max(n_domains, 1))])
    gene_labels = np.asarray(gene_labels).astype(str)
    records: list[dict[str, Any]] = []

    for rank, row in enumerate(plot_rows.itertuples(index=False), start=1):
        gene = str(row.gene)
        class_id = str(row.class_id)
        prefix = f"{rank:02d}__{safe_filename(class_id)}__{safe_filename(gene)}"
        record: dict[str, Any] = {
            "source_stem": sample_stem,
            "rank": rank,
            "gene": gene,
            "class_id": class_id,
            "class_type": getattr(row, "class_type", ""),
            "effect_value": getattr(row, "effect_value", np.nan),
        }
        if gene not in gene_labels:
            warnings.append(f"package_plot_gene_absent_from_full_counts={gene}")
            record["status"] = "missing_from_full_counts"
            records.append(record)
            continue

        try:
            binning_and_plotting.plot_gene_pwlinear(
                gene,
                pw_fit_dict,
                labels,
                isodepth,
                binning_output,
                figsize=(7, 3),
                colors=colors,
                domain_boundary_plotting=True,
                variable_spot_size=True,
                pt_size=0.4,
                ticksize=9,
                lw=1.5,
            )
            record["pwlinear_png"] = save_current_figure(out_dir / f"{prefix}__pwlinear.png")
        except Exception as exc:
            warnings.append(f"package_pwlinear_plot_failed={gene}:{type(exc).__name__}:{exc}")
            plt.close("all")

        try:
            binning_and_plotting.plot_gene_raw(
                gene,
                gene_labels,
                counts_mat,
                coords,
                figsize=(5, 5),
                colorbar=True,
                s=float(np.clip(18000.0 / max(coords.shape[0], 1), 4.0, 14.0)),
                cmap="RdPu",
            )
            orient_image_axis(plt.gca())
            record["raw_expression_png"] = save_current_figure(out_dir / f"{prefix}__raw_expression.png")
        except Exception as exc:
            warnings.append(f"package_raw_plot_failed={gene}:{type(exc).__name__}:{exc}")
            plt.close("all")

        try:
            binning_and_plotting.plot_gene_function(
                gene,
                coords,
                pw_fit_dict,
                labels,
                isodepth,
                binning_output,
                figsize=(5, 5),
                colorbar=True,
                contours=True,
                contour_levels=6,
                contour_lw=0.8,
                contour_fs=7,
                s=float(np.clip(18000.0 / max(coords.shape[0], 1), 4.0, 14.0)),
                cmap="RdPu",
            )
            orient_image_axis(plt.gca())
            record["fitted_function_png"] = save_current_figure(out_dir / f"{prefix}__fitted_function.png")
            record["fitted_function_plotter"] = "gaston.binning_and_plotting.plot_gene_function"
        except Exception as exc:
            plt.close("all")
            try:
                record["fitted_function_png"] = save_gene_function_compatible(
                    out_dir / f"{prefix}__fitted_function.png",
                    gene,
                    coords,
                    pw_fit_dict,
                    labels,
                    isodepth,
                    binning_output,
                    point_size=float(np.clip(18000.0 / max(coords.shape[0], 1), 4.0, 14.0)),
                )
                record["fitted_function_plotter"] = (
                    "package_formula_scalar_index_compatibility_after_plot_gene_function_error"
                )
            except Exception as fallback_exc:
                warnings.append(
                    "package_function_plot_failed="
                    f"{gene}:{type(exc).__name__}:{exc};"
                    f"fallback_failed={type(fallback_exc).__name__}:{fallback_exc}"
                )
                plt.close("all")

        record["status"] = "ok"
        records.append(record)

    index = pd.DataFrame(records)
    index_path = dirs["tables"] / f"{sample_stem}__package_style_gene_panel_index.csv"
    index.to_csv(index_path, index=False)
    return index


def save_gene_function_compatible(
    path: Path,
    gene_name: str,
    coords_mat: np.ndarray,
    pw_fit_dict: dict[str, Any],
    gaston_labels: np.ndarray,
    gaston_isodepth: np.ndarray,
    binning_output: dict[str, Any],
    point_size: float,
    offset: float = 10**6,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gene_labels_idx = np.asarray(binning_output["gene_labels_idx"]).astype(str)
    matches = np.where(gene_labels_idx == gene_name)[0]
    if len(matches) == 0:
        umi_threshold = binning_output["umi_threshold"]
        raise ValueError(f"gene does not have UMI count above threshold {umi_threshold}")
    gene_idx = int(matches[0])
    slope_mat, intercept_mat, _, _ = pw_fit_dict["all_cell_types"]
    outputs = np.zeros(gaston_isodepth.shape[0], dtype=float)
    for i in range(gaston_isodepth.shape[0]):
        dom = int(gaston_labels[i])
        outputs[i] = np.log(offset) + float(intercept_mat[gene_idx, dom]) + float(slope_mat[gene_idx, dom]) * gaston_isodepth[i]

    fig, ax = plt.subplots(figsize=(5, 5))
    im1 = ax.scatter(coords_mat[:, 0], coords_mat[:, 1], c=outputs, cmap="RdPu", s=point_size)
    try:
        contours = ax.tricontour(
            coords_mat[:, 0],
            coords_mat[:, 1],
            outputs,
            levels=6,
            linewidths=0.8,
            colors="k",
            linestyles="solid",
        )
        ax.clabel(contours, contours.levels, inline=True, fontsize=7)
    except Exception:
        pass
    cbar = plt.colorbar(im1)
    cbar.ax.tick_params(labelsize=10)
    orient_image_axis(ax)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def fit_piecewise_score(
    score: np.ndarray,
    labels: np.ndarray,
    isodepth: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    score = np.asarray(score, dtype=float)
    labels = np.asarray(labels, dtype=int)
    isodepth = np.asarray(isodepth, dtype=float)
    fitted = np.full(score.shape, np.nan, dtype=float)
    rows: list[dict[str, Any]] = []
    for segment in sorted(int(x) for x in np.unique(labels)):
        mask = (labels == segment) & np.isfinite(score) & np.isfinite(isodepth)
        n = int(mask.sum())
        if n >= 2 and float(np.nanstd(isodepth[mask])) > 0:
            design = np.column_stack([np.ones(n), isodepth[mask]])
            coef, _, _, _ = np.linalg.lstsq(design, score[mask], rcond=None)
            intercept = float(coef[0])
            slope = float(coef[1])
        elif n >= 1:
            intercept = float(np.nanmean(score[mask]))
            slope = 0.0
        else:
            intercept = float("nan")
            slope = float("nan")
        segment_mask = labels == segment
        if np.isfinite(intercept) and np.isfinite(slope):
            fitted[segment_mask] = intercept + slope * isodepth[segment_mask]
            residual = score[mask] - (intercept + slope * isodepth[mask]) if n else np.array([])
            rss = float(np.sum(residual**2)) if n else float("nan")
        else:
            rss = float("nan")
        rows.append(
            {
                "gradient_segment": segment,
                "n_spots_fitted": n,
                "intercept": intercept,
                "slope": slope,
                "score_min": float(np.nanmin(score[mask])) if n else float("nan"),
                "score_median": float(np.nanmedian(score[mask])) if n else float("nan"),
                "score_max": float(np.nanmax(score[mask])) if n else float("nan"),
                "rss": rss,
            }
        )
    return fitted, pd.DataFrame(rows)


def symmetric_limits(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0
    hi = float(np.nanpercentile(np.abs(values), 98))
    if not np.isfinite(hi) or hi == 0:
        hi = float(np.nanmax(np.abs(values)))
    if not np.isfinite(hi) or hi == 0:
        hi = 1.0
    return -hi, hi


def draw_fitted_score_map(
    ax: Any,
    coords: np.ndarray,
    fitted: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
) -> Any:
    im = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=fitted,
        cmap="RdBu_r",
        s=float(np.clip(18000.0 / max(coords.shape[0], 1), 4.0, 14.0)),
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
    )
    finite = np.isfinite(fitted)
    if int(finite.sum()) >= 3:
        try:
            contours = ax.tricontour(
                coords[finite, 0],
                coords[finite, 1],
                fitted[finite],
                levels=6,
                linewidths=0.8,
                colors="k",
                linestyles="solid",
            )
            ax.clabel(contours, contours.levels, inline=True, fontsize=7)
        except Exception:
            pass
    ax.set_title(title, fontsize=9)
    orient_image_axis(ax)
    return im


def save_snai1ac_score_function_maps(
    dirs: dict[str, Path],
    sample_stem: str,
    alignment: pd.DataFrame,
    labels: np.ndarray,
    isodepth: np.ndarray,
    coords: np.ndarray,
    warnings: list[str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records: list[dict[str, Any]] = []
    outputs: dict[str, str] = {}
    fitted_by_col: dict[str, np.ndarray] = {}
    fit_tables: list[pd.DataFrame] = []

    for column, label in SNAI1AC_SCORE_VARIANTS:
        if column not in alignment.columns:
            warnings.append(f"snai1ac_score_variant_missing={column}")
            continue
        score = pd.to_numeric(alignment[column], errors="coerce").to_numpy(dtype=float)
        fitted, fit_table = fit_piecewise_score(score, labels, isodepth)
        fitted_by_col[column] = fitted
        fit_table.insert(0, "score_column", column)
        fit_table.insert(1, "score_label", label)
        fit_tables.append(fit_table)
        vmin, vmax = symmetric_limits(np.concatenate([score[np.isfinite(score)], fitted[np.isfinite(fitted)]]))
        fig, ax = plt.subplots(figsize=(5, 5))
        im = draw_fitted_score_map(ax, coords, fitted, label, vmin, vmax)
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=10)
        path = dirs["figures"] / "snai1ac_score_fitted_maps" / f"{sample_stem}__{column}__fitted_score_map.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs[column] = str(path)
        records.append(
            {
                "source_stem": sample_stem,
                "score_column": column,
                "score_label": label,
                "fitted_score_map_png": str(path),
                "fit_type": "piecewise_linear_score_on_malignant_oriented_isodepth_by_gradient_segment",
            }
        )

    if fitted_by_col:
        fig, axes = plt.subplots(1, len(fitted_by_col), figsize=(5 * len(fitted_by_col), 5), constrained_layout=True)
        axes_arr = np.asarray(axes).reshape(-1)
        for ax, (column, fitted) in zip(axes_arr, fitted_by_col.items()):
            label = dict(SNAI1AC_SCORE_VARIANTS).get(column, column)
            score = pd.to_numeric(alignment[column], errors="coerce").to_numpy(dtype=float)
            vmin, vmax = symmetric_limits(np.concatenate([score[np.isfinite(score)], fitted[np.isfinite(fitted)]]))
            im = draw_fitted_score_map(ax, coords, fitted, label, vmin, vmax)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=8)
        combined_path = dirs["figures"] / "snai1ac_score_fitted_maps" / f"{sample_stem}__snai1ac_three_variant_fitted_score_maps.png"
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(combined_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs["combined_three_variant"] = str(combined_path)

    index = pd.DataFrame(records)
    index_path = dirs["tables"] / f"{sample_stem}__snai1ac_score_fitted_map_index.csv"
    index.to_csv(index_path, index=False)
    if fit_tables:
        fit_table_all = pd.concat(fit_tables, ignore_index=True)
    else:
        fit_table_all = pd.DataFrame()
    fit_table_path = dirs["tables"] / f"{sample_stem}__snai1ac_score_piecewise_fits.csv"
    fit_table_all.to_csv(fit_table_path, index=False)
    return index, outputs


def save_identity_class_size_plot(path: Path, identity_summary: pd.DataFrame, title: str) -> None:
    if identity_summary.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = identity_summary.copy()
    table["class_order"] = np.where(table["class_type"].astype(str).eq("continuous_slope"), 0, 1)
    table = table.sort_values(["class_order", "class_id"])
    colors = np.where(table["class_type"].astype(str).eq("continuous_slope"), "#2f80b7", "#c44e52")
    fig_height = max(4.0, 0.33 * len(table) + 1.2)
    fig, ax = plt.subplots(figsize=(10, fig_height), constrained_layout=True)
    y = np.arange(len(table))
    ax.barh(y, table["n_genes"].astype(float), color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(table["class_id"].astype(str), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Number of genes in GASTON class")
    ax.set_title(title)
    for i, value in enumerate(table["n_genes"].astype(int)):
        ax.text(value + max(table["n_genes"].max() * 0.01, 0.5), i, str(value), va="center", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def load_manifest(args: argparse.Namespace) -> pd.DataFrame:
    manifest = pd.read_csv(args.alignment_manifest)
    required = [
        "dataset",
        "sample",
        "layer",
        "feature_method",
        "analysis_tier",
        "h5ad_path",
        "result_npz",
        "alignment_table",
    ]
    require_columns(manifest, required, "alignment manifest")
    manifest = manifest[manifest["layer"].astype(str) == "whole"].copy()
    if args.samples:
        manifest = manifest[manifest["sample"].astype(str).isin(set(args.samples))].copy()
    if args.dataset:
        manifest = manifest[manifest["dataset"].astype(str) == args.dataset].copy()
    if args.feature_method:
        manifest = manifest[manifest["feature_method"].astype(str) == args.feature_method].copy()
    if args.analysis_tier:
        manifest = manifest[manifest["analysis_tier"].astype(str) == args.analysis_tier].copy()
    manifest = manifest.sort_values(["dataset", "sample", "feature_method"]).reset_index(drop=True)
    return manifest


def process_sample(
    row: pd.Series,
    args: argparse.Namespace,
    dirs: dict[str, Path],
    modules: Any,
    gene_sets_by_collection: dict[str, dict[str, set[str]]],
    robust_by_gene: dict[str, dict[str, Any]],
    snai1_weights: dict[str, float],
) -> SampleResult:
    ad, segmented_fit, binning_and_plotting, spatial_gene_classification, filter_genes, cluster_plotting = modules
    started = time.time()
    sample_stem = stem(row)
    warnings: list[str] = []

    try:
        h5ad_path = Path(str(row["h5ad_path"]))
        alignment_path = Path(str(row["alignment_table"]))
        if not h5ad_path.exists():
            raise FileNotFoundError(h5ad_path)
        if not alignment_path.exists():
            raise FileNotFoundError(alignment_path)

        alignment = pd.read_csv(alignment_path)
        require_columns(
            alignment,
            ["spot_id", "gaston_isodepth_malignant_oriented", "gaston_domain_selected"],
            f"alignment table {alignment_path}",
        )
        training_coords, display_coords, result_npz_path, model_path, geometry_warnings = load_training_geometry(
            row,
            sample_stem,
            alignment,
        )
        warnings.extend(geometry_warnings)
        adata = ad.read_h5ad(h5ad_path)
        counts_mat, gene_labels, counts_source, count_warnings = read_counts_matrix(
            adata,
            alignment["spot_id"],
            args.counts_layer,
        )
        warnings.extend(count_warnings)
        del adata

        isodepth = alignment["gaston_isodepth_malignant_oriented"].astype(float).to_numpy()
        if np.any(~np.isfinite(isodepth)):
            raise ValueError("Malignant-oriented isodepth contains non-finite values.")
        labels_raw = alignment["gaston_domain_selected"].astype(int).to_numpy()
        labels, original_to_segment, segment_to_original, domain_order_stats = order_labels_by_isodepth(labels_raw, isodepth)
        if any(original != segment for original, segment in original_to_segment.items()):
            warnings.append(f"original_domain_labels_reordered_by_isodepth={original_to_segment}")
        if np.sum(counts_mat, axis=1).min() <= 0:
            warnings.append("one_or_more_spots_have_zero_total_umi")

        n_domains = len(np.unique(labels))
        domain_sizes = pd.Series(labels).value_counts().sort_index().to_dict()
        if any(size <= 10 for size in domain_sizes.values()):
            warnings.append(f"domain_with_10_or_fewer_spots={domain_sizes}")

        idx_kept, gene_labels_idx = filter_genes.filter_genes(
            counts_mat,
            gene_labels,
            umi_threshold=args.umi_threshold,
            exclude_prefix=list(DEFAULT_EXCLUDE_PREFIX),
        )
        if len(idx_kept) == 0:
            raise ValueError("No genes passed GASTON/tumor-tutorial UMI/prefix filtering.")

        print(
            f"[{sample_stem}] spots={counts_mat.shape[0]} domains={n_domains} "
            f"genes={counts_mat.shape[1]} kept={len(idx_kept)} counts={counts_source}",
            flush=True,
        )
        pw_fit_dict = segmented_fit.pw_linear_fit(
            counts_mat,
            labels,
            isodepth,
            None,
            [],
            idx_kept=idx_kept,
            umi_threshold=args.umi_threshold,
            isodepth_mult_factor=args.isodepth_mult_factor,
            t=args.slope_pvalue_t,
        )
        binning_output = binning_and_plotting.bin_data(
            counts_mat,
            labels,
            isodepth,
            None,
            gene_labels,
            idx_kept=idx_kept,
            num_bins=args.num_bins,
            umi_threshold=args.umi_threshold,
        )
        cont_genes = spatial_gene_classification.get_cont_genes(
            pw_fit_dict,
            binning_output,
            q=args.continuous_q,
        )
        discont_genes = spatial_gene_classification.get_discont_genes(
            pw_fit_dict,
            binning_output,
            q=args.discontinuous_q,
        )

        slope_mat, intercept_mat, discont_mat, pv_mat = pw_fit_dict["all_cell_types"]
        counts_idx = counts_mat[:, idx_kept]
        domain_table = build_gene_domain_table(
            row,
            gene_labels_idx,
            counts_idx,
            labels,
            segment_to_original,
            slope_mat,
            intercept_mat,
            pv_mat,
            cont_genes,
            robust_by_gene,
            snai1_weights,
            args.continuous_q,
        )
        boundary_table = build_gene_boundary_table(
            row,
            gene_labels_idx,
            discont_mat,
            segment_to_original,
            discont_genes,
            robust_by_gene,
            snai1_weights,
            args.discontinuous_q,
        )
        pattern_table = build_gene_pattern_table(domain_table, boundary_table)
        for frame in (domain_table, boundary_table, pattern_table):
            if frame.empty or "source_stem" in frame.columns:
                continue
            frame.insert(0, "source_stem", sample_stem)
        continuous_class_table = build_continuous_class_table(domain_table)
        discontinuous_class_table = build_discontinuous_class_table(boundary_table)
        identity_classes = class_gene_sets_from_tables(continuous_class_table, discontinuous_class_table)
        universe = {clean_gene(gene) for gene in gene_labels_idx}
        ora_table = run_ora(identity_classes, gene_sets_by_collection, universe)
        if not ora_table.empty:
            ora_table.insert(0, "source_stem", sample_stem)
            ora_table.insert(1, "dataset", row["dataset"])
            ora_table.insert(2, "sample", row["sample"])
            ora_table.insert(3, "layer", row["layer"])
            ora_table.insert(4, "feature_method", row["feature_method"])
            ora_table.insert(5, "analysis_tier", row["analysis_tier"])
        identity_summary = build_identity_class_summary(
            continuous_class_table,
            discontinuous_class_table,
            ora_table,
        )

        domain_path = dirs["tables"] / f"{sample_stem}__gene_domain_slopes.csv.gz"
        boundary_path = dirs["tables"] / f"{sample_stem}__gene_boundary_discontinuities.csv.gz"
        pattern_path = dirs["tables"] / f"{sample_stem}__gene_gradient_patterns.csv.gz"
        continuous_class_path = dirs["tables"] / f"{sample_stem}__continuous_gene_classes.csv.gz"
        discontinuous_class_path = dirs["tables"] / f"{sample_stem}__discontinuous_gene_classes.csv.gz"
        identity_summary_path = dirs["tables"] / f"{sample_stem}__gradient_identity_class_summary.csv"
        ora_path = dirs["tables"] / f"{sample_stem}__gradient_identity_ora.csv"
        domain_order_path = dirs["tables"] / f"{sample_stem}__domain_order_mapping.csv"
        domain_table.to_csv(domain_path, index=False, compression="gzip")
        boundary_table.to_csv(boundary_path, index=False, compression="gzip")
        pattern_table.to_csv(pattern_path, index=False, compression="gzip")
        continuous_class_table.to_csv(continuous_class_path, index=False, compression="gzip")
        discontinuous_class_table.to_csv(discontinuous_class_path, index=False, compression="gzip")
        identity_summary.to_csv(identity_summary_path, index=False)
        ora_table.to_csv(ora_path, index=False)
        domain_order_stats.to_csv(domain_order_path, index=False)

        object_path = dirs["objects"] / f"{sample_stem}__gaston_gene_gradient_objects.npy"
        np.save(
            object_path,
            {
                "pw_fit_dict": pw_fit_dict,
                "binning_output": binning_output,
                "idx_kept": idx_kept,
                "gene_labels_idx": gene_labels_idx,
                "original_to_gradient_segment": original_to_segment,
                "gradient_segment_to_original_domain": segment_to_original,
                "domain_order_stats": domain_order_stats,
                "training_coords": training_coords,
                "display_coords": display_coords,
                "parameters": vars(args),
            },
            allow_pickle=True,
        )

        topology_outputs: dict[str, str] = {}
        package_panel_index = pd.DataFrame()
        snai1ac_score_panel_index = pd.DataFrame()
        snai1ac_score_outputs: dict[str, str] = {}
        if not args.no_plots:
            topology_outputs = save_topology_plots(
                dirs,
                sample_stem,
                training_coords,
                labels,
                isodepth,
                cluster_plotting,
                model_path,
                warnings,
            )
            save_gene_curve_grid(
                dirs["figures"] / f"{sample_stem}__continuous_class_leader_gene_curves.png",
                f"{sample_stem}: continuous GASTON class leaders",
                class_leaders(continuous_class_table, rank_max=1),
                pw_fit_dict,
                binning_output,
            )
            save_gene_curve_grid(
                dirs["figures"] / f"{sample_stem}__discontinuous_class_leader_gene_curves.png",
                f"{sample_stem}: discontinuous GASTON class leaders",
                class_leaders(discontinuous_class_table, rank_max=1),
                pw_fit_dict,
                binning_output,
            )
            save_identity_class_size_plot(
                dirs["figures"] / f"{sample_stem}__gradient_identity_class_sizes.png",
                identity_summary,
                f"{sample_stem}: GASTON gradient identity class sizes",
            )
            package_rows = select_package_style_gene_rows(
                continuous_class_table,
                discontinuous_class_table,
                args.max_package_style_genes,
            )
            package_panel_index = save_package_style_gene_panels(
                dirs,
                sample_stem,
                package_rows,
                binning_and_plotting,
                pw_fit_dict,
                binning_output,
                labels,
                isodepth,
                counts_mat,
                gene_labels,
                training_coords,
                warnings,
            )
            snai1ac_score_panel_index, snai1ac_score_outputs = save_snai1ac_score_function_maps(
                dirs,
                sample_stem,
                alignment,
                labels,
                isodepth,
                training_coords,
                warnings,
            )

        summary = {
            "dataset": row["dataset"],
            "sample": row["sample"],
            "layer": row["layer"],
            "feature_method": row["feature_method"],
            "analysis_tier": row["analysis_tier"],
            "method_contract": {
                "fit": "gaston.segmented_fit.pw_linear_fit",
                "binning": "gaston.binning_and_plotting.bin_data",
                "continuous_gene_call": "gaston.spatial_gene_classification.get_cont_genes",
                "discontinuous_gene_call": "gaston.spatial_gene_classification.get_discont_genes",
                "topology_plots": "gaston.cluster_plotting.plot_isodepth and gaston.cluster_plotting.plot_clusters",
                "gene_plots": "gaston.binning_and_plotting.plot_gene_pwlinear, plot_gene_raw, and plot_gene_function",
                "snai1ac_score_fitted_maps": "visual score-level piecewise linear fits along malignant-oriented isodepth by gradient segment; not a GASTON piecewise Poisson gene model",
                "cell_type_df": None,
                "count_model": "piecewise Poisson with spot total UMI exposure",
            },
            "parameters": {
                "umi_threshold": args.umi_threshold,
                "exclude_prefix": list(DEFAULT_EXCLUDE_PREFIX),
                "isodepth_mult_factor": args.isodepth_mult_factor,
                "slope_pvalue_t": args.slope_pvalue_t,
                "continuous_q": args.continuous_q,
                "discontinuous_q": args.discontinuous_q,
                "num_bins": args.num_bins,
            },
            "inputs": {
                "h5ad_path": str(h5ad_path),
                "alignment_table": str(alignment_path),
                "result_npz": str(result_npz_path) if result_npz_path is not None else "",
                "model_path": str(model_path) if model_path is not None else "",
                "hallmark_json": str(args.hallmark_json),
                "kegg_json": str(args.kegg_json),
                "robust_core_csv": str(args.robust_core_csv),
                "snai1_weights_json": str(args.snai1_weights_json),
                "counts_source": counts_source,
                "figure_coordinate_source": "result_npz S_scaled with post-plot image y-axis inversion",
                "streamline_gradient_coordinate_source": "result_npz S_scaled and best_model.pt via gaston.cluster_plotting.plot_isodepth",
            },
            "outputs": {
                "gene_domain_slopes": str(domain_path),
                "gene_boundary_discontinuities": str(boundary_path),
                "gene_gradient_patterns": str(pattern_path),
                "continuous_gene_classes": str(continuous_class_path),
                "discontinuous_gene_classes": str(discontinuous_class_path),
                "gradient_identity_class_summary": str(identity_summary_path),
                "gradient_identity_ora": str(ora_path),
                "domain_order_mapping": str(domain_order_path),
                "objects": str(object_path),
                "topology_figures": topology_outputs,
                "package_style_gene_panel_index": str(
                    dirs["tables"] / f"{sample_stem}__package_style_gene_panel_index.csv"
                )
                if not package_panel_index.empty
                else "",
                "snai1ac_score_fitted_map_index": str(
                    dirs["tables"] / f"{sample_stem}__snai1ac_score_fitted_map_index.csv"
                )
                if not snai1ac_score_panel_index.empty
                else "",
                "snai1ac_score_piecewise_fits": str(
                    dirs["tables"] / f"{sample_stem}__snai1ac_score_piecewise_fits.csv"
                )
                if not snai1ac_score_panel_index.empty
                else "",
                "snai1ac_score_fitted_maps": snai1ac_score_outputs,
            },
            "domain_labeling": {
                "gradient_segment_definition": "gradient_segment labels are original GASTON domain labels ordered by malignant-oriented isodepth median, low to high",
                "original_domain_definition": "original_domain_label preserves the selected GASTON domain label from 04_isodepth_score_alignment",
                "original_to_gradient_segment": {str(k): int(v) for k, v in original_to_segment.items()},
                "gradient_segment_to_original_domain": {str(k): int(v) for k, v in segment_to_original.items()},
            },
            "counts": {
                "n_spots": int(counts_mat.shape[0]),
                "n_domains": int(n_domains),
                "domain_sizes": {str(k): int(v) for k, v in domain_sizes.items()},
                "n_genes_total": int(counts_mat.shape[1]),
                "n_genes_kept": int(len(idx_kept)),
                "n_continuous_events": int(domain_table["is_continuous_q"].sum()),
                "n_continuous_genes": int((pattern_table["n_continuous_segments"] > 0).sum()),
                "n_discontinuous_events": int(boundary_table["is_discontinuous_q"].sum()) if not boundary_table.empty else 0,
                "n_discontinuous_genes": int((pattern_table["n_discontinuous_boundaries"] > 0).sum()),
                "n_identity_classes": int(len(identity_classes)),
                "n_robust_core_genes_in_universe": int(pattern_table["is_robust_core_gene"].sum()),
                "n_snai1ac_signature_genes_in_universe": int(pattern_table["is_snai1ac_signature_gene"].sum()),
                "n_ora_rows": int(len(ora_table)),
                "n_package_style_gene_panels": int(len(package_panel_index)),
                "n_snai1ac_score_fitted_maps": int(len(snai1ac_score_panel_index)),
            },
            "warnings": warnings,
            "elapsed_sec": round(time.time() - started, 3),
        }
        write_json(dirs["summaries"] / f"{sample_stem}__gene_gradient_summary.json", summary)
        stale_error = dirs["logs"] / f"{sample_stem}__gene_gradient_error.json"
        if stale_error.exists():
            stale_error.unlink()

        return SampleResult(
            dataset=str(row["dataset"]),
            sample=str(row["sample"]),
            layer=str(row["layer"]),
            feature_method=str(row["feature_method"]),
            analysis_tier=str(row["analysis_tier"]),
            status="ok",
            elapsed_sec=time.time() - started,
            n_spots=int(counts_mat.shape[0]),
            n_domains=int(n_domains),
            n_genes_total=int(counts_mat.shape[1]),
            n_genes_kept=int(len(idx_kept)),
            n_continuous_events=int(domain_table["is_continuous_q"].sum()),
            n_continuous_genes=int((pattern_table["n_continuous_segments"] > 0).sum()),
            n_discontinuous_events=int(boundary_table["is_discontinuous_q"].sum()) if not boundary_table.empty else 0,
            n_discontinuous_genes=int((pattern_table["n_discontinuous_boundaries"] > 0).sum()),
            n_identity_classes=int(len(identity_classes)),
            n_ora_rows=int(len(ora_table)),
            warnings="; ".join(warnings),
        )
    except Exception as exc:
        elapsed = time.time() - started
        error = repr(exc)
        write_json(
            dirs["logs"] / f"{sample_stem}__gene_gradient_error.json",
            {
                "source_stem": sample_stem,
                "error": error,
                "elapsed_sec": elapsed,
            },
        )
        if args.strict:
            raise
        return SampleResult(
            dataset=str(row["dataset"]),
            sample=str(row["sample"]),
            layer=str(row["layer"]),
            feature_method=str(row["feature_method"]),
            analysis_tier=str(row["analysis_tier"]),
            status="error",
            elapsed_sec=elapsed,
            warnings="; ".join(warnings),
            error=error,
        )


def write_run_readme(path: Path) -> None:
    text = f"""# GASTON Native Gene-Gradient Layer

This folder contains the post-training gene-gradient layer for `GASTON_method_v1`.

The analysis starts from selected whole-tissue GASTON domains and
malignant-oriented isodepth already saved in `04_isodepth_score_alignment`.
This stage asks: what is the gene-expression identity of each sample's GASTON
gradient? It does not select genes based on SNAI1-ac and it does not interpret
robust-core/signature genes first. Those labels are carried only as annotations
for later targeted review.

Important labeling convention: original selected GASTON domain labels are
preserved as `original_domain_label`. For the package's segmented gene-gradient
machinery, those labels are relabeled as `gradient_segment` ordered from low to
high malignant-oriented isodepth median. This prevents arbitrary numeric domain
IDs from being mistaken for gradient order.

The implementation follows the package/tutorial machinery:

- `segmented_fit.pw_linear_fit`: piecewise Poisson expression fit with spot total
  UMI exposure
- `binning_and_plotting.bin_data`: binned expression summaries along isodepth
- `spatial_gene_classification.get_cont_genes`: large continuous-slope genes
- `spatial_gene_classification.get_discont_genes`: large boundary-jump genes
- `cluster_plotting.plot_isodepth` / `plot_clusters`: package-style contour,
  streamline/arrow, and ordered-boundary topology plots
- `binning_and_plotting.plot_gene_pwlinear`, `plot_gene_raw`, and
  `plot_gene_function`: package-style review panels for GASTON class-leader
  genes

Default parameters mirror the tumor tutorial where relevant:

- UMI threshold: {DEFAULT_UMI_THRESHOLD}
- Excluded gene prefixes: {", ".join(DEFAULT_EXCLUDE_PREFIX)}
- Isodepth multiplier for numerical stability: {DEFAULT_ISODEPTH_MULT}
- Slope LLR p-value threshold inside `pw_linear_fit`: {DEFAULT_SLOPE_PVALUE_T}
- Continuous-gene quantile: {DEFAULT_CONT_Q}
- Discontinuous-gene quantile: {DEFAULT_DISCONT_Q}
- Bins: {DEFAULT_NUM_BINS}

Primary output groups:

- continuous gradient-segment/direction gene classes
- discontinuous ordered-boundary/direction gene classes
- Hallmark and KEGG ORA for those GASTON-derived classes
- class-leader gene-curve panels for human review
- package-native raw/function/piecewise-linear panels for class-leader genes

The SNAI1-ac fitted-score maps are visual analogs to the fitted gene-function
maps: each score variant is fit piecewise linearly along malignant-oriented
isodepth within each gradient segment. They are not GASTON's piecewise Poisson
gene-expression model and should be interpreted as score-level overlays.

The separate SNAI1-ac score/isodepth association layer and the targeted
robust-core/signature gene review come after this gradient-identity stage.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    modules = import_runtime_modules()
    dirs = ensure_dirs(args.out_dir)
    write_run_readme(args.out_dir / "README.md")

    for input_path in [
        args.alignment_manifest,
        args.hallmark_json,
        args.kegg_json,
        args.robust_core_csv,
        args.snai1_weights_json,
    ]:
        if not input_path.exists():
            raise FileNotFoundError(input_path)

    _, robust_by_gene, snai1_weights = load_gene_annotations(args)
    gene_sets_by_collection = {
        "hallmark": load_gene_sets(args.hallmark_json),
        "kegg_legacy": load_gene_sets(args.kegg_json),
    }
    manifest = load_manifest(args)
    if manifest.empty:
        raise RuntimeError("No manifest rows matched the requested filters.")

    results: list[SampleResult] = []
    for _, row in manifest.iterrows():
        print(f"Starting {stem(row)}", flush=True)
        result = process_sample(
            row,
            args,
            dirs,
            modules,
            gene_sets_by_collection,
            robust_by_gene,
            snai1_weights,
        )
        results.append(result)
        print(
            f"Finished {row['dataset']} / {row['sample']} "
            f"status={result.status} elapsed={result.elapsed_sec:.1f}s",
            flush=True,
        )

    manifest_out = pd.DataFrame([result.__dict__ for result in results])
    manifest_path = dirs["root"] / "gene_gradient_manifest.csv"
    if manifest_path.exists():
        existing = pd.read_csv(manifest_path)
        key_cols = ["dataset", "sample", "layer", "feature_method", "analysis_tier"]
        combined = existing.merge(
            manifest_out[key_cols],
            how="left",
            on=key_cols,
            indicator=True,
        )
        existing_keep = existing.loc[combined["_merge"].to_numpy() == "left_only"].copy()
        manifest_out = pd.concat([existing_keep, manifest_out], ignore_index=True)
    manifest_out.to_csv(manifest_path, index=False)

    print(f"Wrote manifest: {manifest_path}", flush=True)
    return 0 if all(result.status == "ok" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
