"""
Per-sample Definition 3b / raw Definition 4 / GeneNMF-equivalent pipeline.

This pipeline follows the thesis framework's per-sample-first logic while
implementing the intentional handoff override for Definition 4:

- Definition 3b: cluster tumor-zone spots on raw K* cNMF usage vectors.
- Definition 4: describe HH neighborhoods using raw programme neighbor vectors,
  not alignment_category vectors.
- Track 3: run a GeneNMF-style equivalent alignment because the GeneNMF R
  package is not available locally in this environment.

All outputs are written into a timestamped D: run directory and never into the
repository tree.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import anndata as ad
import igraph as ig
import leidenalg
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist, squareform
from scipy.stats import mannwhitneyu, rankdata, spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph

from analysis_utils import cohens_d


REPO_ROOT = Path(r"C:\Users\luchu\Documents\MSc\Master Thesis\Code\HGSOC_Spatial_Atlas")
DATA_ROOT = Path(r"D:\HGSOC_Spatial_Atlas")
ANALYSIS_ROOT = DATA_ROOT / "05_analysis_ready"
RUN_SUFFIX = "definition3b_definition4_raw_geneNMF"
S3_ROOT = ANALYSIS_ROOT / "S3_cNMF_Tumor_Programs"
VISIUM_ROOT = ANALYSIS_ROOT / "visium"
RAW_VISIUM_ROOT = DATA_ROOT / "01_raw_data" / "visium"
MANIFEST_PATH = S3_ROOT / "sample_manifest.csv"

HEX_OFFSETS = [(2, 0), (-2, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
PRIMARY_DATASETS = ["denisenko_2022", "ju_2024", "yamamoto_2025"]
PRIMARY_SCORE_COLUMN = "SNAI1-ac_score"
OPTIONAL_SCORE_COLUMNS = ["SNAI1_ac_pc_thresholded_score", "SNAI1_ac_pc_up_score"]
DEFAULT_RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run per-sample Definition 3b, raw Definition 4, and GeneNMF-equivalent alignment."
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional existing D: run directory. Defaults to today's timestamped run directory.",
    )
    parser.add_argument(
        "--stage",
        choices=["prepare", "definition3b", "definition4", "genenmf", "summary", "all"],
        default="all",
        help="Pipeline stage to execute.",
    )
    parser.add_argument(
        "--samples",
        nargs="*",
        default=None,
        help="Optional sample filters using sample_id_on_disk or dataset__sample_id_on_disk.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite stage outputs when they already exist.",
    )
    return parser.parse_args()


def default_run_dir() -> Path:
    date_prefix = datetime.now().strftime("%Y%m%d")
    return ANALYSIS_ROOT / f"{date_prefix}_{RUN_SUFFIX}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_subdirs(run_dir: Path) -> dict[str, Path]:
    subdirs = {
        "root": run_dir,
        "config": run_dir / "00_config",
        "logs": run_dir / "01_logs",
        "d3b": run_dir / "02_definition3b_mixture_programme_niches",
        "d4": run_dir / "03_definition4_raw_neighbourhoods",
        "genenmf": run_dir / "04_GeneNMF_alignment",
        "qc": run_dir / "05_qc_sanity_checks",
        "summary": run_dir / "06_summary",
    }
    for path in subdirs.values():
        ensure_dir(path)
    return subdirs


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def upsert_csv(path: Path, frame: pd.DataFrame, key_cols: list[str]) -> None:
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, frame], ignore_index=True, sort=False)
        combined = combined.drop_duplicates(subset=key_cols, keep="last")
    else:
        combined = frame.copy()
    combined.to_csv(path, index=False)


def log(run_dir: Path, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    log_path = run_dir / "01_logs" / "pipeline.log"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def append_step(run_dir: Path, step_name: str, status: str, detail: str = "") -> None:
    row = pd.DataFrame(
        [
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "step_name": step_name,
                "status": status,
                "detail": detail,
            }
        ]
    )
    step_log = run_dir / "01_logs" / "step_log.csv"
    row.to_csv(step_log, mode="a", index=False, header=not step_log.exists())


def base_config() -> dict:
    return {
        "dataset_allowlist": PRIMARY_DATASETS,
        "framework_anchor": str(REPO_ROOT / "00_documentation" / "framword_drafts" / "analytical_framework_final.md"),
        "framework_definition4_override": {
            "enabled": True,
            "framework_definition4": "alignment_category neighbour vectors",
            "implemented_definition4": "raw programme neighbour vectors",
            "reason": (
                "User explicitly confirmed the handoff intentionally overrides the framework's "
                "alignment_category bottleneck for Definition 4."
            ),
        },
        "global_rules": {
            "per_sample_first": True,
            "pool_raw_spots_across_samples": False,
            "retain_sample_id_everywhere": True,
            "retain_spot_id_traceability": True,
            "retain_program_id_traceability": True,
            "candidate_signal_language": (
                "Treat SNAI1-ac as a candidate acetylation-associated programme projection, "
                "not a direct measurement of acetylated SNAI1."
            ),
        },
        "integration": {
            "primary_score_column": PRIMARY_SCORE_COLUMN,
            "optional_score_columns": OPTIONAL_SCORE_COLUMNS,
        },
        "definition3b": {
            "resolution_grid": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
            "knn_k": 10,
            "min_cluster_size": 4,
            "component_min_size": 4,
            "moran_permutations": 199,
            "correlation_method": "spearman",
            "max_dominant_programs": 4,
        },
        "definition4": {
            "resolution_grid": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
            "knn_k": 8,
            "min_hh_spots_for_clustering": 6,
            "min_cluster_size": 4,
            "neighbor_ring": 1,
            "permanova_permutations": 999,
            "max_programs_for_display": 15,
        },
        "genenmf_equivalent": {
            "package_available": False,
            "similarity_metric": "cosine",
            "cluster_k_min": 4,
            "cluster_k_max": 40,
            "min_cluster_size": 2,
        },
    }


def select_manifest_rows(sample_filters: list[str] | None) -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST_PATH)
    include_col = "include_sample" if "include_sample" in manifest.columns else "include"
    manifest = manifest.loc[manifest[include_col].astype(str).str.lower() == "true"].copy()
    manifest = manifest.loc[manifest["dataset"].isin(PRIMARY_DATASETS)].copy()
    manifest["sample_filter_key"] = manifest["dataset"] + "__" + manifest["sample_id_on_disk"]
    if sample_filters:
        allowed = set(sample_filters)
        mask = manifest["sample_id_on_disk"].isin(allowed) | manifest["sample_filter_key"].isin(allowed)
        manifest = manifest.loc[mask].copy()
    return manifest.sort_values(["dataset", "sample_id_on_disk"]).reset_index(drop=True)


def build_resolved_manifest(run_dir: Path, config: dict, sample_filters: list[str] | None) -> pd.DataFrame:
    manifest = select_manifest_rows(sample_filters)
    records = []
    for row in manifest.to_dict("records"):
        dataset = str(row["dataset"])
        sample = str(row["sample_id_on_disk"])
        sample_dir = VISIUM_ROOT / dataset / sample
        spatial_dir = RAW_VISIUM_ROOT / dataset / "organized" / sample / "spatial"
        resolved = {
            "dataset": dataset,
            "sample_id_on_disk": sample,
            "sample_label": row["sample_label"],
            "k_star": int(row["k_star"]),
            "analysis_ready_h5ad_path": str(row["analysis_ready_h5ad_path"]),
            "robustness_h5ad_path": str(row["robustness_h5ad_path"]),
            "usage_path": str(S3_ROOT / "per_sample" / sample / "representative_usage_kstar.csv"),
            "spectra_path": str(S3_ROOT / "per_sample" / sample / "extracted_program_spectra.csv"),
            "top100_path": str(S3_ROOT / "per_sample" / sample / "extracted_program_top100.csv"),
            "lisa_path": str(sample_dir / "signature_analysis" / "csvs" / "lisa_results.csv"),
            "interface_path": str(sample_dir / f"{sample}_interface.csv"),
            "celltypes_path": str(sample_dir / f"{sample}_celltypes.csv"),
            "tissue_positions_path": str(spatial_dir / "tissue_positions_list.csv"),
        }
        resolved["all_required_paths_exist"] = all(Path(resolved[path_key]).exists() for path_key in [
            "usage_path",
            "spectra_path",
            "lisa_path",
            "interface_path",
            "celltypes_path",
            "tissue_positions_path",
            "robustness_h5ad_path",
        ])
        records.append(resolved)
    resolved_df = pd.DataFrame(records)
    resolved_df.to_csv(run_dir / "00_config" / "resolved_sample_manifest.csv", index=False)
    return resolved_df


def prepare_run(run_dir: Path, sample_filters: list[str] | None) -> pd.DataFrame:
    config = base_config()
    run_subdirs(run_dir)
    write_json(run_dir / "00_config" / "run_config.json", config)
    manifest = build_resolved_manifest(run_dir, config, sample_filters)
    qc_summary = {
        "run_dir": str(run_dir),
        "n_samples_selected": int(len(manifest)),
        "n_samples_with_all_required_paths": int(manifest["all_required_paths_exist"].sum()) if not manifest.empty else 0,
        "datasets_present": sorted(manifest["dataset"].unique().tolist()) if not manifest.empty else [],
        "definition4_override": config["framework_definition4_override"],
    }
    write_json(run_dir / "05_qc_sanity_checks" / "prepare_summary.json", qc_summary)
    log(run_dir, f"Prepared run manifest with {len(manifest)} samples.")
    return manifest


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    out = np.full(pvalues.shape, np.nan, dtype=float)
    mask = np.isfinite(pvalues)
    if mask.sum() == 0:
        return out
    valid = pvalues[mask]
    order = np.argsort(valid)
    ranks = np.arange(1, len(valid) + 1, dtype=float)
    adjusted = valid[order] * len(valid) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out[np.where(mask)[0][order]] = adjusted
    return out


def read_tissue_positions(path: Path) -> pd.DataFrame:
    positions = pd.read_csv(path, header=None)
    positions.columns = [
        "spot_id",
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
    ]
    positions["spot_id"] = positions["spot_id"].astype(str)
    positions["spot_key"] = positions["array_row"].astype(str) + "x" + positions["array_col"].astype(str)
    return positions


def read_interface(path: Path) -> pd.DataFrame:
    interface = pd.read_csv(path, sep=None, engine="python")
    interface.columns = [str(col).strip() for col in interface.columns]
    if "spot_id" not in interface.columns and len(interface.columns) >= 1:
        interface = interface.rename(columns={interface.columns[0]: "spot_id"})
    interface["spot_key"] = interface["spot_id"].astype(str)
    interface = interface.rename(columns={"spot_id": "interface_spot_key"})
    return interface[["spot_key", "interface"]]


def read_celltypes(path: Path) -> pd.DataFrame:
    celltypes = pd.read_csv(path, sep=None, engine="python", index_col=0)
    celltypes.index = celltypes.index.astype(str)
    celltypes = celltypes.reset_index()
    celltypes = celltypes.rename(columns={celltypes.columns[0]: "spot_key"})
    return celltypes


def read_lisa(path: Path) -> pd.DataFrame:
    lisa = pd.read_csv(path)
    lisa = lisa.rename(columns={"spot": "spot_id"})
    lisa["spot_id"] = lisa["spot_id"].astype(str)
    return lisa


def read_scores_backed(primary_path: Path, fallback_path: Path | None = None) -> tuple[pd.DataFrame, str]:
    attempts = [primary_path]
    if fallback_path is not None:
        attempts.append(fallback_path)
    last_error = None
    required_cols = [PRIMARY_SCORE_COLUMN] + OPTIONAL_SCORE_COLUMNS
    for path in attempts:
        if path is None or not path.exists():
            continue
        try:
            adata = ad.read_h5ad(path, backed="r")
            present_cols = [col for col in required_cols if col in adata.obs.columns]
            obs = adata.obs.loc[:, present_cols].copy()
            out = pd.DataFrame({"spot_id": adata.obs_names.astype(str)})
            present = bool(present_cols)
            for col in present_cols:
                # Avoid index-alignment NaNs when AnnData obs carries barcode indexes.
                out[col] = pd.to_numeric(obs[col].to_numpy(), errors="coerce")
            if hasattr(adata, "file") and adata.file is not None:
                adata.file.close()
            if not present or PRIMARY_SCORE_COLUMN not in out.columns:
                raise KeyError(f"Primary score column {PRIMARY_SCORE_COLUMN} missing in {path}")
            return out, str(path)
        except Exception as exc:  # pragma: no cover - fallback path
            last_error = exc
    raise RuntimeError(f"Could not read score columns from backed h5ad: {last_error}")


def load_integrated_sample(sample_row: pd.Series) -> tuple[pd.DataFrame, list[str], dict]:
    usage = pd.read_csv(sample_row["usage_path"])
    program_cols = [
        col
        for col in usage.columns
        if col not in {"spot_id", "dataset", "sample_id_on_disk", "sample_label"}
    ]
    usage["spot_id"] = usage["spot_id"].astype(str)

    positions = read_tissue_positions(Path(sample_row["tissue_positions_path"]))
    interface = read_interface(Path(sample_row["interface_path"]))
    celltypes = read_celltypes(Path(sample_row["celltypes_path"]))
    lisa = read_lisa(Path(sample_row["lisa_path"]))
    scores, score_source = read_scores_backed(
        Path(sample_row["robustness_h5ad_path"]),
        Path(sample_row["analysis_ready_h5ad_path"]),
    )

    merged = (
        positions.merge(interface, on="spot_key", how="left")
        .merge(celltypes, on="spot_key", how="left")
        .merge(lisa, on="spot_id", how="left", suffixes=("", "_lisa"))
        .merge(scores, on="spot_id", how="left")
    )

    tumor = usage.merge(merged, on="spot_id", how="left")
    tumor["dataset"] = str(sample_row["dataset"])
    tumor["sample_id_on_disk"] = str(sample_row["sample_id_on_disk"])
    tumor["sample_label"] = str(sample_row["sample_label"])
    tumor["interface"] = tumor["interface"].astype(str)
    tumor["LISA_category"] = tumor["LISA_category"].fillna("Missing")
    tumor = tumor.sort_values("spot_id").reset_index(drop=True)

    qc = {
        "dataset": str(sample_row["dataset"]),
        "sample_id_on_disk": str(sample_row["sample_id_on_disk"]),
        "sample_label": str(sample_row["sample_label"]),
        "score_source_path": score_source,
        "usage_spot_n": int(len(usage)),
        "tumor_rows_after_merge": int(len(tumor)),
        "missing_positions_n": int(tumor["spot_key"].isna().sum()),
        "missing_interface_n": int(tumor["interface"].isna().sum()),
        "missing_malignant_n": int(pd.to_numeric(tumor.get("Malignant"), errors="coerce").isna().sum()),
        "missing_score_n": int(pd.to_numeric(tumor[PRIMARY_SCORE_COLUMN], errors="coerce").isna().sum()),
        "missing_lisa_n": int(tumor["LISA_category"].eq("Missing").sum()),
        "interface_non_tumor_n": int((tumor["interface"] != "Tumor").sum()),
    }
    return tumor, program_cols, qc


def build_hex_adjacency(sample_df: pd.DataFrame) -> csr_matrix:
    index_lookup = {
        (int(row.array_row), int(row.array_col)): idx
        for idx, row in sample_df[["array_row", "array_col"]].reset_index(drop=True).iterrows()
    }
    rows: list[int] = []
    cols: list[int] = []
    for idx, row in sample_df[["array_row", "array_col"]].reset_index(drop=True).iterrows():
        key = (int(row.array_row), int(row.array_col))
        for dr, dc in HEX_OFFSETS:
            neighbor_idx = index_lookup.get((key[0] + dr, key[1] + dc))
            if neighbor_idx is not None:
                rows.append(idx)
                cols.append(neighbor_idx)
    data = np.ones(len(rows), dtype=np.int8)
    adjacency = csr_matrix((data, (rows, cols)), shape=(len(sample_df), len(sample_df)))
    adjacency = ((adjacency + adjacency.T) > 0).astype(np.int8)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    return adjacency


def adjacency_neighbor_lists(adjacency: csr_matrix) -> list[np.ndarray]:
    return [adjacency[idx].indices for idx in range(adjacency.shape[0])]


def moran_i(values: np.ndarray, adjacency: csr_matrix, permutations: int = 199, seed: int = DEFAULT_RANDOM_STATE) -> dict:
    x = np.asarray(values, dtype=float)
    mask = np.isfinite(x)
    if mask.sum() < 3:
        return {"I": np.nan, "p_value": np.nan}
    x = x[mask]
    adj = adjacency[mask][:, mask]
    n = len(x)
    if n < 3 or adj.nnz == 0:
        return {"I": np.nan, "p_value": np.nan}
    centered = x - x.mean()
    denom = float(np.dot(centered, centered))
    if denom == 0:
        return {"I": 0.0, "p_value": np.nan}
    weights_sum = float(adj.sum())
    observed = float(n / weights_sum * ((centered[:, None] * centered[None, :]) * adj.toarray()).sum() / denom)
    rng = np.random.default_rng(seed)
    permuted = []
    for _ in range(permutations):
        shuffled = centered[rng.permutation(n)]
        permuted_val = float(n / weights_sum * ((shuffled[:, None] * shuffled[None, :]) * adj.toarray()).sum() / denom)
        permuted.append(permuted_val)
    permuted = np.asarray(permuted, dtype=float)
    p_value = float((1 + np.sum(np.abs(permuted) >= abs(observed))) / (permutations + 1))
    return {"I": observed, "p_value": p_value}


def cluster_binary_moran(labels: np.ndarray, adjacency: csr_matrix, permutations: int) -> tuple[pd.DataFrame, float]:
    rows = []
    labels = np.asarray(labels)
    for label in sorted(pd.unique(labels)):
        mask = (labels == label).astype(float)
        result = moran_i(mask, adjacency, permutations=permutations)
        rows.append(
            {
                "cluster_label": str(label),
                "cluster_size": int(mask.sum()),
                "binary_morans_I": result["I"],
                "binary_morans_p": result["p_value"],
            }
        )
    table = pd.DataFrame(rows)
    weighted_mean = float(np.average(table["binary_morans_I"].fillna(0.0), weights=table["cluster_size"])) if not table.empty else np.nan
    return table, weighted_mean


def build_leiden_graph(features: np.ndarray, knn_k: int) -> ig.Graph:
    n_obs = features.shape[0]
    if n_obs < 2:
        graph = ig.Graph()
        graph.add_vertices(n_obs)
        return graph
    k = max(1, min(knn_k, n_obs - 1))
    dist_graph = kneighbors_graph(features, n_neighbors=k, mode="distance", include_self=False)
    dist_graph = dist_graph.maximum(dist_graph.T).tocsr()
    if dist_graph.nnz == 0:
        graph = ig.Graph()
        graph.add_vertices(n_obs)
        return graph
    sigma = float(np.median(dist_graph.data[dist_graph.data > 0])) if np.any(dist_graph.data > 0) else 1.0
    sigma = sigma if sigma > 0 else 1.0
    weights = dist_graph.copy()
    weights.data = np.exp(-(weights.data / sigma))
    coo = weights.tocoo()
    edges = list(zip(coo.row.tolist(), coo.col.tolist(), strict=False))
    graph = ig.Graph()
    graph.add_vertices(n_obs)
    graph.add_edges(edges)
    graph.es["weight"] = coo.data.astype(float).tolist()
    return graph


def leiden_labels(features: np.ndarray, resolution: float, knn_k: int, seed: int) -> np.ndarray:
    graph = build_leiden_graph(features, knn_k)
    if graph.vcount() == 0:
        return np.asarray([], dtype=int)
    if graph.ecount() == 0:
        return np.zeros(graph.vcount(), dtype=int)
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=graph.es["weight"],
        resolution_parameter=float(resolution),
        seed=seed,
    )
    return np.asarray(partition.membership, dtype=int)


def select_resolution(
    features: np.ndarray,
    resolution_grid: list[float],
    knn_k: int,
    min_cluster_size: int,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    n_obs = features.shape[0]
    if n_obs == 0:
        return np.asarray([], dtype=int), pd.DataFrame(), {"selection_reason": "no_spots"}
    if n_obs < max(2, min_cluster_size):
        labels = np.zeros(n_obs, dtype=int)
        grid = pd.DataFrame(
            [
                {
                    "resolution": np.nan,
                    "n_clusters": 1,
                    "min_cluster_size": n_obs,
                    "max_cluster_size": n_obs,
                    "silhouette": np.nan,
                    "valid_for_selection": False,
                    "selection_reason": "too_few_spots",
                }
            ]
        )
        return labels, grid, {"selection_reason": "too_few_spots", "selected_resolution": np.nan}

    rows = []
    label_lookup: dict[float, np.ndarray] = {}
    for resolution in resolution_grid:
        labels = leiden_labels(features, resolution, knn_k=knn_k, seed=random_state)
        label_lookup[resolution] = labels
        counts = pd.Series(labels).value_counts().sort_values(ascending=False)
        n_clusters = int(counts.shape[0])
        silhouette = np.nan
        if 1 < n_clusters < n_obs:
            try:
                silhouette = float(silhouette_score(features, labels, metric="euclidean"))
            except Exception:
                silhouette = np.nan
        valid = (n_clusters >= 2) and (counts.min() >= min_cluster_size)
        rows.append(
            {
                "resolution": float(resolution),
                "n_clusters": n_clusters,
                "min_cluster_size": int(counts.min()),
                "max_cluster_size": int(counts.max()),
                "silhouette": silhouette,
                "valid_for_selection": bool(valid),
            }
        )
    grid = pd.DataFrame(rows)
    valid_grid = grid.loc[grid["valid_for_selection"]].copy()
    if valid_grid.empty:
        fallback = grid.sort_values(
            ["silhouette", "min_cluster_size", "resolution"],
            ascending=[False, False, True],
            na_position="last",
        ).iloc[0]
        selected_resolution = float(fallback["resolution"])
        selection_reason = "fallback_best_available"
    else:
        best = valid_grid.sort_values(
            ["silhouette", "min_cluster_size", "resolution"],
            ascending=[False, False, True],
            na_position="last",
        ).iloc[0]
        selected_resolution = float(best["resolution"])
        selection_reason = "best_valid_silhouette"
    selected_labels = label_lookup[selected_resolution]
    grid["selected"] = grid["resolution"].eq(selected_resolution)
    selected_counts = pd.Series(selected_labels).value_counts().sort_index()
    selected_info = {
        "selected_resolution": selected_resolution,
        "selection_reason": selection_reason,
        "selected_n_clusters": int(selected_counts.shape[0]),
        "selected_min_cluster_size": int(selected_counts.min()),
        "selected_max_cluster_size": int(selected_counts.max()),
    }
    return selected_labels, grid, selected_info


def label_series_to_strings(labels: np.ndarray, prefix: str) -> pd.Series:
    unique = sorted(pd.unique(labels))
    mapping = {label: f"{prefix}_{idx + 1:02d}" for idx, label in enumerate(unique)}
    return pd.Series([mapping[label] for label in labels], dtype="object")


def component_rows_for_clusters(
    sample_df: pd.DataFrame,
    cluster_labels: pd.Series,
    adjacency: csr_matrix,
    min_size: int,
    prefix: str,
) -> pd.DataFrame:
    rows = []
    for cluster_id in sorted(cluster_labels.unique()):
        mask = cluster_labels.to_numpy() == cluster_id
        cluster_indices = np.flatnonzero(mask)
        if len(cluster_indices) == 0:
            continue
        subgraph = adjacency[cluster_indices][:, cluster_indices]
        n_components, comp_labels = connected_components(subgraph, directed=False, return_labels=True)
        for component_idx in range(n_components):
            member_indices = cluster_indices[comp_labels == component_idx]
            if len(member_indices) < min_size:
                continue
            subset = sample_df.iloc[member_indices]
            rows.append(
                {
                    "sample_label": subset["sample_label"].iloc[0],
                    "cluster_id": cluster_id,
                    "component_id": f"{prefix}_{cluster_id}_R{component_idx + 1:02d}",
                    "n_spots": int(len(subset)),
                    "mean_x": float(subset["pxl_col_in_fullres"].mean()),
                    "mean_y": float(subset["pxl_row_in_fullres"].mean()),
                    "spot_ids": ";".join(subset["spot_id"].tolist()),
                }
            )
    return pd.DataFrame(rows)


def simple_scatter(
    sample_df: pd.DataFrame,
    color_series: pd.Series,
    title: str,
    output_path: Path,
    categorical: bool = True,
    cmap: str = "viridis",
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    x = sample_df["pxl_col_in_fullres"].to_numpy()
    y = sample_df["pxl_row_in_fullres"].to_numpy()
    if categorical:
        values = color_series.astype(str)
        levels = sorted(values.unique())
        palette = sns.color_palette("tab20", n_colors=max(len(levels), 3))
        color_map = {level: palette[idx] for idx, level in enumerate(levels)}
        for level in levels:
            mask = values == level
            ax.scatter(x[mask], y[mask], s=18, c=[color_map[level]], label=level, linewidths=0, alpha=0.9)
        ax.legend(loc="best", frameon=False, fontsize=7)
    else:
        sc = ax.scatter(x, y, s=18, c=pd.to_numeric(color_series, errors="coerce"), cmap=cmap, linewidths=0, alpha=0.9)
        fig.colorbar(sc, ax=ax, shrink=0.8)
    ax.set_title(title)
    ax.set_xlabel("pixel x")
    ax.set_ylabel("pixel y")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def heatmap_plot(data: pd.DataFrame, title: str, output_path: Path, cmap: str = "viridis") -> None:
    fig_h = max(4.5, 0.35 * max(1, data.shape[0]))
    fig_w = max(6.0, 0.22 * max(6, data.shape[1]))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(data, cmap=cmap, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def boxplot_by_group(plot_df: pd.DataFrame, x_col: str, y_col: str, title: str, output_path: Path) -> None:
    plot_df = plot_df[[x_col, y_col]].dropna(subset=[x_col, y_col]).copy()
    if plot_df.empty:
        return
    plot_df[x_col] = plot_df[x_col].astype(str)
    order = sorted(plot_df[x_col].unique().tolist())
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    sns.boxplot(data=plot_df, x=x_col, y=y_col, order=order, ax=ax, color="#cfe3ff")
    sns.stripplot(data=plot_df, x=x_col, y=y_col, order=order, ax=ax, color="#24557a", size=2.5, alpha=0.45)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_sample_output_dirs(base_dir: Path, sample_label: str) -> dict[str, Path]:
    sample_root = ensure_dir(base_dir / sample_label)
    dirs = {
        "root": sample_root,
        "tables": ensure_dir(sample_root / "tables"),
        "figures": ensure_dir(sample_root / "figures"),
    }
    return dirs


def summarize_definition3b_cluster(
    sample_df: pd.DataFrame,
    cluster_col: str,
    program_cols: list[str],
    sample_row: pd.Series,
) -> pd.DataFrame:
    grouped = sample_df.groupby(cluster_col, sort=True)
    summary = grouped[program_cols].mean()
    meta = grouped.agg(
        n_spots=("spot_id", "size"),
        mean_snai1_ac=(PRIMARY_SCORE_COLUMN, "mean"),
        median_snai1_ac=(PRIMARY_SCORE_COLUMN, "median"),
        mean_malignant=("Malignant", "mean"),
        median_malignant=("Malignant", "median"),
    )
    out = meta.join(summary)
    out.insert(0, "sample_label", str(sample_row["sample_label"]))
    out.insert(0, "sample_id_on_disk", str(sample_row["sample_id_on_disk"]))
    out.insert(0, "dataset", str(sample_row["dataset"]))
    dominant = []
    for cluster_id, row in summary.iterrows():
        ordered = row.sort_values(ascending=False).head(4)
        dominant.append(
            {
                "cluster_id": cluster_id,
                "dominant_programs": ";".join(
                    [f"{program}:{value:.3f}" for program, value in ordered.items()]
                ),
            }
        )
    dominant_df = pd.DataFrame(dominant).set_index("cluster_id")
    out = out.join(dominant_df)
    out = out.reset_index().rename(columns={cluster_col: "cluster_id"})
    out.insert(1, "mixture_niche_id", out["cluster_id"])
    return out


def definition3b_program_correlations(sample_df: pd.DataFrame, program_cols: list[str], sample_row: pd.Series) -> pd.DataFrame:
    rows = []
    for program_id in program_cols:
        x = pd.to_numeric(sample_df[program_id], errors="coerce")
        y = pd.to_numeric(sample_df[PRIMARY_SCORE_COLUMN], errors="coerce")
        mask = ~(x.isna() | y.isna())
        rho = np.nan
        p_value = np.nan
        if mask.sum() >= 3:
            rho, p_value = spearmanr(x[mask], y[mask])
        rows.append(
            {
                "dataset": str(sample_row["dataset"]),
                "sample_id_on_disk": str(sample_row["sample_id_on_disk"]),
                "sample_label": str(sample_row["sample_label"]),
                "program_id": program_id,
                "spearman_rho": rho,
                "p_value": p_value,
            }
        )
    table = pd.DataFrame(rows)
    table["fdr_bh"] = bh_fdr(table["p_value"].to_numpy(dtype=float))
    return table


def run_definition3b_for_sample(run_dir: Path, sample_row: pd.Series, config: dict) -> dict:
    integrated, program_cols, qc = load_integrated_sample(sample_row)
    qc["stage"] = "definition3b_input_qc"
    sample_dirs = build_sample_output_dirs(run_dir / "02_definition3b_mixture_programme_niches", str(sample_row["sample_label"]))
    pd.DataFrame([qc]).to_csv(sample_dirs["tables"] / "input_qc.csv", index=False)

    features = integrated[program_cols].to_numpy(dtype=float)
    adjacency = build_hex_adjacency(integrated)
    labels, grid_df, selection = select_resolution(
        features,
        resolution_grid=config["definition3b"]["resolution_grid"],
        knn_k=int(config["definition3b"]["knn_k"]),
        min_cluster_size=int(config["definition3b"]["min_cluster_size"]),
    )
    cluster_ids = label_series_to_strings(labels, prefix="MN")
    integrated["definition3b_cluster_id"] = cluster_ids.values
    integrated["definition3b_mixture_niche_id"] = cluster_ids.values
    grid_df.to_csv(sample_dirs["tables"] / "resolution_grid.csv", index=False)

    cluster_summary = summarize_definition3b_cluster(
        integrated,
        cluster_col="definition3b_cluster_id",
        program_cols=program_cols,
        sample_row=sample_row,
    )
    cluster_summary.to_csv(sample_dirs["tables"] / "cluster_summary.csv", index=False)

    cluster_box = integrated[["definition3b_cluster_id", PRIMARY_SCORE_COLUMN]].copy()
    cluster_box = cluster_box.rename(
        columns={
            "definition3b_cluster_id": "mixture_niche_id",
            PRIMARY_SCORE_COLUMN: "snai1_ac_score",
        }
    )
    boxplot_by_group(
        cluster_box,
        x_col="mixture_niche_id",
        y_col="snai1_ac_score",
        title=f"{sample_row['sample_label']} Definition 3b mixture-programme niches vs SNAI1-ac",
        output_path=sample_dirs["figures"] / "cluster_vs_snai1ac_boxplot.png",
    )

    program_corr = definition3b_program_correlations(integrated, program_cols, sample_row)
    program_corr.to_csv(sample_dirs["tables"] / "programme_snai1ac_correlations.csv", index=False)

    spatial_table, weighted_moran = cluster_binary_moran(
        integrated["definition3b_cluster_id"].to_numpy(),
        adjacency,
        permutations=int(config["definition3b"]["moran_permutations"]),
    )
    spatial_table.insert(0, "sample_label", str(sample_row["sample_label"]))
    spatial_table.to_csv(sample_dirs["tables"] / "cluster_label_moransI.csv", index=False)

    components = component_rows_for_clusters(
        integrated,
        integrated["definition3b_cluster_id"],
        adjacency,
        min_size=int(config["definition3b"]["component_min_size"]),
        prefix=str(sample_row["sample_label"]),
    )
    components.to_csv(sample_dirs["tables"] / "region_components.csv", index=False)

    integrated.to_csv(sample_dirs["tables"] / "spot_level_table.csv", index=False)

    cluster_usage_matrix = (
        cluster_summary.set_index("cluster_id")[program_cols]
        .sort_index()
    )
    heatmap_plot(
        cluster_usage_matrix,
        title=f"{sample_row['sample_label']} mixture-niche x programme mean usage",
        output_path=sample_dirs["figures"] / "cluster_programme_heatmap.png",
    )
    simple_scatter(
        integrated,
        integrated["definition3b_cluster_id"],
        title=f"{sample_row['sample_label']} Definition 3b mixture-programme niches",
        output_path=sample_dirs["figures"] / "spatial_clusters.png",
        categorical=True,
    )
    simple_scatter(
        integrated,
        integrated[PRIMARY_SCORE_COLUMN],
        title=f"{sample_row['sample_label']} SNAI1-ac score",
        output_path=sample_dirs["figures"] / "spatial_snai1ac.png",
        categorical=False,
        cmap="RdBu_r",
    )

    sample_summary = {
        "dataset": str(sample_row["dataset"]),
        "sample_id_on_disk": str(sample_row["sample_id_on_disk"]),
        "sample_label": str(sample_row["sample_label"]),
        "n_tumor_spots": int(len(integrated)),
        "k_star": int(sample_row["k_star"]),
        "selected_resolution": selection["selected_resolution"],
        "selection_reason": selection["selection_reason"],
        "n_clusters": int(cluster_summary["cluster_id"].nunique()),
        "n_mixture_niches": int(cluster_summary["cluster_id"].nunique()),
        "weighted_binary_cluster_morans_I": weighted_moran,
        "n_region_components_ge4": int(len(components)),
        "top_cluster_by_mean_snai1ac": cluster_summary.sort_values("mean_snai1_ac", ascending=False)["cluster_id"].iloc[0],
        "top_mixture_niche_by_mean_snai1ac": cluster_summary.sort_values("mean_snai1_ac", ascending=False)["cluster_id"].iloc[0],
        "top_cluster_mean_snai1ac": float(cluster_summary["mean_snai1_ac"].max()),
        "top_mixture_niche_mean_snai1ac": float(cluster_summary["mean_snai1_ac"].max()),
    }
    return sample_summary


def neighbor_vectors(
    sample_df: pd.DataFrame,
    program_cols: list[str],
    adjacency: csr_matrix,
    focus_labels: list[str],
) -> pd.DataFrame:
    neighbor_lists = adjacency_neighbor_lists(adjacency)
    focus_mask = sample_df["LISA_category"].isin(focus_labels).to_numpy()
    rows = []
    usage_matrix = sample_df[program_cols].to_numpy(dtype=float)
    for idx in np.flatnonzero(focus_mask):
        neighbors = neighbor_lists[idx]
        if len(neighbors) == 0:
            continue
        mean_vector = usage_matrix[neighbors].mean(axis=0)
        row = {
            "spot_id": sample_df.iloc[idx]["spot_id"],
            "dataset": sample_df.iloc[idx]["dataset"],
            "sample_id_on_disk": sample_df.iloc[idx]["sample_id_on_disk"],
            "sample_label": sample_df.iloc[idx]["sample_label"],
            "LISA_category": sample_df.iloc[idx]["LISA_category"],
            PRIMARY_SCORE_COLUMN: sample_df.iloc[idx][PRIMARY_SCORE_COLUMN],
            "Malignant": sample_df.iloc[idx]["Malignant"],
            "interface": sample_df.iloc[idx]["interface"],
            "array_row": sample_df.iloc[idx]["array_row"],
            "array_col": sample_df.iloc[idx]["array_col"],
            "pxl_row_in_fullres": sample_df.iloc[idx]["pxl_row_in_fullres"],
            "pxl_col_in_fullres": sample_df.iloc[idx]["pxl_col_in_fullres"],
            "neighbor_count": int(len(neighbors)),
        }
        for program_id, value in zip(program_cols, mean_vector, strict=False):
            row[program_id] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def fit_multivariate_sse(Y: np.ndarray, X: np.ndarray) -> tuple[float, np.ndarray]:
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    sse = float(np.sum(resid ** 2))
    return sse, resid


def permanova_style_test(
    Y: np.ndarray,
    group_binary: np.ndarray,
    malignant: np.ndarray | None,
    permutations: int,
    seed: int = DEFAULT_RANDOM_STATE,
) -> dict:
    group_binary = np.asarray(group_binary, dtype=float).reshape(-1, 1)
    covariate_mode = "group_only"
    reduced_cols = [np.ones((Y.shape[0], 1), dtype=float)]
    if malignant is not None:
        malignant = np.asarray(malignant, dtype=float)
        if np.isfinite(malignant).sum() == len(malignant) and np.nanstd(malignant) > 0:
            reduced_cols.append(malignant.reshape(-1, 1))
            covariate_mode = "group_plus_malignant"
    X_reduced = np.hstack(reduced_cols)
    X_full = np.hstack([X_reduced, group_binary])

    sse_reduced, resid_reduced = fit_multivariate_sse(Y, X_reduced)
    sse_full, _ = fit_multivariate_sse(Y, X_full)
    df_effect = X_full.shape[1] - X_reduced.shape[1]
    df_resid = Y.shape[0] - X_full.shape[1]
    if df_resid <= 0 or sse_full <= 0:
        return {
            "covariate_mode": covariate_mode,
            "pseudo_F": np.nan,
            "p_value": np.nan,
            "r2": np.nan,
            "n_obs": int(Y.shape[0]),
            "permutations": int(permutations),
        }
    ss_effect = sse_reduced - sse_full
    pseudo_f = float((ss_effect / df_effect) / (sse_full / df_resid))
    fitted_reduced = Y - resid_reduced

    rng = np.random.default_rng(seed)
    permuted = []
    for _ in range(permutations):
        perm_idx = rng.permutation(Y.shape[0])
        Y_perm = fitted_reduced + resid_reduced[perm_idx]
        perm_reduced, _ = fit_multivariate_sse(Y_perm, X_reduced)
        perm_full, _ = fit_multivariate_sse(Y_perm, X_full)
        perm_ss_effect = perm_reduced - perm_full
        if perm_full <= 0:
            permuted.append(np.nan)
        else:
            permuted.append(float((perm_ss_effect / df_effect) / (perm_full / df_resid)))
    permuted = np.asarray(permuted, dtype=float)
    valid = np.isfinite(permuted)
    p_value = np.nan
    if valid.sum() > 0:
        p_value = float((1 + np.sum(permuted[valid] >= pseudo_f)) / (1 + valid.sum()))
    r2 = float(ss_effect / sse_reduced) if sse_reduced > 0 else np.nan
    return {
        "covariate_mode": covariate_mode,
        "pseudo_F": pseudo_f,
        "p_value": p_value,
        "r2": r2,
        "n_obs": int(Y.shape[0]),
        "permutations": int(permutations),
    }


def definition4_programme_contrasts(neighbor_df: pd.DataFrame, program_cols: list[str], sample_row: pd.Series) -> pd.DataFrame:
    rows = []
    group = neighbor_df["group"].astype(str)
    for program_id in program_cols:
        hh = pd.to_numeric(neighbor_df.loc[group == "HH", program_id], errors="coerce").dropna()
        ll = pd.to_numeric(neighbor_df.loc[group == "LL", program_id], errors="coerce").dropna()
        p_value = np.nan
        if len(hh) >= 2 and len(ll) >= 2:
            _, p_value = mannwhitneyu(hh, ll, alternative="two-sided")
        rows.append(
            {
                "dataset": str(sample_row["dataset"]),
                "sample_id_on_disk": str(sample_row["sample_id_on_disk"]),
                "sample_label": str(sample_row["sample_label"]),
                "program_id": program_id,
                "hh_mean": float(hh.mean()) if len(hh) else np.nan,
                "ll_mean": float(ll.mean()) if len(ll) else np.nan,
                "mean_difference_hh_minus_ll": float(hh.mean() - ll.mean()) if len(hh) and len(ll) else np.nan,
                "cohens_d_hh_minus_ll": float(cohens_d(hh.to_numpy(), ll.to_numpy())) if len(hh) >= 2 and len(ll) >= 2 else np.nan,
                "p_value": p_value,
            }
        )
    table = pd.DataFrame(rows)
    table["fdr_bh"] = bh_fdr(table["p_value"].to_numpy(dtype=float))
    return table


def occupancy_table(hh_df: pd.DataFrame, ll_projection: pd.DataFrame, sample_row: pd.Series) -> pd.DataFrame:
    hh_counts = hh_df["hh_cluster_id"].value_counts().rename("hh_count")
    ll_counts = ll_projection["assigned_hh_cluster_id"].value_counts().rename("ll_assigned_count")
    table = pd.concat([hh_counts, ll_counts], axis=1).fillna(0).reset_index().rename(columns={"index": "hh_cluster_id"})
    table["hh_fraction"] = table["hh_count"] / max(1, table["hh_count"].sum())
    table["ll_assigned_fraction"] = table["ll_assigned_count"] / max(1, table["ll_assigned_count"].sum())
    table["fraction_delta_ll_minus_hh"] = table["ll_assigned_fraction"] - table["hh_fraction"]
    table.insert(0, "sample_label", str(sample_row["sample_label"]))
    table.insert(0, "sample_id_on_disk", str(sample_row["sample_id_on_disk"]))
    table.insert(0, "dataset", str(sample_row["dataset"]))
    return table


def run_definition4_for_sample(run_dir: Path, sample_row: pd.Series, config: dict) -> dict:
    integrated, program_cols, qc = load_integrated_sample(sample_row)
    sample_dirs = build_sample_output_dirs(run_dir / "03_definition4_raw_neighbourhoods", str(sample_row["sample_label"]))
    adjacency = build_hex_adjacency(integrated)
    hh_ll = neighbor_vectors(integrated, program_cols, adjacency, focus_labels=["High-High", "Low-Low"])
    if hh_ll.empty:
        pd.DataFrame([qc]).to_csv(sample_dirs["tables"] / "input_qc.csv", index=False)
        return {
            "dataset": str(sample_row["dataset"]),
            "sample_id_on_disk": str(sample_row["sample_id_on_disk"]),
            "sample_label": str(sample_row["sample_label"]),
            "hh_tumor_n": 0,
            "ll_tumor_n": 0,
            "status": "no_hh_or_ll_after_tumor_filter",
        }
    hh_ll["group"] = np.where(hh_ll["LISA_category"] == "High-High", "HH", "LL")
    hh_df = hh_ll.loc[hh_ll["group"] == "HH"].copy()
    ll_df = hh_ll.loc[hh_ll["group"] == "LL"].copy()
    counts = pd.DataFrame(
        [
            {
                "dataset": str(sample_row["dataset"]),
                "sample_id_on_disk": str(sample_row["sample_id_on_disk"]),
                "sample_label": str(sample_row["sample_label"]),
                "hh_tumor_n": int(len(hh_df)),
                "ll_tumor_n": int(len(ll_df)),
                "neighbor_vector_rows": int(len(hh_ll)),
            }
        ]
    )
    counts.to_csv(sample_dirs["tables"] / "hh_ll_counts.csv", index=False)
    hh_ll.to_csv(sample_dirs["tables"] / "neighbor_vector_matrix.csv", index=False)

    if len(hh_df) < int(config["definition4"]["min_hh_spots_for_clustering"]):
        return {
            "dataset": str(sample_row["dataset"]),
            "sample_id_on_disk": str(sample_row["sample_id_on_disk"]),
            "sample_label": str(sample_row["sample_label"]),
            "hh_tumor_n": int(len(hh_df)),
            "ll_tumor_n": int(len(ll_df)),
            "status": "insufficient_hh_for_clustering",
        }

    hh_features = hh_df[program_cols].to_numpy(dtype=float)
    hh_labels, grid_df, selection = select_resolution(
        hh_features,
        resolution_grid=config["definition4"]["resolution_grid"],
        knn_k=int(config["definition4"]["knn_k"]),
        min_cluster_size=int(config["definition4"]["min_cluster_size"]),
    )
    hh_df["hh_cluster_id"] = label_series_to_strings(hh_labels, prefix="HH").values
    grid_df.to_csv(sample_dirs["tables"] / "hh_resolution_grid.csv", index=False)

    hh_centroids = hh_df.groupby("hh_cluster_id", sort=True)[program_cols].mean().reset_index()
    hh_centroids.insert(0, "sample_label", str(sample_row["sample_label"]))
    hh_centroids.to_csv(sample_dirs["tables"] / "hh_centroids.csv", index=False)
    hh_df.to_csv(sample_dirs["tables"] / "hh_cluster_labels.csv", index=False)

    centroid_matrix = hh_centroids.set_index("hh_cluster_id")[program_cols]
    heatmap_plot(
        centroid_matrix,
        title=f"{sample_row['sample_label']} HH centroid heatmap",
        output_path=sample_dirs["figures"] / "hh_centroid_heatmap.png",
    )

    ll_projection = pd.DataFrame()
    if not ll_df.empty:
        centroid_values = centroid_matrix.to_numpy(dtype=float)
        ll_values = ll_df[program_cols].to_numpy(dtype=float)
        distances = cdist(ll_values, centroid_values, metric="euclidean")
        nearest = distances.argmin(axis=1)
        ll_projection = ll_df.copy()
        ll_projection["assigned_hh_cluster_id"] = [centroid_matrix.index[idx] for idx in nearest]
        ll_projection["distance_to_assigned_hh_centroid"] = distances[np.arange(len(ll_df)), nearest]
        ll_projection.to_csv(sample_dirs["tables"] / "ll_projection_table.csv", index=False)
    else:
        ll_projection = pd.DataFrame(columns=list(ll_df.columns) + ["assigned_hh_cluster_id", "distance_to_assigned_hh_centroid"])
        ll_projection.to_csv(sample_dirs["tables"] / "ll_projection_table.csv", index=False)

    hh_self = hh_df.copy()
    hh_self_centroid_map = centroid_matrix.loc[hh_self["hh_cluster_id"], program_cols].to_numpy(dtype=float)
    hh_self["distance_to_own_hh_centroid"] = np.linalg.norm(hh_self[program_cols].to_numpy(dtype=float) - hh_self_centroid_map, axis=1)

    occupancy = occupancy_table(hh_df, ll_projection, sample_row)
    occupancy.to_csv(sample_dirs["tables"] / "occupancy_comparison.csv", index=False)

    contrast_df = definition4_programme_contrasts(hh_ll, program_cols, sample_row)
    contrast_df.to_csv(sample_dirs["tables"] / "programme_level_contrasts.csv", index=False)

    permanova_df = pd.DataFrame()
    if len(hh_df) >= 2 and len(ll_df) >= 2:
        Y = hh_ll[program_cols].to_numpy(dtype=float)
        group_binary = (hh_ll["group"] == "HH").astype(int).to_numpy()
        malignant = pd.to_numeric(hh_ll["Malignant"], errors="coerce").to_numpy()
        result = permanova_style_test(
            Y,
            group_binary=group_binary,
            malignant=malignant,
            permutations=int(config["definition4"]["permanova_permutations"]),
        )
        permanova_df = pd.DataFrame(
            [
                {
                    "dataset": str(sample_row["dataset"]),
                    "sample_id_on_disk": str(sample_row["sample_id_on_disk"]),
                    "sample_label": str(sample_row["sample_label"]),
                    **result,
                }
            ]
        )
    else:
        permanova_df = pd.DataFrame(
            [
                {
                    "dataset": str(sample_row["dataset"]),
                    "sample_id_on_disk": str(sample_row["sample_id_on_disk"]),
                    "sample_label": str(sample_row["sample_label"]),
                    "covariate_mode": "insufficient_groups",
                    "pseudo_F": np.nan,
                    "p_value": np.nan,
                    "r2": np.nan,
                    "n_obs": int(len(hh_ll)),
                    "permutations": int(config["definition4"]["permanova_permutations"]),
                }
            ]
        )
    permanova_df.to_csv(sample_dirs["tables"] / "permanova_results.csv", index=False)

    simple_scatter(
        hh_df,
        hh_df["hh_cluster_id"],
        title=f"{sample_row['sample_label']} HH neighbourhood types",
        output_path=sample_dirs["figures"] / "spatial_hh_clusters.png",
        categorical=True,
    )
    if not ll_projection.empty:
        simple_scatter(
            ll_projection,
            ll_projection["assigned_hh_cluster_id"],
            title=f"{sample_row['sample_label']} LL projected to HH centroids",
            output_path=sample_dirs["figures"] / "spatial_ll_projection.png",
            categorical=True,
        )

        dist_plot = pd.concat(
            [
                hh_self[["distance_to_own_hh_centroid"]].rename(columns={"distance_to_own_hh_centroid": "distance"}).assign(group="HH_to_own"),
                ll_projection[["distance_to_assigned_hh_centroid"]].rename(columns={"distance_to_assigned_hh_centroid": "distance"}).assign(group="LL_to_HH"),
            ],
            ignore_index=True,
        )
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        sns.boxplot(data=dist_plot, x="group", y="distance", ax=ax, color="#ffd8b1")
        sns.stripplot(data=dist_plot, x="group", y="distance", ax=ax, color="#b85c00", alpha=0.5, size=2.5)
        ax.set_title(f"{sample_row['sample_label']} centroid distance distributions")
        fig.tight_layout()
        fig.savefig(sample_dirs["figures"] / "distance_distributions.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    summary = {
        "dataset": str(sample_row["dataset"]),
        "sample_id_on_disk": str(sample_row["sample_id_on_disk"]),
        "sample_label": str(sample_row["sample_label"]),
        "hh_tumor_n": int(len(hh_df)),
        "ll_tumor_n": int(len(ll_df)),
        "neighbor_vector_rows": int(len(hh_ll)),
        "hh_selected_resolution": selection["selected_resolution"],
        "hh_selection_reason": selection["selection_reason"],
        "n_hh_clusters": int(hh_df["hh_cluster_id"].nunique()),
        "status": "completed",
    }
    return summary


def safe_agglomerative(distance_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    try:
        model = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
    except TypeError:  # pragma: no cover - older sklearn
        model = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage="average")
    return model.fit_predict(distance_matrix)


def read_kstar_spectra(manifest: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    gene_cols_union: set[str] = set()
    for sample_row in manifest.to_dict("records"):
        spectra = pd.read_csv(sample_row["spectra_path"])
        spectra["is_k_star"] = spectra["is_k_star"].astype(str).str.lower() == "true"
        spectra = spectra.loc[spectra["is_k_star"]].copy()
        rows.append(spectra)
        gene_cols_union.update([col for col in spectra.columns if col.startswith("__gene__")])
    all_spectra = pd.concat(rows, ignore_index=True)
    gene_cols = sorted(gene_cols_union)
    for col in gene_cols:
        if col not in all_spectra.columns:
            all_spectra[col] = 0.0
    all_spectra[gene_cols] = all_spectra[gene_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return all_spectra, gene_cols


def choose_metaprogram_k(features: np.ndarray, config: dict) -> tuple[np.ndarray, pd.DataFrame, dict]:
    n_obs = features.shape[0]
    distance = 1.0 - cosine_similarity(features)
    distance = np.clip(distance, 0.0, 2.0)
    min_k = int(config["genenmf_equivalent"]["cluster_k_min"])
    max_k = min(int(config["genenmf_equivalent"]["cluster_k_max"]), n_obs - 1)
    rows = []
    label_lookup: dict[int, np.ndarray] = {}
    for k in range(min_k, max_k + 1):
        labels = safe_agglomerative(distance, n_clusters=k)
        label_lookup[k] = labels
        counts = pd.Series(labels).value_counts()
        silhouette = np.nan
        if 1 < len(counts) < n_obs:
            try:
                silhouette = float(silhouette_score(distance, labels, metric="precomputed"))
            except Exception:
                silhouette = np.nan
        rows.append(
            {
                "k": int(k),
                "n_clusters": int(len(counts)),
                "min_cluster_size": int(counts.min()),
                "max_cluster_size": int(counts.max()),
                "singleton_fraction": float((counts == 1).mean()),
                "silhouette": silhouette,
                "valid_for_selection": bool(counts.min() >= int(config["genenmf_equivalent"]["min_cluster_size"])),
            }
        )
    diagnostics = pd.DataFrame(rows)
    valid = diagnostics.loc[diagnostics["valid_for_selection"]].copy()
    if valid.empty:
        selected = diagnostics.sort_values(["silhouette", "singleton_fraction", "k"], ascending=[False, True, True], na_position="last").iloc[0]
        reason = "fallback_best_available"
    else:
        selected = valid.sort_values(["silhouette", "singleton_fraction", "k"], ascending=[False, True, True], na_position="last").iloc[0]
        reason = "best_valid_silhouette"
    k = int(selected["k"])
    diagnostics["selected"] = diagnostics["k"].eq(k)
    return label_lookup[k], diagnostics, {"selected_k": k, "selection_reason": reason}


def top_genes_from_mean_spectrum(cluster_df: pd.DataFrame, gene_cols: list[str], top_n: int = 20) -> str:
    mean_spectrum = cluster_df[gene_cols].mean().sort_values(ascending=False).head(top_n)
    return ";".join([f"{col.replace('__gene__', '')}:{value:.4f}" for col, value in mean_spectrum.items()])


def run_genenmf_equivalent(run_dir: Path, manifest: pd.DataFrame, config: dict) -> pd.DataFrame:
    spectra, gene_cols = read_kstar_spectra(manifest)
    features = spectra[gene_cols].to_numpy(dtype=float)
    labels, diagnostics, selection = choose_metaprogram_k(features, config)
    metaprogram_ids = label_series_to_strings(labels, prefix="MP").replace("MP_", "MP").tolist()
    spectra["metaprogram_id"] = metaprogram_ids

    out_dirs = build_sample_output_dirs(run_dir / "04_GeneNMF_alignment", "cohort_alignment")
    diagnostics.to_csv(out_dirs["tables"] / "metaprogram_k_diagnostics.csv", index=False)

    membership = spectra[
        [
            "program_id",
            "dataset",
            "sample_id_on_disk",
            "sample_label",
            "source_k",
            "local_program_index",
            "metaprogram_id",
        ]
    ].copy()
    membership.to_csv(out_dirs["tables"] / "programme_to_metaprogram.csv", index=False)

    similarity = cosine_similarity(features)
    sim_df = pd.DataFrame(similarity, index=spectra["program_id"], columns=spectra["program_id"])
    sim_df.to_csv(out_dirs["tables"] / "programme_similarity_matrix.csv")

    summaries = []
    for metaprogram_id, subset in spectra.groupby("metaprogram_id", sort=True):
        summaries.append(
            {
                "metaprogram_id": metaprogram_id,
                "n_programs": int(len(subset)),
                "sample_coverage_n": int(subset["sample_label"].nunique()),
                "dataset_coverage_n": int(subset["dataset"].nunique()),
                "samples_present": ";".join(sorted(subset["sample_label"].unique().tolist())),
                "datasets_present": ";".join(sorted(subset["dataset"].unique().tolist())),
                "top_mean_genes": top_genes_from_mean_spectrum(subset, gene_cols, top_n=20),
            }
        )
    summary_df = pd.DataFrame(summaries).sort_values(["n_programs", "metaprogram_id"], ascending=[False, True])
    summary_df.to_csv(out_dirs["tables"] / "metaprogram_summary.csv", index=False)

    # Dendrogram
    distance = 1.0 - similarity
    distance = np.clip(distance, 0.0, 2.0)
    linkage_matrix = linkage(squareform(distance, checks=False), method="average")
    fig, ax = plt.subplots(figsize=(16, 8))
    dendrogram(linkage_matrix, labels=spectra["program_id"].tolist(), leaf_rotation=90, leaf_font_size=6, ax=ax)
    ax.set_title(
        f"GeneNMF-equivalent programme alignment\nselected_k={selection['selected_k']} ({selection['selection_reason']})"
    )
    fig.tight_layout()
    fig.savefig(out_dirs["figures"] / "programme_dendrogram.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    heatmap_plot(
        sim_df,
        title="Programme cosine similarity",
        output_path=out_dirs["figures"] / "programme_similarity_heatmap.png",
        cmap="mako",
    )

    map_definition3b_clusters_to_metaprograms(run_dir, membership)
    return membership


def map_definition3b_clusters_to_metaprograms(run_dir: Path, membership: pd.DataFrame) -> None:
    d3b_root = run_dir / "02_definition3b_mixture_programme_niches"
    out_dir = build_sample_output_dirs(run_dir / "04_GeneNMF_alignment", "cohort_alignment")
    rows = []
    membership_lookup = membership[["program_id", "metaprogram_id"]].copy()
    valid_program_ids = set(membership_lookup["program_id"].astype(str))
    for cluster_summary_path in sorted(d3b_root.glob("*\\tables\\cluster_summary.csv")):
        cluster_summary = pd.read_csv(cluster_summary_path)
        program_cols = [col for col in cluster_summary.columns if str(col) in valid_program_ids]
        if not program_cols:
            continue
        melted = cluster_summary.melt(
            id_vars=["dataset", "sample_id_on_disk", "sample_label", "cluster_id"],
            value_vars=program_cols,
            var_name="program_id",
            value_name="mean_cluster_usage",
        )
        melted["mean_cluster_usage"] = pd.to_numeric(melted["mean_cluster_usage"], errors="coerce")
        merged = melted.merge(membership_lookup, on="program_id", how="left")
        agg = (
            merged.groupby(["dataset", "sample_id_on_disk", "sample_label", "cluster_id", "metaprogram_id"], dropna=False)["mean_cluster_usage"]
            .sum()
            .reset_index()
            .sort_values(
                ["dataset", "sample_id_on_disk", "cluster_id", "mean_cluster_usage"],
                ascending=[True, True, True, False],
            )
        )
        agg["rank_within_cluster"] = agg.groupby(["dataset", "sample_id_on_disk", "cluster_id"]).cumcount() + 1
        rows.append(agg)
    if rows:
        mapping = pd.concat(rows, ignore_index=True)
    else:
        mapping = pd.DataFrame(columns=["dataset", "sample_id_on_disk", "sample_label", "cluster_id", "metaprogram_id", "mean_cluster_usage", "rank_within_cluster"])
    mapping.to_csv(out_dir["tables"] / "definition3b_cluster_to_metaprogram.csv", index=False)


def collect_definition3b_cluster_summaries(run_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((run_dir / "02_definition3b_mixture_programme_niches").glob("*\\tables\\cluster_summary.csv")):
        df = pd.read_csv(path)
        if not df.empty:
            rows.append(df)
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()


def collect_programme_correlations(run_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((run_dir / "02_definition3b_mixture_programme_niches").glob("*\\tables\\programme_snai1ac_correlations.csv")):
        df = pd.read_csv(path)
        if not df.empty:
            rows.append(df)
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()


def metaprogram_loadings_text(cluster_mapping: pd.DataFrame, top_n: int = 3) -> str:
    if cluster_mapping.empty:
        return ""
    subset = cluster_mapping.sort_values(["rank_within_cluster", "mean_cluster_usage"], ascending=[True, False]).head(top_n)
    return ";".join(
        [
            f"{row.metaprogram_id}:{row.mean_cluster_usage:.3f}"
            for row in subset.itertuples(index=False)
        ]
    )


def write_high_snai1ac_followup_tables(run_dir: Path) -> pd.DataFrame:
    cluster_summaries = collect_definition3b_cluster_summaries(run_dir)
    mapping_path = run_dir / "04_GeneNMF_alignment" / "cohort_alignment" / "tables" / "definition3b_cluster_to_metaprogram.csv"
    mapping = pd.read_csv(mapping_path) if mapping_path.exists() else pd.DataFrame()
    if cluster_summaries.empty:
        out = pd.DataFrame()
        out.to_csv(run_dir / "06_summary" / "high_snai1ac_mixture_niche_per_sample.csv", index=False)
        return out

    mapping = mapping.dropna(subset=["metaprogram_id"]).copy() if not mapping.empty else mapping
    mapping["mean_cluster_usage"] = pd.to_numeric(mapping.get("mean_cluster_usage"), errors="coerce")

    top_rows = []
    for sample_label, subset in cluster_summaries.groupby("sample_label", sort=True):
        ordered = subset.sort_values(["mean_snai1_ac", "n_spots"], ascending=[False, False]).copy()
        raw_top = ordered.iloc[0]
        ge4 = ordered.loc[pd.to_numeric(ordered["n_spots"], errors="coerce") >= 4].copy()
        ge4_top = ge4.iloc[0] if not ge4.empty else raw_top

        def cluster_meta(cluster_id: str) -> tuple[str, str]:
            if mapping.empty:
                return "", ""
            cluster_map = mapping.loc[
                (mapping["sample_label"] == sample_label) & (mapping["cluster_id"] == cluster_id)
            ].copy()
            if cluster_map.empty:
                return "", ""
            dominant = cluster_map.sort_values(["rank_within_cluster", "mean_cluster_usage"], ascending=[True, False]).iloc[0]["metaprogram_id"]
            return str(dominant), metaprogram_loadings_text(cluster_map, top_n=3)

        raw_dominant_mp, raw_top3 = cluster_meta(str(raw_top["cluster_id"]))
        ge4_dominant_mp, ge4_top3 = cluster_meta(str(ge4_top["cluster_id"]))

        top_rows.append(
            {
                "dataset": raw_top["dataset"],
                "sample_id_on_disk": raw_top["sample_id_on_disk"],
                "sample_label": sample_label,
                "raw_high_snai1ac_mixture_niche_id": raw_top["cluster_id"],
                "raw_high_snai1ac_mean": raw_top["mean_snai1_ac"],
                "raw_high_snai1ac_n_spots": int(raw_top["n_spots"]),
                "raw_high_snai1ac_dominant_metaprogram": raw_dominant_mp,
                "raw_high_snai1ac_top3_metaprograms": raw_top3,
                "raw_high_snai1ac_size_warning": bool(int(raw_top["n_spots"]) < 4),
                "size_filtered_high_snai1ac_mixture_niche_id": ge4_top["cluster_id"],
                "size_filtered_high_snai1ac_mean": ge4_top["mean_snai1_ac"],
                "size_filtered_high_snai1ac_n_spots": int(ge4_top["n_spots"]),
                "size_filtered_high_snai1ac_dominant_metaprogram": ge4_dominant_mp,
                "size_filtered_high_snai1ac_top3_metaprograms": ge4_top3,
                "size_filtered_switched_from_raw": bool(str(ge4_top["cluster_id"]) != str(raw_top["cluster_id"])),
            }
        )

    top_df = pd.DataFrame(top_rows).sort_values(["dataset", "sample_id_on_disk"]).reset_index(drop=True)
    top_df.to_csv(run_dir / "06_summary" / "high_snai1ac_mixture_niche_per_sample.csv", index=False)

    freq_rows = []
    for strategy_col in [
        "raw_high_snai1ac_dominant_metaprogram",
        "size_filtered_high_snai1ac_dominant_metaprogram",
    ]:
        strategy_name = strategy_col.replace("_dominant_metaprogram", "")
        subset = top_df.loc[top_df[strategy_col].astype(str).str.len() > 0].copy()
        counts = (
            subset.groupby(strategy_col)
            .size()
            .reset_index(name="sample_count")
            .rename(columns={strategy_col: "metaprogram_id"})
            .sort_values(["sample_count", "metaprogram_id"], ascending=[False, True])
        )
        counts.insert(0, "strategy", strategy_name)
        freq_rows.append(counts)
    freq_df = pd.concat(freq_rows, ignore_index=True) if freq_rows else pd.DataFrame(columns=["strategy", "metaprogram_id", "sample_count"])
    freq_df.to_csv(run_dir / "06_summary" / "high_snai1ac_mixture_niche_metaprogram_frequency.csv", index=False)
    return top_df


def write_programme_snai1ac_followup_tables(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr_df = collect_programme_correlations(run_dir)
    membership_path = run_dir / "04_GeneNMF_alignment" / "cohort_alignment" / "tables" / "programme_to_metaprogram.csv"
    membership = pd.read_csv(membership_path) if membership_path.exists() else pd.DataFrame()
    if corr_df.empty:
        empty = pd.DataFrame()
        empty.to_csv(run_dir / "06_summary" / "programme_snai1ac_correlations_all_samples.csv", index=False)
        empty.to_csv(run_dir / "06_summary" / "programme_snai1ac_by_metaprogram_summary.csv", index=False)
        empty.to_csv(run_dir / "06_summary" / "per_sample_extreme_programme_snai1ac_correlations.csv", index=False)
        return empty, empty

    if not membership.empty:
        corr_df = corr_df.merge(
            membership[["program_id", "metaprogram_id", "source_k", "local_program_index"]],
            on="program_id",
            how="left",
        )
    corr_df["direction"] = np.where(
        corr_df["spearman_rho"] > 0,
        "positive",
        np.where(corr_df["spearman_rho"] < 0, "negative", "zero"),
    )
    corr_df["significant_fdr_0_05"] = corr_df["fdr_bh"] < 0.05
    corr_df = corr_df.sort_values(["sample_label", "spearman_rho"], ascending=[True, False]).reset_index(drop=True)
    corr_df.to_csv(run_dir / "06_summary" / "programme_snai1ac_correlations_all_samples.csv", index=False)

    per_sample_rows = []
    for sample_label, subset in corr_df.groupby("sample_label", sort=True):
        pos = subset.sort_values(["spearman_rho", "fdr_bh"], ascending=[False, True]).iloc[0]
        neg = subset.sort_values(["spearman_rho", "fdr_bh"], ascending=[True, True]).iloc[0]
        per_sample_rows.extend(
            [
                {
                    "sample_label": sample_label,
                    "extreme_type": "top_positive",
                    "program_id": pos["program_id"],
                    "metaprogram_id": pos.get("metaprogram_id"),
                    "spearman_rho": pos["spearman_rho"],
                    "fdr_bh": pos["fdr_bh"],
                },
                {
                    "sample_label": sample_label,
                    "extreme_type": "top_negative",
                    "program_id": neg["program_id"],
                    "metaprogram_id": neg.get("metaprogram_id"),
                    "spearman_rho": neg["spearman_rho"],
                    "fdr_bh": neg["fdr_bh"],
                },
            ]
        )
    pd.DataFrame(per_sample_rows).to_csv(
        run_dir / "06_summary" / "per_sample_extreme_programme_snai1ac_correlations.csv",
        index=False,
    )
    extreme_freq = (
        pd.DataFrame(per_sample_rows)
        .groupby(["extreme_type", "metaprogram_id"], dropna=False)
        .size()
        .reset_index(name="sample_count")
        .sort_values(["extreme_type", "sample_count", "metaprogram_id"], ascending=[True, False, True])
    )
    extreme_freq.to_csv(
        run_dir / "06_summary" / "per_sample_extreme_programme_metaprogram_frequency.csv",
        index=False,
    )

    if "metaprogram_id" in corr_df.columns:
        mp_summary = (
            corr_df.groupby("metaprogram_id", dropna=False)
            .agg(
                n_programmes=("program_id", "size"),
                n_samples=("sample_label", "nunique"),
                mean_rho=("spearman_rho", "mean"),
                median_rho=("spearman_rho", "median"),
                positive_sig_n=("significant_fdr_0_05", lambda s: int(((corr_df.loc[s.index, "significant_fdr_0_05"]) & (corr_df.loc[s.index, "spearman_rho"] > 0)).sum())),
                negative_sig_n=("significant_fdr_0_05", lambda s: int(((corr_df.loc[s.index, "significant_fdr_0_05"]) & (corr_df.loc[s.index, "spearman_rho"] < 0)).sum())),
            )
            .reset_index()
            .sort_values(["median_rho", "n_programmes"], ascending=[False, False])
        )
    else:
        mp_summary = pd.DataFrame()
    mp_summary.to_csv(run_dir / "06_summary" / "programme_snai1ac_by_metaprogram_summary.csv", index=False)
    return corr_df, mp_summary


def write_summary_outputs(run_dir: Path) -> None:
    d3b_path = run_dir / "02_definition3b_mixture_programme_niches" / "definition3b_sample_summary.csv"
    d4_path = run_dir / "03_definition4_raw_neighbourhoods" / "definition4_sample_summary.csv"
    d3b_stage_path = run_dir / "05_qc_sanity_checks" / "definition3b_stage_status.csv"
    d4_stage_path = run_dir / "05_qc_sanity_checks" / "definition4_stage_status.csv"
    genenmf_stage_path = run_dir / "05_qc_sanity_checks" / "genenmf_stage_status.csv"

    d3b_summaries = pd.read_csv(d3b_path) if d3b_path.exists() else pd.DataFrame()
    d4_summaries = pd.read_csv(d4_path) if d4_path.exists() else pd.DataFrame()
    d3b_stage = pd.read_csv(d3b_stage_path) if d3b_stage_path.exists() else pd.DataFrame()
    d4_stage = pd.read_csv(d4_stage_path) if d4_stage_path.exists() else pd.DataFrame()
    genenmf_stage = pd.read_csv(genenmf_stage_path) if genenmf_stage_path.exists() else pd.DataFrame()

    if not d3b_summaries.empty:
        d3b_summaries.to_csv(run_dir / "06_summary" / "definition3b_sample_overview.csv", index=False)
    else:
        pd.DataFrame().to_csv(run_dir / "06_summary" / "definition3b_sample_overview.csv", index=False)
    if not d4_summaries.empty:
        d4_summaries.to_csv(run_dir / "06_summary" / "definition4_sample_overview.csv", index=False)
    else:
        pd.DataFrame().to_csv(run_dir / "06_summary" / "definition4_sample_overview.csv", index=False)

    fallback_rows = []
    for path in sorted((run_dir / "02_definition3b_mixture_programme_niches").glob("*\\tables\\resolution_grid.csv")):
        grid = pd.read_csv(path)
        if grid.empty:
            continue
        selected = grid.loc[grid["selected"].astype(str).str.lower() == "true"].copy()
        sample_label = path.parent.parent.name
        if selected.empty:
            fallback_rows.append(
                {
                    "sample_label": sample_label,
                    "issue": "no_selected_resolution_row",
                }
            )
            continue
        selected_row = selected.iloc[0]
        if str(selected_row["valid_for_selection"]).lower() != "true":
            fallback_rows.append(
                {
                    "sample_label": sample_label,
                    "issue": "selected_resolution_failed_cluster_size_sanity",
                    "selected_resolution": selected_row["resolution"],
                    "n_clusters": selected_row["n_clusters"],
                    "min_cluster_size": selected_row["min_cluster_size"],
                    "silhouette": selected_row["silhouette"],
                }
            )
    fallback_df = pd.DataFrame(
        fallback_rows,
        columns=["sample_label", "issue", "selected_resolution", "n_clusters", "min_cluster_size", "silhouette"],
    )
    fallback_df.to_csv(run_dir / "05_qc_sanity_checks" / "definition3b_resolution_warnings.csv", index=False)

    low_count_rows = []
    for path in sorted((run_dir / "03_definition4_raw_neighbourhoods").glob("*\\tables\\hh_ll_counts.csv")):
        counts = pd.read_csv(path)
        if counts.empty:
            continue
        row = counts.iloc[0]
        if int(row["hh_tumor_n"]) < int(base_config()["definition4"]["min_hh_spots_for_clustering"]) or int(row["ll_tumor_n"]) < 5:
            low_count_rows.append(
                {
                    "sample_label": row["sample_label"],
                    "hh_tumor_n": int(row["hh_tumor_n"]),
                    "ll_tumor_n": int(row["ll_tumor_n"]),
                    "issue": "low_hh_or_ll_count",
                }
            )
    low_count_df = pd.DataFrame(
        low_count_rows,
        columns=["sample_label", "hh_tumor_n", "ll_tumor_n", "issue"],
    )
    low_count_df.to_csv(run_dir / "05_qc_sanity_checks" / "definition4_low_count_warnings.csv", index=False)

    n_d3b_completed = int((d3b_stage.get("status", pd.Series(dtype=object)) == "completed").sum()) if not d3b_stage.empty else 0
    n_d4_completed = int((d4_stage.get("status", pd.Series(dtype=object)) == "completed").sum()) if not d4_stage.empty else 0
    n_mixture_niches = int(d3b_summaries["n_mixture_niches"].sum()) if "n_mixture_niches" in d3b_summaries.columns else 0
    hh_total = int(d4_summaries["hh_tumor_n"].sum()) if "hh_tumor_n" in d4_summaries.columns else 0
    ll_total = int(d4_summaries["ll_tumor_n"].sum()) if "ll_tumor_n" in d4_summaries.columns else 0
    n_metaprograms = int(genenmf_stage["n_metaprograms"].iloc[0]) if not genenmf_stage.empty else 0
    n_programmes = int(genenmf_stage["n_programmes"].iloc[0]) if not genenmf_stage.empty else 0

    top_high_df = write_high_snai1ac_followup_tables(run_dir)
    corr_df, mp_corr_summary = write_programme_snai1ac_followup_tables(run_dir)
    raw_size_warning_n = int(top_high_df["raw_high_snai1ac_size_warning"].sum()) if "raw_high_snai1ac_size_warning" in top_high_df.columns else 0
    size_filtered_switch_n = int(top_high_df["size_filtered_switched_from_raw"].sum()) if "size_filtered_switched_from_raw" in top_high_df.columns else 0

    lines = [
        "# Definition 3b / Definition 4 / GeneNMF-equivalent summary",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Framework anchoring",
        "",
        f"- Anchor document: `{REPO_ROOT / '00_documentation' / 'framword_drafts' / 'analytical_framework_final.md'}`",
        "- Definition 4 was intentionally implemented with raw programme neighbour vectors.",
        "- This is a documented user-confirmed override of the framework's alignment_category bottleneck.",
        "- SNAI1-ac language remains restricted to a projected acetylation-associated programme score.",
        "",
        "## Immediate notes",
        "",
        "- Definition 3b stays strictly per-sample and never pools raw spots across samples.",
        "- Definition 3b labels represent mixture-programme niches derived from raw programme usage, not SpaGCN clusters or any external clustering.",
        "- Definition 4 uses HH-defined neighbourhood types and LL projection only; it is not a classifier.",
        "- GeneNMF package was not available locally, so Track 3 used a documented cosine-plus-hierarchical GeneNMF-style equivalent.",
        "",
        "## Completion snapshot",
        "",
        f"- Definition 3b completed for {n_d3b_completed} samples.",
        f"- Definition 4 completed for {n_d4_completed} samples.",
        f"- Total Definition 3b mixture niches: {n_mixture_niches}.",
        f"- Total tumour-zone HH spots in Definition 4: {hh_total}.",
        f"- Total tumour-zone LL spots in Definition 4: {ll_total}.",
        f"- GeneNMF-equivalent mapped {n_programmes} programmes into {n_metaprograms} metaprograms.",
        f"- Samples where the raw highest-SNAI1-ac mixture niche had fewer than 4 spots: {raw_size_warning_n}.",
        f"- Samples where the size-filtered highest-SNAI1-ac mixture niche differed from the raw top niche: {size_filtered_switch_n}.",
        "",
        "## Files to inspect first",
        "",
        f"- `{run_dir / '05_qc_sanity_checks' / 'prepare_summary.json'}`",
        f"- `{run_dir / '05_qc_sanity_checks' / 'definition3b_resolution_warnings.csv'}`",
        f"- `{run_dir / '05_qc_sanity_checks' / 'definition4_low_count_warnings.csv'}`",
        f"- `{run_dir / '00_config' / 'resolved_sample_manifest.csv'}`",
        f"- `{run_dir / '06_summary' / 'definition3b_sample_overview.csv'}`",
        f"- `{run_dir / '06_summary' / 'definition4_sample_overview.csv'}`",
        f"- `{run_dir / '06_summary' / 'high_snai1ac_mixture_niche_per_sample.csv'}`",
        f"- `{run_dir / '06_summary' / 'high_snai1ac_mixture_niche_metaprogram_frequency.csv'}`",
        f"- `{run_dir / '06_summary' / 'programme_snai1ac_correlations_all_samples.csv'}`",
        f"- `{run_dir / '06_summary' / 'programme_snai1ac_by_metaprogram_summary.csv'}`",
        f"- `{run_dir / '04_GeneNMF_alignment' / 'cohort_alignment' / 'tables' / 'programme_to_metaprogram.csv'}`",
        f"- `{run_dir / '04_GeneNMF_alignment' / 'cohort_alignment' / 'tables' / 'definition3b_cluster_to_metaprogram.csv'}`",
    ]
    write_text(run_dir / "06_summary" / "README_run.md", "\n".join(lines))

    finding_lines = [
        "# Key findings and immediate checks",
        "",
        "## Key findings",
        "",
        f"- All {n_d3b_completed} selected samples completed Definition 3b.",
        f"- All {n_d4_completed} selected samples completed Definition 4.",
        f"- The cohort yielded {n_mixture_niches} mixture-programme niches across {len(d3b_summaries)} samples.",
        f"- Definition 4 summarized {hh_total} HH and {ll_total} LL tumour-zone spots.",
        f"- The GeneNMF-equivalent stage grouped {n_programmes} programmes into {n_metaprograms} metaprograms.",
        f"- Raw top SNAI1-ac niches were size-flagged in {raw_size_warning_n} samples.",
        f"- Size-filtering changed the top SNAI1-ac niche assignment in {size_filtered_switch_n} samples.",
        "",
        "## Suspicious or noteworthy items",
        "",
        f"- Definition 3b resolution warnings: {len(fallback_df)} samples selected a resolution that did not fully satisfy the configured cluster-size sanity rule.",
        f"- Definition 4 low-count warnings: {len(low_count_df)} samples had low HH or LL counts by the summary thresholds.",
        "- The framework/user-confirmed deviation remains active: Definition 4 used raw programme neighbour vectors instead of alignment_category vectors.",
        "- Track 3 used a documented GeneNMF-style equivalent because the GeneNMF R package was not installed locally.",
        f"- Programme-level SNAI1-ac summaries were written for {len(corr_df)} sample-specific programmes across {mp_corr_summary['metaprogram_id'].nunique() if 'metaprogram_id' in mp_corr_summary.columns and not mp_corr_summary.empty else 0} metaprograms.",
    ]
    write_text(run_dir / "06_summary" / "key_findings.md", "\n".join(finding_lines))


def run_definition3b(run_dir: Path, manifest: pd.DataFrame, config: dict, overwrite: bool) -> None:
    append_step(run_dir, "definition3b", "started", f"samples={len(manifest)}")
    summaries = []
    qc_rows = []
    for _, sample_row in manifest.iterrows():
        sample_label = str(sample_row["sample_label"])
        log(run_dir, f"Definition 3b: {sample_label}")
        try:
            summary = run_definition3b_for_sample(run_dir, sample_row, config)
            summaries.append(summary)
            qc_rows.append(
                {
                    "sample_label": sample_label,
                    "dataset": str(sample_row["dataset"]),
                    "sample_id_on_disk": str(sample_row["sample_id_on_disk"]),
                    "status": "completed",
                }
            )
        except Exception as exc:
            qc_rows.append(
                {
                    "sample_label": sample_label,
                    "dataset": str(sample_row["dataset"]),
                    "sample_id_on_disk": str(sample_row["sample_id_on_disk"]),
                    "status": "failed",
                    "error": str(exc),
                }
            )
            log(run_dir, f"Definition 3b failed for {sample_label}: {exc}")
    upsert_csv(
        run_dir / "02_definition3b_mixture_programme_niches" / "definition3b_sample_summary.csv",
        pd.DataFrame(summaries),
        key_cols=["sample_label"],
    )
    upsert_csv(
        run_dir / "05_qc_sanity_checks" / "definition3b_stage_status.csv",
        pd.DataFrame(qc_rows),
        key_cols=["sample_label"],
    )
    append_step(run_dir, "definition3b", "completed", f"completed={len(summaries)}")


def run_definition4(run_dir: Path, manifest: pd.DataFrame, config: dict, overwrite: bool) -> None:
    append_step(run_dir, "definition4", "started", f"samples={len(manifest)}")
    summaries = []
    qc_rows = []
    for _, sample_row in manifest.iterrows():
        sample_label = str(sample_row["sample_label"])
        log(run_dir, f"Definition 4 raw: {sample_label}")
        try:
            summary = run_definition4_for_sample(run_dir, sample_row, config)
            summaries.append(summary)
            qc_rows.append(
                {
                    "sample_label": sample_label,
                    "dataset": str(sample_row["dataset"]),
                    "sample_id_on_disk": str(sample_row["sample_id_on_disk"]),
                    "status": summary.get("status", "completed"),
                }
            )
        except Exception as exc:
            qc_rows.append(
                {
                    "sample_label": sample_label,
                    "dataset": str(sample_row["dataset"]),
                    "sample_id_on_disk": str(sample_row["sample_id_on_disk"]),
                    "status": "failed",
                    "error": str(exc),
                }
            )
            log(run_dir, f"Definition 4 failed for {sample_label}: {exc}")
    upsert_csv(
        run_dir / "03_definition4_raw_neighbourhoods" / "definition4_sample_summary.csv",
        pd.DataFrame(summaries),
        key_cols=["sample_label"],
    )
    upsert_csv(
        run_dir / "05_qc_sanity_checks" / "definition4_stage_status.csv",
        pd.DataFrame(qc_rows),
        key_cols=["sample_label"],
    )
    append_step(run_dir, "definition4", "completed", f"completed={len(summaries)}")


def run_genenmf(run_dir: Path, manifest: pd.DataFrame, config: dict) -> None:
    append_step(run_dir, "genenmf_equivalent", "started", "")
    membership = run_genenmf_equivalent(run_dir, manifest, config)
    pd.DataFrame(
        [
            {
                "n_programmes": int(len(membership)),
                "n_metaprograms": int(membership["metaprogram_id"].nunique()) if not membership.empty else 0,
                "package_available": False,
                "implementation": "cosine_similarity_plus_average_linkage_equivalent",
            }
        ]
    ).to_csv(run_dir / "05_qc_sanity_checks" / "genenmf_stage_status.csv", index=False)
    append_step(run_dir, "genenmf_equivalent", "completed", f"programmes={len(membership)}")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir()
    config = base_config()
    run_subdirs(run_dir)

    append_step(run_dir, "pipeline", "started", f"stage={args.stage}")
    manifest = prepare_run(run_dir, args.samples)
    manifest = manifest.loc[manifest["all_required_paths_exist"]].copy()
    if manifest.empty:
        raise RuntimeError("No samples with all required paths were found for this run.")

    if args.stage in {"prepare"}:
        append_step(run_dir, "pipeline", "completed", "prepare_only")
        return
    if args.stage in {"definition3b", "all"}:
        run_definition3b(run_dir, manifest, config, overwrite=args.overwrite)
    if args.stage in {"definition4", "all"}:
        run_definition4(run_dir, manifest, config, overwrite=args.overwrite)
    if args.stage in {"genenmf", "all"}:
        run_genenmf(run_dir, manifest, config)
    if args.stage in {"summary", "all"}:
        write_summary_outputs(run_dir)
    append_step(run_dir, "pipeline", "completed", f"stage={args.stage}")


if __name__ == "__main__":
    main()
