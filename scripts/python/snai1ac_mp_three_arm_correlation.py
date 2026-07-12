from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import enrichmap as em
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr


ROOT = Path(r"D:\HGSOC_Spatial_Atlas")
MANUAL_DIR = (
    ROOT
    / "05_analysis_ready"
    / "S3_cNMF_Tumor_Programs"
    / "jaccard_raw_matrices"
    / "inspection_exports_average"
    / "variantB_nonjunk_manual_cut_v2"
)
SCORING_DIR = MANUAL_DIR / "subcluster_signatures_scoring"
OUT = MANUAL_DIR / "subcluster_snai1ac_correlation"
SCORED_DIR = SCORING_DIR / "scored_h5ad"
SUMMARY_CSV = SCORING_DIR / "manual_subcluster_enrichmap_score_summary.csv"
PREV_MANIFEST = SCORING_DIR / "manual_subcluster_scoring_manifest.json"
MP_SIGNATURES = SCORING_DIR / "signatures" / "manual_subcluster_recurrent_gene_signatures_long.csv"
SNAI_WEIGHTS = ROOT / "05_analysis_ready" / "Signature" / "snai1_ac_weights.json"
SNAI_CONFIG = ROOT / "05_analysis_ready" / "Signature" / "snai1_ac_config.yaml"
R_SCRIPT = Path(__file__).parent / "R" / "snai1ac_mp_meta_analysis.R"
RSCRIPT = Path(r"C:\Program Files\R\R-4.4.3\bin\Rscript.exe")

SNAI_COL = "SNAI1-ac_score"
MALIGNANT_COL = "Malignant"
COMPARTMENT_COL = "interface"
TUMOR_LABEL = "Tumor"
MIN_TUMOR_SPOTS = 50
RANDOM_SEED = 20260528


def stop(message: str) -> None:
    raise SystemExit(f"STOP: {message}")


def require(path: Path) -> None:
    if not path.exists():
        stop(f"Missing input: {path}")


def score_key(subcluster_id: str, label: str) -> str:
    return f"{subcluster_id}_{''.join(ch if ch.isalnum() else '_' for ch in label).strip('_')}"


def load_mp_table() -> pd.DataFrame:
    require(SUMMARY_CSV)
    summary = pd.read_csv(SUMMARY_CSV)
    needed = {"subcluster_id", "label", "score_key", "score_column"}
    if not needed.issubset(summary.columns):
        stop(f"{SUMMARY_CSV} lacks columns: {sorted(needed - set(summary.columns))}")
    mp = summary[list(needed)].drop_duplicates().sort_values("subcluster_id").reset_index(drop=True)
    if len(mp) != 8:
        stop(f"Expected 8 MP score columns; observed {len(mp)}")
    return mp


def scored_h5ads(sample_set: list[str]) -> list[tuple[str, str, Path]]:
    paths = []
    for sample_key in sample_set:
        if "__" not in sample_key:
            stop(f"Malformed sample_set entry in manifest: {sample_key}")
        dataset, sample = sample_key.split("__", 1)
        path = SCORED_DIR / dataset / f"{sample}.manual_jaccard_MP_scores.h5ad"
        require(path)
        paths.append((dataset, sample, path))
    return paths


def load_sample_set() -> list[str]:
    require(PREV_MANIFEST)
    manifest = json.loads(PREV_MANIFEST.read_text(encoding="utf-8"))
    sample_set = manifest.get("sample_set")
    if not isinstance(sample_set, list) or not sample_set:
        stop(f"{PREV_MANIFEST} lacks a non-empty sample_set")
    return sample_set


def validate_columns(paths: list[tuple[str, str, Path]], mp: pd.DataFrame) -> None:
    needed = [SNAI_COL, MALIGNANT_COL, COMPARTMENT_COL, *mp["score_column"].tolist()]
    for dataset, sample, path in paths:
        adata = sc.read_h5ad(path, backed="r")
        missing = [col for col in needed if col not in adata.obs.columns]
        if missing:
            stop(f"{dataset}__{sample} missing obs columns: {missing}")
        labels = set(adata.obs[COMPARTMENT_COL].astype(str).unique())
        if TUMOR_LABEL not in labels:
            stop(f"{dataset}__{sample} lacks tumour label {TUMOR_LABEL!r} in {COMPARTMENT_COL}; observed {sorted(labels)}")
        adata.file.close()


def finite_frame(adata, cols: list[str]) -> pd.DataFrame:
    frame = adata.obs[cols].copy()
    for col in cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna()


def residualize_ranked(values: np.ndarray, covariates: pd.DataFrame) -> np.ndarray:
    y = pd.Series(values).rank(method="average").to_numpy(dtype=float)
    cols = []
    for column in covariates.columns:
        x = pd.to_numeric(covariates[column], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(x).sum() != len(x) or np.nanstd(x) <= 1e-12:
            continue
        xr = pd.Series(x).rank(method="average").to_numpy(dtype=float)
        xr = (xr - xr.mean()) / (xr.std(ddof=0) or 1.0)
        cols.append(xr)
    if not cols:
        return y - y.mean()
    design = np.column_stack([np.ones(len(y)), *cols])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def partial_spearman(frame: pd.DataFrame, x_col: str, y_col: str, covar: str) -> float:
    part = frame[[x_col, y_col, covar]].dropna()
    if len(part) < 4:
        return math.nan
    xr = residualize_ranked(part[x_col].to_numpy(dtype=float), part[[covar]].reset_index(drop=True))
    yr = residualize_ranked(part[y_col].to_numpy(dtype=float), part[[covar]].reset_index(drop=True))
    if np.std(xr) <= 1e-12 or np.std(yr) <= 1e-12:
        return math.nan
    return float(pearsonr(xr, yr).statistic)


def spearman_value(frame: pd.DataFrame, x_col: str, y_col: str) -> float:
    part = frame[[x_col, y_col]].dropna()
    if len(part) < 4 or part[x_col].nunique() < 2 or part[y_col].nunique() < 2:
        return math.nan
    return float(spearmanr(part[x_col], part[y_col]).statistic)


def knn_connectivity(coords: np.ndarray, k: int = 6) -> tuple[int, float]:
    n = coords.shape[0]
    if n < 2:
        return int(n), 1.0 if n == 1 else math.nan
    kk = min(k + 1, n)
    _, idx = cKDTree(coords).query(coords, k=kk)
    idx = np.atleast_2d(idx)
    rows = np.repeat(np.arange(n), idx.shape[1] - 1)
    cols = idx[:, 1:].reshape(-1)
    graph = coo_matrix((np.ones_like(rows), (rows, cols)), shape=(n, n))
    graph = graph.maximum(graph.T).tocsr()
    n_comp, labels = connected_components(graph, directed=False)
    largest = int(np.bincount(labels).max()) if len(labels) else 0
    return int(n_comp), float(largest / n) if n else math.nan


def moran_i(adata, col: str) -> float:
    try:
        import libpysal
        from esda.moran import Moran

        coords = np.asarray(adata.obsm["spatial"])
        if coords.shape[0] < 7:
            return math.nan
        weights = libpysal.weights.KNN.from_array(coords, k=min(6, coords.shape[0] - 1))
        weights.transform = "R"
        return float(Moran(pd.to_numeric(adata.obs[col], errors="coerce").to_numpy(), weights).I)
    except Exception:
        return math.nan


def ensure_hires_alias(adata) -> None:
    if "spatial" not in adata.uns:
        return
    for library in adata.uns["spatial"].values():
        images = library.get("images", {})
        if "hires" not in images and "lowres" in images:
            images["hires"] = images["lowres"]


def load_signatures(mp: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, dict[str, float]]]:
    require(MP_SIGNATURES)
    require(SNAI_WEIGHTS)
    sigs = pd.read_csv(MP_SIGNATURES)
    needed = {"subcluster_id", "label", "gene"}
    if not needed.issubset(sigs.columns):
        stop(f"{MP_SIGNATURES} lacks columns: {sorted(needed - set(sigs.columns))}")
    mp_genes = {}
    for row in mp.itertuples(index=False):
        genes = sigs.loc[sigs["subcluster_id"].eq(row.subcluster_id), "gene"].astype(str).tolist()
        if not genes:
            stop(f"No genes found for {row.subcluster_id} in {MP_SIGNATURES}")
        mp_genes[row.score_key] = genes
    snai_weights = {str(g): float(w) for g, w in json.loads(SNAI_WEIGHTS.read_text(encoding="utf-8")).items()}
    if not snai_weights:
        stop(f"No weights found in {SNAI_WEIGHTS}")
    return mp_genes, {"SNAI1-ac": snai_weights}


def score_tumor_subset(adata, mp: pd.DataFrame, mp_genes: dict[str, list[str]], snai_weights: dict[str, dict[str, float]]):
    snai_present = {g: w for g, w in snai_weights["SNAI1-ac"].items() if g in adata.var_names}
    if not snai_present:
        stop("No SNAI1-ac signature genes present in tumour subset")
    em.tl.score(
        adata=adata,
        gene_set=list(snai_present.keys()),
        gene_weights={"SNAI1-ac": snai_present},
        score_key="SNAI1-ac",
        smoothing=True,
        correct_spatial_covariates=True,
        batch_key=None,
    )
    if SNAI_COL not in adata.obs:
        stop("Tumour-subset SNAI1-ac re-scoring did not create SNAI1-ac_score")
    for row in mp.itertuples(index=False):
        genes = [g for g in mp_genes[row.score_key] if g in adata.var_names]
        if not genes:
            stop(f"No MP genes present for {row.subcluster_id} {row.label}")
        em.tl.score(
            adata=adata,
            gene_set=genes,
            score_key=row.score_key,
            smoothing=True,
            correct_spatial_covariates=True,
            batch_key=None,
        )
        if row.score_column not in adata.obs:
            stop(f"Tumour-subset MP re-scoring did not create {row.score_column}")


def compute_correlations(paths: list[tuple[str, str, Path]], mp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mp_genes, snai_weights = load_signatures(mp)
    rows, diagnostics = [], []
    h5ad_out = OUT / "tumor_subset_scored_h5ad"
    h5ad_out.mkdir(parents=True, exist_ok=True)
    for dataset, sample, path in paths:
        print(f"Processing {dataset}__{sample}")
        adata = sc.read_h5ad(path)
        ensure_hires_alias(adata)
        all_cols = [SNAI_COL, MALIGNANT_COL, *mp["score_column"].tolist()]
        full = finite_frame(adata, all_cols)
        for row in mp.itertuples(index=False):
            raw_frame = full[[SNAI_COL, row.score_column]].dropna()
            part_frame = full[[SNAI_COL, row.score_column, MALIGNANT_COL]].dropna()
            rows.append(
                {
                    "arm": "arm1_raw_all_spots",
                    "MP_id": row.subcluster_id,
                    "MP_label": row.label,
                    "sample": sample,
                    "dataset": dataset,
                    "n_spots_used": int(len(raw_frame)),
                    "spearman_r": spearman_value(raw_frame, SNAI_COL, row.score_column),
                    "included_in_meta": True,
                    "exclusion_reason": "",
                }
            )
            rows.append(
                {
                    "arm": "arm2_partial_all_spots_malignant",
                    "MP_id": row.subcluster_id,
                    "MP_label": row.label,
                    "sample": sample,
                    "dataset": dataset,
                    "n_spots_used": int(len(part_frame)),
                    "spearman_r": partial_spearman(part_frame, SNAI_COL, row.score_column, MALIGNANT_COL),
                    "included_in_meta": True,
                    "exclusion_reason": "",
                }
            )

        tumor_mask = adata.obs[COMPARTMENT_COL].astype(str).eq(TUMOR_LABEL).to_numpy()
        n_tumor = int(tumor_mask.sum())
        diag = {
            "dataset": dataset,
            "sample": sample,
            "n_tumour_spots": n_tumor,
            "k": 6,
            "connected_components": math.nan,
            "largest_component_fraction": math.nan,
            "snai1ac_morans_I": math.nan,
            "included_in_arm3_meta": False,
            "exclusion_reason": "",
        }
        tumor = adata[tumor_mask].copy()
        try:
            if n_tumor == 0:
                raise ValueError("no tumour spots")
            diag["connected_components"], diag["largest_component_fraction"] = knn_connectivity(np.asarray(tumor.obsm["spatial"]), k=6)
            score_tumor_subset(tumor, mp, mp_genes, snai_weights)
            diag["snai1ac_morans_I"] = moran_i(tumor, SNAI_COL)
            out_path = h5ad_out / dataset / f"{sample}.tumor_subset_SNAI1ac_MP_scores.h5ad"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tumor.write_h5ad(out_path)
            diag["tumor_subset_h5ad"] = str(out_path)
            if n_tumor < MIN_TUMOR_SPOTS:
                diag["exclusion_reason"] = f"n_tumour_spots < {MIN_TUMOR_SPOTS}"
            else:
                diag["included_in_arm3_meta"] = True
            for row in mp.itertuples(index=False):
                tframe = finite_frame(tumor, [SNAI_COL, row.score_column])
                rows.append(
                    {
                        "arm": "arm3_tumour_only_rescored",
                        "MP_id": row.subcluster_id,
                        "MP_label": row.label,
                        "sample": sample,
                        "dataset": dataset,
                        "n_spots_used": int(len(tframe)),
                        "spearman_r": spearman_value(tframe, SNAI_COL, row.score_column),
                        "included_in_meta": bool(diag["included_in_arm3_meta"]),
                        "exclusion_reason": diag["exclusion_reason"],
                    }
                )
        except Exception as exc:
            diag["exclusion_reason"] = f"arm3 re-scoring failed: {exc}"
            for row in mp.itertuples(index=False):
                rows.append(
                    {
                        "arm": "arm3_tumour_only_rescored",
                        "MP_id": row.subcluster_id,
                        "MP_label": row.label,
                        "sample": sample,
                        "dataset": dataset,
                        "n_spots_used": n_tumor,
                        "spearman_r": math.nan,
                        "included_in_meta": False,
                        "exclusion_reason": diag["exclusion_reason"],
                    }
                )
        diagnostics.append(diag)
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def write_log(manifest: dict, per_sample: pd.DataFrame, meta: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    files = [p for p in OUT.rglob("*") if p.is_file()]
    lines = [
        "SNAI1-ac x manual MP three-arm correlation log",
        f"Run timestamp: {manifest['run_timestamp']}",
        f"Python version: {manifest['python_version']}",
        f"R version: {manifest.get('r_version', 'unknown')}",
        f"Output directory: {OUT}",
        "",
        "Confirmed columns",
        json.dumps(manifest["confirmed_columns"], indent=2),
        "",
        "Arm definitions",
        json.dumps(manifest["arms"], indent=2),
        "",
        "Per-arm counts",
        per_sample.groupby("arm").agg(
            rows=("spearman_r", "size"),
            samples=("sample", "nunique"),
            median_n_spots_used=("n_spots_used", "median"),
        ).reset_index().to_string(index=False),
        "",
        "Arm-3 diagnostics",
        diagnostics.to_string(index=False),
        "",
        "Meta-analysis settings",
        json.dumps(manifest["meta_analysis"], indent=2),
        "",
        "Output audit",
        f"- per_sample_rows: {len(per_sample)}",
        f"- meta_summary_rows: {len(meta)}",
        f"- arm3_diagnostic_rows: {len(diagnostics)}",
        f"- files_written: {len(files)}",
    ]
    excluded = diagnostics.loc[~diagnostics["included_in_arm3_meta"].astype(bool), ["dataset", "sample", "exclusion_reason"]]
    lines.extend(["", "Arm-3 exclusions", excluded.to_string(index=False) if not excluded.empty else "None"])
    lines.append("")
    lines.append("Spatial-autocorrelation note")
    lines.append(
        "The Fisher-z within-sample variance uses the raw per-arm spot count and is anti-conservative under spatial autocorrelation; the REML between-sample term partly absorbs this."
    )
    (OUT / "snai1ac_mp_three_arm_run_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    np.random.seed(RANDOM_SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), OUT / Path(__file__).name)
    shutil.copy2(R_SCRIPT, OUT / R_SCRIPT.name)

    mp = load_mp_table()
    sample_set = load_sample_set()
    paths = scored_h5ads(sample_set)
    validate_columns(paths, mp)
    per_sample, diagnostics = compute_correlations(paths, mp)
    per_sample.to_csv(OUT / "tables" / "snai1ac_mp_per_sample_correlations.csv", index=False)
    diagnostics.to_csv(OUT / "tables" / "snai1ac_mp_arm3_tumour_rescoring_diagnostics.csv", index=False)

    cmd = [str(RSCRIPT), str(R_SCRIPT), str(OUT)]
    r_result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    (OUT / "snai1ac_mp_meta_analysis_R_stdout.txt").write_text(r_result.stdout, encoding="utf-8")
    (OUT / "snai1ac_mp_meta_analysis_R_stderr.txt").write_text(r_result.stderr, encoding="utf-8")
    meta = pd.read_csv(OUT / "tables" / "snai1ac_mp_meta_analysis_summary.csv")

    r_version = subprocess.run([str(RSCRIPT), "-e", "cat(R.version.string)"], check=False, capture_output=True, text=True).stdout.strip()
    manifest = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "script_path": str(Path(__file__)),
        "r_script_path": str(R_SCRIPT),
        "python_version": sys.version,
        "r_version": r_version,
        "random_seed": RANDOM_SEED,
        "input_paths": {
            "subcluster_scoring_dir": str(SCORING_DIR),
            "scored_h5ad_dir": str(SCORED_DIR),
            "score_summary": str(SUMMARY_CSV),
            "previous_manifest": str(PREV_MANIFEST),
            "mp_signatures": str(MP_SIGNATURES),
            "snai1_ac_weights": str(SNAI_WEIGHTS),
            "snai1_ac_config": str(SNAI_CONFIG),
        },
        "confirmed_columns": {
            "snai1_ac_score": SNAI_COL,
            "malignant_fraction": MALIGNANT_COL,
            "compartment": COMPARTMENT_COL,
            "tumour_label": TUMOR_LABEL,
            "mp_score_columns": mp["score_column"].tolist(),
        },
        "arms": {
            "arm1_raw_all_spots": "Spearman SNAI1-ac vs MP over all finite all-spot scores.",
            "arm2_partial_all_spots_malignant": "Partial Spearman over all finite all-spot scores, controlling continuous Malignant fraction with rank residuals.",
            "arm3_tumour_only_rescored": "Subset interface == Tumor, re-score SNAI1-ac weighted and MPs uniformly, then Spearman.",
        },
        "arm3_rescoring": {
            "smoothing": True,
            "correct_spatial_covariates": True,
            "batch_key": None,
            "SNAI1-ac": "weighted by original log2FC-derived snai1_ac_weights.json",
            "MP1-MP8": "uniform recurrence gene lists; no gene_weights",
            "min_tumour_spots_for_meta": MIN_TUMOR_SPOTS,
        },
        "arm_counts": per_sample.groupby("arm")["sample"].nunique().to_dict(),
        "arm3_exclusions": diagnostics.loc[~diagnostics["included_in_arm3_meta"].astype(bool), ["dataset", "sample", "exclusion_reason"]].to_dict("records"),
        "meta_analysis": {
            "engine": "R metafor::rma",
            "method": "REML",
            "effect_transform": "Fisher z = atanh(r), back-transform with tanh",
            "within_sample_variance": "1 / (n_spots_used - 3), using each row's own arm-specific n_spots_used",
            "spatial_autocorrelation_note": "Anti-conservative under spatial autocorrelation; REML between-sample term partly absorbs this.",
        },
        "output_paths": {
            "per_sample_correlations": str(OUT / "tables" / "snai1ac_mp_per_sample_correlations.csv"),
            "meta_summary": str(OUT / "tables" / "snai1ac_mp_meta_analysis_summary.csv"),
            "arm3_diagnostics": str(OUT / "tables" / "snai1ac_mp_arm3_tumour_rescoring_diagnostics.csv"),
            "tumor_subset_h5ads": str(OUT / "tumor_subset_scored_h5ad"),
            "plots": str(OUT / "plots"),
        },
    }
    (OUT / "snai1ac_mp_three_arm_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_log(manifest, per_sample, meta, diagnostics)
    print(f"Done: {OUT}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        stop(str(exc))
