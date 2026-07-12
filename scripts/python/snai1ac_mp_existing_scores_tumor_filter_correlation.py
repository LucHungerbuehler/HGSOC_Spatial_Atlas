from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
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
SCORED_DIR = SCORING_DIR / "scored_h5ad"
SUMMARY_CSV = SCORING_DIR / "manual_subcluster_enrichmap_score_summary.csv"
PREV_MANIFEST = SCORING_DIR / "manual_subcluster_scoring_manifest.json"
OUT = MANUAL_DIR / "subcluster_snai1ac_correlation_existing_scores_tumor_only"
R_SCRIPT = Path(__file__).parent / "R" / "snai1ac_mp_existing_scores_tumor_filter_meta_analysis.R"
RSCRIPT = Path(r"C:\Program Files\R\R-4.4.3\bin\Rscript.exe")

SNAI_COL = "SNAI1-ac_score"
MALIGNANT_COL = "Malignant"
COMPARTMENT_COL = "interface"
TUMOR_LABEL = "Tumor"
MIN_TUMOR_SPOTS = 50


def stop(message: str) -> None:
    raise SystemExit(f"STOP: {message}")


def require(path: Path) -> None:
    if not path.exists():
        stop(f"Missing input: {path}")


def load_mp_table() -> pd.DataFrame:
    require(SUMMARY_CSV)
    summary = pd.read_csv(SUMMARY_CSV)
    needed = {"subcluster_id", "label", "score_column"}
    if not needed.issubset(summary.columns):
        stop(f"{SUMMARY_CSV} lacks columns: {sorted(needed - set(summary.columns))}")
    mp = summary[list(needed)].drop_duplicates().sort_values("subcluster_id").reset_index(drop=True)
    if len(mp) != 8:
        stop(f"Expected 8 MP score columns; observed {len(mp)}")
    return mp


def load_sample_set() -> list[str]:
    require(PREV_MANIFEST)
    manifest = json.loads(PREV_MANIFEST.read_text(encoding="utf-8"))
    sample_set = manifest.get("sample_set")
    if not isinstance(sample_set, list) or not sample_set:
        stop(f"{PREV_MANIFEST} lacks a non-empty sample_set")
    return sample_set


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


def spearman_value(frame: pd.DataFrame, x_col: str, y_col: str) -> float:
    part = frame[[x_col, y_col]].dropna()
    if len(part) < 4 or part[x_col].nunique() < 2 or part[y_col].nunique() < 2:
        return math.nan
    return float(spearmanr(part[x_col], part[y_col]).statistic)


def partial_spearman(frame: pd.DataFrame, x_col: str, y_col: str, covar: str) -> float:
    part = frame[[x_col, y_col, covar]].dropna()
    if len(part) < 4:
        return math.nan
    xr = residualize_ranked(part[x_col].to_numpy(dtype=float), part[[covar]].reset_index(drop=True))
    yr = residualize_ranked(part[y_col].to_numpy(dtype=float), part[[covar]].reset_index(drop=True))
    if np.std(xr) <= 1e-12 or np.std(yr) <= 1e-12:
        return math.nan
    return float(pearsonr(xr, yr).statistic)


def numeric_frame(obs: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    frame = obs[cols].copy()
    for col in cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def compute(paths: list[tuple[str, str, Path]], mp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    diagnostics = []
    score_cols = mp["score_column"].tolist()
    needed = [SNAI_COL, MALIGNANT_COL, COMPARTMENT_COL, *score_cols]

    for dataset, sample, path in paths:
        print(f"Processing {dataset}__{sample}")
        adata = ad.read_h5ad(path, backed="r")
        try:
            missing = [col for col in needed if col not in adata.obs.columns]
            if missing:
                stop(f"{dataset}__{sample} missing obs columns: {missing}")
            labels = set(adata.obs[COMPARTMENT_COL].astype(str).unique())
            if TUMOR_LABEL not in labels:
                stop(
                    f"{dataset}__{sample} lacks tumour label {TUMOR_LABEL!r} "
                    f"in {COMPARTMENT_COL}; observed {sorted(labels)}"
                )

            obs = adata.obs[needed].copy()
        finally:
            adata.file.close()

        tumor_obs = obs.loc[obs[COMPARTMENT_COL].astype(str).eq(TUMOR_LABEL)].copy()
        n_tumor = int(len(tumor_obs))
        diag = {
            "dataset": dataset,
            "sample": sample,
            "n_tumour_spots": n_tumor,
            "included_in_meta": n_tumor >= MIN_TUMOR_SPOTS,
            "exclusion_reason": "" if n_tumor >= MIN_TUMOR_SPOTS else f"n_tumour_spots < {MIN_TUMOR_SPOTS}",
            "existing_score_source_h5ad": str(path),
        }
        diagnostics.append(diag)

        frame = numeric_frame(tumor_obs, [SNAI_COL, MALIGNANT_COL, *score_cols])
        for row in mp.itertuples(index=False):
            raw_frame = frame[[SNAI_COL, row.score_column]].dropna()
            partial_frame = frame[[SNAI_COL, row.score_column, MALIGNANT_COL]].dropna()
            rows.append(
                {
                    "arm": "arm4_tumour_only_existing_scores",
                    "MP_id": row.subcluster_id,
                    "MP_label": row.label,
                    "sample": sample,
                    "dataset": dataset,
                    "n_spots_used": int(len(raw_frame)),
                    "spearman_r": spearman_value(raw_frame, SNAI_COL, row.score_column),
                    "included_in_meta": bool(diag["included_in_meta"]),
                    "exclusion_reason": diag["exclusion_reason"],
                }
            )
            rows.append(
                {
                    "arm": "arm5_tumour_only_existing_scores_partial_malignant",
                    "MP_id": row.subcluster_id,
                    "MP_label": row.label,
                    "sample": sample,
                    "dataset": dataset,
                    "n_spots_used": int(len(partial_frame)),
                    "spearman_r": partial_spearman(partial_frame, SNAI_COL, row.score_column, MALIGNANT_COL),
                    "included_in_meta": bool(diag["included_in_meta"]),
                    "exclusion_reason": diag["exclusion_reason"],
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def write_log(manifest: dict, per_sample: pd.DataFrame, meta: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    lines = [
        "SNAI1-ac x manual MP existing-score tumor-filter correlation log",
        f"Run timestamp: {manifest['run_timestamp']}",
        f"Python version: {manifest['python_version']}",
        f"R version: {manifest.get('r_version', 'unknown')}",
        f"Output directory: {OUT}",
        "",
        "Analysis purpose",
        (
            "Use the existing whole-sample SNAI1-ac and MP1-MP8 scores, restrict "
            "to interface == Tumor spots, and compute continuous signed associations "
            "without re-scoring."
        ),
        "",
        "Arms",
        json.dumps(manifest["arms"], indent=2),
        "",
        "Per-arm counts",
        per_sample.groupby("arm")
        .agg(rows=("spearman_r", "size"), samples=("sample", "nunique"), median_n_spots_used=("n_spots_used", "median"))
        .reset_index()
        .to_string(index=False),
        "",
        "Tumor-filter diagnostics",
        diagnostics.to_string(index=False),
        "",
        "Meta-analysis summary",
        meta.to_string(index=False),
        "",
        "Output audit",
        f"- per_sample_rows: {len(per_sample)}",
        f"- meta_summary_rows: {len(meta)}",
        f"- diagnostics_rows: {len(diagnostics)}",
    ]
    (OUT / "snai1ac_mp_existing_scores_tumor_filter_run_log.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)
    (OUT / "scripts").mkdir(exist_ok=True)

    shutil.copy2(Path(__file__), OUT / "scripts" / Path(__file__).name)
    shutil.copy2(R_SCRIPT, OUT / "scripts" / R_SCRIPT.name)

    mp = load_mp_table()
    paths = scored_h5ads(load_sample_set())
    per_sample, diagnostics = compute(paths, mp)
    per_sample.to_csv(OUT / "tables" / "snai1ac_mp_existing_scores_tumor_filter_per_sample.csv", index=False)
    diagnostics.to_csv(OUT / "tables" / "snai1ac_mp_existing_scores_tumor_filter_diagnostics.csv", index=False)

    r_result = subprocess.run([str(RSCRIPT), str(R_SCRIPT), str(OUT)], check=True, capture_output=True, text=True)
    (OUT / "snai1ac_mp_existing_scores_tumor_filter_R_stdout.txt").write_text(r_result.stdout, encoding="utf-8")
    (OUT / "snai1ac_mp_existing_scores_tumor_filter_R_stderr.txt").write_text(r_result.stderr, encoding="utf-8")
    meta = pd.read_csv(OUT / "tables" / "snai1ac_mp_existing_scores_tumor_filter_meta_summary.csv")

    r_version = subprocess.run(
        [str(RSCRIPT), "-e", "cat(R.version.string)"], check=False, capture_output=True, text=True
    ).stdout.strip()
    manifest = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "script_path": str(Path(__file__)),
        "r_script_path": str(R_SCRIPT),
        "python_version": sys.version,
        "r_version": r_version,
        "input_paths": {
            "scored_h5ad_dir": str(SCORED_DIR),
            "score_summary": str(SUMMARY_CSV),
            "previous_manifest": str(PREV_MANIFEST),
        },
        "confirmed_columns": {
            "snai1_ac_score": SNAI_COL,
            "malignant_fraction": MALIGNANT_COL,
            "compartment": COMPARTMENT_COL,
            "tumour_label": TUMOR_LABEL,
            "mp_score_columns": mp["score_column"].tolist(),
        },
        "arms": {
            "arm4_tumour_only_existing_scores": (
                "Subset existing whole-sample scored h5ads to interface == Tumor, "
                "then compute Spearman SNAI1-ac vs MP scores without re-scoring."
            ),
            "arm5_tumour_only_existing_scores_partial_malignant": (
                "Same tumor-only existing-score subset, then partial Spearman "
                "controlling continuous Malignant fraction with rank residuals."
            ),
        },
        "meta_analysis": {
            "engine": "R metafor::rma",
            "method": "REML",
            "effect_transform": "Fisher z = atanh(r), back-transform with tanh",
            "within_sample_variance": "1 / (n_spots_used - 3)",
            "spatial_autocorrelation_note": "Anti-conservative under spatial autocorrelation.",
        },
        "output_paths": {
            "per_sample": str(OUT / "tables" / "snai1ac_mp_existing_scores_tumor_filter_per_sample.csv"),
            "meta_summary": str(OUT / "tables" / "snai1ac_mp_existing_scores_tumor_filter_meta_summary.csv"),
            "diagnostics": str(OUT / "tables" / "snai1ac_mp_existing_scores_tumor_filter_diagnostics.csv"),
            "plots": str(OUT / "plots"),
        },
    }
    (OUT / "snai1ac_mp_existing_scores_tumor_filter_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    write_log(manifest, per_sample, meta, diagnostics)
    print(f"Done: {OUT}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        stop(str(exc))
