"""
Build SpottedPy v2 input manifest and visual QC/contact sheets.

This script intentionally does not run SpottedPy, hotspot calling, distance
statistics, neighborhood enrichment, or GEE. It is the step-3 artifact for
human review before hotspot preflight.
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_ROOT = Path(r"D:\HGSOC_Spatial_Atlas")
ANALYSIS_ROOT = DATA_ROOT / "05_analysis_ready"
CODE_ROOT = Path(r"C:\Users\luchu\Documents\MSc\Master Thesis\Code\HGSOC_Spatial_Atlas")
PREFLIGHT_ROOT = CODE_ROOT / "00_documentation" / "spottedpy_v2_preflight_inventory_d_root"
OUT_ROOT = ANALYSIS_ROOT / "SpottedPy_v2_paper_aligned"

CNMF_ROOT = ANALYSIS_ROOT / "S3_cNMF_Tumor_Programs"
MANUAL_DIR = (
    CNMF_ROOT
    / "jaccard_raw_matrices"
    / "inspection_exports_average"
    / "variantB_nonjunk_manual_cut_v2"
)
TUMOR_H5AD_ROOT = MANUAL_DIR / "subcluster_snai1ac_correlation" / "tumor_subset_scored_h5ad"
KSTAR_USAGE_ROOT = CNMF_ROOT / "per_sample"
UNSMOOTHED_PREDICTIONS = (
    CNMF_ROOT
    / "snai1ac_program_decomposition_unsmoothed_uncorrected_v1"
    / "02_per_sample_usage_models"
    / "tables"
    / "per_spot_predictions.csv"
)
GASTON_ALIGNMENT_ROOT = ANALYSIS_ROOT / "GASTON_method_v1" / "04_isodepth_score_alignment"
GASTON_ALIGNMENT_MANIFEST = GASTON_ALIGNMENT_ROOT / "sample_alignment_manifest.csv"
RCTD_REFERENCE = DATA_ROOT / "scRNA_reference" / "rctd_outputs"

OUT_QC = OUT_ROOT / "01_inputs_qc"
TABLE_DIR = OUT_QC / "tables"
CONTACT_DIR = OUT_QC / "figures" / "contact_sheets"
OVERVIEW_DIR = OUT_QC / "figures" / "overview"
SCRIPTS_USED = OUT_QC / "scripts_used"

SNAI_COLS = ["SNAI1-ac_score", "SNAI1_score", "SNAI1-2R_score"]
UNSMOOTHED_COL = "snai1ac_em_unsmoothed_uncorrected"
SPACET_COLS = [
    "Malignant",
    "CAF",
    "Endothelial",
    "Macrophage",
    "B cell",
    "T CD4",
    "T CD8",
    "NK",
    "Plasma",
    "Unidentifiable",
]
MP_COLS = [
    "MP1_angiogenic_vascular_score",
    "MP2_iCAF_stress_score",
    "MP3_complement_CAF_score",
    "MP4_activated_myCAF_score",
    "MP5_IFN_TLS_immune_score",
    "MP6_APC_TAM_myeloid_score",
    "MP7_malignant_hypoxia_score",
    "MP8_malignant_acute_phase_secretory_score",
]
MP_TITLES = {
    "MP1_angiogenic_vascular_score": "MP1 angiogenic/vascular",
    "MP2_iCAF_stress_score": "MP2 iCAF-stress",
    "MP3_complement_CAF_score": "MP3 complement-CAF",
    "MP4_activated_myCAF_score": "MP4 activated-myCAF",
    "MP5_IFN_TLS_immune_score": "MP5 IFN/TLS immune",
    "MP6_APC_TAM_myeloid_score": "MP6 APC/TAM myeloid",
    "MP7_malignant_hypoxia_score": "MP7 malignant hypoxia",
    "MP8_malignant_acute_phase_secretory_score": "MP8 malignant acute-phase/secretory",
}


def stop(message: str) -> None:
    raise SystemExit(f"STOP: {message}")


def require(path: Path) -> None:
    if not path.exists():
        stop(f"Missing required path: {path}")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def decode_h5_values(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind in {"S", "O"}:
        return np.array(
            [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in arr],
            dtype=object,
        )
    return arr


def read_obs_column(obs_group: h5py.Group, column: str) -> np.ndarray:
    obj = obs_group[column]
    if isinstance(obj, h5py.Dataset):
        return decode_h5_values(obj[()])
    if isinstance(obj, h5py.Group) and {"codes", "categories"}.issubset(obj.keys()):
        codes = np.asarray(obj["codes"][()])
        categories = decode_h5_values(obj["categories"][()])
        out = np.empty(codes.shape[0], dtype=object)
        valid = codes >= 0
        out[~valid] = np.nan
        out[valid] = categories[codes[valid]]
        return out
    stop(f"Unsupported h5ad obs column encoding for {column}")


def obs_column_order(handle: h5py.File) -> list[str]:
    obs = handle["obs"]
    if "column-order" in obs.attrs:
        return [str(x) for x in decode_h5_values(obs.attrs["column-order"])]
    return list(obs.keys())


def obs_index(handle: h5py.File) -> np.ndarray:
    obs = handle["obs"]
    key = obs.attrs.get("_index", None)
    if key is None:
        if "_index" in obs:
            key = "_index"
        elif "spot" in obs:
            key = "spot"
        else:
            first = next(iter(obs.keys()))
            key = first
    if isinstance(key, bytes):
        key = key.decode("utf-8")
    return read_obs_column(obs, str(key)).astype(str)


def read_h5ad_obs_spatial(path: Path, columns: list[str]) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    require(path)
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        order = obs_column_order(handle)
        present = [col for col in columns if col in obs]
        frame = pd.DataFrame({col: read_obs_column(obs, col) for col in present})
        index = obs_index(handle)
        frame.index = index
        frame.index.name = "obs_index"
        if "spot" not in frame.columns:
            frame["spot"] = index
        if "obsm" not in handle or "spatial" not in handle["obsm"]:
            stop(f"{path} lacks obsm/spatial")
        spatial = np.asarray(handle["obsm"]["spatial"][()])
        if spatial.shape[0] != frame.shape[0]:
            stop(f"{path} spatial rows do not match obs rows")
    return frame, spatial, order


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def plot_numeric(
    ax: plt.Axes,
    xy: np.ndarray,
    values: pd.Series,
    title: str,
    background_xy: np.ndarray | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    if background_xy is not None and len(background_xy):
        ax.scatter(background_xy[:, 0], background_xy[:, 1], s=4, c="#e5e7eb", linewidths=0)
    vals = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(vals)
    if finite.any():
        sc = ax.scatter(
            xy[finite, 0],
            xy[finite, 1],
            c=vals[finite],
            s=8,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
        )
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    else:
        ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title, fontsize=9)
    style_spatial_axis(ax)


def plot_categorical_interface(ax: plt.Axes, xy: np.ndarray, values: pd.Series, title: str) -> None:
    colors = {
        "Tumor": "#d73027",
        "Interface": "#fdae61",
        "Stroma": "#4575b4",
        "nan": "#d1d5db",
    }
    vals = values.astype(str).fillna("nan")
    for label in sorted(vals.unique()):
        mask = vals.eq(label).to_numpy()
        ax.scatter(xy[mask, 0], xy[mask, 1], s=8, c=colors.get(label, "#6b7280"), label=label, linewidths=0)
    ax.legend(loc="best", fontsize=6, frameon=False, markerscale=2)
    ax.set_title(title, fontsize=9)
    style_spatial_axis(ax)


def style_spatial_axis(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def read_preflight_inventory() -> tuple[pd.DataFrame, pd.DataFrame]:
    require(PREFLIGHT_ROOT / "h5ad_input_inventory.csv")
    require(PREFLIGHT_ROOT / "rctd_candidate_files.csv")
    h5ads = pd.read_csv(PREFLIGHT_ROOT / "h5ad_input_inventory.csv")
    rctd = pd.read_csv(PREFLIGHT_ROOT / "rctd_candidate_files.csv")
    return h5ads, rctd


def sample_from_tumor_name(sample: str) -> str:
    return sample.replace(".tumor_subset_SNAI1ac_MP_scores", "")


def load_unsmoothed_predictions() -> pd.DataFrame:
    require(UNSMOOTHED_PREDICTIONS)
    pred = pd.read_csv(UNSMOOTHED_PREDICTIONS)
    needed = {"dataset", "sample_id_on_disk", "sample_label", "spot_id", UNSMOOTHED_COL}
    missing = needed - set(pred.columns)
    if missing:
        stop(f"Unsmoothed predictions missing columns: {sorted(missing)}")
    pred[UNSMOOTHED_COL] = pd.to_numeric(pred[UNSMOOTHED_COL], errors="coerce")
    return pred[list(needed)].copy()


def load_gaston_alignment_manifest() -> pd.DataFrame:
    require(GASTON_ALIGNMENT_MANIFEST)
    manifest = pd.read_csv(GASTON_ALIGNMENT_MANIFEST)
    needed = {"dataset", "sample", "layer", "alignment_table", "n_spots"}
    missing = needed - set(manifest.columns)
    if missing:
        stop(f"GASTON alignment manifest missing columns: {sorted(missing)}")
    manifest = manifest[manifest["layer"].astype(str).eq("whole")].copy()
    return manifest


def load_all_spot_unsmoothed_scores(alignment_manifest: pd.DataFrame, dataset: str, sample: str) -> pd.DataFrame:
    match = alignment_manifest[
        alignment_manifest["dataset"].astype(str).eq(dataset)
        & alignment_manifest["sample"].astype(str).eq(sample)
    ].copy()
    if len(match) != 1:
        stop(f"Expected one GASTON whole-tissue alignment table for {dataset}__{sample}; found {len(match)}")
    table = Path(str(match["alignment_table"].iloc[0]))
    require(table)
    cols = ["dataset", "sample", "spot_id", UNSMOOTHED_COL]
    frame = pd.read_csv(table, usecols=cols)
    frame[UNSMOOTHED_COL] = pd.to_numeric(frame[UNSMOOTHED_COL], errors="coerce")
    return frame


def kstar_usage_path(sample: str) -> Path:
    return KSTAR_USAGE_ROOT / sample / "representative_usage_kstar.csv"


def load_kstar_usage(path: Path) -> tuple[pd.DataFrame, list[str]]:
    require(path)
    usage = pd.read_csv(path)
    metadata = {"spot_id", "dataset", "sample_id_on_disk", "sample_label"}
    program_cols = [col for col in usage.columns if col not in metadata]
    for col in program_cols:
        usage[col] = pd.to_numeric(usage[col], errors="coerce")
    return usage, program_cols


def build_sample_records(
    h5ads: pd.DataFrame,
    rctd: pd.DataFrame,
    pred: pd.DataFrame,
    alignment_manifest: pd.DataFrame,
) -> pd.DataFrame:
    tumor = h5ads[h5ads["kind"].eq("tumor_subset_scored")].copy()
    full = h5ads[h5ads["kind"].eq("visium_analysis_ready")].copy()
    records = []

    for row in tumor.sort_values(["dataset", "sample"]).itertuples(index=False):
        dataset = str(row.dataset)
        tumor_sample_name = str(row.sample)
        sample = sample_from_tumor_name(tumor_sample_name)
        full_match = full[full["dataset"].eq(dataset) & full["sample"].eq(sample)]
        full_path = str(full_match["path"].iloc[0]) if len(full_match) else ""
        tumor_path = str(row.path)
        kstar_path = kstar_usage_path(sample)
        kstar_cols: list[str] = []
        if kstar_path.exists():
            _, kstar_cols = load_kstar_usage(kstar_path)
        pred_match = pred[pred["dataset"].eq(dataset) & pred["sample_id_on_disk"].eq(sample)]
        align_match = alignment_manifest[
            alignment_manifest["dataset"].astype(str).eq(dataset)
            & alignment_manifest["sample"].astype(str).eq(sample)
        ]
        rctd_match = rctd[
            rctd.astype(str).apply(lambda col: col.str.contains(sample, regex=False, na=False)).any(axis=1)
        ]
        records.append(
            {
                "dataset": dataset,
                "sample": sample,
                "sample_label": f"{dataset}__{sample}",
                "full_h5ad_path": full_path,
                "tumor_subset_h5ad_path": tumor_path,
                "full_h5ad_exists": bool(full_path and Path(full_path).exists()),
                "tumor_subset_h5ad_exists": bool(Path(tumor_path).exists()),
                "tumor_mask_column": "interface",
                "tumor_label": "Tumor",
                "snai_ac_corrected_smoothed_col": "SNAI1-ac_score",
                "snai1_col": "SNAI1_score",
                "snai1_2r_col": "SNAI1-2R_score",
                "snai_ac_unsmoothed_uncorrected_col": UNSMOOTHED_COL,
                "snai_ac_unsmoothed_source": str(GASTON_ALIGNMENT_MANIFEST),
                "snai_ac_unsmoothed_all_spot_alignment_table": str(align_match["alignment_table"].iloc[0])
                if len(align_match) == 1
                else "",
                "snai_ac_unsmoothed_n_all_spots": int(float(align_match["n_spots"].iloc[0]))
                if len(align_match) == 1
                else 0,
                "snai_ac_unsmoothed_n_tumor_prediction_spots": int(len(pred_match)),
                "mp_columns_present": ";".join([col for col in MP_COLS if col in str(getattr(row, "mp_score_columns", ""))])
                if False
                else ";".join(MP_COLS),
                "kstar_usage_path": str(kstar_path),
                "kstar_usage_exists": bool(kstar_path.exists()),
                "kstar_program_columns": ";".join(kstar_cols),
                "kstar_n_programs": int(len(kstar_cols)),
                "spacet_columns": ";".join(SPACET_COLS),
                "rctd_inventory_status": "separate_inventory_needed",
                "rctd_candidate_file_hits_for_sample": int(len(rctd_match)),
            }
        )
    return pd.DataFrame(records)


def value_range(frame: pd.DataFrame, columns: list[str], dataset: str, sample: str, source: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for col in columns:
        if col not in frame.columns:
            records.append(
                {
                    "dataset": dataset,
                    "sample": sample,
                    "source": source,
                    "column": col,
                    "present": False,
                    "n_finite": 0,
                    "min": math.nan,
                    "median": math.nan,
                    "max": math.nan,
                    "n_negative": 0,
                    "n_zero": 0,
                    "raw_zero_scaled_minmax_location": math.nan,
                }
            )
            continue
        vals = pd.to_numeric(frame[col], errors="coerce")
        finite = vals[np.isfinite(vals)]
        if len(finite):
            min_v = float(finite.min())
            max_v = float(finite.max())
            if max_v > min_v:
                zero_loc = float((0.0 - min_v) / (max_v - min_v))
            else:
                zero_loc = math.nan
            records.append(
                {
                    "dataset": dataset,
                    "sample": sample,
                    "source": source,
                    "column": col,
                    "present": True,
                    "n_finite": int(len(finite)),
                    "min": min_v,
                    "median": float(finite.median()),
                    "max": max_v,
                    "n_negative": int((finite < 0).sum()),
                    "n_zero": int((finite == 0).sum()),
                    "raw_zero_scaled_minmax_location": zero_loc,
                }
            )
        else:
            records.append(
                {
                    "dataset": dataset,
                    "sample": sample,
                    "source": source,
                    "column": col,
                    "present": True,
                    "n_finite": 0,
                    "min": math.nan,
                    "median": math.nan,
                    "max": math.nan,
                    "n_negative": 0,
                    "n_zero": 0,
                    "raw_zero_scaled_minmax_location": math.nan,
                }
            )
    return records


def make_core_contact_sheet(
    dataset: str,
    sample: str,
    full_obs: pd.DataFrame,
    full_xy: np.ndarray,
    tumor_obs: pd.DataFrame,
    tumor_xy: np.ndarray,
    all_spot_unsmoothed: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.4), squeeze=False)
    flat = axes.ravel()
    plot_categorical_interface(flat[0], full_xy, full_obs["interface"], "ROI mask: interface")
    plot_numeric(flat[1], full_xy, numeric_series(full_obs, "SNAI1_score"), "SNAI1")
    plot_numeric(flat[2], full_xy, numeric_series(full_obs, "SNAI1-ac_score"), "SNAI1-ac")
    plot_numeric(flat[3], full_xy, numeric_series(full_obs, "Malignant"), "SpaCET Malignant")
    plot_numeric(flat[4], full_xy, numeric_series(full_obs, "SNAI1-2R_score"), "SNAI1-2R")
    if len(all_spot_unsmoothed):
        vals = all_spot_unsmoothed.set_index("spot_id")[UNSMOOTHED_COL]
        full_vals = full_obs["spot"].astype(str).map(vals)
        plot_numeric(
            flat[5],
            full_xy,
            full_vals,
            "SNAI1-ac unsmoothed/uncorrected",
            cmap="viridis",
        )
    else:
        flat[5].text(
            0.5,
            0.5,
            "missing all-spot unsmoothed score",
            ha="center",
            va="center",
            transform=flat[5].transAxes,
        )
        flat[5].set_title("SNAI1-ac unsmoothed/uncorrected", fontsize=9)
        style_spatial_axis(flat[5])

    fig.suptitle(f"{dataset}__{sample}: SpottedPy v2 core QC", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_mp_contact_sheet(
    dataset: str,
    sample: str,
    full_xy: np.ndarray,
    tumor_obs: pd.DataFrame,
    tumor_xy: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16.4, 7.8), squeeze=False)
    mp_values = pd.DataFrame({col: numeric_series(tumor_obs, col) for col in MP_COLS})
    max_abs = float(np.nanmax(np.abs(mp_values.to_numpy(dtype=float))))
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0
    for ax, col in zip(axes.ravel(), MP_COLS):
        plot_numeric(
            ax,
            tumor_xy,
            numeric_series(tumor_obs, col),
            MP_TITLES[col],
            background_xy=full_xy,
            cmap="coolwarm",
            vmin=-max_abs,
            vmax=max_abs,
        )
    fig.suptitle(
        f"{dataset}__{sample}: MP1-MP8 tumor-subset score maps (shared zero-centered scale)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_kstar_contact_sheet(
    dataset: str,
    sample: str,
    full_xy: np.ndarray,
    tumor_obs: pd.DataFrame,
    tumor_xy: np.ndarray,
    kstar: pd.DataFrame,
    program_cols: list[str],
    out_path: Path,
) -> None:
    if not program_cols:
        return
    ncols = 3
    nrows = math.ceil(len(program_cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.2 * nrows), squeeze=False)
    spot_values = kstar.set_index("spot_id")
    for idx, ax in enumerate(axes.ravel()):
        if idx >= len(program_cols):
            ax.axis("off")
            continue
        col = program_cols[idx]
        vals = tumor_obs["spot"].astype(str).map(spot_values[col])
        title = col.replace(f"{dataset}__{sample}__", "")
        plot_numeric(ax, tumor_xy, vals, title, background_xy=full_xy, cmap="plasma")
    fig.suptitle(f"{dataset}__{sample}: sample-specific K* usage maps", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_readme(manifest: pd.DataFrame, plot_manifest: pd.DataFrame) -> None:
    lines = [
        "# SpottedPy v2 Input QC Contact Sheets",
        "",
        "This folder contains the first real visual QC artifact for SpottedPy v2.",
        "No SpottedPy hotspot, neighborhood, distance, or GEE analysis has been run.",
        "",
        "## Scope",
        "",
        "- 23 thesis-cohort tumor-subset scored h5ad files from the final Variant B MP1-MP8 branch.",
        "- Matched full-slide analysis-ready h5ad files for ROI, SNAI1-family, SpaCET, and Hallmark context.",
        "- `SNAI1-ac_score` is treated as the corrected/smoothed production score.",
        "- `snai1ac_em_unsmoothed_uncorrected` is joined for all spots from the GASTON whole-tissue score-alignment tables.",
        "- Sample-specific K* program usage comes from `per_sample/<sample>/representative_usage_kstar.csv`.",
        "",
        "## Tables",
        "",
        "- `tables/spottedpy_v2_live_input_manifest.csv`: one row per sample.",
        "- `tables/spottedpy_v2_score_range_audit.csv`: raw score ranges and min-max zero-location audit.",
        "- `tables/spottedpy_v2_plot_manifest.csv`: generated image paths.",
        "",
        "## Figures",
        "",
        "- `figures/contact_sheets/core_qc`: ROI mask, SNAI1, SNAI1-ac, Malignant, SNAI1-2R, and all-spot unsmoothed SNAI1-ac.",
        "- `figures/contact_sheets/mp_qc`: MP1-MP8 tumor-subset score maps with one zero-centered scale per sample.",
        "- `figures/contact_sheets/kstar_qc`: sample-specific K* usage maps.",
        "- `figures/overview/spottedpy_v2_core_qc_contact_sheet_index.png`: small index of core QC sheets.",
        "",
        "## Review checkpoint",
        "",
        "Review these images before any hotspot preflight. The key decisions are whether column-based ROI logic is acceptable,",
        "whether corrected and unsmoothed SNAI1-ac maps look biologically sane, and whether MP/K* maps are spatially coherent.",
        "",
        "## Counts",
        "",
        f"- Samples in manifest: {len(manifest)}",
        f"- Plot files generated: {len(plot_manifest)}",
        f"- Samples with K* usage files: {int(manifest['kstar_usage_exists'].sum())}",
        f"- Samples with all-spot unsmoothed SNAI1-ac tables: {int((manifest['snai_ac_unsmoothed_n_all_spots'] > 0).sum())}",
        "",
    ]
    (OUT_QC / "README.md").write_text("\n".join(lines), encoding="utf-8")


def make_overview_index(plot_manifest: pd.DataFrame) -> None:
    core = plot_manifest[plot_manifest["figure_type"].eq("core_qc")].copy().head(12)
    if core.empty:
        return
    ncols = 3
    nrows = math.ceil(len(core) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows), squeeze=False)
    for ax, row in zip(axes.ravel(), core.itertuples(index=False)):
        img = plt.imread(row.figure_path)
        ax.imshow(img)
        ax.set_title(row.sample_label, fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[len(core) :]:
        ax.axis("off")
    fig.suptitle("SpottedPy v2 core QC contact sheet index (first 12 samples)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OVERVIEW_DIR / "spottedpy_v2_core_qc_contact_sheet_index.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    for directory in [TABLE_DIR, CONTACT_DIR / "core_qc", CONTACT_DIR / "mp_qc", CONTACT_DIR / "kstar_qc", OVERVIEW_DIR, SCRIPTS_USED]:
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), SCRIPTS_USED / Path(__file__).name)

    h5ads, rctd = read_preflight_inventory()
    pred = load_unsmoothed_predictions()
    alignment_manifest = load_gaston_alignment_manifest()
    manifest = build_sample_records(h5ads, rctd, pred, alignment_manifest)
    manifest_path = TABLE_DIR / "spottedpy_v2_live_input_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    plot_records: list[dict[str, Any]] = []
    range_records: list[dict[str, Any]] = []

    full_columns = ["interface", "Malignant", "EM_ratio", *SNAI_COLS, "array_row", "array_col", "spot"]
    tumor_columns = ["interface", "Malignant", "EM_ratio", *SNAI_COLS, *MP_COLS, "array_row", "array_col", "spot"]

    for row in manifest.itertuples(index=False):
        dataset = row.dataset
        sample = row.sample
        sample_label = row.sample_label
        print(f"Rendering QC sheets for {sample_label}")
        full_path = Path(row.full_h5ad_path)
        tumor_path = Path(row.tumor_subset_h5ad_path)
        full_obs, full_xy, _ = read_h5ad_obs_spatial(full_path, full_columns)
        tumor_obs, tumor_xy, _ = read_h5ad_obs_spatial(tumor_path, tumor_columns)
        tumor_obs["spot"] = tumor_obs["spot"].astype(str)

        all_spot_unsmoothed = load_all_spot_unsmoothed_scores(alignment_manifest, dataset, sample)

        kstar_path = Path(row.kstar_usage_path)
        kstar, kstar_cols = load_kstar_usage(kstar_path)

        core_path = CONTACT_DIR / "core_qc" / f"{safe_name(sample_label)}__core_qc.png"
        mp_path = CONTACT_DIR / "mp_qc" / f"{safe_name(sample_label)}__mp1_mp8_qc.png"
        kstar_path_out = CONTACT_DIR / "kstar_qc" / f"{safe_name(sample_label)}__kstar_qc.png"

        make_core_contact_sheet(dataset, sample, full_obs, full_xy, tumor_obs, tumor_xy, all_spot_unsmoothed, core_path)
        make_mp_contact_sheet(dataset, sample, full_xy, tumor_obs, tumor_xy, mp_path)
        make_kstar_contact_sheet(dataset, sample, full_xy, tumor_obs, tumor_xy, kstar, kstar_cols, kstar_path_out)

        plot_records.extend(
            [
                {
                    "dataset": dataset,
                    "sample": sample,
                    "sample_label": sample_label,
                    "figure_type": "core_qc",
                    "figure_path": str(core_path),
                },
                {
                    "dataset": dataset,
                    "sample": sample,
                    "sample_label": sample_label,
                    "figure_type": "mp_qc",
                    "figure_path": str(mp_path),
                },
                {
                    "dataset": dataset,
                    "sample": sample,
                    "sample_label": sample_label,
                    "figure_type": "kstar_qc",
                    "figure_path": str(kstar_path_out),
                },
            ]
        )

        range_records.extend(value_range(full_obs, [*SNAI_COLS, "Malignant"], dataset, sample, "full_h5ad"))
        range_records.extend(value_range(tumor_obs, MP_COLS, dataset, sample, "tumor_subset_h5ad"))
        unsmoothed_frame = all_spot_unsmoothed.set_index("spot_id")[[UNSMOOTHED_COL]].copy()
        range_records.extend(value_range(unsmoothed_frame, [UNSMOOTHED_COL], dataset, sample, "gaston_all_spot_unsmoothed_alignment"))
        range_records.extend(value_range(kstar.set_index("spot_id"), kstar_cols, dataset, sample, "kstar_usage"))

    plot_manifest = pd.DataFrame(plot_records)
    range_audit = pd.DataFrame(range_records)
    plot_manifest_path = TABLE_DIR / "spottedpy_v2_plot_manifest.csv"
    range_audit_path = TABLE_DIR / "spottedpy_v2_score_range_audit.csv"
    plot_manifest.to_csv(plot_manifest_path, index=False)
    range_audit.to_csv(range_audit_path, index=False)
    make_overview_index(plot_manifest)
    write_readme(manifest, plot_manifest)

    run_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__)),
        "output_root": str(OUT_QC),
        "input_manifest": str(manifest_path),
        "plot_manifest": str(plot_manifest_path),
        "score_range_audit": str(range_audit_path),
        "n_samples": int(len(manifest)),
        "n_plot_files": int(len(plot_manifest)),
        "does_not_run": ["SpottedPy", "hotspot calling", "distance statistics", "GEE"],
        "snai_ac_corrected_smoothed_column": "SNAI1-ac_score",
        "snai_ac_unsmoothed_uncorrected_source": str(GASTON_ALIGNMENT_MANIFEST),
        "kstar_usage_root": str(KSTAR_USAGE_ROOT),
        "rctd_reference_root": str(RCTD_REFERENCE),
        "rctd_status": "inventory separately before sensitivity use",
    }
    (OUT_QC / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(json.dumps(run_manifest, indent=2))


if __name__ == "__main__":
    main()
