"""
Run SpottedPy v2 hotspot preflight and visual review maps.

This script performs only the primary hotspot/coldspot layer for visual review.
It does not run distance statistics, neighborhood enrichment, scale sensitivity,
or GEE. Run as a script file from PowerShell using the conda env Python; do not
use ad hoc `python -c` import probes on Windows.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_ROOT = Path(r"D:\HGSOC_Spatial_Atlas")
ANALYSIS_ROOT = DATA_ROOT / "05_analysis_ready"
CODE_ROOT = Path(r"C:\Users\luchu\Documents\MSc\Master Thesis\Code\HGSOC_Spatial_Atlas")
SPOTTEDPY_CLONE = DATA_ROOT / "git_clones" / "SpottedPy-main"

INPUT_QC = ANALYSIS_ROOT / "SpottedPy_v2_paper_aligned" / "01_inputs_qc"
INPUT_MANIFEST = INPUT_QC / "tables" / "spottedpy_v2_live_input_manifest.csv"
SCORE_AUDIT = INPUT_QC / "tables" / "spottedpy_v2_score_range_audit.csv"

OUT_ROOT = ANALYSIS_ROOT / "SpottedPy_v2_paper_aligned" / "03_hotspots_preflight"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures" / "contact_sheets"
H5AD_DIR = OUT_ROOT / "h5ad"
SCRIPT_DIR = OUT_ROOT / "scripts_used"

NEIGHBOURS_PRIMARY = 10
P_VALUE = 0.05
PERMUTATIONS = 999
SEED = 100

SNAI_COLS = ["SNAI1-ac_score", "SNAI1_score", "SNAI1-2R_score"]
SNAI_TITLES = {
    "SNAI1-ac_score": "SNAI1-ac",
    "SNAI1_score": "SNAI1",
    "SNAI1-2R_score": "SNAI1-2R",
    "snai1ac_em_unsmoothed_uncorrected": "SNAI1-ac unsmoothed/uncorrected",
}
UNSMOOTHED_COL = "snai1ac_em_unsmoothed_uncorrected"

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


@dataclass(frozen=True)
class HotspotSpec:
    family: str
    variable_id: str
    raw_col: str
    scaled_col: str
    title: str
    domain: str
    scale_method: str


def stop(message: str) -> None:
    raise SystemExit(f"STOP: {message}")


def require(path: Path) -> None:
    if not path.exists():
        stop(f"Missing required path: {path}")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def import_spottedpy():
    sys.path.insert(0, str(SPOTTEDPY_CLONE))
    import anndata as ad  # noqa: F401
    import spottedpy as sp

    return sp


def register_anndata_null_reader() -> None:
    """Allow this older anndata env to read newer H5AD null-encoded fields."""
    import h5py
    from anndata._io.specs.registry import IOSpec, _REGISTRY

    null_spec = IOSpec("null", "0.1.0")
    if _REGISTRY.has_reader(h5py.Dataset, null_spec):
        return

    @_REGISTRY.register_read(h5py.Dataset, null_spec)
    def read_null(_elem, _reader):
        return None


def prepare_dirs() -> None:
    for directory in [
        TABLE_DIR,
        FIG_DIR / "core_full_hotcold",
        FIG_DIR / "core_full_numbered",
        FIG_DIR / "core_tumor_hotcold",
        FIG_DIR / "core_tumor_numbered",
        FIG_DIR / "mp_hotcold",
        FIG_DIR / "mp_numbered",
        FIG_DIR / "kstar_hotcold",
        FIG_DIR / "kstar_numbered",
        H5AD_DIR / "core_full",
        H5AD_DIR / "core_tumor",
        H5AD_DIR / "mp_kstar_tumor",
        SCRIPT_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), SCRIPT_DIR / Path(__file__).name)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    require(INPUT_MANIFEST)
    require(SCORE_AUDIT)
    manifest = pd.read_csv(INPUT_MANIFEST)
    score_audit = pd.read_csv(SCORE_AUDIT)
    if len(manifest) != 23:
        stop(f"Expected 23 samples in input manifest; found {len(manifest)}")
    return manifest, score_audit


def load_all_spot_unsmoothed(row: pd.Series) -> pd.DataFrame:
    table = Path(str(row["snai_ac_unsmoothed_all_spot_alignment_table"]))
    require(table)
    frame = pd.read_csv(table, usecols=["spot_id", UNSMOOTHED_COL])
    frame[UNSMOOTHED_COL] = pd.to_numeric(frame[UNSMOOTHED_COL], errors="coerce")
    return frame


def load_kstar_usage(row: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    path = Path(str(row["kstar_usage_path"]))
    require(path)
    usage = pd.read_csv(path)
    metadata = {"spot_id", "dataset", "sample_id_on_disk", "sample_label"}
    program_cols = [col for col in usage.columns if col not in metadata]
    for col in program_cols:
        usage[col] = pd.to_numeric(usage[col], errors="coerce")
    return usage, program_cols


def minmax(values: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return pd.Series(np.nan, index=values.index), {
            "n_finite": 0,
            "raw_min": math.nan,
            "raw_max": math.nan,
            "raw_zero_scaled_location": math.nan,
            "constant_input": True,
        }
    raw_min = float(finite.min())
    raw_max = float(finite.max())
    if raw_max == raw_min:
        scaled = pd.Series(0.5, index=values.index, dtype=float)
        scaled[numeric.isna()] = np.nan
        zero_location = math.nan
        constant = True
    else:
        scaled = (numeric - raw_min) / (raw_max - raw_min)
        zero_location = float((0.0 - raw_min) / (raw_max - raw_min))
        constant = False
    return scaled.astype(float), {
        "n_finite": int(len(finite)),
        "raw_min": raw_min,
        "raw_max": raw_max,
        "raw_zero_scaled_location": zero_location,
        "constant_input": constant,
    }


def nonnegative_raw(values: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    finite = numeric[np.isfinite(numeric)]
    raw_min = float(finite.min()) if len(finite) else math.nan
    raw_max = float(finite.max()) if len(finite) else math.nan
    if len(finite) and raw_min < 0:
        stop(f"Expected nonnegative raw values but observed min {raw_min}")
    return numeric, {
        "n_finite": int(len(finite)),
        "raw_min": raw_min,
        "raw_max": raw_max,
        "raw_zero_scaled_location": 0.0 if len(finite) else math.nan,
        "constant_input": bool(len(finite) and raw_min == raw_max),
    }


def add_spatial_obs(adata, dataset: str, sample: str) -> None:
    if "array_row" not in adata.obs or "array_col" not in adata.obs:
        stop(f"{dataset}__{sample} lacks array_row/array_col")
    if "spot" not in adata.obs:
        adata.obs["spot"] = adata.obs.index.astype(str)
    adata.obs["batch"] = f"{dataset}__{sample}"
    adata.obs["dataset"] = dataset
    adata.obs["sample"] = sample
    adata.obs["array_row"] = pd.to_numeric(adata.obs["array_row"], errors="coerce").astype(int)
    adata.obs["array_col"] = pd.to_numeric(adata.obs["array_col"], errors="coerce").astype(int)


def make_core_full_specs(adata, all_spot_unsmoothed: pd.DataFrame) -> tuple[list[HotspotSpec], list[dict[str, Any]]]:
    specs: list[HotspotSpec] = []
    scaling_records: list[dict[str, Any]] = []
    for col in SNAI_COLS:
        scaled_col = f"spv2_full_{safe_name(col)}_minmax"
        adata.obs[scaled_col], audit = minmax(adata.obs[col])
        specs.append(
            HotspotSpec("core", col, col, scaled_col, SNAI_TITLES[col], "full", "sample_minmax")
        )
        scaling_records.append({"raw_col": col, "scaled_col": scaled_col, "domain": "full", **audit})

    unsmoothed_map = all_spot_unsmoothed.set_index("spot_id")[UNSMOOTHED_COL]
    adata.obs[UNSMOOTHED_COL] = adata.obs["spot"].astype(str).map(unsmoothed_map)
    scaled_col = "spv2_full_snai1ac_unsmoothed_uncorrected_minmax"
    adata.obs[scaled_col], audit = minmax(adata.obs[UNSMOOTHED_COL])
    specs.append(
        HotspotSpec(
            "core",
            UNSMOOTHED_COL,
            UNSMOOTHED_COL,
            scaled_col,
            SNAI_TITLES[UNSMOOTHED_COL],
            "full",
            "sample_minmax",
        )
    )
    scaling_records.append({"raw_col": UNSMOOTHED_COL, "scaled_col": scaled_col, "domain": "full", **audit})
    return specs, scaling_records


def make_core_tumor_specs(adata, all_spot_unsmoothed: pd.DataFrame) -> tuple[list[HotspotSpec], list[dict[str, Any]]]:
    specs: list[HotspotSpec] = []
    scaling_records: list[dict[str, Any]] = []
    tumor_mask = adata.obs["interface"].astype(str).eq("Tumor")

    for col in SNAI_COLS:
        scaled_col = f"spv2_tumor_{safe_name(col)}_minmax"
        scaled = pd.Series(np.nan, index=adata.obs.index, dtype=float)
        scaled.loc[tumor_mask], audit = minmax(adata.obs.loc[tumor_mask, col])
        adata.obs[scaled_col] = scaled
        specs.append(
            HotspotSpec("core", col, col, scaled_col, SNAI_TITLES[col], "tumor", "tumor_minmax")
        )
        scaling_records.append({"raw_col": col, "scaled_col": scaled_col, "domain": "tumor", **audit})

    unsmoothed_map = all_spot_unsmoothed.set_index("spot_id")[UNSMOOTHED_COL]
    adata.obs[UNSMOOTHED_COL] = adata.obs["spot"].astype(str).map(unsmoothed_map)
    scaled_col = "spv2_tumor_snai1ac_unsmoothed_uncorrected_minmax"
    scaled = pd.Series(np.nan, index=adata.obs.index, dtype=float)
    scaled.loc[tumor_mask], audit = minmax(adata.obs.loc[tumor_mask, UNSMOOTHED_COL])
    adata.obs[scaled_col] = scaled
    specs.append(
        HotspotSpec(
            "core",
            UNSMOOTHED_COL,
            UNSMOOTHED_COL,
            scaled_col,
            SNAI_TITLES[UNSMOOTHED_COL],
            "tumor",
            "tumor_minmax",
        )
    )
    scaling_records.append({"raw_col": UNSMOOTHED_COL, "scaled_col": scaled_col, "domain": "tumor", **audit})
    return specs, scaling_records


def make_mp_kstar_specs(adata, kstar_usage: pd.DataFrame, kstar_cols: list[str]) -> tuple[list[HotspotSpec], list[dict[str, Any]]]:
    specs: list[HotspotSpec] = []
    scaling_records: list[dict[str, Any]] = []

    for col in MP_COLS:
        scaled_col = f"spv2_tumor_{safe_name(col)}_minmax"
        adata.obs[scaled_col], audit = minmax(adata.obs[col])
        specs.append(HotspotSpec("mp", col, col, scaled_col, MP_TITLES[col], "tumor", "tumor_minmax"))
        scaling_records.append({"raw_col": col, "scaled_col": scaled_col, "domain": "tumor", **audit})

    kstar_by_spot = kstar_usage.set_index("spot_id")
    for col in kstar_cols:
        raw_col = f"kstar_usage__{safe_name(col)}"
        scaled_col = f"spv2_tumor_kstar_{safe_name(col)}_raw"
        adata.obs[raw_col] = adata.obs["spot"].astype(str).map(kstar_by_spot[col])
        adata.obs[scaled_col], audit = nonnegative_raw(adata.obs[raw_col])
        title = col.split("__", 2)[-1] if "__" in col else col
        specs.append(HotspotSpec("kstar", col, raw_col, scaled_col, title, "tumor", "raw_nonnegative"))
        scaling_records.append({"raw_col": raw_col, "scaled_col": scaled_col, "domain": "tumor", **audit})

    return specs, scaling_records


def run_specs(sp, adata, specs: list[HotspotSpec], sample_label: str) -> tuple[Any, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for spec in specs:
        finite = pd.to_numeric(adata.obs[spec.scaled_col], errors="coerce").notna()
        n_roi = int(finite.sum())
        if n_roi < (NEIGHBOURS_PRIMARY + 2):
            records.append(
                {
                    "sample_label": sample_label,
                    "family": spec.family,
                    "domain": spec.domain,
                    "variable_id": spec.variable_id,
                    "raw_col": spec.raw_col,
                    "scaled_col": spec.scaled_col,
                    "status": "skipped_too_few_spots",
                    "n_roi_spots": n_roi,
                    "n_hot_spots": 0,
                    "n_cold_spots": 0,
                    "n_hot_components": 0,
                    "n_cold_components": 0,
                }
            )
            continue
        t0 = time.time()
        try:
            adata = sp.create_hotspots(
                anndata=adata,
                column_name=spec.scaled_col,
                filter_columns=None,
                filter_value=None,
                neighbours_parameters=NEIGHBOURS_PRIMARY,
                p_value=P_VALUE,
                number_components_return=False,
                relative_to_batch=True,
                number_hotspots=True,
                permutation=PERMUTATIONS,
                seed_number=SEED,
            )
            status = "ok"
            error = ""
        except Exception as exc:
            status = "error"
            error = repr(exc)
        hot_col = f"{spec.scaled_col}_hot"
        cold_col = f"{spec.scaled_col}_cold"
        hot_num = f"{spec.scaled_col}_hot_number"
        cold_num = f"{spec.scaled_col}_cold_number"
        n_hot = int(pd.to_numeric(adata.obs.get(hot_col, pd.Series(dtype=float)), errors="coerce").notna().sum()) if hot_col in adata.obs else 0
        n_cold = int(pd.to_numeric(adata.obs.get(cold_col, pd.Series(dtype=float)), errors="coerce").notna().sum()) if cold_col in adata.obs else 0
        if hot_num in adata.obs:
            hot_labels = adata.obs[hot_num].astype(str)
            n_hot_components = int(hot_labels[valid_component_mask(hot_labels)].nunique())
        else:
            n_hot_components = 0
        if cold_num in adata.obs:
            cold_labels = adata.obs[cold_num].astype(str)
            n_cold_components = int(cold_labels[valid_component_mask(cold_labels)].nunique())
        else:
            n_cold_components = 0
        records.append(
            {
                "sample_label": sample_label,
                "family": spec.family,
                "domain": spec.domain,
                "variable_id": spec.variable_id,
                "title": spec.title,
                "raw_col": spec.raw_col,
                "scaled_col": spec.scaled_col,
                "scale_method": spec.scale_method,
                "status": status,
                "error": error,
                "n_roi_spots": n_roi,
                "n_hot_spots": n_hot,
                "n_cold_spots": n_cold,
                "n_hot_components": n_hot_components,
                "n_cold_components": n_cold_components,
                "seconds": round(time.time() - t0, 3),
            }
        )
    return adata, records


def style_axis(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_hotcold_panel(ax: plt.Axes, adata, spec: HotspotSpec) -> None:
    xy = np.asarray(adata.obsm["spatial"])
    hot_col = f"{spec.scaled_col}_hot"
    cold_col = f"{spec.scaled_col}_cold"
    ax.scatter(xy[:, 0], xy[:, 1], s=4, c="#e5e7eb", linewidths=0)
    if cold_col in adata.obs:
        cold = pd.to_numeric(adata.obs[cold_col], errors="coerce").notna().to_numpy()
        ax.scatter(xy[cold, 0], xy[cold, 1], s=9, c="#2166ac", linewidths=0, label="cold")
    if hot_col in adata.obs:
        hot = pd.to_numeric(adata.obs[hot_col], errors="coerce").notna().to_numpy()
        ax.scatter(xy[hot, 0], xy[hot, 1], s=9, c="#b2182b", linewidths=0, label="hot")
    ax.set_title(spec.title, fontsize=9)
    style_axis(ax)


def compact_component_label(prefix: str, value: str) -> str:
    match = re.match(r"^(\d+)(?:_|$)", str(value))
    if match:
        return f"{prefix}{match.group(1)}"
    return f"{prefix}{str(value)[:6]}"


def valid_component_mask(values: pd.Series) -> pd.Series:
    return values.astype(str).str.match(r"^\d+(?:_|$)", na=False)


def plot_numbered_panel(ax: plt.Axes, adata, spec: HotspotSpec) -> None:
    xy = np.asarray(adata.obsm["spatial"])
    hot_num = f"{spec.scaled_col}_hot_number"
    cold_num = f"{spec.scaled_col}_cold_number"
    ax.scatter(xy[:, 0], xy[:, 1], s=4, c="#e5e7eb", linewidths=0)
    labels = pd.Series("", index=adata.obs.index, dtype=object)
    if hot_num in adata.obs:
        h = adata.obs[hot_num].astype(str)
        h_mask = valid_component_mask(h)
        labels[h_mask] = [compact_component_label("H", value) for value in h[h_mask]]
    if cold_num in adata.obs:
        c = adata.obs[cold_num].astype(str)
        c_mask = valid_component_mask(c)
        labels[c_mask] = [compact_component_label("C", value) for value in c[c_mask]]
    unique = sorted([x for x in labels.unique() if x])
    if unique:
        cmap = plt.get_cmap("tab20", max(len(unique), 1))
        for idx, label in enumerate(unique):
            mask = labels.eq(label).to_numpy()
            ax.scatter(xy[mask, 0], xy[mask, 1], s=9, c=[cmap(idx)], linewidths=0)
            x_mid = float(np.median(xy[mask, 0]))
            y_mid = float(np.median(xy[mask, 1]))
            ax.text(
                x_mid,
                y_mid,
                label,
                ha="center",
                va="center",
                fontsize=5.5,
                fontweight="bold",
                color="#111827",
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
            )
    ax.set_title(spec.title, fontsize=9)
    style_axis(ax)


def plot_contact_sheet(adata, specs: list[HotspotSpec], sample_label: str, out_path: Path, numbered: bool) -> None:
    if not specs:
        return
    ncols = 4 if len(specs) >= 4 else len(specs)
    nrows = int(math.ceil(len(specs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 3.9 * nrows), squeeze=False)
    for idx, ax in enumerate(axes.ravel()):
        if idx >= len(specs):
            ax.axis("off")
            continue
        if numbered:
            plot_numbered_panel(ax, adata, specs[idx])
        else:
            plot_hotcold_panel(ax, adata, specs[idx])
    suffix = "numbered components" if numbered else "hot/cold spots"
    fig.suptitle(f"{sample_label}: {suffix}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_readme(summary: pd.DataFrame, plot_manifest: pd.DataFrame) -> None:
    ok = summary[summary["status"].eq("ok")]
    lines = [
        "# SpottedPy v2 Hotspot Preflight",
        "",
        "This folder contains the primary k=10 hotspot/coldspot preflight for visual review.",
        "It does not contain distance statistics, GEE, neighborhood enrichment, or k=8 sensitivity.",
        "",
        "## Parameters",
        "",
        f"- `neighbours_parameters`: {NEIGHBOURS_PRIMARY}",
        f"- `p_value`: {P_VALUE}",
        f"- `permutations`: {PERMUTATIONS}",
        f"- `seed`: {SEED}",
        "- `relative_to_batch`: True",
        "- batch grain: one sample/patient",
        "",
        "## Scope",
        "",
        "- Core full-slide SNAI1-family hotspots.",
        "- Core tumor-restricted SNAI1-family hotspots.",
        "- MP1-MP8 tumor-region hotspots.",
        "- Sample-specific K* tumor-region hotspots.",
        "",
        "## Scaling",
        "",
        "- Centered EnrichMap scores are min-max scaled within their analysis domain before hotspot calling.",
        "- Raw score columns are preserved in the saved h5ad files.",
        "- K* usage values are used as nonnegative raw usage values.",
        "",
        "## Review",
        "",
        "Review hot/cold maps and numbered component maps before distance statistics.",
        "",
        "## Counts",
        "",
        f"- Hotspot tests attempted: {len(summary)}",
        f"- Successful hotspot tests: {len(ok)}",
        f"- Figure files: {len(plot_manifest)}",
        f"- Tests with any hot spots: {int((summary['n_hot_spots'] > 0).sum())}",
        f"- Tests with any cold spots: {int((summary['n_cold_spots'] > 0).sum())}",
        "",
    ]
    (OUT_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    prepare_dirs()
    manifest, _ = load_inputs()
    sp = import_spottedpy()
    register_anndata_null_reader()

    import anndata as ad

    all_summary: list[dict[str, Any]] = []
    all_scaling: list[dict[str, Any]] = []
    all_plots: list[dict[str, Any]] = []

    for row in manifest.sort_values(["dataset", "sample"]).to_dict("records"):
        dataset = str(row["dataset"])
        sample = str(row["sample"])
        sample_label = str(row["sample_label"])
        print(f"[{datetime.now().isoformat(timespec='seconds')}] Hotspot preflight {sample_label}", flush=True)

        all_spot_unsmoothed = load_all_spot_unsmoothed(pd.Series(row))
        kstar_usage, kstar_cols = load_kstar_usage(pd.Series(row))

        full = ad.read_h5ad(str(row["full_h5ad_path"]))
        full.uns.clear()
        add_spatial_obs(full, dataset, sample)
        core_full_specs, scaling = make_core_full_specs(full, all_spot_unsmoothed)
        all_scaling.extend({"dataset": dataset, "sample": sample, "sample_label": sample_label, **r} for r in scaling)
        full, summary = run_specs(sp, full, core_full_specs, sample_label)
        all_summary.extend(summary)

        core_tumor_specs, scaling = make_core_tumor_specs(full, all_spot_unsmoothed)
        all_scaling.extend({"dataset": dataset, "sample": sample, "sample_label": sample_label, **r} for r in scaling)
        full, summary = run_specs(sp, full, core_tumor_specs, sample_label)
        all_summary.extend(summary)
        full.write_h5ad(H5AD_DIR / "core_full" / f"{safe_name(sample_label)}__core_full_and_tumor_hotspots.h5ad")

        tumor = ad.read_h5ad(str(row["tumor_subset_h5ad_path"]))
        tumor.uns.clear()
        add_spatial_obs(tumor, dataset, sample)
        mp_kstar_specs, scaling = make_mp_kstar_specs(tumor, kstar_usage, kstar_cols)
        all_scaling.extend({"dataset": dataset, "sample": sample, "sample_label": sample_label, **r} for r in scaling)
        tumor, summary = run_specs(sp, tumor, mp_kstar_specs, sample_label)
        all_summary.extend(summary)
        tumor.write_h5ad(H5AD_DIR / "mp_kstar_tumor" / f"{safe_name(sample_label)}__mp_kstar_tumor_hotspots.h5ad")

        core_full_hotcold = FIG_DIR / "core_full_hotcold" / f"{safe_name(sample_label)}__core_full_hotcold.png"
        core_full_numbered = FIG_DIR / "core_full_numbered" / f"{safe_name(sample_label)}__core_full_numbered.png"
        core_tumor_hotcold = FIG_DIR / "core_tumor_hotcold" / f"{safe_name(sample_label)}__core_tumor_hotcold.png"
        core_tumor_numbered = FIG_DIR / "core_tumor_numbered" / f"{safe_name(sample_label)}__core_tumor_numbered.png"
        mp_hotcold = FIG_DIR / "mp_hotcold" / f"{safe_name(sample_label)}__mp_hotcold.png"
        mp_numbered = FIG_DIR / "mp_numbered" / f"{safe_name(sample_label)}__mp_numbered.png"
        kstar_hotcold = FIG_DIR / "kstar_hotcold" / f"{safe_name(sample_label)}__kstar_hotcold.png"
        kstar_numbered = FIG_DIR / "kstar_numbered" / f"{safe_name(sample_label)}__kstar_numbered.png"

        mp_specs = [s for s in mp_kstar_specs if s.family == "mp"]
        kstar_specs = [s for s in mp_kstar_specs if s.family == "kstar"]
        plot_contact_sheet(full, core_full_specs, sample_label, core_full_hotcold, numbered=False)
        plot_contact_sheet(full, core_full_specs, sample_label, core_full_numbered, numbered=True)
        plot_contact_sheet(full, core_tumor_specs, sample_label, core_tumor_hotcold, numbered=False)
        plot_contact_sheet(full, core_tumor_specs, sample_label, core_tumor_numbered, numbered=True)
        plot_contact_sheet(tumor, mp_specs, sample_label, mp_hotcold, numbered=False)
        plot_contact_sheet(tumor, mp_specs, sample_label, mp_numbered, numbered=True)
        plot_contact_sheet(tumor, kstar_specs, sample_label, kstar_hotcold, numbered=False)
        plot_contact_sheet(tumor, kstar_specs, sample_label, kstar_numbered, numbered=True)

        for fig_type, path in [
            ("core_full_hotcold", core_full_hotcold),
            ("core_full_numbered", core_full_numbered),
            ("core_tumor_hotcold", core_tumor_hotcold),
            ("core_tumor_numbered", core_tumor_numbered),
            ("mp_hotcold", mp_hotcold),
            ("mp_numbered", mp_numbered),
            ("kstar_hotcold", kstar_hotcold),
            ("kstar_numbered", kstar_numbered),
        ]:
            all_plots.append(
                {
                    "dataset": dataset,
                    "sample": sample,
                    "sample_label": sample_label,
                    "figure_type": fig_type,
                    "figure_path": str(path),
                }
            )

    summary = pd.DataFrame(all_summary)
    scaling = pd.DataFrame(all_scaling)
    plots = pd.DataFrame(all_plots)
    summary.to_csv(TABLE_DIR / "spottedpy_v2_hotspot_preflight_summary.csv", index=False)
    scaling.to_csv(TABLE_DIR / "spottedpy_v2_hotspot_scaling_audit.csv", index=False)
    plots.to_csv(TABLE_DIR / "spottedpy_v2_hotspot_plot_manifest.csv", index=False)

    run_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__)),
        "output_root": str(OUT_ROOT),
        "spottedpy_clone": str(SPOTTEDPY_CLONE),
        "input_manifest": str(INPUT_MANIFEST),
        "neighbours_parameters": NEIGHBOURS_PRIMARY,
        "p_value": P_VALUE,
        "permutations": PERMUTATIONS,
        "seed": SEED,
        "relative_to_batch": True,
        "batch_grain": "sample/patient",
        "n_samples": int(manifest["sample_label"].nunique()),
        "n_hotspot_tests": int(len(summary)),
        "n_plot_files": int(len(plots)),
        "does_not_run": ["distance statistics", "GEE", "neighborhood enrichment", "k=8 sensitivity"],
    }
    (OUT_ROOT / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    write_readme(summary, plots)
    print(json.dumps(run_manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
