from __future__ import annotations

import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(r"D:\HGSOC_Spatial_Atlas\05_analysis_ready\SpottedPy_v2_paper_aligned")
DIST_ROOT = PROJECT_ROOT / "05_distance_gee"
TABLE_ROOT = DIST_ROOT / "tables"
FIG_ROOT = DIST_ROOT / "figures" / "preflight"
SCRIPT_ROOT = DIST_ROOT / "scripts_used"

DESIGN_PATH = TABLE_ROOT / "spottedpy_v2_distance_availability_preflight.csv"
CONTRAST_PATH = TABLE_ROOT / "spottedpy_v2_distance_contrast_manifest.csv"

COMPONENT_OUT = TABLE_ROOT / "spottedpy_v2_distance_component_summary_preflight.csv"
SAMPLE_OUT = TABLE_ROOT / "spottedpy_v2_distance_sample_target_summary_preflight.csv"
CROSS_SAMPLE_OUT = TABLE_ROOT / "spottedpy_v2_distance_cross_sample_descriptive_summary_preflight.csv"
SAMPLE_AUDIT_OUT = TABLE_ROOT / "spottedpy_v2_distance_input_merge_audit_preflight.csv"
RUN_MANIFEST_OUT = DIST_ROOT / "run_manifest_distance_calculation_preflight.json"
README_OUT = DIST_ROOT / "README_distance_calculation_preflight.md"


PRIMARY_LABELS = {
    "snai1ac_consensus_full_hot": "SNAI1-ac hot",
    "snai1ac_consensus_full_cold": "SNAI1-ac cold",
    "snai12r_full_hot": "SNAI1-2R hot",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_obs(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    a = ad.read_h5ad(path, backed="r")
    try:
        obs = a.obs.copy()
    finally:
        if getattr(a, "file", None) is not None:
            a.file.close()
    return obs


def clean_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().eq("true")


def make_short_target_label(row: pd.Series) -> str:
    state = row["target_state"]
    title = str(row["target_display_label"] or row["target_title"])
    family = row["target_family"]
    if family == "mp":
        first = title.split(" ", 1)[0]
        return f"{first}\n{state}"
    if family == "spacet":
        return f"{title}\n{state}"
    if family == "hallmark":
        title = title.replace("HALLMARK_", "").replace("_", " ")
        return f"{title}\n{state}"
    if family == "kstar":
        return f"{title}\n{state}"
    return f"{title}\n{state}"


def ensure_numeric_coords(obs: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ["array_row", "array_col"] if c not in obs.columns]
    if missing:
        raise ValueError(f"Missing coordinate columns: {missing}")
    obs = obs.copy()
    obs["array_row"] = pd.to_numeric(obs["array_row"], errors="coerce")
    obs["array_col"] = pd.to_numeric(obs["array_col"], errors="coerce")
    if obs[["array_row", "array_col"]].isna().any().any():
        raise ValueError("array_row/array_col contain NA after numeric coercion")
    return obs


def merge_sample_obs(sample_design: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    sample_label = sample_design["sample_label"].iloc[0]
    source_path = sample_design["source_h5ad_path"].iloc[0]
    core_obs = ensure_numeric_coords(read_obs(source_path))
    obs = core_obs[["array_row", "array_col"]].copy()

    needed_core_cols = sorted(
        set(sample_design["source_state_col"])
        | set(sample_design["source_number_col"])
        | set(sample_design["reference_state_col"])
        | set(sample_design["reference_number_col"])
    )
    missing_core = [c for c in needed_core_cols if c not in core_obs.columns]
    if missing_core:
        raise ValueError(f"{sample_label}: missing core columns: {missing_core[:10]}")
    obs = obs.join(core_obs[needed_core_cols], how="left")

    target_paths = sorted(set(sample_design["target_h5ad_path"]))
    merge_audit = {
        "sample_label": sample_label,
        "n_full_obs": int(obs.shape[0]),
        "source_h5ad_path": source_path,
        "target_h5ad_paths": ";".join(target_paths),
        "n_target_h5ad_paths": int(len(target_paths)),
        "missing_target_columns": "",
        "target_index_missing_from_core": "",
        "target_index_extra_not_in_core": "",
    }
    missing_target_cols = []
    target_index_missing = []
    target_index_extra = []

    for path in target_paths:
        target_obs = read_obs(path)
        path_rows = sample_design[sample_design["target_h5ad_path"].eq(path)]
        cols = sorted(set(path_rows["target_state_col"]) | set(path_rows["target_number_col"]))
        missing = [c for c in cols if c not in target_obs.columns]
        if missing:
            missing_target_cols.extend([f"{Path(path).name}:{c}" for c in missing])
            continue
        extra = target_obs.index.difference(obs.index)
        missing_from_target = obs.index.difference(target_obs.index)
        if len(extra) > 0:
            target_index_extra.append(f"{Path(path).name}:{len(extra)}")
        if len(missing_from_target) > 0:
            target_index_missing.append(f"{Path(path).name}:{len(missing_from_target)}")
        obs = obs.join(target_obs[cols], how="left")

    merge_audit["missing_target_columns"] = ";".join(missing_target_cols)
    merge_audit["target_index_missing_from_core"] = ";".join(target_index_missing)
    merge_audit["target_index_extra_not_in_core"] = ";".join(target_index_extra)
    return obs, merge_audit


def extract_region(obs: pd.DataFrame, state_col: str, number_col: str) -> pd.DataFrame:
    if state_col not in obs.columns:
        raise ValueError(f"Missing state column: {state_col}")
    if number_col not in obs.columns:
        raise ValueError(f"Missing number column: {number_col}")
    mask = obs[state_col].notna()
    region = obs.loc[mask, ["array_row", "array_col", number_col]].copy()
    region = region.rename(columns={number_col: "component_number"})
    numeric_components = pd.to_numeric(region["component_number"], errors="coerce")
    missing_numeric = numeric_components.isna()
    if missing_numeric.any():
        parsed = (
            region.loc[missing_numeric, "component_number"]
            .astype(str)
            .str.extract(r"^(\d+)", expand=False)
        )
        numeric_components.loc[missing_numeric] = pd.to_numeric(parsed, errors="coerce")
    region["component_number"] = numeric_components
    region = region.dropna(subset=["component_number"])
    region["component_number"] = region["component_number"].astype(int)
    return region


def nearest_distances(primary_region: pd.DataFrame, target_region: pd.DataFrame) -> np.ndarray:
    if primary_region.empty or target_region.empty:
        return np.array([], dtype=float)
    target_tree = cKDTree(target_region[["array_row", "array_col"]].to_numpy(dtype=float))
    distances, _ = target_tree.query(primary_region[["array_row", "array_col"]].to_numpy(dtype=float), k=1)
    return distances.astype(float)


def summarize_components(
    primary_region: pd.DataFrame,
    distances: np.ndarray,
    metadata: dict,
) -> list[dict]:
    if len(distances) != primary_region.shape[0]:
        raise ValueError("Distance vector length does not match primary region")
    tmp = primary_region.copy()
    tmp["min_distance"] = distances
    rows = []
    grouped = tmp.groupby("component_number", sort=True)
    for component, comp in grouped:
        rows.append(
            {
                **metadata,
                "primary_component_number": int(component),
                "n_primary_component_spots": int(comp.shape[0]),
                "component_min_distance": float(comp["min_distance"].min()),
                "component_median_distance": float(comp["min_distance"].median()),
                "component_mean_distance": float(comp["min_distance"].mean()),
                "component_max_distance": float(comp["min_distance"].max()),
            }
        )
    return rows


def compute_sample(sample_design: pd.DataFrame) -> tuple[list[dict], dict]:
    obs, merge_audit = merge_sample_obs(sample_design)
    sample_label = sample_design["sample_label"].iloc[0]

    primary_specs = {}
    for _, row in sample_design.iterrows():
        primary_specs[row["source_group_id"]] = {
            "group_id": row["source_group_id"],
            "group_label": PRIMARY_LABELS.get(row["source_group_id"], row["source_group_label"]),
            "state_col": row["source_state_col"],
            "number_col": row["source_number_col"],
            "family": row["source_family"],
            "domain": row["source_domain"],
            "state": row["source_state"],
        }
        primary_specs[row["reference_group_id"]] = {
            "group_id": row["reference_group_id"],
            "group_label": PRIMARY_LABELS.get(row["reference_group_id"], row["reference_group_label"]),
            "state_col": row["reference_state_col"],
            "number_col": row["reference_number_col"],
            "family": row["reference_family"],
            "domain": row["reference_domain"],
            "state": row["reference_state"],
        }

    primary_regions = {
        group_id: extract_region(obs, spec["state_col"], spec["number_col"])
        for group_id, spec in primary_specs.items()
    }

    target_rows = (
        sample_design[
            [
                "target_family",
                "target_domain",
                "target_group_class",
                "target_group_order",
                "target_variable_id",
                "target_title",
                "target_display_label",
                "target_state",
                "target_state_col",
                "target_number_col",
                "target_cross_sample_summary_allowed",
                "target_cross_sample_note",
                "kstar_alignment_category_draft",
                "kstar_family_label",
                "kstar_mp1_8_name",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    target_regions = {}
    for _, target in target_rows.iterrows():
        key = (target["target_family"], target["target_variable_id"], target["target_state"])
        target_regions[key] = extract_region(obs, target["target_state_col"], target["target_number_col"])

    component_rows = []
    total_jobs = len(sample_design)
    for job_idx, (_, row) in enumerate(sample_design.iterrows(), start=1):
        if job_idx == 1 or job_idx == total_jobs or job_idx % 50 == 0:
            print(f"[{now_iso()}] {sample_label}: distance job {job_idx}/{total_jobs}", flush=True)

        target_key = (row["target_family"], row["target_variable_id"], row["target_state"])
        target_region = target_regions[target_key]
        for role, group_id in [
            ("source", row["source_group_id"]),
            ("reference", row["reference_group_id"]),
        ]:
            spec = primary_specs[group_id]
            primary_region = primary_regions[group_id]
            distances = nearest_distances(primary_region, target_region)
            metadata = {
                "dataset": row["dataset"],
                "sample": row["sample"],
                "sample_label": sample_label,
                "contrast_id": row["contrast_id"],
                "contrast_label": row["contrast_label"],
                "contrast_role": row["contrast_role"],
                "primary_role": role,
                "primary_group_id": group_id,
                "primary_group_label": spec["group_label"],
                "primary_state_col": spec["state_col"],
                "primary_number_col": spec["number_col"],
                "primary_family": spec["family"],
                "primary_domain": spec["domain"],
                "primary_state": spec["state"],
                "n_primary_spots": int(primary_region.shape[0]),
                "n_primary_components": int(primary_region["component_number"].nunique()),
                "target_family": row["target_family"],
                "target_domain": row["target_domain"],
                "target_group_class": row["target_group_class"],
                "target_group_order": int(row["target_group_order"]),
                "target_variable_id": row["target_variable_id"],
                "target_title": row["target_title"],
                "target_display_label": row["target_display_label"],
                "target_short_label": make_short_target_label(row),
                "target_state": row["target_state"],
                "target_state_col": row["target_state_col"],
                "target_number_col": row["target_number_col"],
                "target_cross_sample_summary_allowed": str(row["target_cross_sample_summary_allowed"]).lower()
                == "true",
                "target_cross_sample_note": row["target_cross_sample_note"],
                "kstar_alignment_category_draft": row["kstar_alignment_category_draft"],
                "kstar_family_label": row["kstar_family_label"],
                "kstar_mp1_8_name": row["kstar_mp1_8_name"],
                "n_target_spots": int(target_region.shape[0]),
                "n_target_components": int(target_region["component_number"].nunique()),
                "distance_metric": "nearest_spot_min_distance",
                "distance_implementation": "scipy.spatial.cKDTree nearest-neighbor query equivalent to SpottedPy distance_matrix min over array_row/array_col",
            }
            component_rows.extend(summarize_components(primary_region, distances, metadata))

    merge_audit["n_component_rows"] = int(len(component_rows))
    merge_audit["n_primary_groups"] = int(len(primary_regions))
    merge_audit["n_target_states"] = int(len(target_regions))
    return component_rows, merge_audit


def summarize_sample_targets(component_df: pd.DataFrame) -> pd.DataFrame:
    sample_primary = (
        component_df.groupby(
            [
                "dataset",
                "sample",
                "sample_label",
                "contrast_id",
                "contrast_label",
                "contrast_role",
                "primary_role",
                "primary_group_id",
                "primary_group_label",
                "target_family",
                "target_domain",
                "target_group_class",
                "target_group_order",
                "target_variable_id",
                "target_title",
                "target_display_label",
                "target_short_label",
                "target_state",
                "target_cross_sample_summary_allowed",
                "target_cross_sample_note",
                "kstar_alignment_category_draft",
                "kstar_family_label",
                "kstar_mp1_8_name",
            ],
            dropna=False,
        )
        .agg(
            n_primary_components=("primary_component_number", "nunique"),
            n_primary_spots=("n_primary_spots", "first"),
            n_target_spots=("n_target_spots", "first"),
            n_target_components=("n_target_components", "first"),
            median_component_median_distance=("component_median_distance", "median"),
            mean_component_median_distance=("component_median_distance", "mean"),
            min_component_min_distance=("component_min_distance", "min"),
            median_component_mean_distance=("component_mean_distance", "median"),
        )
        .reset_index()
    )

    pairs = []
    keys = [
        "dataset",
        "sample",
        "sample_label",
        "contrast_id",
        "contrast_label",
        "contrast_role",
        "target_family",
        "target_domain",
        "target_group_class",
        "target_group_order",
        "target_variable_id",
        "target_title",
        "target_display_label",
        "target_short_label",
        "target_state",
        "target_cross_sample_summary_allowed",
        "target_cross_sample_note",
        "kstar_alignment_category_draft",
        "kstar_family_label",
        "kstar_mp1_8_name",
    ]
    for key_vals, group in sample_primary.groupby(keys, dropna=False):
        rec = dict(zip(keys, key_vals))
        source = group[group["primary_role"].eq("source")]
        reference = group[group["primary_role"].eq("reference")]
        if len(source) != 1 or len(reference) != 1:
            rec["distance_pair_status"] = "missing_source_or_reference"
        else:
            rec["distance_pair_status"] = "ok"
            source_row = source.iloc[0]
            ref_row = reference.iloc[0]
            rec["source_group_id"] = source_row["primary_group_id"]
            rec["source_group_label"] = source_row["primary_group_label"]
            rec["reference_group_id"] = ref_row["primary_group_id"]
            rec["reference_group_label"] = ref_row["primary_group_label"]
            rec["source_n_components"] = source_row["n_primary_components"]
            rec["reference_n_components"] = ref_row["n_primary_components"]
            rec["n_target_spots"] = source_row["n_target_spots"]
            rec["n_target_components"] = source_row["n_target_components"]
            rec["source_median_component_distance"] = source_row["median_component_median_distance"]
            rec["reference_median_component_distance"] = ref_row["median_component_median_distance"]
            rec["delta_source_minus_reference"] = (
                source_row["median_component_median_distance"]
                - ref_row["median_component_median_distance"]
            )
            rec["direction"] = "source_closer" if rec["delta_source_minus_reference"] < 0 else "reference_closer"
        pairs.append(rec)

    sample_delta = pd.DataFrame(pairs)
    return sample_delta


def summarize_cross_sample(sample_df: pd.DataFrame) -> pd.DataFrame:
    ok = sample_df[
        sample_df["distance_pair_status"].eq("ok")
        & clean_bool(sample_df["target_cross_sample_summary_allowed"])
    ].copy()
    grouped = (
        ok.groupby(
            [
                "contrast_id",
                "contrast_label",
                "contrast_role",
                "target_family",
                "target_domain",
                "target_group_class",
                "target_group_order",
                "target_variable_id",
                "target_title",
                "target_display_label",
                "target_short_label",
                "target_state",
                "target_cross_sample_note",
            ],
            dropna=False,
        )
        .agg(
            n_samples=("sample_label", "nunique"),
            median_delta_source_minus_reference=("delta_source_minus_reference", "median"),
            mean_delta_source_minus_reference=("delta_source_minus_reference", "mean"),
            sd_delta_source_minus_reference=("delta_source_minus_reference", "std"),
            min_delta_source_minus_reference=("delta_source_minus_reference", "min"),
            max_delta_source_minus_reference=("delta_source_minus_reference", "max"),
            fraction_samples_source_closer=(
                "delta_source_minus_reference",
                lambda x: float(np.mean(np.asarray(x, dtype=float) < 0)),
            ),
            median_source_distance=("source_median_component_distance", "median"),
            median_reference_distance=("reference_median_component_distance", "median"),
            median_target_spots=("n_target_spots", "median"),
            median_target_components=("n_target_components", "median"),
        )
        .reset_index()
    )
    grouped["abs_median_delta"] = grouped["median_delta_source_minus_reference"].abs()
    grouped = grouped.sort_values(
        ["contrast_id", "target_group_order", "abs_median_delta"],
        ascending=[True, True, False],
    )
    return grouped


def class_palette() -> dict:
    return {
        "MP": "#386cb0",
        "SpaCET": "#7fc97f",
        "Hallmark": "#f0027f",
        "K*": "#bf5b17",
    }


def plot_distance_box(sample_df: pd.DataFrame, target_family: str, contrast_id: str, path: Path) -> None:
    data = sample_df[
        sample_df["target_family"].eq(target_family)
        & sample_df["contrast_id"].eq(contrast_id)
        & sample_df["distance_pair_status"].eq("ok")
    ].copy()
    if data.empty:
        return
    data = data.sort_values(["target_group_order", "target_variable_id", "target_state"])
    order = data["target_short_label"].drop_duplicates().tolist()

    long = data.melt(
        id_vars=["target_short_label", "sample_label"],
        value_vars=["source_median_component_distance", "reference_median_component_distance"],
        var_name="group",
        value_name="median_component_distance",
    )
    group_label = {
        "source_median_component_distance": "SNAI1-ac hot",
        "reference_median_component_distance": data["reference_group_label"].iloc[0],
    }
    long["group"] = long["group"].map(group_label)
    fig_width = max(10, 0.52 * len(order))
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    sns.boxplot(
        data=long,
        x="target_short_label",
        y="median_component_distance",
        hue="group",
        order=order,
        ax=ax,
        fliersize=0,
        linewidth=0.8,
    )
    sns.stripplot(
        data=long,
        x="target_short_label",
        y="median_component_distance",
        hue="group",
        order=order,
        dodge=True,
        ax=ax,
        alpha=0.45,
        size=2.2,
        linewidth=0,
    )
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=False, loc="upper right", fontsize=8)
    ax.set_xlabel("")
    ax.set_ylabel("Median nearest-spot distance per source component")
    ax.set_title(f"{target_family.upper()} distances: {data['contrast_label'].iloc[0]}")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_delta_box(sample_df: pd.DataFrame, target_family: str, contrast_id: str, path: Path) -> None:
    data = sample_df[
        sample_df["target_family"].eq(target_family)
        & sample_df["contrast_id"].eq(contrast_id)
        & sample_df["distance_pair_status"].eq("ok")
    ].copy()
    if data.empty:
        return
    data = data.sort_values(["target_group_order", "target_variable_id", "target_state"])
    order = data["target_short_label"].drop_duplicates().tolist()
    fig_width = max(10, 0.52 * len(order))
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    sns.boxplot(
        data=data,
        x="target_short_label",
        y="delta_source_minus_reference",
        order=order,
        ax=ax,
        color="#d9d9d9",
        fliersize=0,
        linewidth=0.8,
    )
    sns.stripplot(
        data=data,
        x="target_short_label",
        y="delta_source_minus_reference",
        order=order,
        ax=ax,
        hue="direction",
        palette={"source_closer": "#386cb0", "reference_closer": "#b2182b"},
        alpha=0.75,
        size=3,
        linewidth=0,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.set_xlabel("")
    ax.set_ylabel("Delta distance: SNAI1-ac hot - reference")
    ax.set_title(f"{target_family.upper()} distance delta: {data['contrast_label'].iloc[0]}")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_class_overview(sample_df: pd.DataFrame, contrast_id: str, path: Path) -> None:
    data = sample_df[
        sample_df["contrast_id"].eq(contrast_id)
        & sample_df["distance_pair_status"].eq("ok")
    ].copy()
    if data.empty:
        return
    order = ["SpaCET", "MP", "K*", "Hallmark"]
    data["class_plot"] = data["target_group_class"].astype(str)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    sns.boxplot(
        data=data,
        x="class_plot",
        y="delta_source_minus_reference",
        order=order,
        ax=ax,
        color="#e6e6e6",
        fliersize=0,
        linewidth=0.8,
    )
    sns.stripplot(
        data=data,
        x="class_plot",
        y="delta_source_minus_reference",
        order=order,
        hue="class_plot",
        palette=class_palette(),
        ax=ax,
        alpha=0.35,
        size=2.2,
        linewidth=0,
        legend=False,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("")
    ax.set_ylabel("Delta distance: SNAI1-ac hot - reference")
    ax.set_title(f"Distance delta overview: {data['contrast_label'].iloc[0]}")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_readme(component_df: pd.DataFrame, sample_df: pd.DataFrame, cross_df: pd.DataFrame, audit_df: pd.DataFrame) -> None:
    text = f"""# SpottedPy v2 distance calculation preflight

Generated: {now_iso()}

This preflight computes the approved nearest-spot distance design without running final GEE models.

The active target universe is hotspots only. Target coldspots are excluded from this branch.

## Metric

Distances use the paper-aligned hotspot distance: for each spot in a source/reference hotspot component, find the nearest spot in the target hotspot using Visium `array_row`/`array_col`. The script uses a `cKDTree` nearest-neighbor query, which is computationally equivalent to the SpottedPy `distance_matrix(...).min(axis=1)` result for this metric.

The primary preflight summary is the median nearest-spot distance per source/reference component. Sample-level deltas are:

`delta = SNAI1-ac hot median component distance - reference median component distance`

Negative values mean the target region is closer to SNAI1-ac hot than to the reference region.

## Outputs

- component rows: {component_df.shape[0]}
- sample target rows: {sample_df.shape[0]}
- cross-sample descriptive rows excluding sample-specific K*: {cross_df.shape[0]}
- sample merge-audit rows: {audit_df.shape[0]}

No max-distance imputation was used.

Generic boxplots are intentionally not generated here; paper-style discovery figures are generated by `spottedpy_v2_distance_paperstyle_preflight_figures.py`.
"""
    README_OUT.write_text(text, encoding="utf-8")


def main() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    SCRIPT_ROOT.mkdir(parents=True, exist_ok=True)

    design = pd.read_csv(DESIGN_PATH)
    design = design[design["will_calculate_distance"].astype(str).str.lower().eq("true")].copy()
    design["target_group_order"] = pd.to_numeric(design["target_group_order"], errors="coerce").astype(int)
    design = design.sort_values(["dataset", "sample", "contrast_id", "target_group_order", "target_variable_id", "target_state"])

    all_component_rows = []
    audit_rows = []
    sample_groups = list(design.groupby("sample_label", sort=True))
    for sample_idx, (sample_label, sample_design) in enumerate(sample_groups, start=1):
        print(f"[{now_iso()}] sample {sample_idx}/{len(sample_groups)}: {sample_label}", flush=True)
        rows, audit = compute_sample(sample_design.reset_index(drop=True))
        all_component_rows.extend(rows)
        audit_rows.append(audit)

    component_df = pd.DataFrame(all_component_rows)
    audit_df = pd.DataFrame(audit_rows)
    sample_df = summarize_sample_targets(component_df)
    cross_df = summarize_cross_sample(sample_df)

    component_df.to_csv(COMPONENT_OUT, index=False)
    sample_df.to_csv(SAMPLE_OUT, index=False)
    cross_df.to_csv(CROSS_SAMPLE_OUT, index=False)
    audit_df.to_csv(SAMPLE_AUDIT_OUT, index=False)

    figure_paths = []

    if "__file__" in globals():
        shutil.copy2(Path(__file__), SCRIPT_ROOT / Path(__file__).name)

    manifest = {
        "generated_at": now_iso(),
        "input_design": str(DESIGN_PATH),
        "upstream_hotspot_layers": {
            "core_snai1_mp_kstar": str(PROJECT_ROOT / "04_hotspots_preflight_revised_scoring_policy" / "run_manifest.json"),
            "hallmark_spacet": str(PROJECT_ROOT / "04_hotspots_preflight_revised_scoring_policy" / "run_manifest_hallmark_spacet_hotspots.json"),
        },
        "outputs": {
            "component_summary": str(COMPONENT_OUT),
            "sample_target_summary": str(SAMPLE_OUT),
            "cross_sample_descriptive_summary": str(CROSS_SAMPLE_OUT),
            "sample_input_merge_audit": str(SAMPLE_AUDIT_OUT),
            "figures": figure_paths,
        },
        "n_samples": int(design["sample_label"].nunique()),
        "n_design_rows": int(design.shape[0]),
        "n_component_rows": int(component_df.shape[0]),
        "n_sample_target_rows": int(sample_df.shape[0]),
        "n_cross_sample_descriptive_rows": int(cross_df.shape[0]),
        "distance_metric": "nearest_spot_min_distance",
        "component_summary_primary": "component_median_distance",
        "delta_interpretation": "negative means target is closer to SNAI1-ac hot than to reference",
        "no_max_distance_imputation": True,
        "target_state_policy": "target hotspots only",
        "kstar_cross_sample_pooling": "not_allowed_as_same_variable",
    }
    RUN_MANIFEST_OUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_readme(component_df, sample_df, cross_df, audit_df)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
