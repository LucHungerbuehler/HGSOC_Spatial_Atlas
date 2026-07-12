from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"D:\HGSOC_Spatial_Atlas\05_analysis_ready\SpottedPy_v2_paper_aligned")
HOTSPOT_ROOT = PROJECT_ROOT / "04_hotspots_preflight_revised_scoring_policy"
INPUT_QC_ROOT = PROJECT_ROOT / "01_inputs_qc"
OUT_ROOT = PROJECT_ROOT / "05_distance_gee"
OUT_TABLES = OUT_ROOT / "tables"
OUT_SCRIPTS = OUT_ROOT / "scripts_used"

CORE_SUMMARY = HOTSPOT_ROOT / "tables" / "spottedpy_v2_hotspot_preflight_summary.csv"
HALLMARK_SPACET_SUMMARY = HOTSPOT_ROOT / "tables" / "spottedpy_v2_hallmark_spacet_hotspot_summary.csv"
CORE_COMPONENT_AUDIT = HOTSPOT_ROOT / "tables" / "spottedpy_v2_component_numbering_audit.csv"
HALLMARK_SPACET_COMPONENT_AUDIT = (
    HOTSPOT_ROOT / "tables" / "spottedpy_v2_hallmark_spacet_component_numbering_audit.csv"
)
INPUT_MANIFEST = INPUT_QC_ROOT / "tables" / "spottedpy_v2_live_input_manifest.csv"
KSTAR_PROJECTION_TABLE = Path(
    r"D:\HGSOC_Spatial_Atlas\05_analysis_ready\S3_cNMF_Tumor_Programs"
    r"\snai1ac_signature_projection_onto_cnmf_programs_v1\tables"
    r"\kstar_snai1ac_signature_projection_clean.csv"
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def safe_sample_label(label: str) -> str:
    return str(label).replace(" ", "_").replace("/", "_").replace("\\", "_")


def parse_program_p(program_id: str, fallback_title: str) -> str:
    match = re.search(r"__P(\d+)$", str(program_id))
    if match:
        return f"P{match.group(1)}"
    match = re.search(r"P(\d+)$", str(fallback_title))
    if match:
        return f"P{match.group(1)}"
    return str(fallback_title)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def coerce_counts(df: pd.DataFrame) -> pd.DataFrame:
    count_cols = [
        "n_roi_spots",
        "n_hot_spots",
        "n_cold_spots",
        "n_hot_components",
        "n_cold_components",
        "seconds",
    ]
    out = df.copy()
    for col in count_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def h5ad_path_for(row: pd.Series) -> Path:
    sample = safe_sample_label(row["sample_label"])
    family = row["family"]
    if family == "core":
        return HOTSPOT_ROOT / "h5ad" / "core_full" / f"{sample}__core_full_and_tumor_hotspots.h5ad"
    if family in {"mp", "kstar"}:
        return HOTSPOT_ROOT / "h5ad" / "mp_kstar_tumor" / f"{sample}__mp_kstar_tumor_hotspots.h5ad"
    if family in {"hallmark", "spacet"}:
        return HOTSPOT_ROOT / "h5ad" / "hallmark_spacet_full" / f"{sample}__hallmark_spacet_full_hotspots.h5ad"
    raise ValueError(f"Unexpected family: {family}")


def component_audit_key(audit: pd.DataFrame) -> pd.DataFrame:
    cols = ["sample_label", "family", "domain", "variable_id", "state"]
    keep = cols + ["number_col", "n_components", "needs_relabel", "missing_between_min_max"]
    missing = [c for c in keep if c not in audit.columns]
    if missing:
        raise ValueError(f"Component audit missing columns: {missing}")
    out = audit[keep].copy()
    out = out.rename(columns={"number_col": "audit_number_col"})
    out["audit_n_components"] = pd.to_numeric(out["n_components"], errors="coerce")
    out["audit_needs_relabel"] = out["needs_relabel"].astype(str).str.lower().eq("true")
    out = out.drop(columns=["n_components", "needs_relabel"])
    return out


def add_state_rows(summary: pd.DataFrame, component_audit: pd.DataFrame) -> pd.DataFrame:
    state_rows = []
    for _, row in summary.iterrows():
        for state in ["hot", "cold"]:
            prefix = row["scaled_col"]
            state_rows.append(
                {
                    "dataset": row["dataset"],
                    "sample": row["sample"],
                    "sample_label": row["sample_label"],
                    "family": row["family"],
                    "domain": row["domain"],
                    "variable_id": row["variable_id"],
                    "title": row["title"],
                    "raw_col": row["raw_col"],
                    "scaled_col": row["scaled_col"],
                    "scale_method": row["scale_method"],
                    "status": row["status"],
                    "state": state,
                    "state_col": f"{prefix}_{state}",
                    "number_col": f"{prefix}_{state}_number",
                    "n_roi_spots": row.get("n_roi_spots"),
                    "n_state_spots": row.get(f"n_{state}_spots"),
                    "n_state_components": row.get(f"n_{state}_components"),
                    "h5ad_path": str(h5ad_path_for(row)),
                    "h5ad_exists": h5ad_path_for(row).exists(),
                }
            )
    out = pd.DataFrame(state_rows)
    out = out.merge(
        component_audit,
        on=["sample_label", "family", "domain", "variable_id", "state"],
        how="left",
    )
    out["n_state_spots"] = pd.to_numeric(out["n_state_spots"], errors="coerce")
    out["n_state_components"] = pd.to_numeric(out["n_state_components"], errors="coerce")
    out["audit_n_components"] = pd.to_numeric(out["audit_n_components"], errors="coerce")
    out["state_exists"] = (
        out["status"].eq("ok")
        & out["h5ad_exists"].eq(True)
        & out["n_state_spots"].fillna(0).gt(0)
        & out["n_state_components"].fillna(0).gt(0)
        & out["audit_needs_relabel"].fillna(False).eq(False)
    )
    return out


def add_kstar_labels(state_rows: pd.DataFrame) -> pd.DataFrame:
    out = state_rows.copy()
    out["display_label"] = out["title"]
    out["kstar_alignment_category_draft"] = ""
    out["kstar_family_label"] = ""
    out["kstar_mp1_8_name"] = ""

    if not KSTAR_PROJECTION_TABLE.exists():
        return out

    proj = pd.read_csv(KSTAR_PROJECTION_TABLE)
    proj = proj.drop_duplicates(subset=["program_id"])
    label_map = proj.set_index("program_id").to_dict(orient="index")

    is_kstar = out["family"].eq("kstar")
    for idx, row in out[is_kstar].iterrows():
        meta = label_map.get(row["variable_id"], {})
        alignment = str(meta.get("alignment_category_draft", "")).strip()
        family_label = str(meta.get("family_label", "")).strip()
        mp_name = str(meta.get("mp1_8_name", "")).strip()
        p_label = parse_program_p(row["variable_id"], row["title"])
        if alignment and alignment.lower() != "nan":
            display = f"{p_label}: {alignment}"
        else:
            display = str(row["title"])
        out.at[idx, "display_label"] = display
        out.at[idx, "kstar_alignment_category_draft"] = alignment
        out.at[idx, "kstar_family_label"] = family_label
        out.at[idx, "kstar_mp1_8_name"] = mp_name

    return out


def require_one(state_rows: pd.DataFrame, sample_label: str, family: str, domain: str, variable_id: str, state: str) -> pd.Series:
    hit = state_rows[
        state_rows["sample_label"].eq(sample_label)
        & state_rows["family"].eq(family)
        & state_rows["domain"].eq(domain)
        & state_rows["variable_id"].eq(variable_id)
        & state_rows["state"].eq(state)
    ]
    if len(hit) != 1:
        raise ValueError(
            "Expected exactly one state row for "
            f"{sample_label} {family}/{domain}/{variable_id}/{state}, found {len(hit)}"
        )
    return hit.iloc[0]


def build_contrast_rows(samples: pd.DataFrame, state_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, sample in samples.iterrows():
        sample_label = sample["sample_label"]
        source = require_one(state_rows, sample_label, "core", "full", "SNAI1-ac_consensus", "hot")
        references = [
            (
                "snai1ac_hot_vs_snai1ac_cold",
                "SNAI1-ac consensus hot vs SNAI1-ac consensus cold",
                "primary_reference",
                require_one(state_rows, sample_label, "core", "full", "SNAI1-ac_consensus", "cold"),
            ),
            (
                "snai1ac_hot_vs_snai12r_hot",
                "SNAI1-ac consensus hot vs SNAI1-2R hot",
                "specificity_reference",
                require_one(state_rows, sample_label, "core", "full", "SNAI1-2R", "hot"),
            ),
        ]
        for contrast_id, contrast_label, contrast_role, ref in references:
            rows.append(
                {
                    "dataset": sample["dataset"],
                    "sample": sample["sample"],
                    "sample_label": sample_label,
                    "contrast_id": contrast_id,
                    "contrast_label": contrast_label,
                    "contrast_role": contrast_role,
                    "source_group_id": "snai1ac_consensus_full_hot",
                    "source_group_label": "SNAI1-ac consensus full-slide hot",
                    "source_family": source["family"],
                    "source_domain": source["domain"],
                    "source_variable_id": source["variable_id"],
                    "source_state": source["state"],
                    "source_state_col": source["state_col"],
                    "source_number_col": source["number_col"],
                    "source_n_spots": source["n_state_spots"],
                    "source_n_components": source["n_state_components"],
                    "source_h5ad_path": source["h5ad_path"],
                    "source_exists": source["state_exists"],
                    "reference_group_id": (
                        "snai1ac_consensus_full_cold"
                        if ref["variable_id"] == "SNAI1-ac_consensus"
                        else "snai12r_full_hot"
                    ),
                    "reference_group_label": (
                        "SNAI1-ac consensus full-slide cold"
                        if ref["variable_id"] == "SNAI1-ac_consensus"
                        else "SNAI1-2R full-slide hot"
                    ),
                    "reference_family": ref["family"],
                    "reference_domain": ref["domain"],
                    "reference_variable_id": ref["variable_id"],
                    "reference_state": ref["state"],
                    "reference_state_col": ref["state_col"],
                    "reference_number_col": ref["number_col"],
                    "reference_n_spots": ref["n_state_spots"],
                    "reference_n_components": ref["n_state_components"],
                    "reference_h5ad_path": ref["h5ad_path"],
                    "reference_exists": ref["state_exists"],
                }
            )
    return pd.DataFrame(rows)


def build_target_rows(state_rows: pd.DataFrame) -> pd.DataFrame:
    active = state_rows[
        (
            state_rows["family"].eq("mp")
            & state_rows["domain"].eq("tumor")
        )
        | (
            state_rows["family"].eq("kstar")
            & state_rows["domain"].eq("tumor")
        )
        | (
            state_rows["family"].eq("hallmark")
            & state_rows["domain"].eq("full")
        )
        | (
            state_rows["family"].eq("spacet")
            & state_rows["domain"].eq("full")
        )
    ].copy()
    active = active[active["state"].eq("hot")].copy()

    active["target_group_class"] = active["family"].map(
        {
            "spacet": "SpaCET",
            "mp": "MP",
            "kstar": "K*",
            "hallmark": "Hallmark",
        }
    )
    active["target_group_order"] = active["family"].map(
        {
            "spacet": 1,
            "mp": 2,
            "kstar": 3,
            "hallmark": 4,
        }
    )
    active["target_cross_sample_summary_allowed"] = active["family"].ne("kstar")
    active["target_cross_sample_note"] = active["family"].map(
        {
            "kstar": "sample_specific_raw_program_do_not_pool_as_same_variable",
            "mp": "harmonized_recurrent_MP_layer",
            "hallmark": "shared_gene_set_layer",
            "spacet": "shared_deconvolution_fraction_layer",
        }
    )

    rename = {
        "family": "target_family",
        "domain": "target_domain",
        "variable_id": "target_variable_id",
        "title": "target_title",
        "display_label": "target_display_label",
        "state": "target_state",
        "state_col": "target_state_col",
        "number_col": "target_number_col",
        "n_roi_spots": "target_n_roi_spots",
        "n_state_spots": "target_n_spots",
        "n_state_components": "target_n_components",
        "h5ad_path": "target_h5ad_path",
        "h5ad_exists": "target_h5ad_exists",
        "state_exists": "target_exists",
        "scale_method": "target_scale_method",
        "raw_col": "target_raw_col",
        "scaled_col": "target_scaled_col",
    }
    cols = [
        "dataset",
        "sample",
        "sample_label",
        "family",
        "domain",
        "variable_id",
        "title",
        "display_label",
        "state",
        "state_col",
        "number_col",
        "n_roi_spots",
        "n_state_spots",
        "n_state_components",
        "h5ad_path",
        "h5ad_exists",
        "state_exists",
        "scale_method",
        "raw_col",
        "scaled_col",
        "target_group_class",
        "target_group_order",
        "target_cross_sample_summary_allowed",
        "target_cross_sample_note",
        "kstar_alignment_category_draft",
        "kstar_family_label",
        "kstar_mp1_8_name",
    ]
    return active[cols].rename(columns=rename)


def build_specificity_rows(state_rows: pd.DataFrame) -> pd.DataFrame:
    spec = state_rows[
        state_rows["family"].eq("core")
        & state_rows["domain"].eq("full")
        & state_rows["variable_id"].isin(["SNAI1", "SNAI1-2R"])
    ].copy()
    spec["analysis_role"] = "available_specificity_layer_not_headline"
    spec["note"] = (
        "SNAI1 and SNAI1-2R are retained as specificity/comparator layers; "
        "SNAI1-2R hot is also an active reference group."
    )
    return spec[
        [
            "dataset",
            "sample",
            "sample_label",
            "analysis_role",
            "family",
            "domain",
            "variable_id",
            "title",
            "state",
            "state_col",
            "number_col",
            "n_state_spots",
            "n_state_components",
            "state_exists",
            "h5ad_path",
            "note",
        ]
    ]


def build_distance_design(contrasts: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, contrast in contrasts.iterrows():
        sample_targets = targets[targets["sample_label"].eq(contrast["sample_label"])]
        for _, target in sample_targets.iterrows():
            will_calculate = bool(
                contrast["source_exists"] and contrast["reference_exists"] and target["target_exists"]
            )
            missing_reasons = []
            if not contrast["source_exists"]:
                missing_reasons.append("source_missing")
            if not contrast["reference_exists"]:
                missing_reasons.append("reference_missing")
            if not target["target_exists"]:
                missing_reasons.append("target_missing")
            rows.append(
                {
                    **contrast.to_dict(),
                    "target_family": target["target_family"],
                    "target_domain": target["target_domain"],
                    "target_group_class": target["target_group_class"],
                    "target_group_order": target["target_group_order"],
                    "target_variable_id": target["target_variable_id"],
                    "target_title": target["target_title"],
                    "target_display_label": target["target_display_label"],
                    "target_state": target["target_state"],
                    "target_state_col": target["target_state_col"],
                    "target_number_col": target["target_number_col"],
                    "target_n_roi_spots": target["target_n_roi_spots"],
                    "target_n_spots": target["target_n_spots"],
                    "target_n_components": target["target_n_components"],
                    "target_h5ad_path": target["target_h5ad_path"],
                    "target_h5ad_exists": target["target_h5ad_exists"],
                    "target_exists": target["target_exists"],
                    "target_scale_method": target["target_scale_method"],
                    "target_cross_sample_summary_allowed": target["target_cross_sample_summary_allowed"],
                    "target_cross_sample_note": target["target_cross_sample_note"],
                    "kstar_alignment_category_draft": target["kstar_alignment_category_draft"],
                    "kstar_family_label": target["kstar_family_label"],
                    "kstar_mp1_8_name": target["kstar_mp1_8_name"],
                    "distance_spot_metric": "nearest_spot_min_distance",
                    "distance_coordinate_basis": "Visium array_row/array_col Euclidean grid coordinates",
                    "component_summary_primary": "median_of_spotwise_min_distance_within_source_component",
                    "component_summary_sensitivity": "min_and_mean_of_spotwise_min_distance_within_source_component",
                    "empty_hotspot_default_to_max_distance": False,
                    "centroid_to_centroid_primary": False,
                    "will_calculate_distance": will_calculate,
                    "missing_reason": ";".join(missing_reasons),
                }
            )
    return pd.DataFrame(rows)


def write_readme(path: Path, n_design_rows: int, n_available: int) -> None:
    text = f"""# SpottedPy v2 distance/GEE preflight

Generated: {now_iso()}

This branch contains the design and availability preflight for the SpottedPy v2 distance layer.
The downstream preflight scripts compute distances and SpottedPy-native review figures from this design; final report-facing GEE statistics should still come from the full distance/GEE run after the design is frozen.

## Distance measure

Primary distance follows the SpottedPy paper/source implementation as a nearest-spot hotspot distance:

- for each spot in a source hotspot/component, compute the distance to the nearest spot in a target hotspot/component;
- SpottedPy implements this with `scipy.spatial.distance_matrix` over `array_row` and `array_col`;
- the primary component summary for our next step is the median of these spotwise minimum distances per source component;
- minimum and mean summaries are retained as sensitivity;
- centroid-to-centroid distance is **not** the primary design;
- empty hotspots are not imputed with a maximum distance.

## Active source/reference contrasts

The active source group is full-slide `SNAI1-ac_consensus` hot.

Two reference groups are included:

1. full-slide `SNAI1-ac_consensus` cold;
2. full-slide `SNAI1-2R` hot as a specificity/reference layer.

## Active target universe

Targets are MP1-MP8 tumor hotspots, sample-specific K* tumor hotspots, full-slide SpaCET hotspots, and full-slide Hallmark hotspots. Target coldspots are not part of the active distance design. K* targets are kept sample-specific and are flagged as not eligible for naive cross-sample pooling as the same variable.

## Preflight size

- design rows: {n_design_rows}
- rows currently available for distance calculation: {n_available}

## Target-state policy

The target universe is hotspots only. Target coldspots are not included in the active distance design because the discovery question is proximity from SNAI1-ac source/reference regions to other spatially enriched programs or cell-state regions.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_SCRIPTS.mkdir(parents=True, exist_ok=True)

    core_summary = coerce_counts(read_csv(CORE_SUMMARY))
    hallmark_spacet_summary = coerce_counts(read_csv(HALLMARK_SPACET_SUMMARY))
    input_manifest = read_csv(INPUT_MANIFEST)
    core_audit = component_audit_key(read_csv(CORE_COMPONENT_AUDIT))
    hallmark_spacet_audit = component_audit_key(read_csv(HALLMARK_SPACET_COMPONENT_AUDIT))

    summary = pd.concat([core_summary, hallmark_spacet_summary], ignore_index=True)
    component_audit = pd.concat([core_audit, hallmark_spacet_audit], ignore_index=True)
    state_rows = add_state_rows(summary, component_audit)
    state_rows = add_kstar_labels(state_rows)

    samples = input_manifest[["dataset", "sample", "sample_label"]].drop_duplicates().sort_values(
        ["dataset", "sample"]
    )
    contrasts = build_contrast_rows(samples, state_rows)
    targets = build_target_rows(state_rows)
    specificity = build_specificity_rows(state_rows)
    design = build_distance_design(contrasts, targets)

    method = pd.DataFrame(
        [
            {
                "analysis_stage": "distance_preflight",
                "paper_term": "shortest_path_to_hotspot",
                "spottedpy_function_basis": "hotspot_helper.calculateDistances",
                "spottedpy_implementation": "distance_matrix(primary_points[array_row,array_col], comparison_points[array_row,array_col]) followed by per-source-spot min_distance",
                "primary_spot_metric": "minimum distance from each source-region spot to the nearest target-region spot",
                "primary_component_summary_for_next_step": "median of source-spot min_distance values per source component",
                "sensitivity_component_summaries": "min, mean",
                "coordinate_basis": "Visium array_row/array_col grid coordinates",
                "centroid_to_centroid_primary": False,
                "empty_hotspot_default_to_max_distance": False,
                "no_imputation_policy": "if a hotspot/coldspot does not exist, the row is unavailable and no distance is imputed",
            }
        ]
    )

    family_summary = (
        design.groupby(["contrast_id", "target_family", "target_state"], dropna=False)
        .agg(
            n_rows=("will_calculate_distance", "size"),
            n_available=("will_calculate_distance", "sum"),
            n_samples=("sample_label", "nunique"),
            n_target_variables=("target_variable_id", "nunique"),
            median_target_spots=("target_n_spots", "median"),
            median_target_components=("target_n_components", "median"),
        )
        .reset_index()
    )
    family_summary["n_unavailable"] = family_summary["n_rows"] - family_summary["n_available"]

    sample_summary = (
        design.groupby(["sample_label", "dataset", "sample", "contrast_id"], dropna=False)
        .agg(
            n_rows=("will_calculate_distance", "size"),
            n_available=("will_calculate_distance", "sum"),
            source_n_spots=("source_n_spots", "first"),
            source_n_components=("source_n_components", "first"),
            reference_n_spots=("reference_n_spots", "first"),
            reference_n_components=("reference_n_components", "first"),
            n_mp_targets=("target_family", lambda x: int((x == "mp").sum())),
            n_kstar_targets=("target_family", lambda x: int((x == "kstar").sum())),
            n_spacet_targets=("target_family", lambda x: int((x == "spacet").sum())),
            n_hallmark_targets=("target_family", lambda x: int((x == "hallmark").sum())),
        )
        .reset_index()
    )
    sample_summary["n_unavailable"] = sample_summary["n_rows"] - sample_summary["n_available"]

    outputs = {
        "distance_measure_manifest": OUT_TABLES / "spottedpy_v2_distance_measure_manifest.csv",
        "distance_contrast_manifest": OUT_TABLES / "spottedpy_v2_distance_contrast_manifest.csv",
        "distance_target_manifest": OUT_TABLES / "spottedpy_v2_distance_target_manifest.csv",
        "distance_specificity_layer_manifest": OUT_TABLES
        / "spottedpy_v2_distance_specificity_layer_manifest.csv",
        "distance_availability_preflight": OUT_TABLES
        / "spottedpy_v2_distance_availability_preflight.csv",
        "distance_availability_summary_by_family": OUT_TABLES
        / "spottedpy_v2_distance_availability_summary_by_family.csv",
        "distance_availability_summary_by_sample": OUT_TABLES
        / "spottedpy_v2_distance_availability_summary_by_sample.csv",
    }

    method.to_csv(outputs["distance_measure_manifest"], index=False)
    contrasts.to_csv(outputs["distance_contrast_manifest"], index=False)
    targets.to_csv(outputs["distance_target_manifest"], index=False)
    specificity.to_csv(outputs["distance_specificity_layer_manifest"], index=False)
    design.to_csv(outputs["distance_availability_preflight"], index=False)
    family_summary.to_csv(outputs["distance_availability_summary_by_family"], index=False)
    sample_summary.to_csv(outputs["distance_availability_summary_by_sample"], index=False)

    if "__file__" in globals():
        shutil.copy2(Path(__file__), OUT_SCRIPTS / Path(__file__).name)

    manifest = {
        "generated_at": now_iso(),
        "project_root": str(PROJECT_ROOT),
        "out_root": str(OUT_ROOT),
        "inputs": {
            "core_summary": str(CORE_SUMMARY),
            "hallmark_spacet_summary": str(HALLMARK_SPACET_SUMMARY),
            "core_component_audit": str(CORE_COMPONENT_AUDIT),
            "hallmark_spacet_component_audit": str(HALLMARK_SPACET_COMPONENT_AUDIT),
            "input_manifest": str(INPUT_MANIFEST),
            "kstar_projection_table": str(KSTAR_PROJECTION_TABLE),
        },
        "upstream_hotspot_layers": {
            "core_snai1_mp_kstar": {
                "root": str(HOTSPOT_ROOT),
                "summary": str(CORE_SUMMARY),
                "component_numbering_audit": str(CORE_COMPONENT_AUDIT),
                "domains": ["full for SNAI1-family", "tumor for MP1-MP8 and sample-specific K*"],
            },
            "hallmark_spacet": {
                "root": str(HOTSPOT_ROOT),
                "run_manifest": str(HOTSPOT_ROOT / "run_manifest_hallmark_spacet_hotspots.json"),
                "summary": str(HALLMARK_SPACET_SUMMARY),
                "component_numbering_audit": str(HALLMARK_SPACET_COMPONENT_AUDIT),
                "domains": ["full for Hallmark score_genes and SpaCET fractions"],
                "plotting_policy": "calculation-only at hotspot stage; only selected downstream distance results are plotted",
            },
        },
        "outputs": {key: str(value) for key, value in outputs.items()},
        "n_samples": int(samples["sample_label"].nunique()),
        "n_contrast_rows": int(len(contrasts)),
        "n_target_rows": int(len(targets)),
        "n_design_rows": int(len(design)),
        "n_available_design_rows": int(design["will_calculate_distance"].sum()),
        "n_unavailable_design_rows": int((~design["will_calculate_distance"]).sum()),
        "distance_measure": {
            "primary": "nearest_spot_min_distance",
            "component_summary_primary": "median",
            "component_summary_sensitivity": ["min", "mean"],
            "centroid_to_centroid_primary": False,
            "empty_hotspot_default_to_max_distance": False,
        },
        "target_state_policy": "hotspots_only",
        "reference_groups": [
            "SNAI1-ac consensus full-slide cold",
            "SNAI1-2R full-slide hot",
        ],
    }
    (OUT_ROOT / "run_manifest_distance_preflight.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    write_readme(
        OUT_ROOT / "README_distance_preflight.md",
        n_design_rows=int(len(design)),
        n_available=int(design["will_calculate_distance"].sum()),
    )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
