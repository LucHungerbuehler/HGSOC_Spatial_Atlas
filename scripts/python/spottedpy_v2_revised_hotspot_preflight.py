"""
Run SpottedPy v2 hotspot preflight with the revised scoring policy.

This is the primary replacement for the initial EnrichMap/minmax hotspot
preflight. It keeps the same output structure as the renamed 03 provenance
layer, but changes the inputs:

- SNAI1-ac: consensus hot/cold masks from the corrected/smoothed and
  unsmoothed/uncorrected Gi* calls in the 03 layer.
- SNAI1 and SNAI1-2R: scanpy.tl.score_genes on positive-arm genes.
- MP1-MP8: scanpy.tl.score_genes on the final Variant B MP gene lists.
- K* programmes: raw sample-specific K* usage, unchanged.

Run this as a script file from PowerShell using the spottedpy conda env Python.
Do not use ad hoc python -c probes on Windows for scanpy/squidpy/anndata work.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spottedpy_v2_hotspot_preflight import (  # noqa: E402
    HotspotSpec,
    add_spatial_obs,
    import_spottedpy,
    minmax,
    nonnegative_raw,
    plot_contact_sheet,
    register_anndata_null_reader,
    require,
    run_specs,
    safe_name,
    stop,
    valid_component_mask,
)


DATA_ROOT = Path(r"D:\HGSOC_Spatial_Atlas")
ANALYSIS_ROOT = DATA_ROOT / "05_analysis_ready"
CODE_ROOT = Path(r"C:\Users\luchu\Documents\MSc\Master Thesis\Code\HGSOC_Spatial_Atlas")

RUN_ROOT = ANALYSIS_ROOT / "SpottedPy_v2_paper_aligned"
INPUT_QC = RUN_ROOT / "01_inputs_qc"
INPUT_MANIFEST = INPUT_QC / "tables" / "spottedpy_v2_live_input_manifest.csv"
PREVIOUS_ROOT = RUN_ROOT / "03_hotspots_preflight_enrichmap_minmax_initial"

OUT_ROOT = RUN_ROOT / "04_hotspots_preflight_revised_scoring_policy"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures" / "contact_sheets"
H5AD_DIR = OUT_ROOT / "h5ad"
SCRIPT_DIR = OUT_ROOT / "scripts_used"

SIGNATURE_ROOT = ANALYSIS_ROOT / "Signature"
CNMF_ROOT = ANALYSIS_ROOT / "S3_cNMF_Tumor_Programs"
MP_SIGNATURE_DIR = (
    CNMF_ROOT
    / "jaccard_raw_matrices"
    / "inspection_exports_average"
    / "variantB_nonjunk_manual_cut_v2"
    / "subcluster_signatures_scoring"
    / "signatures"
)

NEIGHBOURS_PRIMARY = 10
P_VALUE = 0.05
PERMUTATIONS = 999
SEED = 100
MIN_SCOREGENES_OVERLAP = 5

SNAI1_POSITIVE_ARM = {
    "SNAI1": {
        "weights_file": SIGNATURE_ROOT / "snai1_vs_gfp_weights.json",
        "title": "SNAI1 score_genes",
    },
    "SNAI1-2R": {
        "weights_file": SIGNATURE_ROOT / "snai12r_vs_gfp_weights.json",
        "title": "SNAI1-2R score_genes",
    },
}

MP_GENE_FILES = {
    "MP1_angiogenic_vascular_scoregenes": {
        "file": MP_SIGNATURE_DIR / "MP1_angiogenic_vascular.genes.txt",
        "title": "MP1 angiogenic/vascular",
    },
    "MP2_iCAF_stress_scoregenes": {
        "file": MP_SIGNATURE_DIR / "MP2_iCAF_stress.genes.txt",
        "title": "MP2 iCAF-stress",
    },
    "MP3_complement_CAF_scoregenes": {
        "file": MP_SIGNATURE_DIR / "MP3_complement_CAF.genes.txt",
        "title": "MP3 complement-CAF",
    },
    "MP4_activated_myCAF_scoregenes": {
        "file": MP_SIGNATURE_DIR / "MP4_activated_myCAF.genes.txt",
        "title": "MP4 activated-myCAF",
    },
    "MP5_IFN_TLS_immune_scoregenes": {
        "file": MP_SIGNATURE_DIR / "MP5_IFN_TLS_immune.genes.txt",
        "title": "MP5 IFN/TLS immune",
    },
    "MP6_APC_TAM_myeloid_scoregenes": {
        "file": MP_SIGNATURE_DIR / "MP6_APC_TAM_myeloid.genes.txt",
        "title": "MP6 APC/TAM myeloid",
    },
    "MP7_malignant_hypoxia_scoregenes": {
        "file": MP_SIGNATURE_DIR / "MP7_malignant_hypoxia.genes.txt",
        "title": "MP7 malignant hypoxia",
    },
    "MP8_malignant_acute_phase_secretory_scoregenes": {
        "file": MP_SIGNATURE_DIR / "MP8_malignant_acute_phase_secretory.genes.txt",
        "title": "MP8 malignant acute-phase/secretory",
    },
}


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
    old_script = CODE_ROOT / "scripts" / "spottedpy_v2_hotspot_preflight.py"
    if old_script.exists():
        shutil.copy2(old_script, SCRIPT_DIR / old_script.name)


def load_inputs() -> pd.DataFrame:
    require(INPUT_MANIFEST)
    require(PREVIOUS_ROOT)
    manifest = pd.read_csv(INPUT_MANIFEST)
    if len(manifest) != 23:
        stop(f"Expected 23 samples in input manifest; found {len(manifest)}")
    return manifest


def read_gene_list(path: Path) -> list[str]:
    require(path)
    genes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [gene for gene in genes if gene]


def read_positive_weight_genes(path: Path) -> list[str]:
    require(path)
    weights = json.loads(path.read_text(encoding="utf-8"))
    genes = [gene for gene, value in weights.items() if float(value) > 0]
    if not genes:
        stop(f"No positive-weight genes found in {path}")
    return genes


def load_scoregene_sets() -> dict[str, dict[str, Any]]:
    gene_sets: dict[str, dict[str, Any]] = {}
    for variable_id, meta in SNAI1_POSITIVE_ARM.items():
        genes = read_positive_weight_genes(meta["weights_file"])
        gene_sets[variable_id] = {
            "family": "core",
            "title": meta["title"],
            "source_path": str(meta["weights_file"]),
            "source_logic": "positive_weight_arm_from_signed_json",
            "genes": genes,
        }
    for variable_id, meta in MP_GENE_FILES.items():
        genes = read_gene_list(meta["file"])
        gene_sets[variable_id] = {
            "family": "mp",
            "title": meta["title"],
            "source_path": str(meta["file"]),
            "source_logic": "final_variantB_manual_cut_gene_list",
            "genes": genes,
        }
    return gene_sets


def load_kstar_usage(row: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    path = Path(str(row["kstar_usage_path"]))
    require(path)
    usage = pd.read_csv(path)
    metadata = {"spot_id", "dataset", "sample_id_on_disk", "sample_label"}
    program_cols = [col for col in usage.columns if col not in metadata]
    for col in program_cols:
        usage[col] = pd.to_numeric(usage[col], errors="coerce")
    return usage, program_cols


def score_genes_to_obs(
    adata,
    genes: list[str],
    raw_col: str,
    domain_mask: pd.Series | None,
    gene_meta: dict[str, Any],
    sample_label: str,
) -> dict[str, Any]:
    import scanpy as sc

    if adata.var_names.has_duplicates:
        adata.var_names_make_unique()
    available = [gene for gene in genes if gene in adata.var_names]
    missing = [gene for gene in genes if gene not in adata.var_names]
    audit = {
        "sample_label": sample_label,
        "raw_col": raw_col,
        "variable_id": raw_col,
        "source_path": gene_meta["source_path"],
        "source_logic": gene_meta["source_logic"],
        "n_genes_requested": len(genes),
        "n_genes_available": len(available),
        "n_genes_missing": len(missing),
        "missing_genes_preview": ",".join(missing[:20]),
        "scoregenes_status": "ok",
        "scoregenes_error": "",
    }
    adata.obs[raw_col] = np.nan
    if len(available) < MIN_SCOREGENES_OVERLAP:
        audit["scoregenes_status"] = "skipped_too_few_genes"
        audit["scoregenes_error"] = f"{len(available)} available genes"
        return audit

    try:
        if domain_mask is None:
            work = adata.copy()
            sc.tl.score_genes(
                work,
                gene_list=available,
                score_name=raw_col,
                ctrl_size=200,
                n_bins=25,
                random_state=0,
                use_raw=False,
            )
            adata.obs[raw_col] = pd.to_numeric(work.obs[raw_col], errors="coerce").reindex(adata.obs.index)
        else:
            mask = domain_mask.reindex(adata.obs.index).fillna(False).astype(bool)
            work = adata[mask].copy()
            sc.tl.score_genes(
                work,
                gene_list=available,
                score_name=raw_col,
                ctrl_size=200,
                n_bins=25,
                random_state=0,
                use_raw=False,
            )
            adata.obs.loc[mask, raw_col] = pd.to_numeric(work.obs[raw_col], errors="coerce")
    except Exception as exc:  # keep the run inspectable sample-by-sample
        adata.obs[raw_col] = np.nan
        audit["scoregenes_status"] = "error"
        audit["scoregenes_error"] = repr(exc)
    return audit


def add_scaled_scoregene_spec(
    adata,
    raw_col: str,
    scaled_col: str,
    domain_mask: pd.Series | None,
    spec: HotspotSpec,
) -> tuple[HotspotSpec, dict[str, Any]]:
    if domain_mask is None:
        adata.obs[scaled_col], scale_audit = minmax(adata.obs[raw_col])
    else:
        mask = domain_mask.reindex(adata.obs.index).fillna(False).astype(bool)
        scaled = pd.Series(np.nan, index=adata.obs.index, dtype=float)
        scaled.loc[mask], scale_audit = minmax(adata.obs.loc[mask, raw_col])
        adata.obs[scaled_col] = scaled
    return spec, {
        "raw_col": raw_col,
        "scaled_col": scaled_col,
        "domain": spec.domain,
        "scale_method": spec.scale_method,
        **scale_audit,
    }


def label_consensus_components(adata, mask: pd.Series, value_col: str, out_col: str, out_number_col: str) -> int:
    from spottedpy.hotspot_helper import find_connected_components

    adata.obs[out_col] = np.nan
    adata.obs[out_number_col] = ""
    mask = mask.reindex(adata.obs.index).fillna(False).astype(bool)
    if int(mask.sum()) <= 1:
        return 0
    hotspot = adata.obs.loc[mask].copy()
    labelled, _ = find_connected_components(hotspot, adata)
    if labelled.empty or "hotspot_label" not in labelled.columns:
        return 0
    adata.obs.loc[labelled.index, out_col] = pd.to_numeric(adata.obs.loc[labelled.index, value_col], errors="coerce")
    adata.obs.loc[labelled.index, out_number_col] = labelled["hotspot_label"].astype(str)
    labels = adata.obs[out_number_col].astype(str)
    return int(labels[valid_component_mask(labels)].nunique())


def previous_core_path(sample_label: str) -> Path:
    return (
        PREVIOUS_ROOT
        / "h5ad"
        / "core_full"
        / f"{safe_name(sample_label)}__core_full_and_tumor_hotspots.h5ad"
    )


def add_snai1ac_consensus(adata, previous, sample_label: str, domain: str) -> tuple[HotspotSpec, dict[str, Any], dict[str, Any]]:
    if domain == "full":
        score_source = "SNAI1-ac_score"
        scaled_col = "spv2_full_SNAI1-ac_consensus"
        corrected_prefix = "spv2_full_SNAI1-ac_score_minmax"
        unsmoothed_prefix = "spv2_full_snai1ac_unsmoothed_uncorrected_minmax"
        domain_mask = pd.Series(True, index=adata.obs.index)
    elif domain == "tumor":
        score_source = "SNAI1-ac_score"
        scaled_col = "spv2_tumor_SNAI1-ac_consensus"
        corrected_prefix = "spv2_tumor_SNAI1-ac_score_minmax"
        unsmoothed_prefix = "spv2_tumor_snai1ac_unsmoothed_uncorrected_minmax"
        domain_mask = adata.obs["interface"].astype(str).eq("Tumor")
    else:
        stop(f"Unknown consensus domain: {domain}")

    prev_obs = previous.obs.reindex(adata.obs.index)
    required_cols = [
        f"{corrected_prefix}_hot",
        f"{corrected_prefix}_cold",
        f"{unsmoothed_prefix}_hot",
        f"{unsmoothed_prefix}_cold",
    ]
    missing_cols = [col for col in required_cols if col not in prev_obs]
    if missing_cols:
        stop(f"{sample_label} previous 03 h5ad missing columns: {missing_cols}")

    if domain == "full":
        adata.obs[scaled_col], scale_audit = minmax(adata.obs[score_source])
    else:
        scaled = pd.Series(np.nan, index=adata.obs.index, dtype=float)
        scaled.loc[domain_mask], scale_audit = minmax(adata.obs.loc[domain_mask, score_source])
        adata.obs[scaled_col] = scaled

    corrected_hot = pd.to_numeric(prev_obs[f"{corrected_prefix}_hot"], errors="coerce").notna()
    unsmoothed_hot = pd.to_numeric(prev_obs[f"{unsmoothed_prefix}_hot"], errors="coerce").notna()
    corrected_cold = pd.to_numeric(prev_obs[f"{corrected_prefix}_cold"], errors="coerce").notna()
    unsmoothed_cold = pd.to_numeric(prev_obs[f"{unsmoothed_prefix}_cold"], errors="coerce").notna()
    domain_mask = domain_mask.reindex(adata.obs.index).fillna(False).astype(bool)
    hot_consensus = corrected_hot & unsmoothed_hot & domain_mask
    cold_consensus = corrected_cold & unsmoothed_cold & domain_mask

    hot_col = f"{scaled_col}_hot"
    cold_col = f"{scaled_col}_cold"
    hot_num = f"{scaled_col}_hot_number"
    cold_num = f"{scaled_col}_cold_number"
    t0 = time.time()
    n_hot_components = label_consensus_components(adata, hot_consensus, scaled_col, hot_col, hot_num)
    n_cold_components = label_consensus_components(adata, cold_consensus, scaled_col, cold_col, cold_num)

    n_hot = int(pd.to_numeric(adata.obs[hot_col], errors="coerce").notna().sum())
    n_cold = int(pd.to_numeric(adata.obs[cold_col], errors="coerce").notna().sum())
    spec = HotspotSpec(
        family="core",
        variable_id="SNAI1-ac_consensus",
        raw_col=score_source,
        scaled_col=scaled_col,
        title="SNAI1-ac consensus",
        domain=domain,
        scale_method="consensus_intersection_of_03_corrected_and_unsmoothed_calls",
    )
    summary = {
        "sample_label": sample_label,
        "family": spec.family,
        "domain": spec.domain,
        "variable_id": spec.variable_id,
        "title": spec.title,
        "raw_col": spec.raw_col,
        "scaled_col": spec.scaled_col,
        "scale_method": spec.scale_method,
        "status": "ok",
        "error": "",
        "n_roi_spots": int(domain_mask.sum()),
        "n_hot_spots": n_hot,
        "n_cold_spots": n_cold,
        "n_hot_components": n_hot_components,
        "n_cold_components": n_cold_components,
        "seconds": round(time.time() - t0, 3),
        "consensus_hot_before_component_filter": int(hot_consensus.sum()),
        "consensus_cold_before_component_filter": int(cold_consensus.sum()),
    }
    scaling = {
        "sample_label": sample_label,
        "raw_col": score_source,
        "scaled_col": scaled_col,
        "domain": domain,
        "scale_method": spec.scale_method,
        "source_corrected_prefix": corrected_prefix,
        "source_unsmoothed_prefix": unsmoothed_prefix,
        "corrected_hot_spots": int(corrected_hot.sum()),
        "unsmoothed_hot_spots": int(unsmoothed_hot.sum()),
        "corrected_cold_spots": int(corrected_cold.sum()),
        "unsmoothed_cold_spots": int(unsmoothed_cold.sum()),
        **scale_audit,
    }
    return spec, summary, scaling


def make_core_scoregenes_specs(
    adata,
    gene_sets: dict[str, dict[str, Any]],
    sample_label: str,
    domain: str,
) -> tuple[list[HotspotSpec], list[dict[str, Any]], list[dict[str, Any]]]:
    specs: list[HotspotSpec] = []
    scaling_records: list[dict[str, Any]] = []
    gene_audit: list[dict[str, Any]] = []
    if domain == "full":
        domain_mask = None
    elif domain == "tumor":
        domain_mask = adata.obs["interface"].astype(str).eq("Tumor")
    else:
        stop(f"Unknown score_genes domain: {domain}")

    for variable_id in ["SNAI1", "SNAI1-2R"]:
        meta = gene_sets[variable_id]
        raw_col = f"scoregenes_{safe_name(variable_id)}_positive_arm"
        scaled_col = f"spv2_{domain}_{safe_name(raw_col)}_minmax"
        audit = score_genes_to_obs(adata, meta["genes"], raw_col, domain_mask, meta, sample_label)
        audit.update({"family": "core", "domain": domain, "variable_id": variable_id})
        gene_audit.append(audit)
        spec = HotspotSpec(
            family="core",
            variable_id=variable_id,
            raw_col=raw_col,
            scaled_col=scaled_col,
            title=meta["title"],
            domain=domain,
            scale_method=f"{domain}_score_genes_positive_arm_then_minmax",
        )
        spec, scale = add_scaled_scoregene_spec(adata, raw_col, scaled_col, domain_mask, spec)
        scale.update({"variable_id": variable_id, "scoregenes_status": audit["scoregenes_status"]})
        scaling_records.append(scale)
        specs.append(spec)
    return specs, scaling_records, gene_audit


def make_mp_scoregenes_specs(
    adata,
    gene_sets: dict[str, dict[str, Any]],
    sample_label: str,
) -> tuple[list[HotspotSpec], list[dict[str, Any]], list[dict[str, Any]]]:
    specs: list[HotspotSpec] = []
    scaling_records: list[dict[str, Any]] = []
    gene_audit: list[dict[str, Any]] = []
    for variable_id, meta in gene_sets.items():
        if meta["family"] != "mp":
            continue
        raw_col = variable_id
        scaled_col = f"spv2_tumor_{safe_name(raw_col)}_minmax"
        audit = score_genes_to_obs(adata, meta["genes"], raw_col, None, meta, sample_label)
        audit.update({"family": "mp", "domain": "tumor", "variable_id": variable_id})
        gene_audit.append(audit)
        spec = HotspotSpec(
            family="mp",
            variable_id=variable_id,
            raw_col=raw_col,
            scaled_col=scaled_col,
            title=meta["title"],
            domain="tumor",
            scale_method="tumor_score_genes_final_mp_list_then_minmax",
        )
        spec, scale = add_scaled_scoregene_spec(adata, raw_col, scaled_col, None, spec)
        scale.update({"variable_id": variable_id, "scoregenes_status": audit["scoregenes_status"]})
        scaling_records.append(scale)
        specs.append(spec)
    return specs, scaling_records, gene_audit


def make_kstar_specs(adata, kstar_usage: pd.DataFrame, kstar_cols: list[str]) -> tuple[list[HotspotSpec], list[dict[str, Any]]]:
    specs: list[HotspotSpec] = []
    scaling_records: list[dict[str, Any]] = []
    kstar_by_spot = kstar_usage.set_index("spot_id")
    for col in kstar_cols:
        raw_col = f"kstar_usage__{safe_name(col)}"
        scaled_col = f"spv2_tumor_kstar_{safe_name(col)}_raw"
        adata.obs[raw_col] = adata.obs["spot"].astype(str).map(kstar_by_spot[col])
        adata.obs[scaled_col], audit = nonnegative_raw(adata.obs[raw_col])
        title = col.split("__", 2)[-1] if "__" in col else col
        specs.append(HotspotSpec("kstar", col, raw_col, scaled_col, title, "tumor", "raw_nonnegative"))
        scaling_records.append(
            {"raw_col": raw_col, "scaled_col": scaled_col, "domain": "tumor", "variable_id": col, **audit}
        )
    return specs, scaling_records


def write_readme(summary: pd.DataFrame, plot_manifest: pd.DataFrame, gene_audit: pd.DataFrame) -> None:
    ok = summary[summary["status"].eq("ok")]
    lines = [
        "# SpottedPy v2 Revised Hotspot Preflight",
        "",
        "This folder is the corrected primary hotspot/coldspot preflight layer for visual review.",
        "It mirrors the renamed 03 folder structure but replaces the initial mixed EnrichMap/minmax strategy.",
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
        "## Revised scoring policy",
        "",
        "- SNAI1-ac hot/cold maps are consensus intersections between corrected/smoothed and unsmoothed/uncorrected 03 Gi* calls.",
        "- Consensus components are relabeled after intersection with SpottedPy-like one-ring Visium connectivity and <5 spot filtering.",
        "- SNAI1 and SNAI1-2R use `scanpy.tl.score_genes` on the positive arm of the signed overexpression JSONs, then domain-wise min-max scaling.",
        "- MP1-MP8 use `scanpy.tl.score_genes` on the final Variant B MP gene-list files, then tumor-wise min-max scaling.",
        "- K* programmes remain raw nonnegative sample-specific usage values.",
        "",
        "## Not included yet",
        "",
        "- No Hallmark hotspot calls are run in this stage; those require the narrowed Hallmark set decision before plotting.",
        "- No distance statistics, GEE, neighborhood enrichment, perimeter analysis, or k=8 sensitivity are run here.",
        "",
        "## Counts",
        "",
        f"- Hotspot tests represented: {len(summary)}",
        f"- Successful hotspot tests/consensus rows: {len(ok)}",
        f"- Figure files: {len(plot_manifest)}",
        f"- Gene-set scoring rows: {len(gene_audit)}",
        f"- Tests with any hot spots: {int((summary['n_hot_spots'] > 0).sum())}",
        f"- Tests with any cold spots: {int((summary['n_cold_spots'] > 0).sum())}",
        "",
        "Review hot/cold maps and numbered component maps before distance statistics.",
        "",
    ]
    (OUT_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    prepare_dirs()
    manifest = load_inputs()
    import_spottedpy()
    register_anndata_null_reader()

    import anndata as ad

    gene_sets = load_scoregene_sets()
    all_summary: list[dict[str, Any]] = []
    all_scaling: list[dict[str, Any]] = []
    all_gene_audit: list[dict[str, Any]] = []
    all_plots: list[dict[str, Any]] = []

    for row in manifest.sort_values(["dataset", "sample"]).to_dict("records"):
        dataset = str(row["dataset"])
        sample = str(row["sample"])
        sample_label = str(row["sample_label"])
        print(f"[{datetime.now().isoformat(timespec='seconds')}] Revised hotspot preflight {sample_label}", flush=True)

        kstar_usage, kstar_cols = load_kstar_usage(pd.Series(row))

        full = ad.read_h5ad(str(row["full_h5ad_path"]))
        full.uns.clear()
        add_spatial_obs(full, dataset, sample)
        prev_path = previous_core_path(sample_label)
        require(prev_path)
        previous = ad.read_h5ad(prev_path)

        core_full_specs: list[HotspotSpec] = []
        core_tumor_specs: list[HotspotSpec] = []

        spec, summary, scaling = add_snai1ac_consensus(full, previous, sample_label, "full")
        core_full_specs.append(spec)
        all_summary.append({"dataset": dataset, "sample": sample, **summary})
        all_scaling.append({"dataset": dataset, "sample": sample, **scaling})

        score_specs, scaling_records, gene_audit = make_core_scoregenes_specs(full, gene_sets, sample_label, "full")
        core_full_specs.extend(score_specs)
        all_scaling.extend({"dataset": dataset, "sample": sample, "sample_label": sample_label, **r} for r in scaling_records)
        all_gene_audit.extend({"dataset": dataset, "sample": sample, **r} for r in gene_audit)
        full, summary_records = run_specs(import_spottedpy(), full, score_specs, sample_label)
        all_summary.extend({"dataset": dataset, "sample": sample, **r} for r in summary_records)

        spec, summary, scaling = add_snai1ac_consensus(full, previous, sample_label, "tumor")
        core_tumor_specs.append(spec)
        all_summary.append({"dataset": dataset, "sample": sample, **summary})
        all_scaling.append({"dataset": dataset, "sample": sample, **scaling})

        score_specs, scaling_records, gene_audit = make_core_scoregenes_specs(full, gene_sets, sample_label, "tumor")
        core_tumor_specs.extend(score_specs)
        all_scaling.extend({"dataset": dataset, "sample": sample, "sample_label": sample_label, **r} for r in scaling_records)
        all_gene_audit.extend({"dataset": dataset, "sample": sample, **r} for r in gene_audit)
        full, summary_records = run_specs(import_spottedpy(), full, score_specs, sample_label)
        all_summary.extend({"dataset": dataset, "sample": sample, **r} for r in summary_records)
        full.write_h5ad(H5AD_DIR / "core_full" / f"{safe_name(sample_label)}__core_full_and_tumor_hotspots.h5ad")

        tumor = ad.read_h5ad(str(row["tumor_subset_h5ad_path"]))
        tumor.uns.clear()
        add_spatial_obs(tumor, dataset, sample)
        mp_specs, scaling_records, gene_audit = make_mp_scoregenes_specs(tumor, gene_sets, sample_label)
        all_scaling.extend({"dataset": dataset, "sample": sample, "sample_label": sample_label, **r} for r in scaling_records)
        all_gene_audit.extend({"dataset": dataset, "sample": sample, **r} for r in gene_audit)
        tumor, summary_records = run_specs(import_spottedpy(), tumor, mp_specs, sample_label)
        all_summary.extend({"dataset": dataset, "sample": sample, **r} for r in summary_records)

        kstar_specs, scaling_records = make_kstar_specs(tumor, kstar_usage, kstar_cols)
        all_scaling.extend({"dataset": dataset, "sample": sample, "sample_label": sample_label, **r} for r in scaling_records)
        tumor, summary_records = run_specs(import_spottedpy(), tumor, kstar_specs, sample_label)
        all_summary.extend({"dataset": dataset, "sample": sample, **r} for r in summary_records)
        tumor.write_h5ad(H5AD_DIR / "mp_kstar_tumor" / f"{safe_name(sample_label)}__mp_kstar_tumor_hotspots.h5ad")

        figure_jobs = [
            ("core_full_hotcold", full, core_full_specs, False),
            ("core_full_numbered", full, core_full_specs, True),
            ("core_tumor_hotcold", full, core_tumor_specs, False),
            ("core_tumor_numbered", full, core_tumor_specs, True),
            ("mp_hotcold", tumor, mp_specs, False),
            ("mp_numbered", tumor, mp_specs, True),
            ("kstar_hotcold", tumor, kstar_specs, False),
            ("kstar_numbered", tumor, kstar_specs, True),
        ]
        for fig_type, adata_for_plot, specs, numbered in figure_jobs:
            path = FIG_DIR / fig_type / f"{safe_name(sample_label)}__{fig_type.replace('_hotcold', '').replace('_numbered', '')}_{'numbered' if numbered else 'hotcold'}.png"
            if fig_type == "core_full_hotcold":
                path = FIG_DIR / fig_type / f"{safe_name(sample_label)}__core_full_hotcold.png"
            elif fig_type == "core_full_numbered":
                path = FIG_DIR / fig_type / f"{safe_name(sample_label)}__core_full_numbered.png"
            elif fig_type == "core_tumor_hotcold":
                path = FIG_DIR / fig_type / f"{safe_name(sample_label)}__core_tumor_hotcold.png"
            elif fig_type == "core_tumor_numbered":
                path = FIG_DIR / fig_type / f"{safe_name(sample_label)}__core_tumor_numbered.png"
            elif fig_type == "mp_hotcold":
                path = FIG_DIR / fig_type / f"{safe_name(sample_label)}__mp_hotcold.png"
            elif fig_type == "mp_numbered":
                path = FIG_DIR / fig_type / f"{safe_name(sample_label)}__mp_numbered.png"
            elif fig_type == "kstar_hotcold":
                path = FIG_DIR / fig_type / f"{safe_name(sample_label)}__kstar_hotcold.png"
            elif fig_type == "kstar_numbered":
                path = FIG_DIR / fig_type / f"{safe_name(sample_label)}__kstar_numbered.png"
            plot_contact_sheet(adata_for_plot, specs, sample_label, path, numbered=numbered)
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
    gene_audit = pd.DataFrame(all_gene_audit)
    plots = pd.DataFrame(all_plots)

    summary.to_csv(TABLE_DIR / "spottedpy_v2_hotspot_preflight_summary.csv", index=False)
    scaling.to_csv(TABLE_DIR / "spottedpy_v2_hotspot_scaling_audit.csv", index=False)
    gene_audit.to_csv(TABLE_DIR / "spottedpy_v2_scoregenes_gene_overlap_audit.csv", index=False)
    plots.to_csv(TABLE_DIR / "spottedpy_v2_hotspot_plot_manifest.csv", index=False)

    run_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__)),
        "output_root": str(OUT_ROOT),
        "previous_hotspot_root": str(PREVIOUS_ROOT),
        "input_manifest": str(INPUT_MANIFEST),
        "neighbours_parameters": NEIGHBOURS_PRIMARY,
        "p_value": P_VALUE,
        "permutations": PERMUTATIONS,
        "seed": SEED,
        "relative_to_batch": True,
        "batch_grain": "sample/patient",
        "score_genes_ctrl_size": 200,
        "score_genes_n_bins": 25,
        "score_genes_random_state": 0,
        "min_scoregenes_overlap": MIN_SCOREGENES_OVERLAP,
        "n_samples": int(manifest["sample_label"].nunique()),
        "n_hotspot_rows": int(len(summary)),
        "n_plot_files": int(len(plots)),
        "does_not_run": [
            "distance statistics",
            "GEE",
            "neighborhood enrichment",
            "k=8 sensitivity",
            "Hallmark hotspot preflight",
        ],
    }
    (OUT_ROOT / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    write_readme(summary, plots, gene_audit)
    print(json.dumps(run_manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
