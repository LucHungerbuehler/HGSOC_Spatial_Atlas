"""
Create Hallmark score_genes inputs and run a SpottedPy v2 neighborhood preflight.

This script is intentionally a preflight. It creates reusable Hallmark
score_genes columns for the thesis Hallmark shortlist, then runs a small
neighborhood-enrichment sanity check on one representative sample per dataset.
It does not call Hallmark hotspots, distance statistics, GEE, or sensitivity.

Run from PowerShell using spottedpy_env Python.
"""

from __future__ import annotations

import json
import math
import gc
import faulthandler
import re
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spottedpy_v2_hotspot_preflight import register_anndata_null_reader, require, safe_name, stop  # noqa: E402


DATA_ROOT = Path(r"D:\HGSOC_Spatial_Atlas")
ANALYSIS_ROOT = DATA_ROOT / "05_analysis_ready"
CODE_ROOT = Path(r"C:\Users\luchu\Documents\MSc\Master Thesis\Code\HGSOC_Spatial_Atlas")
SPOTTEDPY_CLONE = DATA_ROOT / "git_clones" / "SpottedPy-main"

RUN_ROOT = ANALYSIS_ROOT / "SpottedPy_v2_paper_aligned"
INPUT_MANIFEST = RUN_ROOT / "01_inputs_qc" / "tables" / "spottedpy_v2_live_input_manifest.csv"
HOTSPOT_ROOT = RUN_ROOT / "04_hotspots_preflight_revised_scoring_policy"
HALLMARK_JSON = ANALYSIS_ROOT / "Signature" / "hallmark_gene_sets.json"

OUT_ROOT = RUN_ROOT / "02_neighborhood_enrichment"
TABLE_DIR = OUT_ROOT / "tables"
SCORE_DIR = OUT_ROOT / "tables" / "hallmark_scoregenes_by_sample"
FIG_DIR = OUT_ROOT / "figures" / "preflight"
SCRIPT_DIR = OUT_ROOT / "scripts_used"
RUN_LOG = OUT_ROOT / "spottedpy_v2_hallmark_neighborhood_preflight.log"
FAULT_LOG = OUT_ROOT / "spottedpy_v2_hallmark_neighborhood_preflight_faults.log"

MIN_GENE_OVERLAP = 5
SCORE_GENES_CTRL_SIZE = 200
SCORE_GENES_N_BINS = 25
SCORE_GENES_RANDOM_STATE = 0
RUN_SPOTTEDPY_HELPER_INNER_OUTER = False

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
    "MP1_angiogenic_vascular_scoregenes",
    "MP2_iCAF_stress_scoregenes",
    "MP3_complement_CAF_scoregenes",
    "MP4_activated_myCAF_scoregenes",
    "MP5_IFN_TLS_immune_scoregenes",
    "MP6_APC_TAM_myeloid_scoregenes",
    "MP7_malignant_hypoxia_scoregenes",
    "MP8_malignant_acute_phase_secretory_scoregenes",
]

HALLMARK_GROUPS = {
    "Proliferation": [
        "HALLMARK_E2F_TARGETS",
        "HALLMARK_G2M_CHECKPOINT",
        "HALLMARK_MYC_TARGETS_V1",
        "HALLMARK_MYC_TARGETS_V2",
    ],
    "Metabolism": [
        "HALLMARK_GLYCOLYSIS",
        "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
        "HALLMARK_FATTY_ACID_METABOLISM",
        "HALLMARK_CHOLESTEROL_HOMEOSTASIS",
        "HALLMARK_ADIPOGENESIS",
    ],
    "Immune Response & Inflammation": [
        "HALLMARK_INFLAMMATORY_RESPONSE",
        "HALLMARK_INTERFERON_ALPHA_RESPONSE",
        "HALLMARK_INTERFERON_GAMMA_RESPONSE",
        "HALLMARK_COMPLEMENT",
        "HALLMARK_IL2_STAT5_SIGNALING",
        "HALLMARK_IL6_JAK_STAT3_SIGNALING",
        "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    ],
    "Cellular Stress & Apoptosis": [
        "HALLMARK_APOPTOSIS",
        "HALLMARK_DNA_REPAIR",
        "HALLMARK_HYPOXIA",
        "HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY",
        "HALLMARK_UNFOLDED_PROTEIN_RESPONSE",
    ],
    "Signaling & Development": [
        "HALLMARK_WNT_BETA_CATENIN_SIGNALING",
        "HALLMARK_NOTCH_SIGNALING",
        "HALLMARK_HEDGEHOG_SIGNALING",
        "HALLMARK_TGF_BETA_SIGNALING",
        "HALLMARK_PI3K_AKT_MTOR_SIGNALING",
        "HALLMARK_MTORC1_SIGNALING",
        "HALLMARK_KRAS_SIGNALING_DN",
        "HALLMARK_KRAS_SIGNALING_UP",
        "HALLMARK_ANGIOGENESIS",
    ],
    "Structure, Adhesion & Cellular Components": [
        "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
        "HALLMARK_APICAL_JUNCTION",
        "HALLMARK_APICAL_SURFACE",
    ],
    "Other Biological States": [
        "HALLMARK_XENOBIOTIC_METABOLISM",
        "HALLMARK_PROTEIN_SECRETION",
        "HALLMARK_ANDROGEN_RESPONSE",
        "HALLMARK_ESTROGEN_RESPONSE_EARLY",
        "HALLMARK_ESTROGEN_RESPONSE_LATE",
        "HALLMARK_PEROXISOME",
    ],
}

PREFLIGHT_CORE_HALLMARKS = [
    "HALLMARK_HYPOXIA",
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
    "HALLMARK_ANGIOGENESIS",
    "HALLMARK_COMPLEMENT",
    "HALLMARK_INTERFERON_GAMMA_RESPONSE",
    "HALLMARK_GLYCOLYSIS",
    "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
    "HALLMARK_APOPTOSIS",
]

_FAULT_HANDLE = None


def log(message: str) -> None:
    text = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(text, flush=True)
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def enable_fault_logging() -> None:
    global _FAULT_HANDLE
    FAULT_LOG.parent.mkdir(parents=True, exist_ok=True)
    _FAULT_HANDLE = FAULT_LOG.open("a", encoding="utf-8")
    faulthandler.enable(file=_FAULT_HANDLE, all_threads=True)


def import_spottedpy():
    sys.path.insert(0, str(SPOTTEDPY_CLONE))
    import spottedpy as sp

    return sp


def prepare_dirs() -> None:
    for directory in [TABLE_DIR, SCORE_DIR, FIG_DIR, SCRIPT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), SCRIPT_DIR / Path(__file__).name)


def load_manifest() -> pd.DataFrame:
    require(INPUT_MANIFEST)
    manifest = pd.read_csv(INPUT_MANIFEST)
    if len(manifest) != 23:
        stop(f"Expected 23 samples; found {len(manifest)}")
    return manifest


def load_hallmark_sets() -> tuple[dict[str, list[str]], pd.DataFrame]:
    require(HALLMARK_JSON)
    raw = json.loads(HALLMARK_JSON.read_text(encoding="utf-8"))
    requested = []
    for group, names in HALLMARK_GROUPS.items():
        for name in names:
            requested.append({"hallmark": name, "group": group})
    request_df = pd.DataFrame(requested)
    missing = sorted(set(request_df["hallmark"]).difference(raw))
    if missing:
        stop(f"Missing requested Hallmarks from local JSON: {missing}")
    return {name: list(raw[name]) for name in request_df["hallmark"]}, request_df


def compact_hallmark_name(name: str) -> str:
    return name.replace("HALLMARK_", "")


def hallmark_col(name: str) -> str:
    return f"hallmark_scoregenes__{name}"


def spot_ids(adata) -> pd.Series:
    if "spot" in adata.obs:
        return adata.obs["spot"].astype(str)
    return pd.Series(adata.obs.index.astype(str), index=adata.obs.index)


def core_h5ad_path(sample_label: str) -> Path:
    return HOTSPOT_ROOT / "h5ad" / "core_full" / f"{safe_name(sample_label)}__core_full_and_tumor_hotspots.h5ad"


def tumor_h5ad_path(sample_label: str) -> Path:
    return HOTSPOT_ROOT / "h5ad" / "mp_kstar_tumor" / f"{safe_name(sample_label)}__mp_kstar_tumor_hotspots.h5ad"


def score_hallmarks_for_sample(adata, gene_sets: dict[str, list[str]], row: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    import scanpy as sc

    if adata.var_names.has_duplicates:
        adata.var_names_make_unique()
    overlap_records = []
    range_records = []
    output = pd.DataFrame(
        {
            "dataset": row["dataset"],
            "sample": row["sample"],
            "sample_label": row["sample_label"],
            "spot_id": spot_ids(adata).to_numpy(),
            "interface": adata.obs["interface"].astype(str).to_numpy() if "interface" in adata.obs else "",
        },
        index=adata.obs.index,
    )
    for hallmark, genes in gene_sets.items():
        col = hallmark_col(hallmark)
        available = [gene for gene in genes if gene in adata.var_names]
        missing = [gene for gene in genes if gene not in adata.var_names]
        status = "ok" if len(available) >= MIN_GENE_OVERLAP else "skipped_too_few_genes"
        output[col] = np.nan
        if status == "ok":
            sc.tl.score_genes(
                adata,
                gene_list=available,
                score_name=col,
                ctrl_size=SCORE_GENES_CTRL_SIZE,
                n_bins=SCORE_GENES_N_BINS,
                random_state=SCORE_GENES_RANDOM_STATE,
                use_raw=False,
            )
            output[col] = pd.to_numeric(adata.obs[col], errors="coerce").to_numpy()
        values = pd.to_numeric(output[col], errors="coerce")
        finite = values[np.isfinite(values)]
        overlap_records.append(
            {
                "dataset": row["dataset"],
                "sample": row["sample"],
                "sample_label": row["sample_label"],
                "hallmark": hallmark,
                "score_col": col,
                "n_genes_requested": len(genes),
                "n_genes_available": len(available),
                "n_genes_missing": len(missing),
                "missing_genes_preview": ",".join(missing[:20]),
                "status": status,
            }
        )
        range_records.append(
            {
                "dataset": row["dataset"],
                "sample": row["sample"],
                "sample_label": row["sample_label"],
                "hallmark": hallmark,
                "score_col": col,
                "n_finite": int(len(finite)),
                "min": float(finite.min()) if len(finite) else math.nan,
                "median": float(finite.median()) if len(finite) else math.nan,
                "max": float(finite.max()) if len(finite) else math.nan,
                "n_negative": int((finite < 0).sum()) if len(finite) else 0,
                "n_positive": int((finite > 0).sum()) if len(finite) else 0,
            }
        )
    return output, overlap_records, range_records


def audit_existing_hallmark_score_table(
    scores: pd.DataFrame,
    gene_sets: dict[str, list[str]],
    row: dict[str, Any],
    var_names: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    overlap_records = []
    range_records = []
    for hallmark, genes in gene_sets.items():
        col = hallmark_col(hallmark)
        values = pd.to_numeric(scores[col], errors="coerce") if col in scores.columns else pd.Series(dtype=float)
        finite = values[np.isfinite(values)]
        available = [gene for gene in genes if var_names is not None and gene in var_names]
        missing = [gene for gene in genes if var_names is not None and gene not in var_names]
        status = "ok_reused_existing_table" if col in scores.columns and len(finite) else "skipped_existing_missing_or_empty"
        overlap_records.append(
            {
                "dataset": row["dataset"],
                "sample": row["sample"],
                "sample_label": row["sample_label"],
                "hallmark": hallmark,
                "score_col": col,
                "n_genes_requested": len(genes),
                "n_genes_available": len(available) if var_names is not None else math.nan,
                "n_genes_missing": len(missing) if var_names is not None else math.nan,
                "missing_genes_preview": ",".join(missing[:20]) if var_names is not None else "",
                "status": status,
            }
        )
        range_records.append(
            {
                "dataset": row["dataset"],
                "sample": row["sample"],
                "sample_label": row["sample_label"],
                "hallmark": hallmark,
                "score_col": col,
                "n_finite": int(len(finite)),
                "min": float(finite.min()) if len(finite) else math.nan,
                "median": float(finite.median()) if len(finite) else math.nan,
                "max": float(finite.max()) if len(finite) else math.nan,
                "n_negative": int((finite < 0).sum()) if len(finite) else 0,
                "n_positive": int((finite > 0).sum()) if len(finite) else 0,
            }
        )
    return overlap_records, range_records


def create_hallmark_scores(manifest: pd.DataFrame, gene_sets: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import anndata as ad

    overlap_all: list[dict[str, Any]] = []
    range_all: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for row in manifest.sort_values(["dataset", "sample"]).to_dict("records"):
        out_path = SCORE_DIR / f"{safe_name(row['sample_label'])}__hallmark_scoregenes.csv.gz"
        if out_path.exists():
            log(f"Reusing existing Hallmark score table {row['sample_label']}")
            scores = pd.read_csv(out_path)
            backed = ad.read_h5ad(str(row["full_h5ad_path"]), backed="r")
            var_names = set(backed.var_names.astype(str))
            backed.file.close()
            overlap, ranges = audit_existing_hallmark_score_table(scores, gene_sets, row, var_names=var_names)
        else:
            log(f"Hallmark score_genes {row['sample_label']}")
            adata = ad.read_h5ad(str(row["full_h5ad_path"]))
            adata.uns.clear()
            scores, overlap, ranges = score_hallmarks_for_sample(adata, gene_sets, row)
            scores.to_csv(out_path, index=False)
            del adata
        overlap_all.extend(overlap)
        range_all.extend(ranges)
        manifest_rows.append(
            {
                "dataset": row["dataset"],
                "sample": row["sample"],
                "sample_label": row["sample_label"],
                "score_table": str(out_path),
                "n_spots": int(len(scores)),
                "n_hallmarks": int(len(gene_sets)),
            }
        )
        del scores
        gc.collect()
    score_manifest = pd.DataFrame(manifest_rows)
    overlap_df = pd.DataFrame(overlap_all)
    range_df = pd.DataFrame(range_all)
    score_manifest.to_csv(TABLE_DIR / "hallmark_scoregenes_sample_manifest.csv", index=False)
    overlap_df.to_csv(TABLE_DIR / "hallmark_scoregenes_overlap_audit.csv", index=False)
    range_df.to_csv(TABLE_DIR / "hallmark_scoregenes_range_audit.csv", index=False)
    return score_manifest, overlap_df, range_df


def load_sample_scores(sample_label: str) -> pd.DataFrame:
    path = SCORE_DIR / f"{safe_name(sample_label)}__hallmark_scoregenes.csv.gz"
    require(path)
    return pd.read_csv(path)


def build_neighborhood_variables(full, core, tumor, hallmark_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[dict[str, Any]]]:
    obs = full.obs.copy()
    obs["spot_id"] = spot_ids(full)
    variables = pd.DataFrame(index=full.obs.index)
    variable_manifest: list[dict[str, Any]] = []

    def add_variable(var_id: str, title: str, family: str, values: pd.Series | np.ndarray, scope: str) -> None:
        numeric = pd.to_numeric(pd.Series(values, index=variables.index), errors="coerce")
        variables[var_id] = numeric
        finite = numeric[np.isfinite(numeric)]
        variable_manifest.append(
            {
                "variable_id": var_id,
                "title": title,
                "family": family,
                "scope": scope,
                "n_finite": int(len(finite)),
                "min": float(finite.min()) if len(finite) else math.nan,
                "max": float(finite.max()) if len(finite) else math.nan,
            }
        )

    add_variable("SNAI1_ac", "SNAI1-ac production score", "core", full.obs["SNAI1-ac_score"], "full")

    core_by_spot = core.obs.copy()
    core_by_spot["spot_id"] = spot_ids(core)
    core_by_spot = core_by_spot.set_index("spot_id")
    spot_index = obs["spot_id"].astype(str)
    snai1_col = "spv2_full_scoregenes_SNAI1_positive_arm_minmax"
    snai12r_col = "spv2_full_scoregenes_SNAI1-2R_positive_arm_minmax"
    for required in [snai1_col, snai12r_col]:
        if required not in core_by_spot.columns:
            stop(f"Missing required core score column in 04 h5ad: {required}")
    add_variable("SNAI1_scoregenes", "SNAI1 score_genes", "core", spot_index.map(core_by_spot[snai1_col]), "full")
    add_variable("SNAI1_2R_scoregenes", "SNAI1-2R score_genes", "core", spot_index.map(core_by_spot[snai12r_col]), "full")

    hallmark_by_spot = hallmark_scores.set_index("spot_id")
    for hallmark in [name for names in HALLMARK_GROUPS.values() for name in names]:
        col = hallmark_col(hallmark)
        if col not in hallmark_by_spot.columns:
            stop(f"Missing Hallmark score column: {col}")
        add_variable(compact_hallmark_name(hallmark), compact_hallmark_name(hallmark), "hallmark", spot_index.map(hallmark_by_spot[col]), "full")

    for col in SPACET_COLS:
        if col in full.obs.columns:
            add_variable(f"SpaCET_{safe_name(col)}", f"SpaCET {col}", "spacet", full.obs[col], "full")

    tumor_by_spot = tumor.obs.copy()
    tumor_by_spot["spot_id"] = spot_ids(tumor)
    tumor_by_spot = tumor_by_spot.set_index("spot_id")
    for col in MP_COLS:
        if col in tumor_by_spot.columns:
            add_variable(col.replace("_scoregenes", ""), col.replace("_scoregenes", ""), "mp", spot_index.map(tumor_by_spot[col]), "tumor_only")

    mp_variable_ids = [record["variable_id"] for record in variable_manifest if record["family"] == "mp"]
    return variables, obs, mp_variable_ids, variable_manifest


def pearson_pair(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    frame = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame.iloc[:, 0].nunique() < 2 or frame.iloc[:, 1].nunique() < 2:
        return math.nan, math.nan, int(len(frame))
    corr, pval = pearsonr(frame.iloc[:, 0], frame.iloc[:, 1])
    return float(corr), float(pval), int(len(frame))


def flatten_corr_matrix(corr: pd.DataFrame, pval: pd.DataFrame, sample_label: str, mode: str, ring: int) -> list[dict[str, Any]]:
    rows = []
    cols = list(corr.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            rows.append(
                {
                    "sample_label": sample_label,
                    "mode": mode,
                    "ring": ring,
                    "variable_a": left,
                    "variable_b": right,
                    "corr": float(corr.loc[left, right]) if pd.notna(corr.loc[left, right]) else math.nan,
                    "pval": float(pval.loc[left, right]) if pd.notna(pval.loc[left, right]) else math.nan,
                }
            )
    return rows


def custom_central_to_ring1(full, variables: pd.DataFrame, obs: pd.DataFrame, source_nodes: list[str], mp_variables: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    import squidpy as sq

    if "spatial_connectivities" not in full.obsp:
        sq.gr.spatial_neighbors(full, n_rings=1, coord_type="grid", n_neighs=6)
    conn = full.obsp["spatial_connectivities"].tocsr()
    obs_names = np.asarray(full.obs_names)
    obs_pos = pd.Series(np.arange(len(obs_names)), index=obs_names)
    source_set = set(source_nodes)
    source_positions = obs_pos.loc[source_nodes].to_numpy(dtype=int)
    central = variables.loc[source_nodes].copy()
    variable_values = variables.apply(pd.to_numeric, errors="coerce")
    outer_rows = []
    for node, node_pos in zip(source_nodes, source_positions):
        start = conn.indptr[node_pos]
        end = conn.indptr[node_pos + 1]
        neighbor_positions = conn.indices[start:end]
        neighbor_nodes = obs_names[neighbor_positions]
        row = {}
        for var in variables.columns:
            if var in mp_variables:
                nodes = [idx for idx in neighbor_nodes if idx in source_set]
            else:
                nodes = neighbor_nodes
            row[var] = variable_values.loc[nodes, var].mean() if len(nodes) else np.nan
        outer_rows.append(row)
    outer = pd.DataFrame(outer_rows, index=source_nodes, columns=variables.columns)
    return central, outer


def run_neighborhood_preflight(manifest: pd.DataFrame, score_manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import anndata as ad

    sp = import_spottedpy()
    preflight_samples = (
        manifest.sort_values(["dataset", "sample"])
        .groupby("dataset", as_index=False)
        .head(1)
        .sort_values(["dataset", "sample"])
    )
    allinone_rows: list[dict[str, Any]] = []
    inner_outer_rows: list[dict[str, Any]] = []
    helper_rows: list[dict[str, Any]] = []
    variable_rows: list[dict[str, Any]] = []

    for row in preflight_samples.to_dict("records"):
        sample_label = row["sample_label"]
        log(f"Neighborhood preflight {sample_label}")
        full = ad.read_h5ad(str(row["full_h5ad_path"]))
        full.uns.clear()
        core = ad.read_h5ad(core_h5ad_path(sample_label))
        tumor = ad.read_h5ad(tumor_h5ad_path(sample_label))
        hallmark_scores = load_sample_scores(sample_label)
        variables, obs, mp_vars, var_manifest = build_neighborhood_variables(full, core, tumor, hallmark_scores)
        for record in var_manifest:
            variable_rows.append({"dataset": row["dataset"], "sample": row["sample"], "sample_label": sample_label, **record})

        source_nodes = list(obs.index[obs["interface"].astype(str).eq("Tumor")])
        if len(source_nodes) < 20:
            stop(f"{sample_label} has too few tumor source nodes for neighborhood preflight: {len(source_nodes)}")

        preflight_vars = (
            ["SNAI1_ac", "SNAI1_scoregenes", "SNAI1_2R_scoregenes"]
            + [compact_hallmark_name(name) for name in PREFLIGHT_CORE_HALLMARKS]
            + [f"SpaCET_{safe_name(col)}" for col in ["Malignant", "CAF", "Macrophage", "T CD8", "Endothelial"] if f"SpaCET_{safe_name(col)}" in variables.columns]
            + [col.replace("_scoregenes", "") for col in MP_COLS if col.replace("_scoregenes", "") in variables.columns]
        )
        preflight_vars = [var for var in preflight_vars if var in variables.columns]
        neighbour_variables = variables[preflight_vars].copy()
        mp_preflight_vars = [var for var in preflight_vars if var in mp_vars]

        results = sp.calculate_neighbourhood_correlation(
            rings_range=[1],
            adata_vis=full,
            neighbour_variables=neighbour_variables,
            source_nodes=source_nodes,
            neighbourhood_variable_filter_for_tumour_cells=mp_preflight_vars if mp_preflight_vars else None,
            split_by_batch=False,
        )
        corr, pval = results[1]
        allinone_rows.extend(flatten_corr_matrix(corr, pval, sample_label, "all_in_one_ring1_spottedpy", 1))
        pd.DataFrame(variable_rows).to_csv(TABLE_DIR / "neighborhood_preflight_variable_manifest.csv", index=False)
        pd.DataFrame(allinone_rows).to_csv(TABLE_DIR / "neighborhood_preflight_allinone_ring1_correlations.csv", index=False)

        central, outer = custom_central_to_ring1(full, neighbour_variables, obs, source_nodes, mp_preflight_vars)
        for key in ["SNAI1_ac", "SNAI1_scoregenes", "SNAI1_2R_scoregenes"]:
            if key not in central.columns:
                continue
            for var in outer.columns:
                corr_val, pval_val, n_pairs = pearson_pair(central[key], outer[var])
                inner_outer_rows.append(
                    {
                        "sample_label": sample_label,
                        "mode": "custom_central_to_immediate_ring1",
                        "ring": 1,
                        "central_variable": key,
                        "outer_variable": var,
                        "corr": corr_val,
                        "pval": pval_val,
                        "n_pairs": n_pairs,
                    }
                )

        if RUN_SPOTTEDPY_HELPER_INNER_OUTER:
            helper = sp.calculate_inner_outer_neighbourhood_enrichment(
                rings_range=[1],
                adata_vis=full,
                neighbour_variables=neighbour_variables,
                source_nodes=source_nodes,
            )
            helper_corr = sp.calculate_corr_pvalue_for_inner_outer_neighbourhood_enrichment(
                helper,
                correlation_key_variable="SNAI1_ac",
                rings_range=[1],
                average_by_batch=False,
            )
            helper_corr_df, helper_pval_df = helper_corr[1]
            key = "SNAI1_ac_inner_values"
            if key in helper_corr_df.columns:
                for var in helper_corr_df.index:
                    if var == key:
                        continue
                    helper_rows.append(
                        {
                            "sample_label": sample_label,
                            "mode": "spottedpy_inner_outer_ring1_inner_vs_outer",
                            "ring": 1,
                            "central_variable": "SNAI1_ac",
                            "outer_variable": var,
                            "corr": float(helper_corr_df.loc[var, key]) if pd.notna(helper_corr_df.loc[var, key]) else math.nan,
                            "pval": float(helper_pval_df.loc[var, key]) if pd.notna(helper_pval_df.loc[var, key]) else math.nan,
                            "status": "ok",
                        }
                    )
        else:
            helper_rows.append(
                {
                    "sample_label": sample_label,
                    "mode": "spottedpy_inner_outer_ring1_inner_vs_outer",
                    "ring": 1,
                    "central_variable": "SNAI1_ac",
                    "outer_variable": "",
                    "corr": math.nan,
                    "pval": math.nan,
                    "status": "skipped_in_preflight",
                    "reason": "Disabled after repeated hard Python exits on Windows during helper calculation; custom central-to-ring1 table is the paper-facing inner-outer preflight.",
                }
            )

        plot_heatmap(corr, sample_label, FIG_DIR / f"{safe_name(sample_label)}__all_in_one_ring1_heatmap.png")
        plot_top_inner_outer(
            pd.DataFrame([r for r in inner_outer_rows if r["sample_label"] == sample_label and r["central_variable"] == "SNAI1_ac"]),
            sample_label,
            FIG_DIR / f"{safe_name(sample_label)}__central_SNAI1_ac_to_ring1_top_correlations.png",
        )
        pd.DataFrame(variable_rows).to_csv(TABLE_DIR / "neighborhood_preflight_variable_manifest.csv", index=False)
        pd.DataFrame(allinone_rows).to_csv(TABLE_DIR / "neighborhood_preflight_allinone_ring1_correlations.csv", index=False)
        pd.DataFrame(inner_outer_rows).to_csv(TABLE_DIR / "neighborhood_preflight_inner_outer_custom_central_ring1_correlations.csv", index=False)
        pd.DataFrame(helper_rows).to_csv(TABLE_DIR / "neighborhood_preflight_inner_outer_spottedpy_ring1_correlations.csv", index=False)
        del full, core, tumor, hallmark_scores, variables, obs, neighbour_variables
        gc.collect()

    variable_df = pd.DataFrame(variable_rows)
    allinone_df = pd.DataFrame(allinone_rows)
    inner_outer_df = pd.DataFrame(inner_outer_rows)
    helper_df = pd.DataFrame(helper_rows)
    variable_df.to_csv(TABLE_DIR / "neighborhood_preflight_variable_manifest.csv", index=False)
    allinone_df.to_csv(TABLE_DIR / "neighborhood_preflight_allinone_ring1_correlations.csv", index=False)
    inner_outer_df.to_csv(TABLE_DIR / "neighborhood_preflight_inner_outer_custom_central_ring1_correlations.csv", index=False)
    helper_df.to_csv(TABLE_DIR / "neighborhood_preflight_inner_outer_spottedpy_ring1_correlations.csv", index=False)
    return variable_df, allinone_df, inner_outer_df, helper_df


def plot_heatmap(corr: pd.DataFrame, sample_label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    data = corr.to_numpy(dtype=float)
    im = ax.imshow(data, vmin=-1, vmax=1, cmap="vlag" if "vlag" in plt.colormaps() else "coolwarm")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(corr.index, fontsize=6)
    ax.set_title(f"{sample_label}: all-in-one neighborhood correlation, ring 1")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Pearson r")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_top_inner_outer(df: pd.DataFrame, sample_label: str, out_path: Path) -> None:
    frame = df.dropna(subset=["corr"]).copy()
    frame["abs_corr"] = frame["corr"].abs()
    frame = frame.sort_values("abs_corr", ascending=False).head(20).sort_values("corr")
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = np.where(frame["corr"] >= 0, "#b2182b", "#2166ac")
    ax.barh(frame["outer_variable"], frame["corr"], color=colors)
    ax.axvline(0, color="#111827", linewidth=0.8)
    ax.set_xlabel("Pearson r")
    ax.set_title(f"{sample_label}: central SNAI1-ac vs immediate ring variables")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_readme(
    request_df: pd.DataFrame,
    score_manifest: pd.DataFrame,
    overlap_df: pd.DataFrame,
    variable_df: pd.DataFrame,
    allinone_df: pd.DataFrame,
    inner_outer_df: pd.DataFrame,
    helper_df: pd.DataFrame,
) -> None:
    lines = [
        "# SpottedPy v2 Neighborhood Enrichment Preflight",
        "",
        "This folder creates the missing Hallmark score_genes inputs and runs a small neighborhood-enrichment preflight.",
        "It does not run full neighborhood enrichment, Hallmark hotspot calling, distance statistics, or GEE.",
        "",
        "## Hallmark scoring",
        "",
        f"- Hallmark shortlist size: {request_df['hallmark'].nunique()}",
        f"- Samples scored: {score_manifest['sample_label'].nunique()}",
        f"- Score genes rows: {len(overlap_df)}",
        f"- Non-ok score rows: {int((~overlap_df['status'].astype(str).str.startswith('ok')).sum())}",
        "- Method: scanpy.tl.score_genes with ctrl_size=200, n_bins=25, random_state=0, use_raw=False.",
        "",
        "## Neighborhood preflight",
        "",
        "- One representative sample per dataset is used.",
        "- Source nodes are tumor spots.",
        "- All-in-one uses SpottedPy calculate_neighbourhood_correlation with ring 1.",
        "- Inner-outer preflight includes a custom central-to-immediate-ring1 implementation to match the paper's direct-neighborhood wording.",
        "- SpottedPy's tutorial-style inner_outer ring 1 is recorded as skipped in this preflight after repeated hard Python exits on Windows with the full preflight variable set.",
        "- MP variables are tumor-defined; their neighborhood means are calculated over tumor neighbors only.",
        "",
        "## Outputs",
        "",
        "- `hallmark_scoregenes_by_sample/`: per-sample Hallmark score tables.",
        "- `hallmark_scoregenes_overlap_audit.csv`: gene overlap and scoring status.",
        "- `neighborhood_preflight_allinone_ring1_correlations.csv`: all-in-one correlation preflight.",
        "- `neighborhood_preflight_inner_outer_custom_central_ring1_correlations.csv`: paper-facing direct central-to-ring1 preflight.",
        "- `neighborhood_preflight_inner_outer_spottedpy_ring1_correlations.csv`: SpottedPy helper skip status for this preflight.",
        "- `figures/preflight/`: compact preflight heatmaps and SNAI1-ac top-correlation plots.",
        "",
        "## Counts",
        "",
        f"- Variable manifest rows: {len(variable_df)}",
        f"- All-in-one pair rows: {len(allinone_df)}",
        f"- Custom inner-outer rows: {len(inner_outer_df)}",
        f"- SpottedPy helper inner-outer rows: {len(helper_df)}",
        "",
        "Review these preflight outputs before running full neighborhood enrichment.",
        "",
    ]
    (OUT_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    prepare_dirs()
    enable_fault_logging()
    log("Starting SpottedPy v2 Hallmark score_genes and neighborhood preflight")
    register_anndata_null_reader()
    manifest = load_manifest()
    gene_sets, request_df = load_hallmark_sets()
    request_df.to_csv(TABLE_DIR / "hallmark_scoregenes_requested_shortlist.csv", index=False)

    score_manifest, overlap_df, range_df = create_hallmark_scores(manifest, gene_sets)
    variable_df, allinone_df, inner_outer_df, helper_df = run_neighborhood_preflight(manifest, score_manifest)

    run_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__)),
        "output_root": str(OUT_ROOT),
        "input_manifest": str(INPUT_MANIFEST),
        "hotspot_root_used_for_core_and_mp_scoregenes": str(HOTSPOT_ROOT),
        "hallmark_json": str(HALLMARK_JSON),
        "n_samples_scored": int(score_manifest["sample_label"].nunique()),
        "n_hallmarks_scored": int(request_df["hallmark"].nunique()),
        "preflight_samples": sorted(variable_df["sample_label"].unique().tolist()),
        "run_spottedpy_helper_inner_outer": RUN_SPOTTEDPY_HELPER_INNER_OUTER,
        "spottedpy_helper_inner_outer_note": "Skipped in this preflight after repeated hard Python exits on Windows with the full variable set; custom central-to-ring1 table is the paper-facing inner-outer preflight.",
        "does_not_run": [
            "full neighborhood enrichment",
            "Hallmark hotspot calling",
            "distance statistics",
            "GEE",
            "scale sensitivity",
        ],
    }
    (OUT_ROOT / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    write_readme(request_df, score_manifest, overlap_df, variable_df, allinone_df, inner_outer_df, helper_df)
    log("Completed SpottedPy v2 Hallmark score_genes and neighborhood preflight")
    print(json.dumps(run_manifest, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("ERROR: run failed")
        log(traceback.format_exc())
        raise
