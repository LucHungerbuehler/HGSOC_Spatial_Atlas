from __future__ import annotations

import gc
import json
import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spottedpy_v2_hotspot_preflight import register_anndata_null_reader  # noqa: E402


DATA_ROOT = Path(r"D:\HGSOC_Spatial_Atlas")
ANALYSIS_ROOT = DATA_ROOT / "05_analysis_ready"
CODE_ROOT = Path(r"C:\Users\luchu\Documents\MSc\Master Thesis\Code\HGSOC_Spatial_Atlas")

RUN_ROOT = ANALYSIS_ROOT / "SpottedPy_v2_paper_aligned"
INPUT_MANIFEST = RUN_ROOT / "01_inputs_qc" / "tables" / "spottedpy_v2_live_input_manifest.csv"
HOTSPOT_ROOT = RUN_ROOT / "04_hotspots_preflight_revised_scoring_policy"
HALLMARK_SCORE_DIR = RUN_ROOT / "02_neighborhood_enrichment" / "tables" / "hallmark_scoregenes_by_sample"
KSTAR_PROJECTION_TABLE = (
    ANALYSIS_ROOT
    / "S3_cNMF_Tumor_Programs"
    / "snai1ac_signature_projection_onto_cnmf_programs_v1"
    / "tables"
    / "kstar_snai1ac_signature_projection_clean.csv"
)

RUN_MODE = os.environ.get("SPOTTEDPY_V2_NEIGHBORHOOD_RUN_MODE", "preflight").strip().lower()
DEFAULT_OUT_SUBDIR = "consensus_source_full" if RUN_MODE == "full" else "consensus_source_preflight"
OUT_SUBDIR = os.environ.get("SPOTTEDPY_V2_CONSENSUS_OUT_SUBDIR", DEFAULT_OUT_SUBDIR).strip()

OUT_ROOT = RUN_ROOT / "02_neighborhood_enrichment" / OUT_SUBDIR
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
SCRIPT_DIR = OUT_ROOT / "scripts_used"


def compact_label(value: str) -> str:
    return str(value).strip().replace("\n", " ").replace("  ", " ")


def load_display_label_map() -> dict[str, str]:
    label_map = {
        "SNAI1_ac": "SNAI1-ac",
        "SNAI1_scoregenes": "SNAI1",
        "SNAI1_2R_scoregenes": "SNAI1-2R",
    }
    if KSTAR_PROJECTION_TABLE.exists():
        kstar = pd.read_csv(KSTAR_PROJECTION_TABLE)
        for row in kstar.itertuples(index=False):
            program_id = str(row.program_id)
            suffix = program_id.rsplit("__", 1)[-1]
            category = compact_label(getattr(row, "alignment_category_draft", ""))
            if not category or category.lower() == "nan":
                continue
            label_map[f"Kstar_{program_id}"] = f"{suffix}_{category}"
    return label_map


def display_label(variable: str, label_map: dict[str, str]) -> str:
    return label_map.get(variable, variable)

CORE_SOURCE_COLUMNS = {
    "snai1ac_consensus_hot": "spv2_full_SNAI1-ac_consensus_hot",
    "snai1ac_consensus_cold": "spv2_full_SNAI1-ac_consensus_cold",
    "snai12r_hot": "spv2_full_scoregenes_SNAI1-2R_positive_arm_minmax_hot",
}

CORE_VARIABLES = {
    "SNAI1_ac": "SNAI1-ac_score",
    "SNAI1_scoregenes": "spv2_full_scoregenes_SNAI1_positive_arm_minmax",
    "SNAI1_2R_scoregenes": "spv2_full_scoregenes_SNAI1-2R_positive_arm_minmax",
}

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


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))


def spot_ids(adata) -> pd.Series:
    if "spot" in adata.obs:
        return adata.obs["spot"].astype(str)
    return pd.Series(adata.obs.index.astype(str), index=adata.obs.index)


def hallmark_col(name: str) -> str:
    return f"hallmark_scoregenes__{name}"


def compact_hallmark_name(name: str) -> str:
    return name.replace("HALLMARK_", "")


def core_h5ad_path(sample_label: str) -> Path:
    return HOTSPOT_ROOT / "h5ad" / "core_full" / f"{safe_name(sample_label)}__core_full_and_tumor_hotspots.h5ad"


def tumor_h5ad_path(sample_label: str) -> Path:
    return HOTSPOT_ROOT / "h5ad" / "mp_kstar_tumor" / f"{safe_name(sample_label)}__mp_kstar_tumor_hotspots.h5ad"


def hallmark_scores_path(sample_label: str) -> Path:
    return HALLMARK_SCORE_DIR / f"{safe_name(sample_label)}__hallmark_scoregenes.csv.gz"


def truthy_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).astype(float) > 0
    text = series.astype(str).str.lower()
    return text.isin(["true", "1", "yes", "hot", "cold"])


def bh_qvalues(pvals: pd.Series) -> pd.Series:
    values = pd.to_numeric(pvals, errors="coerce")
    q = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    n = len(valid)
    if n == 0:
        return q
    ranks = np.arange(1, n + 1)
    adjusted = (valid.to_numpy() * n) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    q.loc[valid.index] = adjusted
    return q


def pearson_pair(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    frame = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame.iloc[:, 0].nunique() < 2 or frame.iloc[:, 1].nunique() < 2:
        return math.nan, math.nan, int(len(frame))
    corr, pval = pearsonr(frame.iloc[:, 0], frame.iloc[:, 1])
    return float(corr), float(pval), int(len(frame))


def spearman_pair(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    frame = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame.iloc[:, 0].nunique() < 2 or frame.iloc[:, 1].nunique() < 2:
        return math.nan, math.nan, int(len(frame))
    corr, pval = spearmanr(frame.iloc[:, 0], frame.iloc[:, 1])
    return float(corr), float(pval), int(len(frame))


def make_corr_tables(values: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    variables = list(values.columns)
    corr = pd.DataFrame(np.nan, index=variables, columns=variables, dtype=float)
    pval = corr.copy()
    for left in variables:
        for right in variables:
            r, p, _ = pearson_pair(values[left], values[right])
            corr.loc[left, right] = r
            pval.loc[left, right] = p
    return corr, pval


def clustered_order(corr: pd.DataFrame) -> list[str]:
    if len(corr) <= 2:
        return list(corr.index)
    matrix = corr.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1, 1)
    dist = 1 - matrix
    np.fill_diagonal(dist.values, 0)
    condensed = squareform(dist.values, checks=False)
    order = leaves_list(linkage(condensed, method="average"))
    return list(corr.index[order])


def plot_corr_heatmap(corr: pd.DataFrame, out_path: Path, title: str, label_map: dict[str, str]) -> None:
    order = clustered_order(corr)
    data = corr.loc[order, order].to_numpy(dtype=float)
    n = len(order)
    labels = [display_label(variable, label_map) for variable in order]
    fig_size = max(7, min(20, 0.36 * n + 4))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(data, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Pearson r")
    fig.subplots_adjust(left=0.26, right=0.88, bottom=0.34, top=0.94)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_ranked_bars(df: pd.DataFrame, out_path: Path, title: str, label_map: dict[str, str]) -> None:
    if df.empty:
        return
    frame = df.sort_values("corr")
    height = max(4, 0.28 * len(frame) + 1.8)
    fig, ax = plt.subplots(figsize=(10.5, height))
    colors = np.where(frame["corr"] >= 0, "#b2182b", "#2166ac")
    labels = [display_label(row.outer_variable, label_map) for row in frame.itertuples(index=False)]
    bars = ax.barh(labels, frame["corr"], color=colors)
    ax.axvline(0, color="#111827", linewidth=0.8)
    ax.set_xlabel("Pearson r")
    ax.set_title(title, fontsize=10)
    for bar, marker, corr_value in zip(bars, frame["sig_marker"], frame["corr"]):
        marker_text = "" if pd.isna(marker) else str(marker).strip()
        if not marker_text:
            continue
        ax.text(
            corr_value / 2,
            bar.get_y() + bar.get_height() / 2,
            marker_text,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="#ffffff",
        )
    ax.text(
        0.01,
        0.01,
        "Markers indicate BH q-value: **** <=1e-4, *** <=1e-3, ** <=1e-2, * <=0.05",
        transform=ax.transAxes,
        fontsize=6,
        ha="left",
        va="bottom",
        color="#374151",
    )
    ax.tick_params(axis="y", labelsize=7)
    fig.subplots_adjust(left=0.34, right=0.98, bottom=0.16, top=0.9)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def significance_marker(qval: float) -> str:
    if not np.isfinite(qval):
        return ""
    if qval <= 1e-4:
        return "****"
    if qval <= 1e-3:
        return "***"
    if qval <= 1e-2:
        return "**"
    if qval <= 0.05:
        return "*"
    return ""


def clustered_row_order(matrix: pd.DataFrame) -> list[str]:
    if len(matrix) <= 2:
        return list(matrix.index)
    values = matrix.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1, 1)
    if values.nunique(axis=1).max() <= 1:
        return list(matrix.index)
    distances = 1 - np.corrcoef(values.to_numpy(dtype=float))
    distances = np.nan_to_num(distances, nan=1.0, posinf=1.0, neginf=1.0)
    np.fill_diagonal(distances, 0)
    condensed = squareform(distances, checks=False)
    order = leaves_list(linkage(condensed, method="average"))
    return list(matrix.index[order])


def plot_source_group_comparison(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    source_order: list[str],
    label_map: dict[str, str],
) -> None:
    if df.empty:
        return
    corr = df.pivot(index="outer_variable", columns="source_group", values="corr").reindex(columns=source_order)
    qval = df.pivot(index="outer_variable", columns="source_group", values="qval").reindex(columns=source_order)
    corr = corr.dropna(how="all")
    qval = qval.reindex(corr.index)
    if corr.empty:
        return
    order = clustered_row_order(corr)
    corr = corr.loc[order]
    qval = qval.loc[order]
    display = corr.where(qval <= 0.05)
    height = max(4.5, min(24, 0.26 * len(corr) + 2.2))
    row_labels = [display_label(variable, label_map) for variable in corr.index]
    max_label_len = max((len(label) for label in row_labels), default=10)
    width = max(10, min(17, 8 + 0.07 * max_label_len))
    fig, ax = plt.subplots(figsize=(width, height))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#ffffff")
    im = ax.imshow(display.to_numpy(dtype=float), vmin=-1, vmax=1, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(source_order)))
    ax.set_xticklabels(source_order, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(row_labels, fontsize=6)
    ax.set_title(title, fontsize=10)
    for row_idx, variable in enumerate(corr.index):
        for col_idx, source in enumerate(source_order):
            marker = significance_marker(qval.loc[variable, source])
            if marker:
                ax.text(col_idx, row_idx, marker, ha="center", va="center", fontsize=6, color="#111827")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label="Pearson r, significant cells only")
    ax.text(
        0.0,
        -0.05,
        "White cells did not pass BH q <= 0.05 within source group and variable class.",
        transform=ax.transAxes,
        fontsize=6,
        ha="left",
        va="top",
        color="#374151",
    )
    fig.subplots_adjust(left=0.44, right=0.82, bottom=0.14, top=0.96)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def load_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(INPUT_MANIFEST)
    if len(manifest) != 23:
        raise RuntimeError(f"Expected 23 samples; found {len(manifest)}")
    return manifest


def preflight_samples(manifest: pd.DataFrame) -> pd.DataFrame:
    return (
        manifest.sort_values(["dataset", "sample"])
        .groupby("dataset", as_index=False)
        .head(1)
        .sort_values(["dataset", "sample"])
    )


def selected_samples(manifest: pd.DataFrame) -> pd.DataFrame:
    if RUN_MODE == "full":
        return manifest.sort_values(["dataset", "sample"]).reset_index(drop=True)
    if RUN_MODE == "preflight":
        return preflight_samples(manifest).reset_index(drop=True)
    raise RuntimeError(f"Unsupported SPOTTEDPY_V2_NEIGHBORHOOD_RUN_MODE={RUN_MODE!r}")


def build_variables(full, core, tumor, hallmark_scores: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]], list[dict[str, Any]]]:
    variables = pd.DataFrame(index=full.obs.index)
    records: list[dict[str, Any]] = []

    full_spots = spot_ids(full).astype(str)
    core_obs = core.obs.copy()
    core_obs["spot_id"] = spot_ids(core)
    core_by_spot = core_obs.set_index("spot_id")
    tumor_obs = tumor.obs.copy()
    tumor_obs["spot_id"] = spot_ids(tumor)
    tumor_by_spot = tumor_obs.set_index("spot_id")
    hallmark_by_spot = hallmark_scores.set_index("spot_id")

    families: dict[str, list[str]] = {"core": [], "hallmark": [], "spacet": [], "mp_kstar": []}

    def add_variable(var_id: str, family: str, values: pd.Series | np.ndarray) -> None:
        numeric = pd.to_numeric(pd.Series(values, index=variables.index), errors="coerce")
        variables[var_id] = numeric
        families.setdefault(family, []).append(var_id)
        finite = numeric[np.isfinite(numeric)]
        records.append(
            {
                "variable_id": var_id,
                "family": family,
                "n_finite": int(len(finite)),
                "min": float(finite.min()) if len(finite) else math.nan,
                "max": float(finite.max()) if len(finite) else math.nan,
            }
        )

    for var_id, col in CORE_VARIABLES.items():
        if col not in core_by_spot.columns:
            raise RuntimeError(f"Missing core variable {col}")
        add_variable(var_id, "core", full_spots.map(core_by_spot[col]))

    for hallmark in [name for names in HALLMARK_GROUPS.values() for name in names]:
        col = hallmark_col(hallmark)
        if col not in hallmark_by_spot.columns:
            raise RuntimeError(f"Missing Hallmark score column {col}")
        add_variable(compact_hallmark_name(hallmark), "hallmark", full_spots.map(hallmark_by_spot[col]))

    for col in SPACET_COLS:
        if col in full.obs.columns:
            add_variable(f"SpaCET_{safe_name(col)}", "spacet", full.obs[col])

    for col in MP_COLS:
        if col in tumor_by_spot.columns:
            add_variable(col.replace("_scoregenes", ""), "mp_kstar", full_spots.map(tumor_by_spot[col]))

    kstar_cols = [
        col
        for col in tumor_by_spot.columns
        if col.startswith("spv2_tumor_kstar_") and col.endswith("_raw")
    ]
    for col in kstar_cols:
        var_id = col.replace("spv2_tumor_kstar_", "Kstar_").replace("_raw", "")
        add_variable(var_id, "mp_kstar", full_spots.map(tumor_by_spot[col]))

    return variables, families, records


def source_nodes_for_group(full, core, source_col: str) -> list[str]:
    full_spots = spot_ids(full).astype(str)
    core_obs = core.obs.copy()
    core_obs["spot_id"] = spot_ids(core)
    core_by_spot = core_obs.set_index("spot_id")
    if source_col not in core_by_spot.columns:
        raise RuntimeError(f"Missing source column {source_col}")
    mask = truthy_series(full_spots.map(core_by_spot[source_col]))
    return list(full.obs.index[mask.to_numpy()])


def ring1_neighbor_means(full, variables: pd.DataFrame, source_nodes: list[str]) -> pd.DataFrame:
    if "spatial_connectivities" not in full.obsp:
        import squidpy as sq

        sq.gr.spatial_neighbors(full, n_rings=1, coord_type="grid", n_neighs=6)
    conn = full.obsp["spatial_connectivities"].tocsr()
    obs_names = np.asarray(full.obs_names)
    obs_pos = pd.Series(np.arange(len(obs_names)), index=obs_names)
    source_positions = obs_pos.loc[source_nodes].to_numpy(dtype=int)
    variable_values = variables.apply(pd.to_numeric, errors="coerce")
    rows = []
    for node_pos in source_positions:
        start = conn.indptr[node_pos]
        end = conn.indptr[node_pos + 1]
        neighbor_nodes = obs_names[conn.indices[start:end]]
        rows.append(variable_values.loc[neighbor_nodes].mean(axis=0))
    return pd.DataFrame(rows, index=source_nodes, columns=variables.columns)


def class_variable_sets(families: dict[str, list[str]]) -> dict[str, list[str]]:
    core = ["SNAI1_ac", "SNAI1_scoregenes", "SNAI1_2R_scoregenes"]
    return {
        "all_variables": core + families["hallmark"] + families["spacet"] + families["mp_kstar"],
        "spacet": ["SNAI1_ac"] + families["spacet"],
        "mp_kstar": ["SNAI1_ac"] + families["mp_kstar"],
        "hallmark": ["SNAI1_ac"] + families["hallmark"],
    }


def flatten_matrix(corr: pd.DataFrame, pval: pd.DataFrame, meta: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    cols = list(corr.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            rows.append(
                {
                    **meta,
                    "variable_a": left,
                    "variable_b": right,
                    "corr": corr.loc[left, right],
                    "pval": pval.loc[left, right],
                }
            )
    return rows


def central_to_ring1(
    central: pd.DataFrame,
    ring1: pd.DataFrame,
    variables: list[str],
    meta: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pearson_rows = []
    spearman_rows = []
    for var in variables:
        pr, pp, pn = pearson_pair(central["SNAI1_ac"], ring1[var])
        sr, sp, sn = spearman_pair(central["SNAI1_ac"], ring1[var])
        pearson_rows.append(
            {
                **meta,
                "central_variable": "SNAI1_ac",
                "outer_variable": var,
                "corr": pr,
                "pval": pp,
                "n_pairs": pn,
            }
        )
        spearman_rows.append(
            {
                **meta,
                "central_variable": "SNAI1_ac",
                "outer_variable": var,
                "spearman_corr": sr,
                "spearman_pval": sp,
                "n_pairs": sn,
            }
        )
    return pearson_rows, spearman_rows


def select_ranked(df: pd.DataFrame) -> pd.DataFrame:
    frame = df[df["outer_variable"] != "SNAI1_ac"].dropna(subset=["corr", "pval"]).copy()
    if frame.empty:
        return frame
    frame["qval"] = bh_qvalues(frame["pval"])
    frame["is_significant"] = frame["qval"] <= 0.05
    frame["sig_marker"] = frame["qval"].apply(significance_marker)
    pos = frame[frame["corr"] > 0].sort_values("corr", ascending=False).head(6)
    neg = frame[frame["corr"] < 0].sort_values("corr", ascending=True).head(6)
    out = pd.concat([neg, pos], ignore_index=True)
    out["selection_basis"] = "strongest_6_positive_and_6_negative_by_pearson_r"
    return out


def compare_neighborhood_means(sample_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pivot_groups = {
        group: frame.set_index("variable_id")
        for group, frame in sample_rows.groupby("source_group")
    }
    for reference in ["snai1ac_consensus_cold", "snai12r_hot"]:
        if "snai1ac_consensus_hot" not in pivot_groups or reference not in pivot_groups:
            continue
        hot = pivot_groups["snai1ac_consensus_hot"]
        ref = pivot_groups[reference]
        for var in sorted(set(hot.index).intersection(ref.index)):
            rows.append(
                {
                    "sample_label": hot.loc[var, "sample_label"],
                    "reference_group": reference,
                    "variable_id": var,
                    "family": hot.loc[var, "family"],
                    "hot_mean": hot.loc[var, "ring1_mean"],
                    "reference_mean": ref.loc[var, "ring1_mean"],
                    "delta_hot_minus_reference": hot.loc[var, "ring1_mean"] - ref.loc[var, "ring1_mean"],
                }
            )
    return pd.DataFrame(rows)


def prepare_dirs() -> None:
    for directory in [TABLE_DIR, FIG_DIR, SCRIPT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(Path(__file__), SCRIPT_DIR / Path(__file__).name)
    except PermissionError as exc:
        log(f"WARNING: Could not refresh scripts_used copy: {exc}")


def main() -> None:
    prepare_dirs()
    register_anndata_null_reader()
    manifest = selected_samples(load_manifest())
    label_map = load_display_label_map()
    log(f"Run mode: {RUN_MODE}; samples: {len(manifest)}; output: {OUT_ROOT}")
    all_matrix_rows: list[dict[str, Any]] = []
    all_inner_rows: list[dict[str, Any]] = []
    all_spearman_rows: list[dict[str, Any]] = []
    all_ranked_rows: list[dict[str, Any]] = []
    all_variable_rows: list[dict[str, Any]] = []
    all_source_rows: list[dict[str, Any]] = []
    all_ring1_summary_rows: list[dict[str, Any]] = []

    for row in manifest.to_dict("records"):
        sample_label = row["sample_label"]
        log(f"Consensus-source neighborhood {RUN_MODE} {sample_label}")
        full = ad.read_h5ad(row["full_h5ad_path"])
        full.uns.clear()
        core = ad.read_h5ad(core_h5ad_path(sample_label))
        tumor = ad.read_h5ad(tumor_h5ad_path(sample_label))
        hallmark_scores = pd.read_csv(hallmark_scores_path(sample_label))
        variables, families, variable_records = build_variables(full, core, tumor, hallmark_scores)
        for record in variable_records:
            all_variable_rows.append({"sample_label": sample_label, **record})

        variable_sets = class_variable_sets(families)
        for source_group, source_col in CORE_SOURCE_COLUMNS.items():
            nodes = source_nodes_for_group(full, core, source_col)
            all_source_rows.append(
                {
                    "sample_label": sample_label,
                    "source_group": source_group,
                    "source_column": source_col,
                    "n_source_nodes": len(nodes),
                }
            )
            if len(nodes) < 3:
                continue

            ring1_all = ring1_neighbor_means(full, variables, nodes)
            central_all = variables.loc[nodes]
            for var, family in [(v, f) for f, vs in families.items() for v in vs]:
                values = pd.to_numeric(ring1_all[var], errors="coerce").dropna()
                all_ring1_summary_rows.append(
                    {
                        "sample_label": sample_label,
                        "source_group": source_group,
                        "variable_id": var,
                        "family": family,
                        "ring1_n": int(len(values)),
                        "ring1_mean": float(values.mean()) if len(values) else math.nan,
                        "ring1_median": float(values.median()) if len(values) else math.nan,
                    }
                )

            for class_name, vars_for_class in variable_sets.items():
                vars_for_class = [v for v in vars_for_class if v in ring1_all.columns]
                if len(vars_for_class) < 2:
                    continue
                class_ring1 = ring1_all[vars_for_class]
                corr, pval = make_corr_tables(class_ring1)
                meta = {
                    "sample_label": sample_label,
                    "source_group": source_group,
                    "variable_class": class_name,
                    "mode": "all_in_one_ring1_neighbor_means",
                    "n_source_nodes": len(nodes),
                }
                all_matrix_rows.extend(flatten_matrix(corr, pval, meta))
                plot_corr_heatmap(
                    corr,
                    FIG_DIR / f"{safe_name(sample_label)}__{source_group}__{class_name}__clustered_heatmap.png",
                    f"{sample_label}: {source_group}, {class_name}",
                    label_map,
                )

                pearson_rows, spearman_rows = central_to_ring1(central_all, ring1_all, vars_for_class, meta)
                pearson_df = pd.DataFrame(pearson_rows)
                pearson_df["qval"] = bh_qvalues(pearson_df["pval"])
                all_inner_rows.extend(pearson_df.to_dict("records"))
                all_spearman_rows.extend(spearman_rows)

                ranked = select_ranked(pearson_df)
                if len(ranked):
                    all_ranked_rows.extend(ranked.to_dict("records"))
                    plot_ranked_bars(
                        ranked,
                        FIG_DIR / f"{safe_name(sample_label)}__{source_group}__{class_name}__SNAI1ac_ring1_ranked12.png",
                        f"{sample_label}: {source_group}, {class_name}, SNAI1-ac to ring 1",
                        label_map,
                    )

        del full, core, tumor, hallmark_scores, variables
        gc.collect()

    variable_df = pd.DataFrame(all_variable_rows)
    source_df = pd.DataFrame(all_source_rows)
    matrix_df = pd.DataFrame(all_matrix_rows)
    inner_df = pd.DataFrame(all_inner_rows)
    spearman_df = pd.DataFrame(all_spearman_rows)
    ranked_df = pd.DataFrame(all_ranked_rows)
    ring1_summary_df = pd.DataFrame(all_ring1_summary_rows)
    comparison_df = compare_neighborhood_means(ring1_summary_df)
    source_group_comparison_df = inner_df.copy()
    source_order = list(CORE_SOURCE_COLUMNS.keys())
    n_source_group_comparison_heatmaps = 0
    for (sample_label, class_name), group in source_group_comparison_df.groupby(["sample_label", "variable_class"]):
        if group["source_group"].nunique() < 2:
            continue
        plot_source_group_comparison(
            group,
            FIG_DIR / f"{safe_name(sample_label)}__{class_name}__source_group_comparison_paperstyle.png",
            f"{sample_label}: source-group comparison, {class_name}",
            source_order,
            label_map,
        )
        n_source_group_comparison_heatmaps += 1

    variable_df.to_csv(TABLE_DIR / "consensus_source_variable_manifest.csv", index=False)
    source_df.to_csv(TABLE_DIR / "consensus_source_node_counts.csv", index=False)
    matrix_df.to_csv(TABLE_DIR / "consensus_source_allinone_pearson_correlations.csv", index=False)
    inner_df.to_csv(TABLE_DIR / "consensus_source_central_snai1ac_to_ring1_pearson.csv", index=False)
    spearman_df.to_csv(TABLE_DIR / "consensus_source_central_snai1ac_to_ring1_spearman_sensitivity.csv", index=False)
    ranked_df.to_csv(TABLE_DIR / "consensus_source_central_snai1ac_ranked12_for_plots.csv", index=False)
    source_group_comparison_df.to_csv(TABLE_DIR / "consensus_source_sourcegroup_comparison_long.csv", index=False)
    ring1_summary_df.to_csv(TABLE_DIR / "consensus_source_ring1_variable_summary.csv", index=False)
    comparison_df.to_csv(TABLE_DIR / "consensus_source_hot_vs_reference_ring1_mean_deltas.csv", index=False)

    manifest_out = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__)),
        "output_root": str(OUT_ROOT),
        "source_groups": CORE_SOURCE_COLUMNS,
        "run_mode": RUN_MODE,
        "samples": source_df["sample_label"].drop_duplicates().tolist(),
        "correlation_primary": "Pearson, matching SpottedPy default correlation behavior",
        "sensitivity": "Spearman central SNAI1-ac to ring-1 variable correlations exported as table",
        "paper_alignment_note": (
            "SpottedPy tutorials use source_nodes as a flexible subset for neighborhood calculations; "
            "formal reference-vs-comparison testing is better developed for distance/GEE. "
            "This preflight therefore runs separate SNAI1-ac hot/cold/SNAI1-2R-hot source sets "
            "and exports hot-vs-reference ring-1 mean deltas for review."
        ),
        "n_heatmaps": len(list(FIG_DIR.glob("*clustered_heatmap.png"))),
        "n_ranked_barplots": len(list(FIG_DIR.glob("*ranked12.png"))),
        "n_source_group_comparison_heatmaps": n_source_group_comparison_heatmaps,
    }
    (OUT_ROOT / "run_manifest.json").write_text(json.dumps(manifest_out, indent=2), encoding="utf-8")

    readme = [
        f"# SpottedPy v2 Consensus-Source Neighborhood {RUN_MODE.title()}",
        "",
        "Primary source nodes are SNAI1-ac full-slide consensus hotspots.",
        "Reference/source comparison sets are SNAI1-ac full-slide consensus coldspots and SNAI1-2R full-slide hotspots.",
        "",
        "Pearson is the primary correlation because it matches SpottedPy's default neighborhood implementation.",
        "Spearman is exported as sensitivity for central SNAI1-ac to ring-1 variables.",
        "",
        "SNAI1-ac remains in every heatmap row/column. It is excluded only from ranked barplot outcomes.",
        "",
        "Figures are generated for all variables and for SpaCET, MP/K*, and Hallmark classes separately.",
        "Ranked barplots always show the strongest positive and negative SNAI1-ac ring-1 correlations and mark BH-significant variables.",
        "",
        "Paper-style source-group comparison heatmaps use SNAI1-ac consensus hot, SNAI1-ac consensus cold, and SNAI1-2R hot as source-state columns.",
        "Only BH-significant Pearson correlations are colored, matching the logic of SpottedPy paper Fig. 6b; non-significant cells are white.",
        "",
        "The paper/tutorial guidance supports flexible source-node subsets for neighborhood analysis, but formal reference testing is primarily part of distance/GEE.",
        f"This {RUN_MODE} run therefore keeps hot/cold/SNAI1-2R-hot source groups separate and adds descriptive hot-vs-reference ring-1 mean deltas.",
        "",
        f"Heatmaps: {manifest_out['n_heatmaps']}",
        f"Ranked barplots: {manifest_out['n_ranked_barplots']}",
        f"Source-group comparison heatmaps: {manifest_out['n_source_group_comparison_heatmaps']}",
        "",
    ]
    (OUT_ROOT / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(json.dumps(manifest_out, indent=2), flush=True)


if __name__ == "__main__":
    main()
