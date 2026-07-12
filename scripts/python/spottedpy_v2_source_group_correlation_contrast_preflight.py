from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import spottedpy_v2_consensus_source_neighborhood_preflight as nb  # noqa: E402
from spottedpy_v2_hotspot_preflight import register_anndata_null_reader  # noqa: E402


RUN_ROOT = nb.RUN_ROOT
RUN_MODE = os.environ.get("SPOTTEDPY_V2_NEIGHBORHOOD_RUN_MODE", "preflight").strip().lower()
DEFAULT_OUT_SUBDIR = "source_group_correlation_contrast_full" if RUN_MODE == "full" else "source_group_correlation_contrast_preflight"
OUT_SUBDIR = os.environ.get("SPOTTEDPY_V2_CONTRAST_OUT_SUBDIR", DEFAULT_OUT_SUBDIR).strip()
OUT_ROOT = RUN_ROOT / "02_neighborhood_enrichment" / OUT_SUBDIR
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
HEATMAP_DIR = FIG_DIR / "01_delta_z_heatmaps"
BARPLOT_DIR = FIG_DIR / "02_ranked_delta_z_barplots"
SCRIPT_DIR = OUT_ROOT / "scripts_used"

KSTAR_PROJECTION_TABLE = Path(
    r"D:\HGSOC_Spatial_Atlas\05_analysis_ready\S3_cNMF_Tumor_Programs"
    r"\snai1ac_signature_projection_onto_cnmf_programs_v1\tables"
    r"\kstar_snai1ac_signature_projection_clean.csv"
)

SOURCE_ORDER = [
    "snai1ac_consensus_hot",
    "snai1ac_consensus_cold",
    "snai12r_hot",
    "background_non_snai1ac_hot_cold",
]

CONTRASTS = {
    "hot_vs_cold": "snai1ac_consensus_cold",
    "hot_vs_snai12r_hot": "snai12r_hot",
    "hot_vs_background": "background_non_snai1ac_hot_cold",
}

CLASS_ORDER = ["core", "spacet", "mp", "kstar", "hallmark"]
N_PERMUTATIONS = int(os.environ.get("SPOTTEDPY_V2_N_PERMUTATIONS", "999" if RUN_MODE == "full" else "199"))
MIN_PAIRS_FOR_CONTRAST = 6


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def prepare_dirs() -> None:
    for directory in [TABLE_DIR, HEATMAP_DIR, BARPLOT_DIR, SCRIPT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), SCRIPT_DIR / Path(__file__).name)


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def fisher_z(r: float) -> float:
    if pd.isna(r):
        return math.nan
    clipped = float(np.clip(r, -0.999999, 0.999999))
    return float(np.arctanh(clipped))


def fisher_contrast_p(r_hot: float, n_hot: int, r_ref: float, n_ref: int) -> tuple[float, float]:
    if pd.isna(r_hot) or pd.isna(r_ref) or n_hot <= 3 or n_ref <= 3:
        return math.nan, math.nan
    z_hot = fisher_z(r_hot)
    z_ref = fisher_z(r_ref)
    se = math.sqrt((1 / (n_hot - 3)) + (1 / (n_ref - 3)))
    if se <= 0 or pd.isna(se):
        return math.nan, math.nan
    z_stat = (z_hot - z_ref) / se
    pval = 2 * norm.sf(abs(z_stat))
    return float(z_stat), float(pval)


def corr_from_arrays(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return math.nan
    x_valid = x[mask]
    y_valid = y[mask]
    if np.unique(x_valid).size < 2 or np.unique(y_valid).size < 2:
        return math.nan
    return float(pearsonr(x_valid, y_valid)[0])


def permutation_delta_p(
    hot_frame: pd.DataFrame,
    ref_frame: pd.DataFrame,
    observed_delta_z: float,
    seed: int,
    n_permutations: int = N_PERMUTATIONS,
) -> tuple[float, int]:
    if pd.isna(observed_delta_z):
        return math.nan, 0
    hot = hot_frame[["central", "outer"]].replace([np.inf, -np.inf], np.nan).dropna()
    ref = ref_frame[["central", "outer"]].replace([np.inf, -np.inf], np.nan).dropna()
    n_hot = len(hot)
    n_ref = len(ref)
    if n_hot < MIN_PAIRS_FOR_CONTRAST or n_ref < MIN_PAIRS_FOR_CONTRAST:
        return math.nan, 0
    combined = pd.concat([hot, ref], ignore_index=True)
    x = combined["central"].to_numpy(dtype=float)
    y = combined["outer"].to_numpy(dtype=float)
    n_total = len(combined)
    if n_total < (2 * MIN_PAIRS_FOR_CONTRAST):
        return math.nan, 0
    rng = np.random.default_rng(seed)
    exceed = 0
    valid = 0
    all_idx = np.arange(n_total)
    for _ in range(n_permutations):
        hot_idx = rng.choice(all_idx, size=n_hot, replace=False)
        hot_mask = np.zeros(n_total, dtype=bool)
        hot_mask[hot_idx] = True
        r_perm_hot = corr_from_arrays(x[hot_mask], y[hot_mask])
        r_perm_ref = corr_from_arrays(x[~hot_mask], y[~hot_mask])
        if pd.isna(r_perm_hot) or pd.isna(r_perm_ref):
            continue
        delta_perm = fisher_z(r_perm_hot) - fisher_z(r_perm_ref)
        if abs(delta_perm) >= abs(observed_delta_z):
            exceed += 1
        valid += 1
    if valid == 0:
        return math.nan, 0
    return float((exceed + 1) / (valid + 1)), int(valid)


def progress_log(sample_label: str, message: str) -> None:
    log(f"{sample_label}: {message}")


def bh_qvalues(values: pd.Series) -> pd.Series:
    return nb.bh_qvalues(values)


def significance_marker(qval: float) -> str:
    if pd.isna(qval):
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


def compact_label(value: str) -> str:
    return str(value).strip().replace("\n", " ").replace("  ", " ")


def load_display_label_map() -> dict[str, str]:
    label_map = {
        "SNAI1_ac": "SNAI1_ac",
        "SNAI1_scoregenes": "SNAI1_scoregenes",
        "SNAI1_2R_scoregenes": "SNAI1_2R_scoregenes",
    }
    if KSTAR_PROJECTION_TABLE.exists():
        kstar = pd.read_csv(KSTAR_PROJECTION_TABLE)
        for row in kstar.itertuples(index=False):
            program_id = str(row.program_id)
            suffix = program_id.rsplit("__", 1)[-1]
            prefix = suffix.replace("P", "P", 1)
            category = compact_label(getattr(row, "alignment_category_draft", ""))
            if not category or category.lower() == "nan":
                continue
            label_map[f"Kstar_{program_id}"] = f"{prefix}_{category}"
    return label_map


def display_label(variable: str, label_map: dict[str, str]) -> str:
    return label_map.get(variable, variable)


def variable_family(variable: str, source_family: str | None = None) -> str:
    if variable in ["SNAI1_ac", "SNAI1_scoregenes", "SNAI1_2R_scoregenes"]:
        return "core"
    if variable.startswith("SpaCET_"):
        return "spacet"
    if variable.startswith("MP"):
        return "mp"
    if variable.startswith("Kstar_"):
        return "kstar"
    if source_family:
        return source_family
    return "hallmark"


def variable_order(
    variables: list[str],
    variable_manifest: pd.DataFrame,
    label_map: dict[str, str],
    variable_class: str,
) -> tuple[list[str], list[tuple[str, int, int]]]:
    if variable_class != "all_variables":
        ordered = [v for v in variables if v != "SNAI1_ac"]
        ordered = ["SNAI1_ac"] + sorted(
            ordered,
            key=lambda value: (variable_family(value), display_label(value, label_map)),
        )
    else:
        family_lookup = (
            variable_manifest.drop_duplicates("variable_id").set_index("variable_id")["family"].to_dict()
            if len(variable_manifest)
            else {}
        )
        ordered = sorted(
            variables,
            key=lambda value: (
                CLASS_ORDER.index(variable_family(value, family_lookup.get(value)))
                if variable_family(value, family_lookup.get(value)) in CLASS_ORDER
                else len(CLASS_ORDER),
                display_label(value, label_map),
            ),
        )

    group_labels: list[tuple[str, int, int]] = []
    current = None
    start = 0
    for idx, variable in enumerate(ordered):
        family = variable_family(variable)
        if current is None:
            current = family
            start = idx
        elif family != current:
            group_labels.append((current, start, idx - 1))
            current = family
            start = idx
    if current is not None:
        group_labels.append((current, start, len(ordered) - 1))
    label_lookup = {"core": "core", "spacet": "SpaCET", "mp": "MP", "kstar": "K*", "hallmark": "Hallmark"}
    return ordered, [(label_lookup.get(label, label), start, end) for label, start, end in group_labels]


def source_node_sets(full, core, variables: pd.DataFrame) -> dict[str, list[str]]:
    source_nodes = {
        group: nb.source_nodes_for_group(full, core, col)
        for group, col in nb.CORE_SOURCE_COLUMNS.items()
    }
    hot = set(source_nodes["snai1ac_consensus_hot"])
    cold = set(source_nodes["snai1ac_consensus_cold"])
    eligible = set(variables.index[pd.to_numeric(variables["SNAI1_ac"], errors="coerce").notna()])
    background = sorted(eligible.difference(hot).difference(cold))
    source_nodes["background_non_snai1ac_hot_cold"] = background
    return source_nodes


def central_ring_frames(
    full,
    variables: pd.DataFrame,
    source_nodes: dict[str, list[str]],
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    frames: dict[str, pd.DataFrame] = {}
    counts: list[dict[str, Any]] = []
    for group in SOURCE_ORDER:
        nodes = source_nodes[group]
        if len(nodes) < 3:
            continue
        ring1 = nb.ring1_neighbor_means(full, variables, nodes)
        central = variables.loc[nodes, ["SNAI1_ac"]].rename(columns={"SNAI1_ac": "central"})
        frame = ring1.copy()
        frame.insert(0, "central", central["central"])
        frames[group] = frame
        counts.append({"source_group": group, "n_source_nodes": len(nodes)})
    return frames, counts


def inner_outer_rows(
    frames: dict[str, pd.DataFrame],
    variable_sets: dict[str, list[str]],
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_group, frame in frames.items():
        for variable_class, variables in variable_sets.items():
            for variable in [v for v in variables if v in frame.columns]:
                r, p, n = nb.pearson_pair(frame["central"], frame[variable])
                rows.append(
                    {
                        **meta,
                        "source_group": source_group,
                        "variable_class": variable_class,
                        "central_variable": "SNAI1_ac",
                        "outer_variable": variable,
                        "corr": r,
                        "pval": p,
                        "n_pairs": n,
                    }
                )
    return rows


def contrast_rows(
    corr_df: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    key_cols = ["sample_label", "variable_class", "outer_variable"]
    corr_lookup = corr_df.set_index(key_cols + ["source_group"])
    hot_group = "snai1ac_consensus_hot"
    permutation_cache: dict[tuple[str, str], tuple[float, int]] = {}
    sample_label_for_log = str(meta["sample_label"])
    total_unique = len(
        {
            (contrast_name, variable)
            for (_, _, variable), _ in corr_df.groupby(key_cols)
            for contrast_name in CONTRASTS
        }
    )
    completed_unique = 0
    for (sample_label, variable_class, variable), _ in corr_df.groupby(key_cols):
        if sample_label != meta["sample_label"]:
            continue
        hot_key = (sample_label, variable_class, variable, hot_group)
        if hot_key not in corr_lookup.index:
            continue
        hot_row = corr_lookup.loc[hot_key]
        r_hot = float(hot_row["corr"])
        n_hot = int(hot_row["n_pairs"])
        z_hot = fisher_z(r_hot)
        for contrast_name, ref_group in CONTRASTS.items():
            ref_key = (sample_label, variable_class, variable, ref_group)
            if ref_key not in corr_lookup.index:
                continue
            ref_row = corr_lookup.loc[ref_key]
            r_ref = float(ref_row["corr"])
            n_ref = int(ref_row["n_pairs"])
            z_ref = fisher_z(r_ref)
            delta_z = z_hot - z_ref if not pd.isna(z_hot) and not pd.isna(z_ref) else math.nan
            delta_r = r_hot - r_ref if not pd.isna(r_hot) and not pd.isna(r_ref) else math.nan
            z_stat, fisher_p = fisher_contrast_p(r_hot, n_hot, r_ref, n_ref)
            hot_frame = pd.DataFrame(
                {"central": frames[hot_group]["central"], "outer": frames[hot_group][variable]}
            )
            ref_frame = pd.DataFrame(
                {"central": frames[ref_group]["central"], "outer": frames[ref_group][variable]}
            )
            overlap = len(set(frames[hot_group].index).intersection(set(frames[ref_group].index)))
            cache_key = (contrast_name, variable)
            if cache_key in permutation_cache:
                perm_p, n_valid_perm = permutation_cache[cache_key]
            else:
                completed_unique += 1
                if completed_unique == 1 or completed_unique % 25 == 0 or completed_unique == total_unique:
                    progress_log(
                        sample_label_for_log,
                        f"permutation contrasts {completed_unique}/{total_unique}",
                    )
                perm_p, n_valid_perm = permutation_delta_p(
                    hot_frame,
                    ref_frame,
                    delta_z,
                    seed=stable_seed(sample_label, variable, contrast_name),
                )
                permutation_cache[cache_key] = (perm_p, n_valid_perm)
            rows.append(
                {
                    **meta,
                    "contrast": contrast_name,
                    "hot_group": hot_group,
                    "reference_group": ref_group,
                    "variable_class": variable_class,
                    "outer_variable": variable,
                    "r_hot": r_hot,
                    "r_reference": r_ref,
                    "delta_r": delta_r,
                    "z_hot": z_hot,
                    "z_reference": z_ref,
                    "delta_z": delta_z,
                    "n_hot_pairs": n_hot,
                    "n_reference_pairs": n_ref,
                    "source_node_overlap": overlap,
                    "fisher_z_stat": z_stat,
                    "fisher_pval": fisher_p,
                    "permutation_pval": perm_p,
                    "n_valid_permutations": n_valid_perm,
                    "n_requested_permutations": N_PERMUTATIONS,
                }
            )
    return rows


def add_qvalues(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["fisher_qval"] = np.nan
    out["permutation_qval"] = np.nan
    out["sig_marker"] = ""
    group_cols = ["sample_label", "variable_class", "contrast"]
    for _, idx in out.groupby(group_cols).groups.items():
        out.loc[idx, "fisher_qval"] = bh_qvalues(out.loc[idx, "fisher_pval"])
        out.loc[idx, "permutation_qval"] = bh_qvalues(out.loc[idx, "permutation_pval"])
    out["sig_marker"] = out["permutation_qval"].apply(significance_marker)
    return out


def plot_delta_heatmap(
    df: pd.DataFrame,
    variable_manifest: pd.DataFrame,
    label_map: dict[str, str],
    out_path: Path,
    title: str,
    variable_class: str,
) -> None:
    if df.empty:
        return
    matrix = df.pivot(index="contrast", columns="outer_variable", values="delta_z").reindex(index=list(CONTRASTS))
    qvals = df.pivot(index="contrast", columns="outer_variable", values="permutation_qval").reindex(index=list(CONTRASTS))
    matrix = matrix.dropna(axis=1, how="all")
    if matrix.empty:
        return
    order, group_labels = variable_order(list(matrix.columns), variable_manifest, label_map, variable_class)
    order = [var for var in order if var in matrix.columns]
    matrix = matrix[order]
    qvals = qvals[order]
    labels = [display_label(var, label_map) for var in order]
    max_abs = float(np.nanmax(np.abs(matrix.to_numpy(dtype=float))))
    vmax = max(0.2, min(2.5, max_abs))
    width = max(9.5, min(42, 0.34 * len(order) + 4.0))
    max_label_len = max((len(label) for label in labels), default=0)
    height = 7.4 if len(order) > 35 or max_label_len > 32 else 5.8
    bottom = 0.56 if height >= 7 else 0.48
    fig, ax = plt.subplots(figsize=(width, height))
    left = 0.18 if width < 14 else 0.10
    right = 0.88 if width < 14 else 0.94
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=0.76)
    im = ax.imshow(matrix.to_numpy(dtype=float), vmin=-vmax, vmax=vmax, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=6)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=8)
    fig.suptitle(title, fontsize=10, y=0.96)
    for label, start, end in group_labels:
        if start >= len(order):
            continue
        end = min(end, len(order) - 1)
        if start > 0:
            ax.axvline(start - 0.5, color="#111827", linewidth=0.8)
        ax.text(
            (start + end) / 2,
            1.08,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            clip_on=False,
        )
    for row_idx, contrast in enumerate(matrix.index):
        for col_idx, variable in enumerate(order):
            marker = significance_marker(qvals.loc[contrast, variable])
            if marker:
                ax.text(col_idx, row_idx, marker, ha="center", va="center", fontsize=6, color="#111827")
    cbar = fig.colorbar(im, ax=ax, fraction=0.022 if len(order) > 35 else 0.04, pad=0.018)
    cbar.set_label("Delta Fisher z\nhot - reference", fontsize=7)
    fig.text(
        left,
        0.025,
        "Stars: permutation BH q; positive means stronger/more positive coupling in SNAI1-ac hot.",
        fontsize=6,
        ha="left",
        va="bottom",
        color="#374151",
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def select_ranked_contrasts(df: pd.DataFrame, label_map: dict[str, str]) -> pd.DataFrame:
    frame = df[df["outer_variable"] != "SNAI1_ac"].dropna(subset=["delta_z"]).copy()
    if frame.empty:
        return frame
    pos = frame[frame["delta_z"] > 0].sort_values("delta_z", ascending=False).head(6)
    neg = frame[frame["delta_z"] < 0].sort_values("delta_z", ascending=True).head(6)
    out = pd.concat([neg, pos], ignore_index=True)
    out["outer_variable_display"] = out["outer_variable"].map(lambda value: display_label(value, label_map))
    out["selection_basis"] = "strongest_6_positive_and_6_negative_delta_fisher_z"
    return out


def plot_ranked_delta(df: pd.DataFrame, out_path: Path, title: str) -> None:
    if df.empty:
        return
    frame = df.sort_values("delta_z")
    height = max(4, 0.3 * len(frame) + 1.8)
    fig, ax = plt.subplots(figsize=(8.5, height))
    colors = np.where(frame["delta_z"] >= 0, "#b2182b", "#2166ac")
    bars = ax.barh(frame["outer_variable_display"], frame["delta_z"], color=colors)
    ax.axvline(0, color="#111827", linewidth=0.8)
    ax.set_xlabel("Delta Fisher z (SNAI1-ac hot - reference)")
    ax.set_title(title, fontsize=10)
    for bar, marker, delta_value in zip(bars, frame["sig_marker"], frame["delta_z"]):
        if not marker:
            continue
        ax.text(
            delta_value / 2,
            bar.get_y() + bar.get_height() / 2,
            marker,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="#ffffff",
        )
    ax.tick_params(axis="y", labelsize=7)
    fig.subplots_adjust(bottom=0.22)
    fig.text(
        0.02,
        0.035,
        "Stars indicate permutation BH q-value: **** <=1e-4, *** <=1e-3, ** <=1e-2, * <=0.05",
        fontsize=6,
        ha="left",
        va="bottom",
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    prepare_dirs()
    register_anndata_null_reader()
    manifest = nb.selected_samples(nb.load_manifest())
    log(
        f"Run mode: {RUN_MODE}; samples: {len(manifest)}; "
        f"permutations: {N_PERMUTATIONS}; output: {OUT_ROOT}"
    )
    label_map = load_display_label_map()
    all_source_counts: list[dict[str, Any]] = []
    all_corr_rows: list[dict[str, Any]] = []
    all_contrast_rows: list[dict[str, Any]] = []
    all_variable_rows: list[dict[str, Any]] = []

    for row in manifest.to_dict("records"):
        sample_label = row["sample_label"]
        log(f"Source-group correlation contrast {RUN_MODE} {sample_label}")
        full = ad.read_h5ad(row["full_h5ad_path"])
        full.uns.clear()
        core = ad.read_h5ad(nb.core_h5ad_path(sample_label))
        tumor = ad.read_h5ad(nb.tumor_h5ad_path(sample_label))
        hallmark_scores = pd.read_csv(nb.hallmark_scores_path(sample_label))
        variables, families, variable_records = nb.build_variables(full, core, tumor, hallmark_scores)
        variable_sets = nb.class_variable_sets(families)
        for record in variable_records:
            all_variable_rows.append({"sample_label": sample_label, **record})
        source_nodes = source_node_sets(full, core, variables)
        progress_log(
            sample_label,
            "source nodes "
            + ", ".join(f"{group}={len(nodes)}" for group, nodes in source_nodes.items()),
        )
        progress_log(sample_label, "computing ring-1 neighbor means")
        frames, count_rows = central_ring_frames(full, variables, source_nodes)
        for count_row in count_rows:
            all_source_counts.append({"sample_label": sample_label, **count_row})
        progress_log(sample_label, "computing inner-outer correlations")
        corr_rows = inner_outer_rows(frames, variable_sets, {"sample_label": sample_label})
        corr_df = pd.DataFrame(corr_rows)
        corr_df["qval"] = np.nan
        for _, idx in corr_df.groupby(["sample_label", "source_group", "variable_class"]).groups.items():
            corr_df.loc[idx, "qval"] = bh_qvalues(corr_df.loc[idx, "pval"])
        all_corr_rows.extend(corr_df.to_dict("records"))
        progress_log(sample_label, "computing source-group contrast tests")
        sample_contrasts = contrast_rows(corr_df, frames, {"sample_label": sample_label})
        all_contrast_rows.extend(sample_contrasts)
        progress_log(sample_label, "finished sample")
        del full, core, tumor, hallmark_scores, variables, frames
        gc.collect()

    source_counts_df = pd.DataFrame(all_source_counts)
    variable_manifest_df = pd.DataFrame(all_variable_rows)
    corr_df = pd.DataFrame(all_corr_rows)
    contrast_df = add_qvalues(pd.DataFrame(all_contrast_rows))
    contrast_df["outer_variable_display"] = contrast_df["outer_variable"].map(lambda value: display_label(value, label_map))

    ranked_rows = []
    n_heatmaps = 0
    n_barplots = 0
    for (sample_label, variable_class), group in contrast_df.groupby(["sample_label", "variable_class"]):
        plot_delta_heatmap(
            group,
            variable_manifest_df[variable_manifest_df["sample_label"] == sample_label],
            label_map,
            HEATMAP_DIR / f"{nb.safe_name(sample_label)}__{variable_class}__delta_fisher_z_contrasts.png",
            f"{sample_label}: source-group inner-outer correlation contrasts, {variable_class}",
            variable_class,
        )
        n_heatmaps += 1
        for contrast_name, contrast_group in group.groupby("contrast"):
            ranked = select_ranked_contrasts(contrast_group, label_map)
            if ranked.empty:
                continue
            ranked_rows.append(ranked)
            plot_ranked_delta(
                ranked,
                BARPLOT_DIR
                / f"{nb.safe_name(sample_label)}__{variable_class}__{contrast_name}__ranked_delta_fisher_z.png",
                f"{sample_label}: {contrast_name}, {variable_class}",
            )
            n_barplots += 1

    ranked_df = pd.concat(ranked_rows, ignore_index=True) if ranked_rows else pd.DataFrame()

    source_counts_df.to_csv(TABLE_DIR / "source_group_contrast_source_counts.csv", index=False)
    variable_manifest_df.to_csv(TABLE_DIR / "source_group_contrast_variable_manifest.csv", index=False)
    corr_df.to_csv(TABLE_DIR / "source_group_inner_outer_correlations_with_background.csv", index=False)
    contrast_df.to_csv(TABLE_DIR / "source_group_correlation_contrasts.csv", index=False)
    ranked_df.to_csv(TABLE_DIR / "source_group_correlation_contrast_ranked12_for_plots.csv", index=False)

    manifest_out = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__)),
        "output_root": str(OUT_ROOT),
        "run_mode": RUN_MODE,
        "samples": manifest["sample_label"].tolist(),
        "source_groups": {
            **nb.CORE_SOURCE_COLUMNS,
            "background_non_snai1ac_hot_cold": "all spots with finite SNAI1-ac, excluding SNAI1-ac consensus hot and cold source spots",
        },
        "contrasts": CONTRASTS,
        "correlation": "Pearson central SNAI1-ac against ring-1 neighbor variable mean",
        "contrast_scale": "delta Fisher z = atanh(r_hot) - atanh(r_reference)",
        "significance_primary": f"label-permutation p-values with {N_PERMUTATIONS} requested permutations, BH-corrected within sample/variable_class/contrast",
        "significance_secondary": "analytic Fisher z contrast p-values exported for comparison",
        "n_heatmaps": n_heatmaps,
        "n_ranked_barplots": n_barplots,
    }
    (OUT_ROOT / "run_manifest.json").write_text(json.dumps(manifest_out, indent=2), encoding="utf-8")
    readme = [
        f"# SpottedPy v2 Source-Group Correlation Contrast {RUN_MODE.title()}",
        "",
        "This branch tests whether inner-outer correlations in SNAI1-ac consensus hotspots differ from reference source groups.",
        "",
        "Primary contrasts:",
        "",
        "- `hot_vs_cold`: SNAI1-ac consensus hot minus SNAI1-ac consensus cold.",
        "- `hot_vs_snai12r_hot`: SNAI1-ac consensus hot minus SNAI1-2R hotspot source spots.",
        "- `hot_vs_background`: SNAI1-ac consensus hot minus all finite-SNAI1-ac spots excluding SNAI1-ac hot and cold spots.",
        "",
        "The plotted effect is delta Fisher z, `atanh(r_hot) - atanh(r_reference)`. Positive values mean the inner-outer coupling is stronger or more positive in SNAI1-ac hotspots; negative values mean weaker or more negative coupling.",
        "",
        "Stars in figures use permutation BH q-values. Analytic Fisher-z p-values are exported as a secondary, faster reference.",
        "",
        f"This is the `{RUN_MODE}` run.",
        "",
        "## Outputs",
        "",
        "- `tables/source_group_inner_outer_correlations_with_background.csv`",
        "- `tables/source_group_correlation_contrasts.csv`",
        "- `figures/01_delta_z_heatmaps/`",
        "- `figures/02_ranked_delta_z_barplots/`",
        "",
        f"Heatmaps: {n_heatmaps}",
        f"Ranked barplots: {n_barplots}",
        "",
    ]
    (OUT_ROOT / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(json.dumps(manifest_out, indent=2), flush=True)


if __name__ == "__main__":
    main()
