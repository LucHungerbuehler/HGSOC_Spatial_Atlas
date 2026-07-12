from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.optimize import linear_sum_assignment
from scipy.stats import mannwhitneyu, wilcoxon


ROOT = Path(r"D:\HGSOC_Spatial_Atlas\05_analysis_ready")
RUN_ROOT = ROOT / "20260424_definition3b_definition4_raw_geneNMF"
HH_ROOT = RUN_ROOT / "08_hh_programme_characterization"
D3B_ROOT = RUN_ROOT / "02_definition3b_mixture_programme_niches"
MP_H5AD_ROOT = (
    ROOT
    / "S3_cNMF_Tumor_Programs"
    / "jaccard_raw_matrices"
    / "inspection_exports_average"
    / "variantB_nonjunk_manual_cut_v2"
    / "subcluster_signatures_scoring"
    / "scored_h5ad"
)
MP_SCORE_MANIFEST = MP_H5AD_ROOT.parent / "manual_subcluster_scoring_manifest.json"

OUT_ROOT = HH_ROOT / "report_assets_mp_enrichmap"
OUT_TABLES = OUT_ROOT / "tables"
OUT_FIGURES = OUT_ROOT / "figures"

HEX_OFFSETS = [(2, 0), (-2, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
CONTEXT_ORDER = [
    ("spot_level", "Center spots"),
    ("neighborhood", "Immediate neighborhood"),
]
CONTEXT_LABELS = {
    "spot_level": "Centre spots",
    "neighborhood": "Immediate neighbours",
}

MP_SCORES = [
    ("MP1", "MP1 angiogenic/vascular", "MP1_angiogenic_vascular_score"),
    ("MP2", "MP2 iCAF-stress", "MP2_iCAF_stress_score"),
    ("MP3", "MP3 complement-CAF", "MP3_complement_CAF_score"),
    ("MP4", "MP4 activated-myCAF", "MP4_activated_myCAF_score"),
    ("MP5", "MP5 IFN/TLS immune", "MP5_IFN_TLS_immune_score"),
    ("MP6", "MP6 APC/TAM myeloid", "MP6_APC_TAM_myeloid_score"),
    ("MP7", "MP7 malignant hypoxia", "MP7_malignant_hypoxia_score"),
    ("MP8", "MP8 malignant acute-phase/secretory", "MP8_malignant_acute_phase_secretory_score"),
]
MP_ORDER = [row[0] for row in MP_SCORES]
MP_LABELS = {mp_id: label for mp_id, label, _ in MP_SCORES}
MP_COLUMNS = [column for _, _, column in MP_SCORES]

CONTRAST_ORDER = [
    ("hh_vs_ll_unmatched", "HH versus LL unmatched", "LL"),
    ("hh_vs_matched_ll", "HH versus matched LL", "LL"),
    ("hh_vs_matched_nonhh", "HH versus matched non-HH", "non-HH"),
]

INK = "#1F2430"
MUTED = "#5F6675"
GRID = "#E6E8F0"
AXIS = "#C8CDD8"
HH_COLOR = "#bf3f3f"
CONTROL_COLOR = "#3f6fb5"
PLOT_X_LIMIT = 10


def ensure_dirs() -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)


def decode_h5_values(values: np.ndarray) -> list[object]:
    out: list[object] = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(value)
    return out


def read_h5ad_obs_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    with h5py.File(path, "r") as handle:
        obs_group = handle["obs"]
        index_key = obs_group.attrs.get("_index", "_index")
        if isinstance(index_key, bytes):
            index_key = index_key.decode("utf-8")
        if index_key in obs_group:
            spot_index = decode_h5_values(obs_group[index_key][()])
        elif "_index" in obs_group:
            spot_index = decode_h5_values(obs_group["_index"][()])
        else:
            raise KeyError(f"Could not find obs index in {path}")

        data: dict[str, object] = {"spot_id": spot_index}
        missing: list[str] = []
        for column in columns:
            if column not in obs_group:
                missing.append(column)
                continue
            node = obs_group[column]
            if not isinstance(node, h5py.Dataset):
                raise TypeError(f"Expected numeric dataset for {column} in {path}")
            data[column] = node[()]
    if missing:
        raise KeyError(f"Missing MP score columns in {path}: {missing}")
    return pd.DataFrame(data)


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    out = np.full(pvalues.shape, np.nan, dtype=float)
    mask = np.isfinite(pvalues)
    if mask.sum() == 0:
        return out
    valid = pvalues[mask]
    order = np.argsort(valid)
    ranks = np.arange(1, len(valid) + 1, dtype=float)
    adjusted = valid[order] * len(valid) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out[np.where(mask)[0][order]] = adjusted
    return out


def pooled_sd(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 and len(b) < 2:
        return math.nan
    var_a = float(np.var(a, ddof=1)) if len(a) >= 2 else 0.0
    var_b = float(np.var(b, ddof=1)) if len(b) >= 2 else 0.0
    return math.sqrt((var_a + var_b) / 2.0)


def standardized_mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    denom = pooled_sd(a, b)
    if not np.isfinite(denom) or denom == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / denom)


def sample_label_from_path(path: Path) -> str:
    return path.parent.parent.name


def mp_h5ad_path(sample_label: str) -> Path:
    dataset, sample = sample_label.split("__", 1)
    return MP_H5AD_ROOT / dataset / f"{sample}.manual_jaccard_MP_scores.h5ad"


def load_sample_frame(sample_path: Path) -> pd.DataFrame:
    sample_label = sample_label_from_path(sample_path)
    path = mp_h5ad_path(sample_label)
    if not path.exists():
        raise FileNotFoundError(f"Missing MP EnrichMap h5ad for {sample_label}: {path}")

    frame = pd.read_csv(sample_path)
    scores = read_h5ad_obs_columns(path, MP_COLUMNS)
    merged = frame.merge(scores, on="spot_id", how="left", validate="one_to_one")
    missing = merged[MP_COLUMNS].isna().any(axis=1).sum()
    if missing:
        raise ValueError(f"{sample_label} has {int(missing)} tumor rows missing MP scores after spot_id join")

    needed = [
        "spot_id",
        "dataset",
        "sample_id_on_disk",
        "sample_label",
        "LISA_category",
        "Malignant",
        "array_row",
        "array_col",
        *MP_COLUMNS,
    ]
    missing_cols = [column for column in needed if column not in merged.columns]
    if missing_cols:
        raise KeyError(f"{sample_path} lacks required columns after MP join: {missing_cols}")
    out = merged[needed].copy()
    out["Malignant"] = pd.to_numeric(out["Malignant"], errors="coerce")
    out["array_row"] = pd.to_numeric(out["array_row"], errors="coerce")
    out["array_col"] = pd.to_numeric(out["array_col"], errors="coerce")
    for column in MP_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    valid_mask = (
        np.isfinite(out["Malignant"].to_numpy(dtype=float))
        & np.isfinite(out["array_row"].to_numpy(dtype=float))
        & np.isfinite(out["array_col"].to_numpy(dtype=float))
        & np.isfinite(out[MP_COLUMNS].to_numpy(dtype=float)).all(axis=1)
    )
    return out.loc[valid_mask].reset_index(drop=True)


def build_hex_neighbor_lists(frame: pd.DataFrame) -> list[np.ndarray]:
    coords = frame[["array_row", "array_col"]].copy()
    coords["array_row"] = coords["array_row"].astype(int)
    coords["array_col"] = coords["array_col"].astype(int)
    lookup = {
        (int(row.array_row), int(row.array_col)): idx
        for idx, row in coords.reset_index(drop=True).iterrows()
    }
    neighbor_lists: list[np.ndarray] = []
    for _, row in coords.reset_index(drop=True).iterrows():
        key = (int(row.array_row), int(row.array_col))
        neighbors = []
        for dr, dc in HEX_OFFSETS:
            idx = lookup.get((key[0] + dr, key[1] + dc))
            if idx is not None:
                neighbors.append(int(idx))
        neighbor_lists.append(np.asarray(sorted(set(neighbors)), dtype=int))
    return neighbor_lists


def compute_neighbor_vectors(frame: pd.DataFrame) -> pd.DataFrame:
    neighbor_lists = build_hex_neighbor_lists(frame)
    scores = frame[MP_COLUMNS].to_numpy(dtype=float)
    keep_cols = [
        "spot_id",
        "dataset",
        "sample_id_on_disk",
        "sample_label",
        "LISA_category",
        "Malignant",
        "array_row",
        "array_col",
    ]
    rows: list[dict[str, object]] = []
    for idx, neighbors in enumerate(neighbor_lists):
        if len(neighbors) == 0:
            continue
        row = {column: frame.iloc[idx][column] for column in keep_cols}
        row["neighbor_count"] = int(len(neighbors))
        mean_vector = scores[neighbors].mean(axis=0)
        for column, value in zip(MP_COLUMNS, mean_vector, strict=True):
            row[column] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def optimal_malignant_matching(
    hh_frame: pd.DataFrame,
    control_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    hh_values = hh_frame["Malignant"].to_numpy(dtype=float)
    control_values = control_frame["Malignant"].to_numpy(dtype=float)
    cost = np.abs(hh_values[:, None] - control_values[None, :])
    hh_idx, control_idx = linear_sum_assignment(cost)
    order = np.argsort(hh_idx)
    hh_idx = hh_idx[order]
    control_idx = control_idx[order]
    matched_hh = hh_frame.iloc[hh_idx].reset_index(drop=True).copy()
    matched_control = control_frame.iloc[control_idx].reset_index(drop=True).copy()
    pair_diff = np.abs(
        matched_hh["Malignant"].to_numpy(dtype=float)
        - matched_control["Malignant"].to_numpy(dtype=float)
    )
    diagnostics = {
        "matched_pairs_n": int(len(matched_hh)),
        "mean_abs_malignant_diff": float(np.mean(pair_diff)) if len(pair_diff) else math.nan,
        "median_abs_malignant_diff": float(np.median(pair_diff)) if len(pair_diff) else math.nan,
        "max_abs_malignant_diff": float(np.max(pair_diff)) if len(pair_diff) else math.nan,
    }
    return matched_hh, matched_control, diagnostics


def summarize_context(
    hh_all: pd.DataFrame,
    control_all: pd.DataFrame,
    context_type: str,
    contrast_id: str,
    contrast_label: str,
    control_label: str,
    matched_hh: pd.DataFrame | None = None,
    matched_control: pd.DataFrame | None = None,
    diagnostics: dict[str, float] | None = None,
) -> dict[str, object]:
    sample_info = hh_all.iloc[0]
    row: dict[str, object] = {
        "dataset": str(sample_info["dataset"]),
        "sample_id_on_disk": str(sample_info["sample_id_on_disk"]),
        "sample_label": str(sample_info["sample_label"]),
        "contrast_id": contrast_id,
        "contrast_label": contrast_label,
        "control_label": control_label,
        "context_type": context_type,
        "hh_pool_n": int(len(hh_all)),
        "control_pool_n": int(len(control_all)),
        "hh_malignant_mean_pre": float(np.mean(hh_all["Malignant"])),
        "control_malignant_mean_pre": float(np.mean(control_all["Malignant"])),
        "malignant_smd_pre": standardized_mean_difference(
            hh_all["Malignant"].to_numpy(dtype=float),
            control_all["Malignant"].to_numpy(dtype=float),
        ),
    }
    if matched_hh is not None and matched_control is not None and diagnostics is not None:
        row.update(
            {
                "matched_pairs_n": int(diagnostics["matched_pairs_n"]),
                "hh_malignant_mean_post": float(np.mean(matched_hh["Malignant"])),
                "control_malignant_mean_post": float(np.mean(matched_control["Malignant"])),
                "malignant_smd_post": standardized_mean_difference(
                    matched_hh["Malignant"].to_numpy(dtype=float),
                    matched_control["Malignant"].to_numpy(dtype=float),
                ),
                "mean_abs_malignant_diff": diagnostics["mean_abs_malignant_diff"],
                "median_abs_malignant_diff": diagnostics["median_abs_malignant_diff"],
                "max_abs_malignant_diff": diagnostics["max_abs_malignant_diff"],
            }
        )
    return row


def paired_mp_contrasts(
    hh_frame: pd.DataFrame,
    control_frame: pd.DataFrame,
    context_type: str,
    contrast_id: str,
    contrast_label: str,
    control_label: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sample_info = hh_frame.iloc[0]
    for mp_id, mp_label, column in MP_SCORES:
        hh_vals = hh_frame[column].to_numpy(dtype=float)
        control_vals = control_frame[column].to_numpy(dtype=float)
        diff = hh_vals - control_vals
        p_value = math.nan
        statistic = math.nan
        nonzero_diff = diff[np.abs(diff) > 1e-12]
        if len(nonzero_diff) >= 5:
            result = wilcoxon(diff, zero_method="wilcox", alternative="two-sided", correction=False)
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
        diff_sd = float(np.std(diff, ddof=1)) if len(diff) >= 2 else math.nan
        cohens_dz = float(np.mean(diff) / diff_sd) if np.isfinite(diff_sd) and diff_sd > 0 else 0.0
        rows.append(
            {
                "dataset": str(sample_info["dataset"]),
                "sample_id_on_disk": str(sample_info["sample_id_on_disk"]),
                "sample_label": str(sample_info["sample_label"]),
                "contrast_id": contrast_id,
                "contrast_label": contrast_label,
                "control_label": control_label,
                "context_type": context_type,
                "context_label": CONTEXT_LABELS[context_type],
                "mp_id": mp_id,
                "mp_label": mp_label,
                "score_column": column,
                "n_pairs": int(len(diff)),
                "hh_mean": float(np.mean(hh_vals)),
                "control_mean": float(np.mean(control_vals)),
                "mean_difference_hh_minus_control": float(np.mean(diff)),
                "median_difference_hh_minus_control": float(np.median(diff)),
                "cohens_dz_hh_minus_control": cohens_dz,
                "test": "paired_wilcoxon_signed_rank",
                "test_statistic": statistic,
                "p_value": p_value,
            }
        )
    out = pd.DataFrame(rows)
    out["fdr_bh"] = bh_fdr(out["p_value"].to_numpy(dtype=float))
    out["direction"] = np.where(
        out["mean_difference_hh_minus_control"] > 0,
        "HH_enriched",
        np.where(out["mean_difference_hh_minus_control"] < 0, "HH_depleted", "no_difference"),
    )
    return out


def unpaired_mp_contrasts(
    hh_frame: pd.DataFrame,
    control_frame: pd.DataFrame,
    context_type: str,
    contrast_id: str,
    contrast_label: str,
    control_label: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sample_info = hh_frame.iloc[0]
    for mp_id, mp_label, column in MP_SCORES:
        hh_vals = hh_frame[column].to_numpy(dtype=float)
        control_vals = control_frame[column].to_numpy(dtype=float)
        p_value = math.nan
        statistic = math.nan
        if len(hh_vals) >= 3 and len(control_vals) >= 3:
            result = mannwhitneyu(hh_vals, control_vals, alternative="two-sided", method="auto")
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
        rows.append(
            {
                "dataset": str(sample_info["dataset"]),
                "sample_id_on_disk": str(sample_info["sample_id_on_disk"]),
                "sample_label": str(sample_info["sample_label"]),
                "contrast_id": contrast_id,
                "contrast_label": contrast_label,
                "control_label": control_label,
                "context_type": context_type,
                "context_label": CONTEXT_LABELS[context_type],
                "mp_id": mp_id,
                "mp_label": mp_label,
                "score_column": column,
                "hh_n": int(len(hh_vals)),
                "control_n": int(len(control_vals)),
                "hh_mean": float(np.mean(hh_vals)),
                "control_mean": float(np.mean(control_vals)),
                "mean_difference_hh_minus_control": float(np.mean(hh_vals) - np.mean(control_vals)),
                "median_difference_hh_minus_control": float(np.median(hh_vals) - np.median(control_vals)),
                "cohens_d_hh_minus_control": standardized_mean_difference(hh_vals, control_vals),
                "test": "mannwhitneyu",
                "test_statistic": statistic,
                "p_value": p_value,
            }
        )
    out = pd.DataFrame(rows)
    out["fdr_bh"] = bh_fdr(out["p_value"].to_numpy(dtype=float))
    out["direction"] = np.where(
        out["mean_difference_hh_minus_control"] > 0,
        "HH_enriched",
        np.where(out["mean_difference_hh_minus_control"] < 0, "HH_depleted", "no_difference"),
    )
    return out


def analyze_context(
    frame: pd.DataFrame,
    context_type: str,
    contrast_id: str,
    contrast_label: str,
    control_label: str,
    matched: bool,
) -> tuple[dict[str, object], pd.DataFrame]:
    hh = frame.loc[frame["LISA_category"].astype(str).eq("High-High")].reset_index(drop=True)
    if control_label == "LL":
        control = frame.loc[frame["LISA_category"].astype(str).eq("Low-Low")].reset_index(drop=True)
    else:
        control = frame.loc[~frame["LISA_category"].astype(str).eq("High-High")].reset_index(drop=True)
    if hh.empty or control.empty:
        sample_label = frame["sample_label"].iloc[0] if "sample_label" in frame else "unknown"
        raise ValueError(f"{sample_label}: unavailable HH/{control_label} pools for {context_type}")

    if matched:
        matched_hh, matched_control, diagnostics = optimal_malignant_matching(hh, control)
        summary = summarize_context(
            hh_all=hh,
            control_all=control,
            context_type=context_type,
            contrast_id=contrast_id,
            contrast_label=contrast_label,
            control_label=control_label,
            matched_hh=matched_hh,
            matched_control=matched_control,
            diagnostics=diagnostics,
        )
        contrasts = paired_mp_contrasts(
            matched_hh,
            matched_control,
            context_type=context_type,
            contrast_id=contrast_id,
            contrast_label=contrast_label,
            control_label=control_label,
        )
    else:
        summary = summarize_context(
            hh_all=hh,
            control_all=control,
            context_type=context_type,
            contrast_id=contrast_id,
            contrast_label=contrast_label,
            control_label=control_label,
        )
        contrasts = unpaired_mp_contrasts(
            hh,
            control,
            context_type=context_type,
            contrast_id=contrast_id,
            contrast_label=contrast_label,
            control_label=control_label,
        )
    return summary, contrasts


def analyze_sample(sample_path: Path) -> tuple[list[dict[str, object]], list[pd.DataFrame], list[dict[str, str]]]:
    sample_label = sample_label_from_path(sample_path)
    frame = load_sample_frame(sample_path)
    neighbor = compute_neighbor_vectors(frame)
    contexts = [("spot_level", frame), ("neighborhood", neighbor)]
    summaries: list[dict[str, object]] = []
    contrasts: list[pd.DataFrame] = []
    skipped: list[dict[str, str]] = []

    contrast_specs = [
        ("hh_vs_ll_unmatched", "HH versus LL unmatched", "LL", False),
        ("hh_vs_matched_ll", "HH versus matched LL", "LL", True),
        ("hh_vs_matched_nonhh", "HH versus matched non-HH", "non-HH", True),
    ]
    for contrast_id, contrast_label, control_label, matched in contrast_specs:
        for context_type, context_frame in contexts:
            try:
                summary, table = analyze_context(
                    context_frame,
                    context_type=context_type,
                    contrast_id=contrast_id,
                    contrast_label=contrast_label,
                    control_label=control_label,
                    matched=matched,
                )
            except ValueError as exc:
                skipped.append(
                    {
                        "sample_label": sample_label,
                        "contrast_id": contrast_id,
                        "context_type": context_type,
                        "reason": str(exc),
                    }
                )
                continue
            summaries.append(summary)
            contrasts.append(table)
    return summaries, contrasts, skipped


def build_contrasts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_tables = sorted(D3B_ROOT.glob("*/tables/spot_level_table.csv"))
    if not sample_tables:
        raise FileNotFoundError(f"No spot_level_table.csv files under {D3B_ROOT}")

    summary_rows: list[dict[str, object]] = []
    contrast_tables: list[pd.DataFrame] = []
    skipped_rows: list[dict[str, str]] = []
    for sample_path in sample_tables:
        print(f"Analysing {sample_label_from_path(sample_path)}", flush=True)
        summaries, tables, skipped = analyze_sample(sample_path)
        summary_rows.extend(summaries)
        contrast_tables.extend(tables)
        skipped_rows.extend(skipped)

    if not contrast_tables:
        raise RuntimeError("No MP EnrichMap contrast tables were produced")
    summary = pd.DataFrame(summary_rows).sort_values(
        ["contrast_id", "context_type", "sample_label"]
    ).reset_index(drop=True)
    contrasts = pd.concat(contrast_tables, ignore_index=True).sort_values(
        ["contrast_id", "context_type", "sample_label", "mp_id"]
    ).reset_index(drop=True)
    skipped = pd.DataFrame(skipped_rows)
    return summary, contrasts, skipped


def significant_table(contrasts: pd.DataFrame) -> pd.DataFrame:
    sig = contrasts.copy()
    sig["fdr_bh"] = pd.to_numeric(sig["fdr_bh"], errors="coerce")
    sig = sig.loc[sig["fdr_bh"] < 0.05].copy()
    sig["enriched_state"] = np.where(sig["direction"].eq("HH_enriched"), "HH", sig["control_label"])
    sig["signed_count"] = np.where(sig["enriched_state"].eq("HH"), 1, -1)
    return sig.sort_values(
        ["contrast_id", "context_type", "mp_id", "enriched_state", "sample_label"]
    ).reset_index(drop=True)


def make_counts(sig: pd.DataFrame) -> pd.DataFrame:
    if sig.empty:
        return pd.DataFrame(
            columns=[
                "contrast_id",
                "contrast_label",
                "control_label",
                "context_type",
                "context_label",
                "mp_id",
                "mp_label",
                "enriched_state",
                "n_significant_sample_contrasts",
            ]
        )
    return (
        sig.groupby(
            [
                "contrast_id",
                "contrast_label",
                "control_label",
                "context_type",
                "context_label",
                "mp_id",
                "mp_label",
                "enriched_state",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="n_significant_sample_contrasts")
    )


def count_for(
    counts: pd.DataFrame,
    *,
    contrast_id: str,
    context_type: str,
    mp_id: str,
    state: str,
) -> int:
    if counts.empty:
        return 0
    value = counts.loc[
        counts["contrast_id"].eq(contrast_id)
        & counts["context_type"].eq(context_type)
        & counts["mp_id"].eq(mp_id)
        & counts["enriched_state"].eq(state),
        "n_significant_sample_contrasts",
    ].sum()
    return int(value)


def symmetric_limit(counts: pd.DataFrame) -> int:
    if counts.empty:
        return 5
    max_count = int(counts["n_significant_sample_contrasts"].max())
    return max(5, int(np.ceil((max_count + 1) / 5) * 5))


def tick_values(limit: int) -> np.ndarray:
    if limit % 5 == 0:
        return np.arange(-limit, limit + 1, 5)
    step = max(1, int(limit / 3))
    return np.arange(-limit, limit + 1, step)


def style_axis(ax: plt.Axes) -> None:
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="both", labelsize=8, colors=INK, length=0)


def draw_panel(
    ax: plt.Axes,
    counts: pd.DataFrame,
    *,
    contrast_id: str,
    control_label: str,
    context_type: str,
    x_limit: int,
    show_mp_labels: bool,
) -> None:
    y = np.arange(len(MP_ORDER))
    hh_counts = [
        count_for(counts, contrast_id=contrast_id, context_type=context_type, mp_id=mp_id, state="HH")
        for mp_id in MP_ORDER
    ]
    control_counts = [
        count_for(
            counts,
            contrast_id=contrast_id,
            context_type=context_type,
            mp_id=mp_id,
            state=control_label,
        )
        for mp_id in MP_ORDER
    ]
    labels = [MP_LABELS[mp_id] for mp_id in MP_ORDER]
    ax.barh(y, [-count for count in control_counts], color=CONTROL_COLOR, height=0.72)
    ax.barh(y, hh_counts, color=HH_COLOR, height=0.72)
    ax.set_xlim(-x_limit, x_limit)
    ticks = tick_values(x_limit)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(abs(int(tick))) for tick in ticks])
    ax.set_yticks(y)
    ax.set_yticklabels(labels if show_mp_labels else [""] * len(labels), fontsize=8, color=INK)
    ax.tick_params(axis="y", labelleft=show_mp_labels)
    ax.invert_yaxis()
    style_axis(ax)


def plot_contrast(counts: pd.DataFrame, contrast_id: str, control_label: str, contrast_label: str, path: Path) -> None:
    x_limit = PLOT_X_LIMIT
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), sharey=True)
    for ax, (context_type, context_label) in zip(axes, CONTEXT_ORDER, strict=True):
        draw_panel(
            ax,
            counts,
            contrast_id=contrast_id,
            control_label=control_label,
            context_type=context_type,
            x_limit=x_limit,
            show_mp_labels=ax is axes[0],
        )
        ax.set_title(context_label, fontsize=12, color=INK)
        ax.set_xlabel("Significant sample-level MP contrasts", fontsize=9, color=INK)
    handles = [
        Patch(facecolor=CONTROL_COLOR, label=f"{control_label} enriched"),
        Patch(facecolor=HH_COLOR, label="HH enriched"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(contrast_label, fontsize=13, y=0.98)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_composite(counts: pd.DataFrame, out_png: Path, out_svg: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    x_limit = PLOT_X_LIMIT
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16.8, 6.2), sharex=True, sharey=False)
    for col_idx, (contrast_id, contrast_label, control_label) in enumerate(CONTRAST_ORDER):
        axes[0, col_idx].set_title(contrast_label, fontsize=11, fontweight="bold", color=INK, pad=10)
        for row_idx, (context_type, _) in enumerate(CONTEXT_ORDER):
            draw_panel(
                axes[row_idx, col_idx],
                counts,
                contrast_id=contrast_id,
                control_label=control_label,
                context_type=context_type,
                x_limit=x_limit,
                show_mp_labels=col_idx == 0,
            )
            if row_idx == 1:
                axes[row_idx, col_idx].set_xlabel("Significant sample-level MP contrasts", fontsize=9, color=INK)

    fig.text(0.974, 0.70, CONTEXT_ORDER[0][1], rotation=270, ha="center", va="center", fontsize=11, color=INK)
    fig.text(0.974, 0.30, CONTEXT_ORDER[1][1], rotation=270, ha="center", va="center", fontsize=11, color=INK)
    handles = [
        Patch(facecolor=CONTROL_COLOR, label="Comparison enriched"),
        Patch(facecolor=HH_COLOR, label="HH enriched"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0.055, 0.055, 0.955, 0.965), w_pad=1.8, h_pad=1.6)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_svg)
    plt.close(fig)


def write_manifest(
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    sig: pd.DataFrame,
    counts: pd.DataFrame,
    skipped: pd.DataFrame,
) -> None:
    scoring_manifest = json.loads(MP_SCORE_MANIFEST.read_text(encoding="utf-8")) if MP_SCORE_MANIFEST.exists() else {}
    manifest = {
        "analysis": "HH state contrasts using EnrichMap-scored MP1-MP8 metaprograms",
        "purpose": (
            "Mirror the raw K* program-family significant-count composite, but replace "
            "sample-specific K* usage columns with harmonized MP1-MP8 EnrichMap score columns."
        ),
        "source_hh_spot_tables": str(D3B_ROOT / "<sample>" / "tables" / "spot_level_table.csv"),
        "source_mp_h5ad_root": str(MP_H5AD_ROOT),
        "source_mp_scoring_manifest": str(MP_SCORE_MANIFEST),
        "mp_scoring_manifest": scoring_manifest,
        "output_root": str(OUT_ROOT),
        "mp_scores": [
            {"mp_id": mp_id, "mp_label": label, "score_column": column}
            for mp_id, label, column in MP_SCORES
        ],
        "contexts": {
            "spot_level": "HH and comparison center tumor spots from Def3b spot_level_table.csv",
            "neighborhood": (
                "Mean MP score across immediate hex-grid neighbors present in the same tumor spot table; "
                "center spot excluded."
            ),
        },
        "contrasts": [
            {
                "contrast_id": "hh_vs_ll_unmatched",
                "control_label": "LL",
                "test": "Mann-Whitney U per sample/context across unmatched HH and LL spots",
            },
            {
                "contrast_id": "hh_vs_matched_ll",
                "control_label": "LL",
                "test": "Paired Wilcoxon signed-rank after 1:1 malignant-fraction matching",
            },
            {
                "contrast_id": "hh_vs_matched_nonhh",
                "control_label": "non-HH",
                "test": "Paired Wilcoxon signed-rank after 1:1 malignant-fraction matching",
            },
        ],
        "fdr_scope": "Benjamini-Hochberg over the eight MP tests within each sample, context, and contrast.",
        "plot_count_unit": "one significant sample-level MP contrast with fdr_bh < 0.05",
        "rows": {
            "context_summary": int(len(summary)),
            "all_contrasts": int(len(contrasts)),
            "significant_contrasts": int(len(sig)),
            "direction_counts": int(len(counts)),
            "skipped": int(len(skipped)),
        },
        "samples": {
            "all_contrast_samples": sorted(contrasts["sample_label"].unique().tolist()),
            "n_all_contrast_samples": int(contrasts["sample_label"].nunique()),
            "skipped_samples": sorted(skipped["sample_label"].unique().tolist()) if not skipped.empty else [],
        },
    }
    (OUT_ROOT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    shutil.copy2(Path(__file__), OUT_ROOT / Path(__file__).name)
    summary, contrasts, skipped = build_contrasts()
    sig = significant_table(contrasts)
    counts = make_counts(sig)

    summary.to_csv(OUT_TABLES / "mp_enrichmap_state_context_summary.csv", index=False)
    contrasts.to_csv(OUT_TABLES / "mp_enrichmap_state_contrasts_long.csv", index=False)
    sig.to_csv(OUT_TABLES / "mp_enrichmap_state_significant_mps.csv", index=False)
    counts.to_csv(OUT_TABLES / "mp_enrichmap_state_direction_counts.csv", index=False)
    skipped.to_csv(OUT_TABLES / "mp_enrichmap_state_skipped_contexts.csv", index=False)

    for contrast_id, contrast_label, control_label in CONTRAST_ORDER:
        plot_contrast(
            counts,
            contrast_id=contrast_id,
            control_label=control_label,
            contrast_label=contrast_label,
            path=OUT_FIGURES / f"{contrast_id}_mp_enrichmap_significant_counts.png",
        )
    plot_composite(
        counts,
        OUT_FIGURES / "mp_enrichmap_state_significant_counts_composite.png",
        OUT_FIGURES / "mp_enrichmap_state_significant_counts_composite.svg",
    )
    write_manifest(summary, contrasts, sig, counts, skipped)
    print(f"Wrote {OUT_ROOT}")
    print(f"All contrast rows: {len(contrasts)}")
    print(f"Significant rows: {len(sig)}")
    print(f"Skipped contexts: {len(skipped)}")


if __name__ == "__main__":
    main()
