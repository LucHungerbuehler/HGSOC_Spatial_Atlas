"""
Per-sample HH hotspot programme characterization using malignant-matched non-HH controls.

This script answers two related questions within each sample:

1. What programme usage characterizes HH spots themselves?
2. What programme composition surrounds HH hotspots?

For each sample, it performs 1:1 optimal matching of HH tumor spots to non-HH
tumor spots using malignant fraction as the matching variable. It then computes
paired programme contrasts on:

- Spot-level raw programme usages.
- Immediate-neighborhood mean programme vectors.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.optimize import linear_sum_assignment
from scipy.stats import mannwhitneyu, wilcoxon


HEX_OFFSETS = [(2, 0), (-2, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
PROGRAM_PATTERN = re.compile(r"__K\d+__P\d+$")
SHORT_PROGRAM_PATTERN = re.compile(r"(K\d+__P\d+)$")
CONTEXT_ORDER = ["spot_level", "neighborhood"]
CONTEXT_LABELS = {
    "spot_level": "Spot-level",
    "neighborhood": "Neighborhood",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Characterize HH spots and HH neighborhoods using malignant-matched non-HH controls."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Definition 3b / Definition 4 run directory on D: drive.",
    )
    parser.add_argument(
        "--output-subdir",
        default="08_hh_programme_characterization",
        help="Name of the output folder to create under the run directory.",
    )
    parser.add_argument(
        "--comparison-mode",
        default="malignant_matched_nonhh",
        choices=["malignant_matched_nonhh", "ll_unmatched", "ll_malignant_matched"],
        help=(
            "Comparison branch to run. The default reproduces the original malignant-matched "
            "HH-vs-non-HH analysis."
        ),
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def programme_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if PROGRAM_PATTERN.search(str(column))]


def pooled_sd(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 and len(b) < 2:
        return math.nan
    var_a = float(np.var(a, ddof=1)) if len(a) >= 2 else 0.0
    var_b = float(np.var(b, ddof=1)) if len(b) >= 2 else 0.0
    denom = math.sqrt((var_a + var_b) / 2.0)
    return denom


def standardized_mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    denom = pooled_sd(a, b)
    if not np.isfinite(denom) or denom == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / denom)


def build_hex_neighbor_lists(frame: pd.DataFrame) -> list[np.ndarray]:
    coords = frame[["array_row", "array_col"]].copy()
    coords["array_row"] = pd.to_numeric(coords["array_row"], errors="raise").astype(int)
    coords["array_col"] = pd.to_numeric(coords["array_col"], errors="raise").astype(int)
    index_lookup = {
        (int(row.array_row), int(row.array_col)): idx
        for idx, row in coords.reset_index(drop=True).iterrows()
    }
    neighbor_lists: list[np.ndarray] = []
    for _, row in coords.reset_index(drop=True).iterrows():
        key = (int(row.array_row), int(row.array_col))
        neighbors: list[int] = []
        for dr, dc in HEX_OFFSETS:
            neighbor_idx = index_lookup.get((key[0] + dr, key[1] + dc))
            if neighbor_idx is not None:
                neighbors.append(int(neighbor_idx))
        neighbor_lists.append(np.asarray(sorted(set(neighbors)), dtype=int))
    return neighbor_lists


def compute_neighbor_vectors(frame: pd.DataFrame, program_cols: list[str]) -> pd.DataFrame:
    neighbor_lists = build_hex_neighbor_lists(frame)
    usage = frame[program_cols].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    keep_cols = [
        "spot_id",
        "dataset",
        "sample_id_on_disk",
        "sample_label",
        "LISA_category",
        "Malignant",
        "SNAI1-ac_score",
        "array_row",
        "array_col",
    ]
    for idx, neighbors in enumerate(neighbor_lists):
        if len(neighbors) == 0:
            continue
        mean_vector = usage[neighbors].mean(axis=0)
        row = {column: frame.iloc[idx][column] for column in keep_cols}
        row["neighbor_count"] = int(len(neighbors))
        for program_id, value in zip(program_cols, mean_vector, strict=False):
            row[program_id] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def make_valid_frame(frame: pd.DataFrame, program_cols: list[str]) -> pd.DataFrame:
    needed_cols = ["spot_id", "dataset", "sample_id_on_disk", "sample_label", "LISA_category", "Malignant"]
    extra_cols = [column for column in ["SNAI1-ac_score", "array_row", "array_col"] if column in frame.columns]
    subset = frame[needed_cols + extra_cols + program_cols].copy()
    subset["Malignant"] = pd.to_numeric(subset["Malignant"], errors="coerce")
    if "SNAI1-ac_score" in subset.columns:
        subset["SNAI1-ac_score"] = pd.to_numeric(subset["SNAI1-ac_score"], errors="coerce")
    for column in program_cols:
        subset[column] = pd.to_numeric(subset[column], errors="coerce")
    if "array_row" in subset.columns:
        subset["array_row"] = pd.to_numeric(subset["array_row"], errors="coerce")
    if "array_col" in subset.columns:
        subset["array_col"] = pd.to_numeric(subset["array_col"], errors="coerce")
    valid_mask = np.isfinite(subset["Malignant"].to_numpy()) & np.isfinite(subset[program_cols].to_numpy()).all(axis=1)
    if "array_row" in subset.columns and "array_col" in subset.columns:
        valid_mask &= np.isfinite(subset["array_row"].to_numpy()) & np.isfinite(subset["array_col"].to_numpy())
    subset = subset.loc[valid_mask].reset_index(drop=True)
    return subset


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
        matched_hh["Malignant"].to_numpy(dtype=float) - matched_control["Malignant"].to_numpy(dtype=float)
    )
    diagnostics = {
        "matched_pairs_n": int(len(matched_hh)),
        "mean_abs_malignant_diff": float(np.mean(pair_diff)) if len(pair_diff) else math.nan,
        "median_abs_malignant_diff": float(np.median(pair_diff)) if len(pair_diff) else math.nan,
        "max_abs_malignant_diff": float(np.max(pair_diff)) if len(pair_diff) else math.nan,
    }
    return matched_hh, matched_control, diagnostics


def paired_programme_contrasts(
    hh_frame: pd.DataFrame,
    control_frame: pd.DataFrame,
    program_cols: list[str],
    context_type: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sample_info = hh_frame.iloc[0]
    for program_id in program_cols:
        hh_vals = hh_frame[program_id].to_numpy(dtype=float)
        control_vals = control_frame[program_id].to_numpy(dtype=float)
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
                "context_type": context_type,
                "program_id": program_id,
                "n_pairs": int(len(diff)),
                "hh_mean": float(np.mean(hh_vals)),
                "matched_nonhh_mean": float(np.mean(control_vals)),
                "mean_difference_hh_minus_nonhh": float(np.mean(diff)),
                "median_difference_hh_minus_nonhh": float(np.median(diff)),
                "cohens_dz_hh_minus_nonhh": cohens_dz,
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            }
        )
    out = pd.DataFrame(rows)
    out["fdr_bh"] = bh_fdr(out["p_value"].to_numpy(dtype=float))
    out["direction"] = np.where(
        out["mean_difference_hh_minus_nonhh"] > 0,
        "HH_enriched",
        np.where(out["mean_difference_hh_minus_nonhh"] < 0, "HH_depleted", "no_difference"),
    )
    return out


def unpaired_programme_contrasts(
    hh_frame: pd.DataFrame,
    control_frame: pd.DataFrame,
    program_cols: list[str],
    context_type: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sample_info = hh_frame.iloc[0]
    for program_id in program_cols:
        hh_vals = hh_frame[program_id].to_numpy(dtype=float)
        control_vals = control_frame[program_id].to_numpy(dtype=float)
        p_value = math.nan
        statistic = math.nan
        if len(hh_vals) >= 3 and len(control_vals) >= 3:
            result = mannwhitneyu(hh_vals, control_vals, alternative="two-sided", method="auto")
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
        smd = standardized_mean_difference(hh_vals, control_vals)
        rows.append(
            {
                "dataset": str(sample_info["dataset"]),
                "sample_id_on_disk": str(sample_info["sample_id_on_disk"]),
                "sample_label": str(sample_info["sample_label"]),
                "context_type": context_type,
                "program_id": program_id,
                "hh_n": int(len(hh_vals)),
                "ll_n": int(len(control_vals)),
                "hh_mean": float(np.mean(hh_vals)),
                "ll_mean": float(np.mean(control_vals)),
                "mean_difference_hh_minus_ll": float(np.mean(hh_vals) - np.mean(control_vals)),
                "median_difference_hh_minus_ll": float(np.median(hh_vals) - np.median(control_vals)),
                "cohens_d_hh_minus_ll": smd,
                "mannwhitneyu_statistic": statistic,
                "p_value": p_value,
            }
        )
    out = pd.DataFrame(rows)
    out["fdr_bh"] = bh_fdr(out["p_value"].to_numpy(dtype=float))
    out["direction"] = np.where(
        out["mean_difference_hh_minus_ll"] > 0,
        "HH_enriched",
        np.where(out["mean_difference_hh_minus_ll"] < 0, "HH_depleted", "no_difference"),
    )
    return out


def summarize_context(
    hh_all: pd.DataFrame,
    control_all: pd.DataFrame,
    matched_hh: pd.DataFrame,
    matched_control: pd.DataFrame,
    diagnostics: dict[str, float],
    context_type: str,
) -> dict[str, object]:
    sample_info = hh_all.iloc[0]
    pre_smd = standardized_mean_difference(
        hh_all["Malignant"].to_numpy(dtype=float),
        control_all["Malignant"].to_numpy(dtype=float),
    )
    post_smd = standardized_mean_difference(
        matched_hh["Malignant"].to_numpy(dtype=float),
        matched_control["Malignant"].to_numpy(dtype=float),
    )
    return {
        "dataset": str(sample_info["dataset"]),
        "sample_id_on_disk": str(sample_info["sample_id_on_disk"]),
        "sample_label": str(sample_info["sample_label"]),
        "context_type": context_type,
        "hh_pool_n": int(len(hh_all)),
        "nonhh_pool_n": int(len(control_all)),
        "matched_pairs_n": int(diagnostics["matched_pairs_n"]),
        "hh_malignant_mean_pre": float(np.mean(hh_all["Malignant"])),
        "nonhh_malignant_mean_pre": float(np.mean(control_all["Malignant"])),
        "hh_malignant_mean_post": float(np.mean(matched_hh["Malignant"])),
        "matched_nonhh_malignant_mean_post": float(np.mean(matched_control["Malignant"])),
        "malignant_smd_pre": pre_smd,
        "malignant_smd_post": post_smd,
        "mean_abs_malignant_diff": diagnostics["mean_abs_malignant_diff"],
        "median_abs_malignant_diff": diagnostics["median_abs_malignant_diff"],
        "max_abs_malignant_diff": diagnostics["max_abs_malignant_diff"],
    }


def summarize_unmatched_ll_context(
    hh_all: pd.DataFrame,
    ll_all: pd.DataFrame,
    context_type: str,
) -> dict[str, object]:
    sample_info = hh_all.iloc[0]
    return {
        "dataset": str(sample_info["dataset"]),
        "sample_id_on_disk": str(sample_info["sample_id_on_disk"]),
        "sample_label": str(sample_info["sample_label"]),
        "context_type": context_type,
        "hh_pool_n": int(len(hh_all)),
        "ll_pool_n": int(len(ll_all)),
        "hh_malignant_mean": float(np.mean(hh_all["Malignant"])),
        "ll_malignant_mean": float(np.mean(ll_all["Malignant"])),
        "malignant_smd_hh_vs_ll": standardized_mean_difference(
            hh_all["Malignant"].to_numpy(dtype=float),
            ll_all["Malignant"].to_numpy(dtype=float),
        ),
    }


def summarize_matched_ll_context(
    hh_all: pd.DataFrame,
    ll_all: pd.DataFrame,
    matched_hh: pd.DataFrame,
    matched_ll: pd.DataFrame,
    diagnostics: dict[str, float],
    context_type: str,
) -> dict[str, object]:
    sample_info = hh_all.iloc[0]
    pre_smd = standardized_mean_difference(
        hh_all["Malignant"].to_numpy(dtype=float),
        ll_all["Malignant"].to_numpy(dtype=float),
    )
    post_smd = standardized_mean_difference(
        matched_hh["Malignant"].to_numpy(dtype=float),
        matched_ll["Malignant"].to_numpy(dtype=float),
    )
    return {
        "dataset": str(sample_info["dataset"]),
        "sample_id_on_disk": str(sample_info["sample_id_on_disk"]),
        "sample_label": str(sample_info["sample_label"]),
        "context_type": context_type,
        "hh_pool_n": int(len(hh_all)),
        "ll_pool_n": int(len(ll_all)),
        "matched_pairs_n": int(diagnostics["matched_pairs_n"]),
        "hh_malignant_mean_pre": float(np.mean(hh_all["Malignant"])),
        "ll_malignant_mean_pre": float(np.mean(ll_all["Malignant"])),
        "hh_malignant_mean_post": float(np.mean(matched_hh["Malignant"])),
        "matched_ll_malignant_mean_post": float(np.mean(matched_ll["Malignant"])),
        "malignant_smd_pre": pre_smd,
        "malignant_smd_post": post_smd,
        "mean_abs_malignant_diff": diagnostics["mean_abs_malignant_diff"],
        "median_abs_malignant_diff": diagnostics["median_abs_malignant_diff"],
        "max_abs_malignant_diff": diagnostics["max_abs_malignant_diff"],
    }


def top_hit_summary(contrast_df: pd.DataFrame, effect_col: str = "mean_difference_hh_minus_nonhh") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (sample_label, context_type), subset in contrast_df.groupby(["sample_label", "context_type"], sort=True):
        pos = subset.sort_values([effect_col, "fdr_bh"], ascending=[False, True]).iloc[0]
        neg = subset.sort_values([effect_col, "fdr_bh"], ascending=[True, True]).iloc[0]
        rows.append(
            {
                "dataset": pos["dataset"],
                "sample_id_on_disk": pos["sample_id_on_disk"],
                "sample_label": sample_label,
                "context_type": context_type,
                "top_positive_program_id": pos["program_id"],
                "top_positive_mean_difference": pos[effect_col],
                "top_positive_fdr_bh": pos["fdr_bh"],
                "top_negative_program_id": neg["program_id"],
                "top_negative_mean_difference": neg[effect_col],
                "top_negative_fdr_bh": neg["fdr_bh"],
                "significant_program_n_fdr_0_05": int((subset["fdr_bh"] < 0.05).sum()),
            }
        )
    return pd.DataFrame(rows)


def short_program_id(program_id: str) -> str:
    match = SHORT_PROGRAM_PATTERN.search(str(program_id))
    if match:
        return match.group(1)
    return str(program_id)


def build_annotation(effect: float, fdr_bh: float) -> str:
    if not np.isfinite(effect):
        return ""
    marker = "*" if np.isfinite(fdr_bh) and fdr_bh < 0.05 else ""
    return f"{effect:+.2f}{marker}"


def significance_stars(fdr_bh: float) -> str:
    if not np.isfinite(fdr_bh):
        return "n/a"
    if fdr_bh < 1e-4:
        return "****"
    if fdr_bh < 1e-3:
        return "***"
    if fdr_bh < 1e-2:
        return "**"
    if fdr_bh < 0.05:
        return "*"
    return "ns"


def style_violin(parts: dict[str, object], color: str) -> None:
    for body in parts.get("bodies", []):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.45)
        body.set_linewidth(0.8)


def style_boxplot(boxplot: dict[str, object], facecolor: str, edgecolor: str) -> None:
    for patch in boxplot.get("boxes", []):
        patch.set_facecolor(facecolor)
        patch.set_edgecolor(edgecolor)
        patch.set_linewidth(0.9)
        patch.set_alpha(0.9)
    for key in ["whiskers", "caps", "medians"]:
        for artist in boxplot.get(key, []):
            artist.set_color(edgecolor)
            artist.set_linewidth(0.9)


def plot_context_distribution(
    matched_hh: pd.DataFrame,
    matched_control: pd.DataFrame,
    contrast_df: pd.DataFrame,
    program_cols: list[str],
    context_type: str,
    output_path: Path,
    *,
    effect_col: str = "mean_difference_hh_minus_nonhh",
    title_suffix: str = "HH vs malignant-matched non-HH",
    control_label: str = "Matched non-HH",
) -> None:
    order_map = (
        contrast_df.sort_values([effect_col, "fdr_bh"], ascending=[False, True])["program_id"]
        .tolist()
    )
    program_order = [program_id for program_id in order_map if program_id in program_cols]
    hh_data = [matched_hh[program_id].to_numpy(dtype=float) for program_id in program_order]
    control_data = [matched_control[program_id].to_numpy(dtype=float) for program_id in program_order]
    positions = np.arange(len(program_order), dtype=float)
    hh_positions = positions - 0.18
    control_positions = positions + 0.18

    all_values = np.concatenate(hh_data + control_data) if program_order else np.asarray([0.0])
    finite_values = all_values[np.isfinite(all_values)]
    data_min = float(np.min(finite_values)) if len(finite_values) else 0.0
    data_max = float(np.max(finite_values)) if len(finite_values) else 1.0
    data_range = max(data_max - data_min, 0.08)

    width = max(8.0, 0.95 * len(program_order) + 3.5)
    fig, ax = plt.subplots(figsize=(width, 4.6))

    hh_violin = ax.violinplot(hh_data, positions=hh_positions, widths=0.28, showmeans=False, showmedians=False, showextrema=False)
    control_violin = ax.violinplot(
        control_data,
        positions=control_positions,
        widths=0.28,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    style_violin(hh_violin, "#c44536")
    style_violin(control_violin, "#4a6fa5")

    hh_box = ax.boxplot(
        hh_data,
        positions=hh_positions,
        widths=0.10,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
    )
    control_box = ax.boxplot(
        control_data,
        positions=control_positions,
        widths=0.10,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
    )
    style_boxplot(hh_box, "#f4d7d1", "#8f2f24")
    style_boxplot(control_box, "#d6e1f2", "#33527c")

    contrast_lookup = contrast_df.set_index("program_id")
    annotation_tops: list[float] = []
    for idx, program_id in enumerate(program_order):
        pair_top = float(max(np.max(hh_data[idx]), np.max(control_data[idx])))
        bracket_y = pair_top + 0.05 * data_range
        ax.plot(
            [hh_positions[idx], hh_positions[idx], control_positions[idx], control_positions[idx]],
            [bracket_y - 0.015 * data_range, bracket_y, bracket_y, bracket_y - 0.015 * data_range],
            color="black",
            linewidth=0.8,
        )
        ax.text(
            positions[idx],
            bracket_y + 0.01 * data_range,
            significance_stars(float(contrast_lookup.loc[program_id, "fdr_bh"])),
            ha="center",
            va="bottom",
            fontsize=8,
        )
        annotation_tops.append(bracket_y + 0.07 * data_range)

    y_lower = min(0.0, data_min - 0.06 * data_range)
    y_upper = max(annotation_tops) if annotation_tops else data_max + 0.1 * data_range
    y_upper = max(y_upper, data_max + 0.12 * data_range)
    ax.set_ylim(y_lower, y_upper)

    ax.set_xticks(positions)
    ax.set_xticklabels([short_program_id(program_id) for program_id in program_order], rotation=45, ha="right")
    ax.set_ylabel("Programme usage score")
    ax.set_xlabel("Programme")
    ax.set_title(
        f"{matched_hh['sample_label'].iat[0]} | {CONTEXT_LABELS[context_type]}\n{title_suffix}",
        fontsize=11,
        pad=10,
    )
    ax.grid(axis="y", alpha=0.18, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=[
            Patch(facecolor="#c44536", edgecolor="#8f2f24", alpha=0.55, label="HH"),
            Patch(facecolor="#4a6fa5", edgecolor="#33527c", alpha=0.55, label=control_label),
        ],
        loc="upper right",
        frameon=False,
    )
    fig.text(0.99, 0.015, "Labels show FDR-adjusted significance: ns, *, **, ***, ****", ha="right", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_sample_heatmap(
    sample_df: pd.DataFrame,
    output_path: Path,
    global_abs_max: float,
    *,
    effect_col: str = "mean_difference_hh_minus_nonhh",
    n_col: str = "n_pairs",
    title_suffix: str = "HH minus malignant-matched non-HH programme usage",
) -> None:
    program_order = (
        sample_df.groupby("program_id")[effect_col]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    effect = (
        sample_df.pivot(index="context_type", columns="program_id", values=effect_col)
        .reindex(index=CONTEXT_ORDER, columns=program_order)
    )
    fdr = sample_df.pivot(index="context_type", columns="program_id", values="fdr_bh").reindex(
        index=CONTEXT_ORDER,
        columns=program_order,
    )
    pairs = sample_df.groupby("context_type")[n_col].first().to_dict()

    width = max(7.5, 0.9 * len(program_order) + 3.5)
    fig, ax = plt.subplots(figsize=(width, 3.2))
    im = ax.imshow(
        effect.to_numpy(dtype=float),
        cmap="RdBu_r",
        vmin=-global_abs_max,
        vmax=global_abs_max,
        aspect="auto",
        interpolation="nearest",
    )

    for row_idx in range(effect.shape[0]):
        for col_idx in range(effect.shape[1]):
            value = float(effect.iat[row_idx, col_idx])
            pval = float(fdr.iat[row_idx, col_idx])
            label = build_annotation(value, pval)
            text_color = "white" if np.isfinite(value) and abs(value) >= 0.55 * global_abs_max else "black"
            ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=8, color=text_color)

    ax.set_xticks(range(len(program_order)))
    ax.set_xticklabels([short_program_id(program_id) for program_id in program_order], rotation=45, ha="right")
    ax.set_yticks(range(len(CONTEXT_ORDER)))
    ax.set_yticklabels(
        [f"{CONTEXT_LABELS[context]} (n={int(pairs.get(context, 0))})" for context in CONTEXT_ORDER]
    )
    ax.set_title(
        f"{sample_df['sample_label'].iat[0]}\n{title_suffix}",
        fontsize=11,
        pad=10,
    )

    ax.set_xticks(np.arange(-0.5, len(program_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CONTEXT_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Mean difference in programme usage", rotation=90)
    fig.text(0.99, 0.02, "* FDR < 0.05", ha="right", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_heatmaps(
    spot_df: pd.DataFrame,
    neighbor_df: pd.DataFrame,
    figures_dir: Path,
    *,
    effect_col: str = "mean_difference_hh_minus_nonhh",
    n_col: str = "n_pairs",
    output_suffix: str = "hh_vs_matched_nonhh_programme_heatmap",
    title_suffix: str = "HH minus malignant-matched non-HH programme usage",
) -> None:
    combined = pd.concat([spot_df, neighbor_df], ignore_index=True)
    global_abs_max = float(np.nanmax(np.abs(combined[effect_col].to_numpy(dtype=float))))
    global_abs_max = max(global_abs_max, 0.05)
    for sample_label, sample_df in combined.groupby("sample_label", sort=True):
        plot_sample_heatmap(
            sample_df=sample_df,
            output_path=figures_dir / f"{sample_label}__{output_suffix}.png",
            global_abs_max=global_abs_max,
            effect_col=effect_col,
            n_col=n_col,
            title_suffix=title_suffix,
        )


def analyze_sample(
    sample_path: Path,
    violin_boxplot_dir: Path,
) -> tuple[list[dict[str, object]], pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(sample_path)
    program_cols = programme_columns(frame)
    if not program_cols:
        raise ValueError(f"No programme columns found in {sample_path}")

    spot_frame = make_valid_frame(frame, program_cols)
    hh_spot = spot_frame.loc[spot_frame["LISA_category"].astype(str) == "High-High"].reset_index(drop=True)
    nonhh_spot = spot_frame.loc[spot_frame["LISA_category"].astype(str) != "High-High"].reset_index(drop=True)
    if hh_spot.empty or nonhh_spot.empty:
        raise ValueError(f"HH/non-HH spot pools unavailable in {sample_path}")
    matched_hh_spot, matched_nonhh_spot, spot_diag = optimal_malignant_matching(hh_spot, nonhh_spot)
    spot_summary = summarize_context(
        hh_all=hh_spot,
        control_all=nonhh_spot,
        matched_hh=matched_hh_spot,
        matched_control=matched_nonhh_spot,
        diagnostics=spot_diag,
        context_type="spot_level",
    )
    spot_contrasts = paired_programme_contrasts(
        matched_hh_spot,
        matched_nonhh_spot,
        program_cols=program_cols,
        context_type="spot_level",
    )
    plot_context_distribution(
        matched_hh=matched_hh_spot,
        matched_control=matched_nonhh_spot,
        contrast_df=spot_contrasts,
        program_cols=program_cols,
        context_type="spot_level",
        output_path=violin_boxplot_dir / f"{spot_frame['sample_label'].iat[0]}__spot_level__hh_vs_matched_nonhh_violin_boxplot.png",
    )

    neighbor_frame = compute_neighbor_vectors(spot_frame, program_cols)
    hh_neighbor = neighbor_frame.loc[neighbor_frame["LISA_category"].astype(str) == "High-High"].reset_index(drop=True)
    nonhh_neighbor = neighbor_frame.loc[neighbor_frame["LISA_category"].astype(str) != "High-High"].reset_index(drop=True)
    if hh_neighbor.empty or nonhh_neighbor.empty:
        raise ValueError(f"HH/non-HH neighbor pools unavailable in {sample_path}")
    matched_hh_neighbor, matched_nonhh_neighbor, neighbor_diag = optimal_malignant_matching(hh_neighbor, nonhh_neighbor)
    neighbor_summary = summarize_context(
        hh_all=hh_neighbor,
        control_all=nonhh_neighbor,
        matched_hh=matched_hh_neighbor,
        matched_control=matched_nonhh_neighbor,
        diagnostics=neighbor_diag,
        context_type="neighborhood",
    )
    neighbor_contrasts = paired_programme_contrasts(
        matched_hh_neighbor,
        matched_nonhh_neighbor,
        program_cols=program_cols,
        context_type="neighborhood",
    )
    plot_context_distribution(
        matched_hh=matched_hh_neighbor,
        matched_control=matched_nonhh_neighbor,
        contrast_df=neighbor_contrasts,
        program_cols=program_cols,
        context_type="neighborhood",
        output_path=violin_boxplot_dir / f"{spot_frame['sample_label'].iat[0]}__neighborhood__hh_vs_matched_nonhh_violin_boxplot.png",
    )

    return [spot_summary, neighbor_summary], spot_contrasts, neighbor_contrasts


def analyze_sample_hh_vs_ll_unmatched(
    sample_path: Path,
    violin_boxplot_dir: Path,
) -> tuple[list[dict[str, object]], pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(sample_path)
    program_cols = programme_columns(frame)
    if not program_cols:
        raise ValueError(f"No programme columns found in {sample_path}")

    spot_frame = make_valid_frame(frame, program_cols)
    hh_spot = spot_frame.loc[spot_frame["LISA_category"].astype(str) == "High-High"].reset_index(drop=True)
    ll_spot = spot_frame.loc[spot_frame["LISA_category"].astype(str) == "Low-Low"].reset_index(drop=True)
    if hh_spot.empty or ll_spot.empty:
        raise ValueError(f"HH/LL spot pools unavailable in {sample_path}")
    spot_summary = summarize_unmatched_ll_context(
        hh_all=hh_spot,
        ll_all=ll_spot,
        context_type="spot_level",
    )
    spot_contrasts = unpaired_programme_contrasts(
        hh_spot,
        ll_spot,
        program_cols=program_cols,
        context_type="spot_level",
    )
    plot_context_distribution(
        matched_hh=hh_spot,
        matched_control=ll_spot,
        contrast_df=spot_contrasts,
        program_cols=program_cols,
        context_type="spot_level",
        output_path=violin_boxplot_dir / f"{spot_frame['sample_label'].iat[0]}__spot_level__hh_vs_ll_violin_boxplot.png",
        effect_col="mean_difference_hh_minus_ll",
        title_suffix="HH vs LL",
        control_label="LL",
    )

    neighbor_frame = compute_neighbor_vectors(spot_frame, program_cols)
    hh_neighbor = neighbor_frame.loc[neighbor_frame["LISA_category"].astype(str) == "High-High"].reset_index(drop=True)
    ll_neighbor = neighbor_frame.loc[neighbor_frame["LISA_category"].astype(str) == "Low-Low"].reset_index(drop=True)
    if hh_neighbor.empty or ll_neighbor.empty:
        raise ValueError(f"HH/LL neighbor pools unavailable in {sample_path}")
    neighbor_summary = summarize_unmatched_ll_context(
        hh_all=hh_neighbor,
        ll_all=ll_neighbor,
        context_type="neighborhood",
    )
    neighbor_contrasts = unpaired_programme_contrasts(
        hh_neighbor,
        ll_neighbor,
        program_cols=program_cols,
        context_type="neighborhood",
    )
    plot_context_distribution(
        matched_hh=hh_neighbor,
        matched_control=ll_neighbor,
        contrast_df=neighbor_contrasts,
        program_cols=program_cols,
        context_type="neighborhood",
        output_path=violin_boxplot_dir / f"{spot_frame['sample_label'].iat[0]}__neighborhood__hh_vs_ll_violin_boxplot.png",
        effect_col="mean_difference_hh_minus_ll",
        title_suffix="HH vs LL",
        control_label="LL",
    )

    return [spot_summary, neighbor_summary], spot_contrasts, neighbor_contrasts


def analyze_sample_hh_vs_ll_matched(
    sample_path: Path,
    violin_boxplot_dir: Path,
) -> tuple[list[dict[str, object]], pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(sample_path)
    program_cols = programme_columns(frame)
    if not program_cols:
        raise ValueError(f"No programme columns found in {sample_path}")

    spot_frame = make_valid_frame(frame, program_cols)
    hh_spot = spot_frame.loc[spot_frame["LISA_category"].astype(str) == "High-High"].reset_index(drop=True)
    ll_spot = spot_frame.loc[spot_frame["LISA_category"].astype(str) == "Low-Low"].reset_index(drop=True)
    if hh_spot.empty or ll_spot.empty:
        raise ValueError(f"HH/LL spot pools unavailable in {sample_path}")
    matched_hh_spot, matched_ll_spot, spot_diag = optimal_malignant_matching(hh_spot, ll_spot)
    spot_summary = summarize_matched_ll_context(
        hh_all=hh_spot,
        ll_all=ll_spot,
        matched_hh=matched_hh_spot,
        matched_ll=matched_ll_spot,
        diagnostics=spot_diag,
        context_type="spot_level",
    )
    spot_contrasts = paired_programme_contrasts(
        matched_hh_spot,
        matched_ll_spot,
        program_cols=program_cols,
        context_type="spot_level",
    )
    spot_contrasts = spot_contrasts.rename(
        columns={
            "matched_nonhh_mean": "matched_ll_mean",
            "mean_difference_hh_minus_nonhh": "mean_difference_hh_minus_ll",
            "median_difference_hh_minus_nonhh": "median_difference_hh_minus_ll",
            "cohens_dz_hh_minus_nonhh": "cohens_dz_hh_minus_ll",
        }
    )
    plot_context_distribution(
        matched_hh=matched_hh_spot,
        matched_control=matched_ll_spot,
        contrast_df=spot_contrasts,
        program_cols=program_cols,
        context_type="spot_level",
        output_path=violin_boxplot_dir / f"{spot_frame['sample_label'].iat[0]}__spot_level__hh_vs_matched_ll_violin_boxplot.png",
        effect_col="mean_difference_hh_minus_ll",
        title_suffix="HH vs malignant-matched LL",
        control_label="Matched LL",
    )

    neighbor_frame = compute_neighbor_vectors(spot_frame, program_cols)
    hh_neighbor = neighbor_frame.loc[neighbor_frame["LISA_category"].astype(str) == "High-High"].reset_index(drop=True)
    ll_neighbor = neighbor_frame.loc[neighbor_frame["LISA_category"].astype(str) == "Low-Low"].reset_index(drop=True)
    if hh_neighbor.empty or ll_neighbor.empty:
        raise ValueError(f"HH/LL neighbor pools unavailable in {sample_path}")
    matched_hh_neighbor, matched_ll_neighbor, neighbor_diag = optimal_malignant_matching(hh_neighbor, ll_neighbor)
    neighbor_summary = summarize_matched_ll_context(
        hh_all=hh_neighbor,
        ll_all=ll_neighbor,
        matched_hh=matched_hh_neighbor,
        matched_ll=matched_ll_neighbor,
        diagnostics=neighbor_diag,
        context_type="neighborhood",
    )
    neighbor_contrasts = paired_programme_contrasts(
        matched_hh_neighbor,
        matched_ll_neighbor,
        program_cols=program_cols,
        context_type="neighborhood",
    )
    neighbor_contrasts = neighbor_contrasts.rename(
        columns={
            "matched_nonhh_mean": "matched_ll_mean",
            "mean_difference_hh_minus_nonhh": "mean_difference_hh_minus_ll",
            "median_difference_hh_minus_nonhh": "median_difference_hh_minus_ll",
            "cohens_dz_hh_minus_nonhh": "cohens_dz_hh_minus_ll",
        }
    )
    plot_context_distribution(
        matched_hh=matched_hh_neighbor,
        matched_control=matched_ll_neighbor,
        contrast_df=neighbor_contrasts,
        program_cols=program_cols,
        context_type="neighborhood",
        output_path=violin_boxplot_dir / f"{spot_frame['sample_label'].iat[0]}__neighborhood__hh_vs_matched_ll_violin_boxplot.png",
        effect_col="mean_difference_hh_minus_ll",
        title_suffix="HH vs malignant-matched LL",
        control_label="Matched LL",
    )

    return [spot_summary, neighbor_summary], spot_contrasts, neighbor_contrasts


def main(run_dir: Path, output_subdir: str, comparison_mode: str) -> None:
    d3b_root = run_dir / "02_definition3b_mixture_programme_niches"
    sample_tables = sorted(d3b_root.glob("*\\tables\\spot_level_table.csv"))
    if not sample_tables:
        raise FileNotFoundError(f"No spot-level tables found under {d3b_root}")

    output_root = ensure_dir(run_dir / output_subdir)
    tables_dir = ensure_dir(output_root / "tables")
    figures_dir = ensure_dir(output_root / "figures")
    violin_boxplot_dir = ensure_dir(figures_dir / "violin_boxplot")

    summary_rows: list[dict[str, object]] = []
    spot_tables: list[pd.DataFrame] = []
    neighbor_tables: list[pd.DataFrame] = []
    skipped_rows: list[dict[str, str]] = []

    if comparison_mode == "malignant_matched_nonhh":
        analyzer = analyze_sample
        summary_name = "per_sample_matching_summary.csv"
        spot_name = "hh_spot_level_programme_contrasts.csv"
        neighbor_name = "hh_neighborhood_programme_contrasts.csv"
        top_effect_col = "mean_difference_hh_minus_nonhh"
        heatmap_kwargs = {}
    elif comparison_mode == "ll_unmatched":
        analyzer = analyze_sample_hh_vs_ll_unmatched
        summary_name = "per_sample_hh_ll_summary.csv"
        spot_name = "hh_ll_spot_level_programme_contrasts.csv"
        neighbor_name = "hh_ll_neighborhood_programme_contrasts.csv"
        top_effect_col = "mean_difference_hh_minus_ll"
        heatmap_kwargs = {
            "effect_col": "mean_difference_hh_minus_ll",
            "n_col": "hh_n",
            "output_suffix": "hh_vs_ll_programme_heatmap",
            "title_suffix": "HH minus LL programme usage",
        }
    elif comparison_mode == "ll_malignant_matched":
        analyzer = analyze_sample_hh_vs_ll_matched
        summary_name = "per_sample_hh_ll_matching_summary.csv"
        spot_name = "hh_ll_matched_spot_level_programme_contrasts.csv"
        neighbor_name = "hh_ll_matched_neighborhood_programme_contrasts.csv"
        top_effect_col = "mean_difference_hh_minus_ll"
        heatmap_kwargs = {
            "effect_col": "mean_difference_hh_minus_ll",
            "n_col": "n_pairs",
            "output_suffix": "hh_vs_matched_ll_programme_heatmap",
            "title_suffix": "HH minus malignant-matched LL programme usage",
        }
    else:
        raise ValueError(f"Unsupported comparison mode: {comparison_mode}")

    for sample_path in sample_tables:
        print(f"Analysing {sample_path.parent.parent.name} ...", flush=True)
        try:
            sample_summaries, spot_contrasts, neighbor_contrasts = analyzer(
                sample_path,
                violin_boxplot_dir=violin_boxplot_dir,
            )
        except ValueError as exc:
            print(f"Skipping {sample_path.parent.parent.name}: {exc}", flush=True)
            skipped_rows.append({"sample_label": sample_path.parent.parent.name, "reason": str(exc)})
            continue
        summary_rows.extend(sample_summaries)
        spot_tables.append(spot_contrasts)
        neighbor_tables.append(neighbor_contrasts)

    if not spot_tables or not neighbor_tables:
        raise RuntimeError(f"No samples produced contrasts for comparison mode {comparison_mode}")

    summary_df = pd.DataFrame(summary_rows).sort_values(["context_type", "sample_label"]).reset_index(drop=True)
    spot_df = pd.concat(spot_tables, ignore_index=True).sort_values(["sample_label", "program_id"]).reset_index(drop=True)
    neighbor_df = pd.concat(neighbor_tables, ignore_index=True).sort_values(["sample_label", "program_id"]).reset_index(drop=True)
    top_hits_df = pd.concat([spot_df, neighbor_df], ignore_index=True)
    top_hits_df = top_hit_summary(top_hits_df, effect_col=top_effect_col)

    summary_df.to_csv(tables_dir / summary_name, index=False)
    spot_df.to_csv(tables_dir / spot_name, index=False)
    neighbor_df.to_csv(tables_dir / neighbor_name, index=False)
    top_hits_df.to_csv(tables_dir / "per_sample_context_top_programmes.csv", index=False)
    if skipped_rows:
        pd.DataFrame(skipped_rows).to_csv(tables_dir / "skipped_samples.csv", index=False)
    write_heatmaps(spot_df=spot_df, neighbor_df=neighbor_df, figures_dir=figures_dir, **heatmap_kwargs)
    print(f"Wrote outputs to {output_root}", flush=True)


if __name__ == "__main__":
    args = parse_args()
    main(
        run_dir=Path(args.run_dir),
        output_subdir=str(args.output_subdir),
        comparison_mode=str(args.comparison_mode),
    )
