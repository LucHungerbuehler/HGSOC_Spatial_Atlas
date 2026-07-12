from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


OUT_ROOT = Path(
    r"D:\HGSOC_Spatial_Atlas\05_analysis_ready\SpottedPy_v2_paper_aligned"
    r"\02_neighborhood_enrichment\consensus_source_preflight"
)
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures_paperstyle_v2"

INNER_TABLE = TABLE_DIR / "consensus_source_central_snai1ac_to_ring1_pearson.csv"
VARIABLE_MANIFEST = TABLE_DIR / "consensus_source_variable_manifest.csv"
RANKED_TABLE = TABLE_DIR / "consensus_source_central_snai1ac_ranked12_for_plots_paperstyle_v2.csv"
SOURCE_COMPARISON_TABLE = TABLE_DIR / "consensus_source_sourcegroup_comparison_long_paperstyle_v2.csv"
KSTAR_PROJECTION_TABLE = Path(
    r"D:\HGSOC_Spatial_Atlas\05_analysis_ready\S3_cNMF_Tumor_Programs"
    r"\snai1ac_signature_projection_onto_cnmf_programs_v1\tables"
    r"\kstar_snai1ac_signature_projection_clean.csv"
)

SOURCE_ORDER = ["snai1ac_consensus_hot", "snai1ac_consensus_cold", "snai12r_hot"]
FAMILY_ORDER = ["core", "spacet", "mp_kstar", "hallmark"]
FAMILY_LABELS = {
    "core": "core",
    "spacet": "SpaCET",
    "mp_kstar": "MP/K*",
    "hallmark": "Hallmark",
}


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))


def compact_label(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


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


def load_inner() -> pd.DataFrame:
    df = pd.read_csv(INNER_TABLE)
    for col in ["corr", "pval", "qval", "n_pairs"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["is_significant"] = df["qval"] <= 0.05
    df["sig_marker"] = df["qval"].apply(significance_marker)
    return df


def load_variable_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(VARIABLE_MANIFEST)
    return manifest[["sample_label", "variable_id", "family"]].drop_duplicates()


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
            match = re.search(r"__P(\d+)$", program_id)
            prefix = f"P{match.group(1)}" if match else program_id.rsplit("__", 1)[-1]
            category = compact_label(getattr(row, "alignment_category_draft", ""))
            if not category or category.lower() == "nan":
                category = "unannotated"
            label_map[f"Kstar_{program_id}"] = f"{prefix}_{category}"
    return label_map


def display_label(variable: str, label_map: dict[str, str]) -> str:
    return label_map.get(variable, variable)


def variable_order_for_plot(
    sample_label: str,
    variable_class: str,
    available_variables: list[str],
    variable_manifest: pd.DataFrame,
) -> tuple[list[str], list[tuple[str, int, int]]]:
    available = list(dict.fromkeys(available_variables))
    sample_manifest = variable_manifest[variable_manifest["sample_label"] == sample_label].copy()
    order: list[str] = []
    groups: list[tuple[str, int, int]] = []
    families = FAMILY_ORDER if variable_class == "all_variables" else ["core", variable_class]

    for family in families:
        family_vars = [
            variable
            for variable in sample_manifest.loc[sample_manifest["family"] == family, "variable_id"].tolist()
            if variable in available
        ]
        if not family_vars:
            continue
        start = len(order)
        order.extend([variable for variable in family_vars if variable not in order])
        end = len(order) - 1
        if variable_class == "all_variables":
            groups.append((FAMILY_LABELS.get(family, family), start, end))

    leftovers = [variable for variable in available if variable not in order]
    order.extend(leftovers)
    if variable_class == "all_variables" and leftovers:
        groups.append(("Other", len(order) - len(leftovers), len(order) - 1))
    return order, groups


def select_ranked(df: pd.DataFrame, label_map: dict[str, str]) -> pd.DataFrame:
    frame = df[df["outer_variable"] != "SNAI1_ac"].dropna(subset=["corr"]).copy()
    if frame.empty:
        return frame
    pos = frame[frame["corr"] > 0].sort_values("corr", ascending=False).head(6)
    neg = frame[frame["corr"] < 0].sort_values("corr", ascending=True).head(6)
    out = pd.concat([neg, pos], ignore_index=True)
    out["selection_basis"] = "strongest_6_positive_and_6_negative_by_pearson_r"
    out["outer_variable_display"] = out["outer_variable"].map(lambda value: display_label(value, label_map))
    return out


def plot_ranked_bars(df: pd.DataFrame, out_path: Path, title: str) -> None:
    if df.empty:
        return
    frame = df.sort_values("corr")
    height = max(4, 0.28 * len(frame) + 1.8)
    fig, ax = plt.subplots(figsize=(8, height))
    colors = np.where(frame["corr"] >= 0, "#b2182b", "#2166ac")
    label_col = "outer_variable_display" if "outer_variable_display" in frame.columns else "outer_variable"
    labels = frame[label_col].tolist()
    bars = ax.barh(labels, frame["corr"], color=colors)
    ax.axvline(0, color="#111827", linewidth=0.8)
    ax.set_xlabel("Pearson r")
    ax.set_title(title, fontsize=10)
    for bar, marker, corr_value in zip(bars, frame["sig_marker"], frame["corr"]):
        if not marker:
            continue
        ax.text(
            corr_value / 2,
            bar.get_y() + bar.get_height() / 2,
            marker,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="#ffffff",
        )
    fig.subplots_adjust(bottom=0.22)
    fig.text(
        0.02,
        0.035,
        "Stars indicate BH q-value: **** <=1e-4, *** <=1e-3, ** <=1e-2, * <=0.05",
        fontsize=6,
        ha="left",
        va="bottom",
        color="#374151",
    )
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


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


def plot_source_group_comparison(df: pd.DataFrame, out_path: Path, title: str) -> None:
    if df.empty:
        return
    corr = df.pivot(index="outer_variable", columns="source_group", values="corr").reindex(columns=SOURCE_ORDER)
    qval = df.pivot(index="outer_variable", columns="source_group", values="qval").reindex(columns=SOURCE_ORDER)
    corr = corr.dropna(how="all")
    qval = qval.reindex(corr.index)
    if corr.empty:
        return
    order = clustered_row_order(corr)
    corr = corr.loc[order]
    qval = qval.loc[order]
    display = corr.where(qval <= 0.05)
    height = max(4.5, min(24, 0.26 * len(corr) + 2.2))
    fig, ax = plt.subplots(figsize=(7.2, height))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#ffffff")
    im = ax.imshow(display.to_numpy(dtype=float), vmin=-1, vmax=1, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(SOURCE_ORDER)))
    ax.set_xticklabels(SOURCE_ORDER, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=6)
    ax.set_title(title, fontsize=10)
    for row_idx, variable in enumerate(corr.index):
        for col_idx, source in enumerate(SOURCE_ORDER):
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
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_source_group_comparison_allcorr(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    variable_order: list[str],
    group_labels: list[tuple[str, int, int]],
    label_map: dict[str, str],
) -> None:
    if df.empty:
        return
    corr = df.pivot(index="source_group", columns="outer_variable", values="corr").reindex(index=SOURCE_ORDER)
    qval = df.pivot(index="source_group", columns="outer_variable", values="qval").reindex(index=SOURCE_ORDER)
    corr = corr.dropna(axis=1, how="all")
    qval = qval.reindex(columns=corr.columns)
    if corr.empty:
        return
    order = [variable for variable in variable_order if variable in corr.columns]
    order.extend([variable for variable in corr.columns if variable not in order])
    corr = corr[order]
    qval = qval[order]
    display_labels = [display_label(variable, label_map) for variable in corr.columns]
    max_label_len = max((len(label) for label in display_labels), default=0)
    width = max(9.5, min(42, 0.34 * len(corr.columns) + 3.8))
    height = 7.4 if len(corr.columns) > 35 or max_label_len > 32 else 5.6
    bottom_margin = 0.56 if height >= 7 else 0.48
    top_margin = 0.76 if group_labels else 0.84
    left_margin = 0.17 if width < 14 else 0.10
    right_margin = 0.88 if width < 14 else 0.94
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=left_margin, right=right_margin, bottom=bottom_margin, top=top_margin)
    im = ax.imshow(corr.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(
        display_labels,
        rotation=90,
        ha="center",
        fontsize=6,
    )
    ax.set_yticks(range(len(SOURCE_ORDER)))
    ax.set_yticklabels(SOURCE_ORDER, fontsize=8)
    fig.suptitle(title, fontsize=10, y=0.96)
    for label, start, end in group_labels:
        if start >= len(corr.columns):
            continue
        end = min(end, len(corr.columns) - 1)
        if start > 0:
            ax.axvline(start - 0.5, color="#111827", linewidth=0.8)
        midpoint = (start + end) / 2
        ax.text(
            midpoint,
            1.08,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            clip_on=False,
        )
    for row_idx, source in enumerate(SOURCE_ORDER):
        for col_idx, variable in enumerate(corr.columns):
            marker = significance_marker(qval.loc[source, variable])
            if marker:
                ax.text(col_idx, row_idx, marker, ha="center", va="center", fontsize=6, color="#111827")
    cbar = fig.colorbar(im, ax=ax, fraction=0.022 if len(corr.columns) > 35 else 0.04, pad=0.018)
    cbar.set_label("Pearson r\nStars: BH q", fontsize=7)
    fig.text(
        left_margin,
        0.025,
        "BH stars: **** <=1e-4, *** <=1e-3, ** <=1e-2, * <=0.05",
        fontsize=6,
        ha="left",
        va="bottom",
        color="#374151",
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_inner()
    variable_manifest = load_variable_manifest()
    label_map = load_display_label_map()
    df["outer_variable_display"] = df["outer_variable"].map(lambda value: display_label(value, label_map))

    ranked_rows = []
    ranked_count = 0
    for (sample_label, source_group, variable_class), group in df.groupby(
        ["sample_label", "source_group", "variable_class"]
    ):
        ranked = select_ranked(group, label_map)
        if ranked.empty:
            continue
        ranked_rows.append(ranked)
        ranked_count += 1
        plot_ranked_bars(
            ranked,
            FIG_DIR / f"{safe_name(sample_label)}__{source_group}__{variable_class}__SNAI1ac_ring1_ranked12.png",
            f"{sample_label}: {source_group}, {variable_class}, SNAI1-ac to ring 1",
        )

    ranked_df = pd.concat(ranked_rows, ignore_index=True) if ranked_rows else pd.DataFrame()
    ranked_df.to_csv(RANKED_TABLE, index=False)
    df.to_csv(SOURCE_COMPARISON_TABLE, index=False)

    comparison_allcorr_count = 0
    for (sample_label, variable_class), group in df.groupby(["sample_label", "variable_class"]):
        if group["source_group"].nunique() < 2:
            continue
        variable_order, group_labels = variable_order_for_plot(
            sample_label,
            variable_class,
            group["outer_variable"].drop_duplicates().tolist(),
            variable_manifest,
        )
        comparison_allcorr_count += 1
        plot_source_group_comparison_allcorr(
            group,
            FIG_DIR / f"{safe_name(sample_label)}__{variable_class}__source_group_comparison_all_pearson_with_sigstars.png",
            f"{sample_label}: source-group comparison, {variable_class}",
            variable_order,
            group_labels,
            label_map,
        )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__)),
        "input_table": str(INNER_TABLE),
        "ranked_barplots_regenerated": ranked_count,
        "source_group_comparison_allcorr_heatmaps_generated": comparison_allcorr_count,
        "ranked_selection": "strongest 6 positive and strongest 6 negative Pearson correlations, with BH q-value markers",
        "source_group_comparison_allcorr": "Source groups on y-axis, variables on x-axis, all Pearson r values colored, and BH q-value stars overlaid",
        "all_variable_order": "core/SpaCET/MP/K*/Hallmark, using consensus_source_variable_manifest.csv",
        "kstar_label_source": str(KSTAR_PROJECTION_TABLE),
        "kstar_label_format": "P#_<alignment_category_draft>",
    }
    (OUT_ROOT / "postprocess_ranked_and_source_comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    readme_path = OUT_ROOT / "README.md"
    with readme_path.open("a", encoding="utf-8") as handle:
        handle.write("\n## Postprocess update\n\n")
        handle.write(f"Updated at: {manifest['created_at']}\n\n")
        handle.write("- Ranked barplots now always show strongest positive/negative correlations and mark BH-significant variables.\n")
        handle.write("- Source-group comparison heatmaps now use source groups on the y-axis and variables on the x-axis.\n")
        handle.write("- Source-group comparison heatmaps now show all Pearson values with significance stars; significant-only white-cell versions are deprecated.\n")
        handle.write("- All-variable source-group plots use fixed column order: core/SpaCET/MP/K*/Hallmark.\n")
        handle.write(f"- Ranked barplots regenerated: {ranked_count}\n")
        handle.write(f"- All-Pearson source-group comparison heatmaps generated: {comparison_allcorr_count}\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
