from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


SPOTTEDPY_ROOT = Path(r"D:\HGSOC_Spatial_Atlas\git_clones\SpottedPy-main")
PROJECT_ROOT = Path(r"D:\HGSOC_Spatial_Atlas\05_analysis_ready\SpottedPy_v2_paper_aligned")
DIST_ROOT = PROJECT_ROOT / "05_distance_gee"
TABLE_ROOT = DIST_ROOT / "tables"
FIG_ROOT = DIST_ROOT / "figures" / "preflight_spottedpy_native"
SCRIPT_ROOT = DIST_ROOT / "scripts_used"

COMPONENT_SUMMARY = TABLE_ROOT / "spottedpy_v2_distance_component_summary_preflight.csv"
NATIVE_LONG_OUT = TABLE_ROOT / "spottedpy_v2_distance_spottedpy_native_long_preflight.csv"
NATIVE_STATS_OUT = TABLE_ROOT / "spottedpy_v2_distance_spottedpy_native_gee_stats_preflight.csv"
RUN_MANIFEST_OUT = DIST_ROOT / "run_manifest_distance_spottedpy_native_preflight.json"
README_OUT = FIG_ROOT / "README_spottedpy_native_preflight.md"
HALLMARK_GROUP_MAP_OUT = TABLE_ROOT / "spottedpy_v2_distance_hallmark_group_map_preflight.csv"
HALLMARK_BATCH_INCLUSION_OUT = TABLE_ROOT / "spottedpy_v2_distance_hallmark_batch_plot_inclusion_preflight.csv"
DISTANCE_DESIGN_MANIFEST = DIST_ROOT / "run_manifest_distance_preflight.json"
DISTANCE_CALCULATION_MANIFEST = DIST_ROOT / "run_manifest_distance_calculation_preflight.json"
CORE_HOTSPOT_MANIFEST = PROJECT_ROOT / "04_hotspots_preflight_revised_scoring_policy" / "run_manifest.json"
HALLMARK_SPACET_HOTSPOT_MANIFEST = (
    PROJECT_ROOT / "04_hotspots_preflight_revised_scoring_policy" / "run_manifest_hallmark_spacet_hotspots.json"
)
BRANCH_README_OUT = DIST_ROOT / "README_distance_preflight.md"

PRIMARY_LABELS = {
    "snai1ac_consensus_full_hot": "SNAI1ac_hot",
    "snai1ac_consensus_full_cold": "SNAI1ac_cold",
    "snai12r_full_hot": "SNAI12R_hot",
}

FAMILY_LABELS = {
    "mp": "MP",
    "spacet": "SpaCET",
    "hallmark": "Hallmark",
}

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
    "Immune_Response_Inflammation": [
        "HALLMARK_INFLAMMATORY_RESPONSE",
        "HALLMARK_INTERFERON_ALPHA_RESPONSE",
        "HALLMARK_INTERFERON_GAMMA_RESPONSE",
        "HALLMARK_COMPLEMENT",
        "HALLMARK_IL2_STAT5_SIGNALING",
        "HALLMARK_IL6_JAK_STAT3_SIGNALING",
        "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    ],
    "Cellular_Stress_Apoptosis": [
        "HALLMARK_APOPTOSIS",
        "HALLMARK_DNA_REPAIR",
        "HALLMARK_HYPOXIA",
        "HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY",
        "HALLMARK_UNFOLDED_PROTEIN_RESPONSE",
    ],
    "Signaling_Development": [
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
    "Structure_Adhesion_CellComponents": [
        "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
        "HALLMARK_APICAL_JUNCTION",
        "HALLMARK_APICAL_SURFACE",
    ],
    "Other_Biological_States": [
        "HALLMARK_XENOBIOTIC_METABOLISM",
        "HALLMARK_PROTEIN_SECRETION",
        "HALLMARK_ANDROGEN_RESPONSE",
        "HALLMARK_ESTROGEN_RESPONSE_EARLY",
        "HALLMARK_ESTROGEN_RESPONSE_LATE",
        "HALLMARK_PEROXISOME",
    ],
}

HALLMARK_GROUP_BY_ID = {
    hallmark_id: group_name
    for group_name, hallmark_ids in HALLMARK_GROUPS.items()
    for hallmark_id in hallmark_ids
}

MP_DISPLAY_LABELS = {
    "MP1_angiogenic_vascular_scoregenes": "MP1 angiogenic/vascular",
    "MP2_iCAF_stress_scoregenes": "MP2 inflammatory / immediate-early stress CAF",
    "MP3_complement_CAF_scoregenes": "MP3 complement/anti-migratory CAF",
    "MP4_activated_myCAF_scoregenes": "MP4 ECM producing myCAF",
    "MP5_IFN_TLS_immune_scoregenes": "MP5 type 1/2 IFN response",
    "MP6_APC_TAM_myeloid_scoregenes": "MP6 APC/LAM/TAM",
    "MP7_malignant_hypoxia_scoregenes": "MP7 hypoxia-glycolysis",
    "MP8_malignant_acute_phase_secretory_scoregenes": "MP8 acute-phase/alarmin secretory",
}

MP_LEGACY_DISPLAY_LABELS = {
    "MP1 angiogenic/vascular": "MP1 angiogenic/vascular",
    "MP2 iCAF-stress": "MP2 inflammatory / immediate-early stress CAF",
    "MP3 complement-CAF": "MP3 complement/anti-migratory CAF",
    "MP4 activated-myCAF": "MP4 ECM producing myCAF",
    "MP5 IFN/TLS immune": "MP5 type 1/2 IFN response",
    "MP6 APC/TAM myeloid": "MP6 APC/LAM/TAM",
    "MP7 malignant hypoxia": "MP7 hypoxia-glycolysis",
    "MP8 malignant acute-phase/secretory": "MP8 acute-phase/alarmin secretory",
}

CONTRAST_SHORT = {
    "snai1ac_hot_vs_snai1ac_cold": "acHot_vs_acCold",
    "snai1ac_hot_vs_snai12r_hot": "acHot_vs_2rHot",
}

HALLMARK_GROUP_SHORT = {
    "Proliferation": "prolif",
    "Metabolism": "metab",
    "Immune_Response_Inflammation": "immune",
    "Cellular_Stress_Apoptosis": "stress",
    "Signaling_Development": "signal",
    "Structure_Adhesion_CellComponents": "adhesion",
    "Other_Biological_States": "other",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def make_comparison_label(row: pd.Series) -> str:
    family = row["target_family"]
    if family == "mp":
        target_id = str(row.get("target_variable_id", ""))
        if target_id in MP_DISPLAY_LABELS:
            return MP_DISPLAY_LABELS[target_id]
        return MP_LEGACY_DISPLAY_LABELS.get(str(row["target_display_label"]), str(row["target_display_label"]))
    if family == "spacet":
        return str(row["target_display_label"]).replace("SpaCET ", "")
    if family == "hallmark":
        return str(row["target_display_label"]).replace("HALLMARK_", "").replace("_", " ")
    return str(row["target_display_label"])


def slugify(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("&", "and")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def contrast_slug(contrast_id: str) -> str:
    return CONTRAST_SHORT.get(contrast_id, slugify(contrast_id))


def family_slug(family: str) -> str:
    if family == "hallmark":
        return "hm_all"
    return slugify(family)


def hallmark_group_slug(group_name: str) -> str:
    return f"hm_{HALLMARK_GROUP_SHORT.get(group_name, slugify(group_name))}"


def figure_path(plot_kind: str, variable_set: str, contrast_id: str) -> Path:
    return FIG_ROOT / f"{plot_kind}__{variable_set}__{contrast_slug(contrast_id)}.png"


def build_native_long(component_df: pd.DataFrame) -> pd.DataFrame:
    use = component_df[component_df["target_state"].eq("hot")].copy()
    use = use[use["target_family"].isin(["mp", "spacet", "hallmark"])].copy()
    use["primary_variable"] = use["primary_group_id"].map(PRIMARY_LABELS)
    is_mp = use["target_family"].eq("mp")
    use.loc[is_mp, "target_display_label"] = use.loc[is_mp].apply(make_comparison_label, axis=1)
    use["comparison_variable"] = use.apply(make_comparison_label, axis=1)
    out = pd.DataFrame(
        {
            "min_distance": pd.to_numeric(use["component_median_distance"], errors="coerce"),
            "primary_variable": use["primary_variable"],
            "comparison_variable": use["comparison_variable"],
            "primary_index": use["sample_label"].astype(str)
            + "__"
            + use["primary_variable"].astype(str)
            + "__H"
            + use["primary_component_number"].astype(str),
            "batch": use["sample_label"],
            "hotspot_number": pd.to_numeric(use["primary_component_number"], errors="coerce").astype("Int64"),
            "contrast_id": use["contrast_id"],
            "contrast_label": use["contrast_label"],
            "target_family": use["target_family"],
            "target_variable_id": use["target_variable_id"],
            "target_display_label": use["target_display_label"],
            "hallmark_group": use["target_variable_id"].map(HALLMARK_GROUP_BY_ID).fillna(""),
            "target_state": use["target_state"],
            "source_note": "component_median_distance_used_as_min_distance_for_spottedpy_plotting",
        }
    )
    out = out.dropna(subset=["min_distance", "primary_variable", "comparison_variable", "hotspot_number"])
    return out


def ordered_comparisons(native_df: pd.DataFrame, contrast_id: str, family: str) -> list[str]:
    data = native_df[
        native_df["contrast_id"].eq(contrast_id)
        & native_df["target_family"].eq(family)
        & native_df["primary_variable"].eq("SNAI1ac_hot")
    ].copy()
    order = (
        data.groupby("comparison_variable", dropna=False)["min_distance"]
        .median()
        .sort_values(ascending=True)
        .index.tolist()
    )
    return order


def ordered_hallmark_group_comparisons(native_df: pd.DataFrame, contrast_id: str, group_name: str) -> list[str]:
    data = native_df[
        native_df["contrast_id"].eq(contrast_id)
        & native_df["target_family"].eq("hallmark")
        & native_df["hallmark_group"].eq(group_name)
        & native_df["primary_variable"].eq("SNAI1ac_hot")
    ].copy()
    if data.empty:
        return []
    return (
        data.groupby("comparison_variable", dropna=False)["min_distance"]
        .median()
        .sort_values(ascending=True)
        .index.tolist()
    )


def primary_vars_for_contrast(contrast_id: str) -> list[str]:
    if contrast_id == "snai1ac_hot_vs_snai1ac_cold":
        return ["SNAI1ac_hot", "SNAI1ac_cold"]
    if contrast_id == "snai1ac_hot_vs_snai12r_hot":
        return ["SNAI1ac_hot", "SNAI12R_hot"]
    raise ValueError(f"Unexpected contrast_id: {contrast_id}")


def plot_bubble_plot_mean_distances_with_legend(
    distances_df: pd.DataFrame,
    primary_vars: list[str],
    comparison_vars: list[str],
    fig_size: tuple[float, float],
    save_path: str,
    title: str,
) -> None:
    filtered = distances_df[
        distances_df["primary_variable"].isin(primary_vars)
        & distances_df["comparison_variable"].isin(comparison_vars)
    ].copy()
    if filtered.empty:
        return

    mean_df = (
        filtered.groupby(["primary_variable", "comparison_variable"], as_index=False)["min_distance"]
        .mean()
        .rename(columns={"min_distance": "mean_distance"})
    )
    mean_df["primary_variable"] = pd.Categorical(mean_df["primary_variable"], categories=primary_vars, ordered=True)
    mean_df["comparison_variable"] = pd.Categorical(
        mean_df["comparison_variable"], categories=comparison_vars, ordered=True
    )
    mean_df = mean_df.dropna(subset=["primary_variable", "comparison_variable", "mean_distance"])
    if mean_df.empty:
        return

    x_lookup = {name: idx for idx, name in enumerate(primary_vars)}
    y_lookup = {name: idx for idx, name in enumerate(comparison_vars)}
    x = mean_df["primary_variable"].map(x_lookup).astype(float)
    y = mean_df["comparison_variable"].map(y_lookup).astype(float)
    values = mean_df["mean_distance"].astype(float).to_numpy()
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if np.isclose(vmin, vmax):
        sizes = np.full_like(values, 520.0, dtype=float)
    else:
        sizes = 160 + 1240 * (values - vmin) / (vmax - vmin)

    fig, ax = plt.subplots(figsize=fig_size)
    scatter = ax.scatter(
        x,
        y,
        c=values,
        s=sizes,
        cmap="viridis",
        edgecolor="white",
        linewidth=0.6,
        alpha=0.88,
    )
    ax.set_xticks(range(len(primary_vars)))
    ax.set_xticklabels(primary_vars, rotation=35, ha="right")
    ax.set_yticks(range(len(comparison_vars)))
    ax.set_yticklabels(comparison_vars)
    ax.invert_yaxis()
    ax.set_xlim(-0.5, len(primary_vars) - 0.5)
    ax.set_ylim(len(comparison_vars) - 0.5, -0.5)
    ax.set_xlabel("Primary region")
    ax.set_ylabel("Target hotspot")
    ax.set_title(title, pad=14)
    ax.grid(axis="x", color="0.9", linewidth=0.8)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.08)
    cbar.set_label("Color: mean distance")

    legend_values = np.unique(np.round(np.linspace(vmin, vmax, min(3, len(np.unique(values)))), 2))
    handles = []
    for val in legend_values:
        if np.isclose(vmin, vmax):
            size = 520.0
        else:
            size = 160 + 1240 * (val - vmin) / (vmax - vmin)
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="0.65",
                markeredgecolor="white",
                markersize=np.sqrt(size) * 0.48,
                label=f"{val:g}",
            )
        )
    ax.legend(
        handles=handles,
        title="Size: mean distance",
        loc="upper left",
        bbox_to_anchor=(1.34, 1.0),
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_bubble_chart_by_batch_all_with_significance(
    prepare_data_hotspot_fn,
    df: pd.DataFrame,
    primary_variable_value: str,
    comparison_variable_values: list[str],
    reference_variable: str,
    save_path: str,
    pval_cutoff: float = 0.05,
    fig_size: tuple[float, float] = (12, 10),
    bubble_size: float = 24,
    slide_order: dict[str, int] | None = None,
) -> None:
    grouped_data = prepare_data_hotspot_fn(
        df,
        primary_variable_value,
        comparison_variable_values,
        reference_variable,
    )
    data = grouped_data[grouped_data["primary_variable"].ne(reference_variable)].copy()
    if data.empty:
        return

    if slide_order:
        slide_positions = slide_order
    else:
        slides = list(df["batch"].dropna().astype(str).unique())
        slide_positions = {slide: idx for idx, slide in enumerate(slides)}
    strict_pval_cutoff = pval_cutoff / max(len(slide_positions), 1)

    comp_positions = {comp: idx for idx, comp in enumerate(comparison_variable_values)}
    data["batch"] = data["batch"].astype(str)
    data = data[data["comparison_variable"].isin(comp_positions)]
    if data.empty:
        return

    data["x"] = data["comparison_variable"].map(comp_positions).astype(float)
    data["y"] = data["batch"].map(slide_positions).astype(float)
    data["Difference"] = pd.to_numeric(data["Difference"], errors="coerce")
    data["Pvalue"] = pd.to_numeric(data["Pvalue"], errors="coerce")
    data = data.dropna(subset=["x", "y", "Difference"])
    if data.empty:
        return

    data["is_significant"] = data["Pvalue"].lt(pval_cutoff)
    data["is_slide_corrected_significant"] = data["Pvalue"].lt(strict_pval_cutoff)
    data["direction"] = np.where(
        data["Difference"].lt(0),
        f"Closer to {primary_variable_value}",
        np.where(data["Difference"].gt(0), f"Closer to {reference_variable}", "No difference"),
    )
    color_map = {
        f"Closer to {primary_variable_value}": "#d7191c",
        f"Closer to {reference_variable}": "#2c7bb6",
        "No difference": "0.65",
    }
    data["color"] = data["direction"].map(color_map).fillna("0.65")
    max_abs = max(float(data["Difference"].abs().max()), 1e-6)
    data["size"] = 30 + bubble_size * 42 * data["Difference"].abs() / max_abs
    data["alpha"] = np.where(data["is_significant"], 0.92, 0.28)
    data["edgecolor"] = np.where(data["is_significant"], "black", "white")
    data["linewidth"] = np.where(data["is_significant"], 0.7, 0.25)

    fig, ax = plt.subplots(figsize=fig_size)
    for _, row in data.iterrows():
        ax.scatter(
            row["x"],
            row["y"],
            s=row["size"],
            color=row["color"],
            alpha=row["alpha"],
            edgecolors=row["edgecolor"],
            linewidth=row["linewidth"],
        )
    strict_data = data[data["is_slide_corrected_significant"]].copy()
    if not strict_data.empty:
        ax.scatter(
            strict_data["x"],
            strict_data["y"],
            marker="*",
            s=72,
            facecolors="white",
            edgecolors="black",
            linewidths=0.65,
            zorder=4,
        )

    ax.set_xticks(range(len(comparison_variable_values)))
    ax.set_xticklabels(comparison_variable_values, rotation=90)
    ax.set_yticks(list(slide_positions.values()))
    ax.set_yticklabels(list(slide_positions.keys()))
    ax.set_xlim(-0.5, len(comparison_variable_values) - 0.5)
    ax.set_ylim(len(slide_positions) - 0.5, -0.5)
    ax.set_xlabel("Target hotspot")
    ax.set_ylabel("Sample")
    ax.grid(axis="x", color="0.92", linewidth=0.8)

    direction_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color, label=label, markersize=8)
        for label, color in color_map.items()
        if label in set(data["direction"])
    ]
    sig_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="0.75",
            markeredgecolor="black",
            label=f"per-slide p < {pval_cutoff:g}",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="0.75",
            markeredgecolor="white",
            alpha=0.35,
            label=f"per-slide p >= {pval_cutoff:g}",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="black",
            label=f"p < {strict_pval_cutoff:.3g} (0.05 / n slides)",
            markersize=10,
        ),
    ]
    first_legend = ax.legend(
        handles=direction_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.07),
        ncol=max(1, len(direction_handles)),
        frameon=True,
    )
    ax.add_artist(first_legend)
    ax.legend(handles=sig_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_current_gee_plot_with_external_legend(save_path: Path) -> None:
    fig = plt.gcf()
    ax = plt.gca()
    legend = ax.get_legend()
    if legend is not None:
        handles = getattr(legend, "legend_handles", None)
        if handles is None:
            handles = getattr(legend, "legendHandles", [])
        labels = [text.get_text() for text in legend.get_texts()]
        title = legend.get_title().get_text() or "GEE p-value"
        legend.remove()
        ax.legend(
            handles=handles,
            labels=labels,
            title=title,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0,
            frameon=False,
        )
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def significant_hallmarks_for_contrast(stats: pd.DataFrame, contrast_id: str, alpha: float = 0.05) -> list[str]:
    if stats is None or stats.empty:
        return []
    use = stats[
        stats["contrast_id"].eq(contrast_id)
        & stats["target_family"].eq("hallmark")
        & pd.to_numeric(stats["p_value"], errors="coerce").lt(alpha)
    ].copy()
    return sorted(use["comparison_variable"].dropna().astype(str).unique().tolist())


def write_readme(figures: list[str]) -> None:
    text = f"""# SpottedPy-native distance preflight plots

Generated: {now_iso()}

These plots use SpottedPy's distance-plotting semantics on a hot-target-only adapter table.

Filename convention:

- `gee__...`: cohort-level GEE/differential distance dotplot from `plot_custom_scatter`.
- `mean__...`: descriptive mean-distance bubble matrix.
- `batch__...`: slide-level by-batch diagnostic bubble plot.
- variable-set tags: `mp`, `spacet`, `hm_all`, `hm_sig`, `hm_prolif`, `hm_metab`, `hm_immune`, `hm_stress`, `hm_signal`, `hm_adhesion`, `hm_other`.
- contrast tags: `acHot_vs_acCold` and `acHot_vs_2rHot`.

Used helpers:

- `plot_custom_scatter(..., compare_distribution_metric='median')` for the paper-style differential distance/GEE dotplot. These are cohort-level plots: SpottedPy fits a GEE grouped by `batch`.
- `batch__...` plots use SpottedPy's by-batch distance-preparation semantics, but the local wrapper draws all bubbles for included variables and marks nominal per-slide significance instead of hiding non-significant values behind SpottedPy's built-in Bonferroni-by-slide visibility gate.
- `mean__...` plots use the same grouping definition as SpottedPy's `plot_bubble_plot_mean_distances`, but the local wrapper adds legends because the package helper hard-codes `legend=False`. These are descriptive cohort-level aggregates, not per-sample plots.

Plot scope:

- `gee__...`: cohort-level GEE summary over multiple samples.
- `batch__...`: per-sample/slide diagnostic; rows are batches. All included variables are drawn. Black outlines mark nominal per-slide `p < 0.05`; stars mark the stricter slide-corrected threshold `p < 0.05 / n_slides`.
- `mean__...`: descriptive aggregate across all source components and samples; no GEE/statistical clustering.

Hallmark grouping:

The per-group Hallmark plots use the manifest groups: Proliferation, Metabolism, Immune_Response_Inflammation, Cellular_Stress_Apoptosis, Signaling_Development, Structure_Adhesion_CellComponents, and Other_Biological_States.

For Hallmark `bubble_by_batch` plots, the all-Hallmark version is restricted to Hallmarks with `plot_custom_scatter` GEE `p < 0.05` for that contrast. Per-group Hallmark batch plots are not prefiltered; they include every pathway in the relevant manifest group. Once a variable is included, the same bubble drawing and significance marking is used for MP, SpaCET, and Hallmarks.

Adapter note:

SpottedPy's plotting helpers expect a long distance table with `primary_variable`, `comparison_variable`, `min_distance`, `batch`, and `hotspot_number`. Our distance calculation has already summarized each source/reference component, so `component_median_distance` is passed as `min_distance`, and `primary_component_number` is passed as `hotspot_number`.

The target universe is hotspots only. Target coldspots are excluded.

The native GEE statistics in `spottedpy_v2_distance_spottedpy_native_gee_stats_preflight.csv` are preflight statistics for review; final report-facing statistics should come from the full distance/GEE run once we approve the design.

Figures written: {len(figures)}
"""
    README_OUT.write_text(text, encoding="utf-8")


def write_branch_readme(native_df: pd.DataFrame, stats_df: pd.DataFrame, figures: list[str]) -> None:
    text = f"""# SpottedPy v2 Distance/GEE Preflight

Generated: {now_iso()}

This branch is the distance-statistics preflight for the paper-aligned SpottedPy v2 workflow. It now contains design tables, calculated nearest-hotspot distances, and SpottedPy-native review figures. It is still a preflight branch: final report-facing statistics should be generated by the full distance/GEE run after this design is frozen.

## Upstream hotspot provenance

- Core/SNAI1-family/MP/K*: `{CORE_HOTSPOT_MANIFEST}`
- Hallmark/SpaCET calculation-only hotspots: `{HALLMARK_SPACET_HOTSPOT_MANIFEST}`

Hallmark/SpaCET hotspots were calculated before distance statistics but were not plotted at the hotspot stage. They enter here as distance targets.

## Active design

- Source group: full-slide `SNAI1-ac_consensus` hotspots.
- Reference groups: full-slide `SNAI1-ac_consensus` coldspots and full-slide `SNAI1-2R` hotspots.
- Target universe: MP1-MP8 tumor hotspots, sample-specific K* tumor hotspots, full-slide SpaCET hotspots, and full-slide Hallmark hotspots.
- Target coldspots are excluded.
- K* programs remain sample-specific and are not pooled as identical cross-sample variables.

## Distance metric

Distances use nearest-spot hotspot distance over Visium `array_row`/`array_col` coordinates. The primary component summary is `component_median_distance`; min and mean component summaries remain sensitivity fields in the component table. Empty hotspots are not imputed.

## Current preflight outputs

- Native long table rows: {native_df.shape[0]}
- Native GEE/stat rows: {stats_df.shape[0]}
- Active SpottedPy-native figures: {len(figures)}
- Figure root: `{FIG_ROOT}`

Figure prefixes:

- `gee__...`: cohort-level SpottedPy `plot_custom_scatter` GEE/differential-distance plots.
- `mean__...`: descriptive cohort aggregate mean-distance bubble plots.
- `batch__...`: per-slide diagnostic bubble plots; black outlines mark nominal per-slide `p < 0.05`, stars mark `p < 0.05 / n_slides`.

Hallmark group plots use the manifest groups agreed for the v2 design. All-Hallmark by-batch plots are restricted to Hallmarks with cohort-level GEE `p < 0.05`; Hallmark group by-batch plots include all pathways in that group.

## Manifest chain

- Distance design manifest: `{DISTANCE_DESIGN_MANIFEST}`
- Distance calculation manifest: `{DISTANCE_CALCULATION_MANIFEST}`
- SpottedPy-native plot/GEE manifest: `{RUN_MANIFEST_OUT}`
"""
    BRANCH_README_OUT.write_text(text, encoding="utf-8")


def main() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    SCRIPT_ROOT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(SPOTTEDPY_ROOT))
    from spottedpy.sp_plotting import plot_custom_scatter, prepare_data_hotspot

    component_df = pd.read_csv(COMPONENT_SUMMARY, low_memory=False)
    native_df = build_native_long(component_df)
    native_df.to_csv(NATIVE_LONG_OUT, index=False)
    pd.DataFrame(
        [
            {
                "hallmark_group": group_name,
                "target_variable_id": hallmark_id,
                "comparison_variable": hallmark_id.replace("HALLMARK_", "").replace("_", " "),
            }
            for group_name, hallmark_ids in HALLMARK_GROUPS.items()
            for hallmark_id in hallmark_ids
        ]
    ).to_csv(HALLMARK_GROUP_MAP_OUT, index=False)

    stats_rows = []
    figures: list[str] = []
    for contrast_id in sorted(native_df["contrast_id"].unique()):
        primary_vars = primary_vars_for_contrast(contrast_id)
        for family in ["mp", "spacet", "hallmark"]:
            comparison_vars = ordered_comparisons(native_df, contrast_id, family)
            if not comparison_vars:
                continue
            family_label = FAMILY_LABELS[family]
            plot_df = native_df[
                native_df["contrast_id"].eq(contrast_id)
                & native_df["target_family"].eq(family)
            ].copy()

            custom_path = figure_path("gee", family_slug(family), contrast_id)
            fig_width = max(8, 0.32 * len(comparison_vars))
            stats = plot_custom_scatter(
                plot_df,
                primary_vars=primary_vars,
                comparison_vars=comparison_vars,
                fig_size=(fig_width, 4.8),
                bubble_size=(70, 420),
                sort_by_difference=True,
                compare_distribution_metric="median",
                statistical_test=True,
                save_path=None,
            )
            save_current_gee_plot_with_external_legend(custom_path)
            plt.close("all")
            if stats is not None:
                stats = stats.copy()
                stats["contrast_id"] = contrast_id
                stats["target_family"] = family
                stats["plot_scope"] = "family_all"
                stats["spottedpy_helper"] = "plot_custom_scatter_median"
                stats_rows.append(stats)
            figures.append(str(custom_path))

            bubble_path = figure_path("mean", family_slug(family), contrast_id)
            if family == "hallmark":
                bubble_fig_size = (max(8.5, 0.28 * len(comparison_vars)), max(10, 0.34 * len(comparison_vars)))
            else:
                bubble_fig_size = (
                    max(6, 0.2 * len(comparison_vars)),
                    max(6, 0.18 * len(comparison_vars)),
                )
            plot_bubble_plot_mean_distances_with_legend(
                plot_df,
                primary_vars=primary_vars,
                comparison_vars=comparison_vars,
                fig_size=bubble_fig_size,
                save_path=str(bubble_path),
                title=f"{family_label} hotspot mean distances\n{contrast_id}",
            )
            figures.append(str(bubble_path))

            if family in ["mp", "spacet"]:
                by_batch_path = figure_path("batch", family_slug(family), contrast_id)
                plot_bubble_chart_by_batch_all_with_significance(
                    prepare_data_hotspot,
                    plot_df,
                    primary_variable_value=primary_vars[0],
                    comparison_variable_values=comparison_vars,
                    reference_variable=primary_vars[1],
                    save_path=str(by_batch_path),
                    pval_cutoff=0.05,
                    fig_size=(max(9, 0.45 * len(comparison_vars)), 7.5),
                    bubble_size=12,
                    slide_order=None,
                )
                plt.close("all")
                figures.append(str(by_batch_path))

            if family == "hallmark":
                for group_name in HALLMARK_GROUPS:
                    group_vars = ordered_hallmark_group_comparisons(native_df, contrast_id, group_name)
                    if not group_vars:
                        continue
                    group_df = plot_df[plot_df["hallmark_group"].eq(group_name)].copy()

                    group_custom_path = figure_path("gee", hallmark_group_slug(group_name), contrast_id)
                    group_stats = plot_custom_scatter(
                        group_df,
                        primary_vars=primary_vars,
                        comparison_vars=group_vars,
                        fig_size=(max(7, 0.46 * len(group_vars)), 4.6),
                        bubble_size=(70, 420),
                        sort_by_difference=True,
                        compare_distribution_metric="median",
                        statistical_test=True,
                        save_path=None,
                    )
                    save_current_gee_plot_with_external_legend(group_custom_path)
                    plt.close("all")
                    if group_stats is not None:
                        group_stats = group_stats.copy()
                        group_stats["contrast_id"] = contrast_id
                        group_stats["target_family"] = family
                        group_stats["plot_scope"] = f"hallmark_group:{group_name}"
                        group_stats["hallmark_group"] = group_name
                        group_stats["spottedpy_helper"] = "plot_custom_scatter_median"
                        stats_rows.append(group_stats)
                    figures.append(str(group_custom_path))

                    group_bubble_path = figure_path("mean", hallmark_group_slug(group_name), contrast_id)
                    plot_bubble_plot_mean_distances_with_legend(
                        group_df,
                        primary_vars=primary_vars,
                        comparison_vars=group_vars,
                        fig_size=(max(6.5, 0.55 * len(group_vars)), max(4.8, 0.48 * len(group_vars))),
                        save_path=str(group_bubble_path),
                        title=f"Hallmark mean distances: {group_name.replace('_', ' ')}\n{contrast_id}",
                    )
                    figures.append(str(group_bubble_path))

    stats_df = pd.concat(stats_rows, ignore_index=True) if stats_rows else pd.DataFrame()
    stats_df.to_csv(NATIVE_STATS_OUT, index=False)

    hallmark_batch_inclusion_rows = []
    for contrast_id in sorted(native_df["contrast_id"].unique()):
        primary_vars = primary_vars_for_contrast(contrast_id)
        plot_df = native_df[
            native_df["contrast_id"].eq(contrast_id)
            & native_df["target_family"].eq("hallmark")
        ].copy()
        sig_vars = significant_hallmarks_for_contrast(
            stats_df[stats_df["plot_scope"].eq("family_all")], contrast_id, alpha=0.05
        )
        if sig_vars:
            by_batch_path = figure_path("batch", "hm_sig", contrast_id)
            plot_bubble_chart_by_batch_all_with_significance(
                prepare_data_hotspot,
                plot_df,
                primary_variable_value=primary_vars[0],
                comparison_variable_values=sig_vars,
                reference_variable=primary_vars[1],
                save_path=str(by_batch_path),
                pval_cutoff=0.05,
                fig_size=(max(10, 0.45 * len(sig_vars)), 8.2),
                bubble_size=12,
                slide_order=None,
            )
            plt.close("all")
            figures.append(str(by_batch_path))
            for comp in sig_vars:
                hallmark_batch_inclusion_rows.append(
                    {
                        "contrast_id": contrast_id,
                        "plot_scope": "all_hallmarks_prefiltered_by_family_gee_p_lt_0.05",
                        "hallmark_group": "all_significant",
                        "comparison_variable": comp,
                    }
                )

        for group_name in HALLMARK_GROUPS:
            group_vars = ordered_hallmark_group_comparisons(native_df, contrast_id, group_name)
            if not group_vars:
                continue
            group_df = plot_df[plot_df["hallmark_group"].eq(group_name)].copy()
            group_by_batch_path = figure_path("batch", hallmark_group_slug(group_name), contrast_id)
            plot_bubble_chart_by_batch_all_with_significance(
                prepare_data_hotspot,
                group_df,
                primary_variable_value=primary_vars[0],
                comparison_variable_values=group_vars,
                reference_variable=primary_vars[1],
                save_path=str(group_by_batch_path),
                pval_cutoff=0.05,
                fig_size=(max(8, 0.65 * len(group_vars)), 7.4),
                bubble_size=12,
                slide_order=None,
            )
            plt.close("all")
            figures.append(str(group_by_batch_path))
            for comp in group_vars:
                hallmark_batch_inclusion_rows.append(
                    {
                        "contrast_id": contrast_id,
                        "plot_scope": "hallmark_group_all_members_no_cohort_prefilter",
                        "hallmark_group": group_name,
                        "comparison_variable": comp,
                    }
                )

    pd.DataFrame(hallmark_batch_inclusion_rows).to_csv(HALLMARK_BATCH_INCLUSION_OUT, index=False)

    if "__file__" in globals():
        shutil.copy2(Path(__file__), SCRIPT_ROOT / Path(__file__).name)

    figures = [f for f in figures if Path(f).exists()]
    write_readme(figures)
    write_branch_readme(native_df, stats_df, figures)
    manifest = {
        "generated_at": now_iso(),
        "inputs": {
            "component_summary": str(COMPONENT_SUMMARY),
            "distance_design_manifest": str(DISTANCE_DESIGN_MANIFEST),
            "distance_calculation_manifest": str(DISTANCE_CALCULATION_MANIFEST),
            "core_hotspot_manifest": str(CORE_HOTSPOT_MANIFEST),
            "hallmark_spacet_hotspot_manifest": str(HALLMARK_SPACET_HOTSPOT_MANIFEST),
        },
        "outputs": {
            "native_long": str(NATIVE_LONG_OUT),
            "native_stats": str(NATIVE_STATS_OUT),
            "hallmark_group_map": str(HALLMARK_GROUP_MAP_OUT),
            "hallmark_batch_plot_inclusion": str(HALLMARK_BATCH_INCLUSION_OUT),
            "figure_root": str(FIG_ROOT),
            "figures": figures,
            "readme": str(README_OUT),
            "branch_readme": str(BRANCH_README_OUT),
        },
        "n_native_long_rows": int(native_df.shape[0]),
        "n_stats_rows": int(stats_df.shape[0]),
        "n_figures": len(figures),
        "target_state_policy": "hotspots_only",
        "active_source_group": "SNAI1-ac consensus full-slide hotspots",
        "active_reference_groups": [
            "SNAI1-ac consensus full-slide coldspots",
            "SNAI1-2R full-slide hotspots",
        ],
        "target_families": ["mp", "spacet", "hallmark"],
        "kstar_note": "K* distances are calculated in component/sample tables but excluded from SpottedPy-native cross-sample plot families because program IDs are sample-specific.",
        "native_helpers": [
            "plot_custom_scatter",
            "plot_bubble_plot_mean_distances_semantics_with_local_legend_wrapper",
            "plot_bubble_chart_by_batch_semantics_with_local_all_bubbles_wrapper",
        ],
    }
    RUN_MANIFEST_OUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
