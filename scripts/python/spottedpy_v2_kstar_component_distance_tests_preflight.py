from __future__ import annotations

import json
import math
import shutil
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu


PROJECT_ROOT = Path(r"D:\HGSOC_Spatial_Atlas\05_analysis_ready\SpottedPy_v2_paper_aligned")
DIST_ROOT = PROJECT_ROOT / "05_distance_gee"
TABLE_ROOT = DIST_ROOT / "tables"
FIG_ROOT = DIST_ROOT / "figures" / "kstar_component_distance_preflight"
SCRIPT_ROOT = DIST_ROOT / "scripts_used"

COMPONENT_SUMMARY = TABLE_ROOT / "spottedpy_v2_distance_component_summary_preflight.csv"
LONG_OUT = TABLE_ROOT / "spottedpy_v2_kstar_component_distance_long_preflight.csv"
TESTS_OUT = TABLE_ROOT / "spottedpy_v2_kstar_component_distance_tests_preflight.csv"
RUN_MANIFEST_OUT = DIST_ROOT / "run_manifest_kstar_component_distance_tests_preflight.json"
README_OUT = FIG_ROOT / "README_kstar_component_distance_preflight.md"

SOURCE_GROUP = "snai1ac_consensus_full_hot"
REFERENCE_GROUPS = {
    "snai1ac_hot_vs_snai1ac_cold": "snai1ac_consensus_full_cold",
    "snai1ac_hot_vs_snai12r_hot": "snai12r_full_hot",
}
GROUP_LABELS = {
    "snai1ac_consensus_full_hot": "SNAI1ac_hot",
    "snai1ac_consensus_full_cold": "SNAI1ac_cold",
    "snai12r_full_hot": "SNAI1-2R_hot",
}
GROUP_ORDER = ["SNAI1ac_hot", "SNAI1ac_cold", "SNAI1-2R_hot"]
GROUP_COLORS = {
    "SNAI1ac_hot": "#c83f49",
    "SNAI1ac_cold": "#3b78b8",
    "SNAI1-2R_hot": "#5f5f5f",
}

METRICS = {
    "median": "component_median_distance",
    "mean": "component_mean_distance",
    "min": "component_min_distance",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def safe_name(value: str) -> str:
    return (
        str(value)
        .replace("\\", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
        .replace(" ", "_")
    )


def bh_adjust(values: pd.Series) -> pd.Series:
    p = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=values.index, dtype=float)
    valid = p.dropna()
    if valid.empty:
        return out
    order = valid.sort_values().index.to_list()
    ranked = valid.loc[order].to_numpy(dtype=float)
    n = len(ranked)
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        q = min(prev, ranked[i] * n / rank)
        adjusted[i] = q
        prev = q
    out.loc[order] = np.clip(adjusted, 0, 1)
    return out


def p_to_stars(q: float) -> str:
    if pd.isna(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def wrap_label(text: str, width: int = 24) -> str:
    compact = str(text).replace("\n", " ")
    return "\n".join(textwrap.wrap(compact, width=width, break_long_words=False)) or compact


def build_long(component_df: pd.DataFrame) -> pd.DataFrame:
    kstar = component_df[
        component_df["target_family"].eq("kstar")
        & component_df["target_state"].eq("hot")
        & component_df["primary_group_id"].isin(
            [SOURCE_GROUP, "snai1ac_consensus_full_cold", "snai12r_full_hot"]
        )
    ].copy()
    keep = [
        "dataset",
        "sample",
        "sample_label",
        "contrast_id",
        "contrast_label",
        "primary_role",
        "primary_group_id",
        "primary_group_label",
        "primary_component_number",
        "n_primary_component_spots",
        "target_variable_id",
        "target_title",
        "target_display_label",
        "target_short_label",
        "kstar_alignment_category_draft",
        "kstar_family_label",
        "kstar_mp1_8_name",
        "n_target_spots",
        "n_target_components",
        *METRICS.values(),
    ]
    missing = [col for col in keep if col not in kstar.columns]
    if missing:
        raise ValueError(f"Component summary missing columns: {missing}")
    kstar = kstar[keep].copy()
    for col in ["primary_component_number", "n_primary_component_spots", "n_target_spots", "n_target_components"]:
        kstar[col] = pd.to_numeric(kstar[col], errors="coerce")
    for col in METRICS.values():
        kstar[col] = pd.to_numeric(kstar[col], errors="coerce")
    kstar["primary_group_plot"] = kstar["primary_group_id"].map(GROUP_LABELS).fillna(kstar["primary_group_label"])
    kstar["component_key"] = (
        kstar["sample_label"].astype(str)
        + "__"
        + kstar["primary_group_id"].astype(str)
        + "__H"
        + kstar["primary_component_number"].astype("Int64").astype(str)
    )

    source = kstar[
        kstar["contrast_id"].eq("snai1ac_hot_vs_snai1ac_cold")
        & kstar["primary_group_id"].eq(SOURCE_GROUP)
    ].copy()
    source["source_for_plot"] = "source_deduplicated_from_acHot_vs_acCold"
    cold = kstar[
        kstar["contrast_id"].eq("snai1ac_hot_vs_snai1ac_cold")
        & kstar["primary_group_id"].eq("snai1ac_consensus_full_cold")
    ].copy()
    cold["source_for_plot"] = "reference_from_acHot_vs_acCold"
    two_r = kstar[
        kstar["contrast_id"].eq("snai1ac_hot_vs_snai12r_hot")
        & kstar["primary_group_id"].eq("snai12r_full_hot")
    ].copy()
    two_r["source_for_plot"] = "reference_from_acHot_vs_2R"
    plot_long = pd.concat([source, cold, two_r], ignore_index=True)
    plot_long = plot_long.drop_duplicates(
        subset=["sample_label", "target_variable_id", "primary_group_id", "primary_component_number"]
    )
    return plot_long


def test_metric(source_vals: pd.Series, reference_vals: pd.Series) -> tuple[float, float, str]:
    source_vals = pd.to_numeric(source_vals, errors="coerce").dropna()
    reference_vals = pd.to_numeric(reference_vals, errors="coerce").dropna()
    if len(source_vals) < 2 or len(reference_vals) < 2:
        return np.nan, np.nan, "not_tested_lt_2_components_in_one_group"
    delta = float(source_vals.median() - reference_vals.median())
    try:
        p_value = float(mannwhitneyu(source_vals, reference_vals, alternative="two-sided").pvalue)
        status = "ok"
    except ValueError as exc:
        p_value = np.nan
        status = f"not_tested_{exc}"
    return delta, p_value, status


def build_tests(component_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    kstar = component_df[
        component_df["target_family"].eq("kstar")
        & component_df["target_state"].eq("hot")
    ].copy()
    for col in METRICS.values():
        kstar[col] = pd.to_numeric(kstar[col], errors="coerce")
    group_cols = ["sample_label", "dataset", "sample", "target_variable_id", "contrast_id"]
    for keys, group in kstar.groupby(group_cols, dropna=False, sort=True):
        sample_label, dataset, sample, target_variable_id, contrast_id = keys
        reference_group = REFERENCE_GROUPS.get(str(contrast_id))
        if reference_group is None:
            continue
        source = group[group["primary_group_id"].eq(SOURCE_GROUP)].copy()
        reference = group[group["primary_group_id"].eq(reference_group)].copy()
        if source.empty or reference.empty:
            continue
        meta = group.iloc[0].to_dict()
        row = {
            "dataset": dataset,
            "sample": sample,
            "sample_label": sample_label,
            "contrast_id": contrast_id,
            "contrast_label": meta.get("contrast_label", ""),
            "target_variable_id": target_variable_id,
            "target_title": meta.get("target_title", ""),
            "target_display_label": meta.get("target_display_label", ""),
            "kstar_alignment_category_draft": meta.get("kstar_alignment_category_draft", ""),
            "kstar_family_label": meta.get("kstar_family_label", ""),
            "kstar_mp1_8_name": meta.get("kstar_mp1_8_name", ""),
            "n_target_spots": pd.to_numeric(meta.get("n_target_spots"), errors="coerce"),
            "n_target_components": pd.to_numeric(meta.get("n_target_components"), errors="coerce"),
            "source_group_id": SOURCE_GROUP,
            "source_group_label": GROUP_LABELS[SOURCE_GROUP],
            "reference_group_id": reference_group,
            "reference_group_label": GROUP_LABELS[reference_group],
            "n_source_components": int(source["primary_component_number"].nunique()),
            "n_reference_components": int(reference["primary_component_number"].nunique()),
        }
        for metric_name, metric_col in METRICS.items():
            delta, p_value, status = test_metric(source[metric_col], reference[metric_col])
            row[f"delta_{metric_name}"] = delta
            row[f"p_{metric_name}"] = p_value
            row[f"test_status_{metric_name}"] = status
            row[f"source_group_{metric_name}_of_component_values"] = float(source[metric_col].median())
            row[f"reference_group_{metric_name}_of_component_values"] = float(reference[metric_col].median())
        row["direction_primary_median"] = (
            "source_closer"
            if pd.notna(row["delta_median"]) and row["delta_median"] < 0
            else "reference_closer"
            if pd.notna(row["delta_median"]) and row["delta_median"] > 0
            else "tie_or_not_tested"
        )
        rows.append(row)
    tests = pd.DataFrame(rows)
    if tests.empty:
        return tests
    for metric_name in METRICS:
        q_col = f"q_{metric_name}_bh_within_sample_contrast"
        tests[q_col] = np.nan
        for _, idx in tests.groupby(["sample_label", "contrast_id"], dropna=False).groups.items():
            tests.loc[idx, q_col] = bh_adjust(tests.loc[idx, f"p_{metric_name}"])
    tests["significant_median_bh_q05"] = tests["q_median_bh_within_sample_contrast"].lt(0.05)
    tests["nominal_median_p05"] = tests["p_median"].lt(0.05)
    return tests


def draw_sig_text(ax, x_position: int, y_base: float, tests_for_target: pd.DataFrame, y_step: float) -> None:
    labels = []
    for contrast_id, prefix in [
        ("snai1ac_hot_vs_snai1ac_cold", "C"),
        ("snai1ac_hot_vs_snai12r_hot", "2R"),
    ]:
        hit = tests_for_target[tests_for_target["contrast_id"].eq(contrast_id)]
        if hit.empty:
            continue
        row = hit.iloc[0]
        star = p_to_stars(row.get("q_median_bh_within_sample_contrast", np.nan))
        nominal = pd.notna(row.get("p_median", np.nan)) and row.get("p_median", np.nan) < 0.05
        if star:
            labels.append(f"{prefix}{star}")
        elif nominal:
            labels.append(f"{prefix}o")
    if not labels:
        return
    ax.text(
        x_position,
        y_base + y_step,
        " ".join(labels),
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color="black",
        clip_on=False,
    )


def plot_sample(sample_label: str, plot_long: pd.DataFrame, tests: pd.DataFrame) -> Path | None:
    sample_df = plot_long[plot_long["sample_label"].eq(sample_label)].copy()
    if sample_df.empty:
        return None
    sample_tests = tests[tests["sample_label"].eq(sample_label)].copy()
    target_order = (
        sample_tests[sample_tests["contrast_id"].eq("snai1ac_hot_vs_snai1ac_cold")]
        .sort_values(["delta_median", "target_display_label"], ascending=[True, True])["target_variable_id"]
        .drop_duplicates()
        .tolist()
    )
    if not target_order:
        target_order = sorted(sample_df["target_variable_id"].dropna().unique().tolist())
    label_map = (
        sample_df.drop_duplicates("target_variable_id")
        .set_index("target_variable_id")["target_display_label"]
        .to_dict()
    )
    sample_df["target_plot_label"] = sample_df["target_variable_id"].map(lambda x: wrap_label(label_map.get(x, x)))
    order_labels = [wrap_label(label_map.get(t, t)) for t in target_order]
    sample_df["primary_group_plot"] = pd.Categorical(
        sample_df["primary_group_plot"], categories=GROUP_ORDER, ordered=True
    )

    fig_width = max(11.0, 1.05 * len(target_order))
    fig_height = 7.2
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.boxplot(
        data=sample_df,
        x="target_plot_label",
        y="component_median_distance",
        hue="primary_group_plot",
        order=order_labels,
        hue_order=GROUP_ORDER,
        palette=GROUP_COLORS,
        fliersize=0,
        linewidth=0.8,
        width=0.75,
        ax=ax,
    )
    sns.stripplot(
        data=sample_df,
        x="target_plot_label",
        y="component_median_distance",
        hue="primary_group_plot",
        order=order_labels,
        hue_order=GROUP_ORDER,
        palette=GROUP_COLORS,
        dodge=True,
        alpha=0.72,
        size=3.1,
        linewidth=0.2,
        edgecolor="white",
        ax=ax,
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[: len(GROUP_ORDER)], labels[: len(GROUP_ORDER)], frameon=False, loc="upper right")

    y_max = sample_df["component_median_distance"].max()
    y_min = sample_df["component_median_distance"].min()
    y_range = max(1.0, y_max - y_min)
    y_step = 0.04 * y_range
    ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.18 * y_range)
    for idx, target_id in enumerate(target_order):
        draw_sig_text(
            ax,
            idx,
            sample_df.loc[sample_df["target_variable_id"].eq(target_id), "component_median_distance"].max(),
            sample_tests[sample_tests["target_variable_id"].eq(target_id)],
            y_step,
        )

    ax.set_xlabel("")
    ax.set_ylabel("Component-level median nearest-spot distance to K* hotspot")
    ax.set_title(f"{sample_label}: K* hotspot proximity to SNAI1-ac source/reference components")
    ax.tick_params(axis="x", labelrotation=70)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.text(
        0.01,
        0.99,
        "C* = SNAI1ac_hot vs SNAI1ac_cold BH q<0.05; 2R* = SNAI1ac_hot vs SNAI1-2R_hot BH q<0.05; o = nominal p<0.05 only",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2},
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    out = FIG_ROOT / f"kstar_component_distance__{safe_name(sample_label)}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def write_readme(figures: list[Path], tests: pd.DataFrame) -> None:
    n_sig = int(tests["significant_median_bh_q05"].sum()) if "significant_median_bh_q05" in tests else 0
    lines = [
        "# K* Component-Level Distance Preflight",
        "",
        f"Generated: {now_iso()}",
        "",
        "This layer tests sample-specific K* hotspot proximity. It is not a cross-sample GEE layer because raw K* programs are sample-local.",
        "",
        "Primary report-facing statistic:",
        "",
        "- component-level median nearest-spot distance to each K* hotspot",
        "- `delta_median = median(SNAI1ac_hot component distances) - median(reference component distances)`",
        "- negative delta means the K* hotspot is closer to `SNAI1ac_hot` than to the reference",
        "",
        "Primary test:",
        "",
        "- two-sided Mann-Whitney U test on component-level median distances",
        "- BH correction within each `sample_label + contrast_id` across K* programs",
        "- `q_median_bh_within_sample_contrast < 0.05` is the report-facing significance flag",
        "",
        "Additional table fields include mean and minimum component-distance effects/tests as sensitivity.",
        "",
        "Plot annotations:",
        "",
        "- `C*`, `C**`, `C***`: SNAI1ac_hot vs SNAI1ac_cold BH-significant at q<0.05/0.01/0.001",
        "- `2R*`, `2R**`, `2R***`: SNAI1ac_hot vs SNAI1-2R_hot BH-significant at q<0.05/0.01/0.001",
        "- `Co` or `2Ro`: nominal p<0.05 only",
        "",
        "Outputs:",
        "",
        f"- long component rows: `{LONG_OUT}`",
        f"- test table: `{TESTS_OUT}`",
        f"- figures: {len(figures)} PNG files in `{FIG_ROOT}`",
        f"- BH-significant median tests: {n_sig}",
        "",
    ]
    README_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    SCRIPT_ROOT.mkdir(parents=True, exist_ok=True)

    component_df = pd.read_csv(COMPONENT_SUMMARY, low_memory=False)
    long_df = build_long(component_df)
    tests = build_tests(component_df)
    long_df.to_csv(LONG_OUT, index=False)
    tests.to_csv(TESTS_OUT, index=False)

    figures: list[Path] = []
    for sample_label in sorted(long_df["sample_label"].dropna().unique()):
        path = plot_sample(sample_label, long_df, tests)
        if path is not None:
            figures.append(path)

    write_readme(figures, tests)
    shutil.copy2(Path(__file__), SCRIPT_ROOT / Path(__file__).name)
    manifest = {
        "generated_at": now_iso(),
        "input_component_summary": str(COMPONENT_SUMMARY),
        "outputs": {
            "long_component_table": str(LONG_OUT),
            "test_table": str(TESTS_OUT),
            "figure_root": str(FIG_ROOT),
            "figures": [str(path) for path in figures],
            "readme": str(README_OUT),
        },
        "n_component_long_rows": int(long_df.shape[0]),
        "n_test_rows": int(tests.shape[0]),
        "n_samples": int(long_df["sample_label"].nunique()),
        "n_figures": len(figures),
        "primary_metric": "component_median_distance",
        "primary_test": "two_sided_mannwhitneyu_component_level",
        "bh_family": "within sample_label + contrast_id across K* programs",
        "kstar_cross_sample_pooling": "not_used_raw_Kstar_programs_are_sample_specific",
    }
    RUN_MANIFEST_OUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
