from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"D:\HGSOC_Spatial_Atlas\05_analysis_ready\SpottedPy_v2_paper_aligned")
CODE_ROOT = Path(r"C:\Users\luchu\Documents\MSc\Master Thesis\Code\HGSOC_Spatial_Atlas")
DIST_ROOT = PROJECT_ROOT / "05_distance_gee"
TABLE_ROOT = DIST_ROOT / "tables"
FIG_ROOT = DIST_ROOT / "figures"
FULL_ROOT = DIST_ROOT / "full_distance_gee"
FULL_TABLE_ROOT = FULL_ROOT / "tables"
FULL_FIG_ROOT = FULL_ROOT / "figures"
FULL_SCRIPT_ROOT = FULL_ROOT / "scripts_used"
DIST_SCRIPT_ROOT = DIST_ROOT / "scripts_used"

NATIVE_FIG_ROOT = FIG_ROOT / "preflight_spottedpy_native"
KSTAR_FIG_ROOT = FIG_ROOT / "kstar_component_distance_preflight"

NATIVE_STATS = TABLE_ROOT / "spottedpy_v2_distance_spottedpy_native_gee_stats_preflight.csv"
NATIVE_LONG = TABLE_ROOT / "spottedpy_v2_distance_spottedpy_native_long_preflight.csv"
HALLMARK_GROUP_MAP = TABLE_ROOT / "spottedpy_v2_distance_hallmark_group_map_preflight.csv"
HALLMARK_BATCH_INCLUSION = TABLE_ROOT / "spottedpy_v2_distance_hallmark_batch_plot_inclusion_preflight.csv"
KSTAR_TESTS = TABLE_ROOT / "spottedpy_v2_kstar_component_distance_tests_preflight.csv"
KSTAR_LONG = TABLE_ROOT / "spottedpy_v2_kstar_component_distance_long_preflight.csv"
COMPONENT_SUMMARY = TABLE_ROOT / "spottedpy_v2_distance_component_summary_preflight.csv"
SAMPLE_TARGET_SUMMARY = TABLE_ROOT / "spottedpy_v2_distance_sample_target_summary_preflight.csv"
DESCRIPTIVE_SUMMARY = TABLE_ROOT / "spottedpy_v2_distance_cross_sample_descriptive_summary_preflight.csv"

PAPER_AUDIT = (
    Path(r"C:\Users\luchu\Documents\MSc\Master Thesis")
    / "tmp"
    / "pdfs"
    / "spottedpy_paper_tutorial_distance_audit.md"
)
PAPER_ALIGNMENT_OUT = DIST_ROOT / "paper_alignment_go_no_go_distance_full.md"
RUN_MANIFEST_OUT = DIST_ROOT / "run_manifest_distance_full_promotion.json"
README_OUT = FULL_ROOT / "README_distance_full_gee.md"

CONTRAST_LABELS = {
    "snai1ac_hot_vs_snai1ac_cold": "SNAI1-ac consensus hot vs SNAI1-ac consensus cold",
    "snai1ac_hot_vs_snai12r_hot": "SNAI1-ac consensus hot vs SNAI1-2R hot",
}

FINAL_TABLES = {
    "native_stats": FULL_TABLE_ROOT / "distance_full_cohort_gee_stats.csv",
    "native_significant_raw": FULL_TABLE_ROOT / "distance_full_cohort_gee_significant_raw_p05.csv",
    "native_significant_bh": FULL_TABLE_ROOT / "distance_full_cohort_gee_significant_bh_q05.csv",
    "native_long": FULL_TABLE_ROOT / "distance_full_spottedpy_native_long.csv",
    "hallmark_shortlist_raw": FULL_TABLE_ROOT / "distance_full_hallmark_shortlist_raw_p05.csv",
    "hallmark_shortlist_bh": FULL_TABLE_ROOT / "distance_full_hallmark_shortlist_bh_q05.csv",
    "hallmark_group_map": FULL_TABLE_ROOT / "distance_full_hallmark_group_map.csv",
    "hallmark_batch_inclusion": FULL_TABLE_ROOT / "distance_full_hallmark_batch_plot_inclusion.csv",
    "kstar_tests": FULL_TABLE_ROOT / "distance_full_kstar_per_sample_component_tests.csv",
    "kstar_significant_bh": FULL_TABLE_ROOT / "distance_full_kstar_per_sample_component_significant_bh_q05.csv",
    "kstar_nominal": FULL_TABLE_ROOT / "distance_full_kstar_per_sample_component_nominal_p05.csv",
    "kstar_long": FULL_TABLE_ROOT / "distance_full_kstar_per_sample_component_long.csv",
    "input_manifest": FULL_TABLE_ROOT / "distance_full_input_table_manifest.csv",
    "summary_counts": FULL_TABLE_ROOT / "distance_full_summary_counts.csv",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file is missing: {path}")


def row_count(path: Path) -> int:
    if not path.exists():
        return 0
    return int(pd.read_csv(path, usecols=[0]).shape[0])


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


def direction_from_delta(delta: float) -> str:
    if pd.isna(delta):
        return "not_available"
    if delta < 0:
        return "SNAI1ac_hot_closer"
    if delta > 0:
        return "reference_closer"
    return "tie"


def prepare_dirs() -> None:
    for path in [
        FULL_ROOT,
        FULL_TABLE_ROOT,
        FULL_FIG_ROOT / "gee",
        FULL_FIG_ROOT / "mean_bubble",
        FULL_FIG_ROOT / "batch_bubble",
        FULL_FIG_ROOT / "kstar_component_distance",
        FULL_SCRIPT_ROOT,
        DIST_SCRIPT_ROOT,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def annotate_native_stats() -> pd.DataFrame:
    stats = pd.read_csv(NATIVE_STATS)
    required = [
        "comparison_variable",
        "p_value",
        "difference",
        "contrast_id",
        "target_family",
        "plot_scope",
        "hallmark_group",
    ]
    missing = [col for col in required if col not in stats.columns]
    if missing:
        raise ValueError(f"Native GEE stats table missing columns: {missing}")
    stats["p_value"] = pd.to_numeric(stats["p_value"], errors="coerce")
    stats["difference"] = pd.to_numeric(stats["difference"], errors="coerce")
    stats["contrast_label"] = stats["contrast_id"].map(CONTRAST_LABELS).fillna(stats["contrast_id"])
    stats["bh_family"] = (
        stats["contrast_id"].astype(str)
        + "|"
        + stats["target_family"].astype(str)
        + "|"
        + stats["plot_scope"].astype(str)
    )
    stats["q_value_bh_within_contrast_family_scope"] = stats.groupby("bh_family", group_keys=False)[
        "p_value"
    ].apply(bh_adjust)
    stats["significant_raw_p05"] = stats["p_value"] < 0.05
    stats["significant_bh_q05"] = stats["q_value_bh_within_contrast_family_scope"] < 0.05
    stats["distance_effect_direction"] = stats["difference"].map(direction_from_delta)
    stats["distance_effect_note"] = (
        "difference < 0 means the target hotspot is closer to SNAI1-ac consensus hot "
        "than to the reference group for that contrast"
    )
    return stats


def write_native_tables(stats: pd.DataFrame) -> dict[str, int]:
    hallmark_map = pd.read_csv(HALLMARK_GROUP_MAP)
    stats.to_csv(FINAL_TABLES["native_stats"], index=False)
    stats[stats["significant_raw_p05"]].to_csv(FINAL_TABLES["native_significant_raw"], index=False)
    stats[stats["significant_bh_q05"]].to_csv(FINAL_TABLES["native_significant_bh"], index=False)
    shutil.copy2(NATIVE_LONG, FINAL_TABLES["native_long"])
    shutil.copy2(HALLMARK_GROUP_MAP, FINAL_TABLES["hallmark_group_map"])
    shutil.copy2(HALLMARK_BATCH_INCLUSION, FINAL_TABLES["hallmark_batch_inclusion"])

    hallmark_family = stats[
        stats["target_family"].eq("hallmark") & stats["plot_scope"].eq("family_all")
    ].copy()
    hallmark_family = hallmark_family.merge(
        hallmark_map[["comparison_variable", "hallmark_group"]],
        on="comparison_variable",
        how="left",
        suffixes=("", "_manifest"),
    )
    if "hallmark_group_manifest" in hallmark_family.columns:
        hallmark_family["hallmark_group"] = hallmark_family["hallmark_group"].replace("", np.nan)
        hallmark_family["hallmark_group"] = hallmark_family["hallmark_group"].fillna(
            hallmark_family["hallmark_group_manifest"]
        )
        hallmark_family = hallmark_family.drop(columns=["hallmark_group_manifest"])
    hallmark_family[hallmark_family["significant_raw_p05"]].to_csv(
        FINAL_TABLES["hallmark_shortlist_raw"], index=False
    )
    hallmark_family[hallmark_family["significant_bh_q05"]].to_csv(
        FINAL_TABLES["hallmark_shortlist_bh"], index=False
    )
    return {
        "n_native_stats_rows": len(stats),
        "n_native_raw_p05_rows": int(stats["significant_raw_p05"].sum()),
        "n_native_bh_q05_rows": int(stats["significant_bh_q05"].sum()),
        "n_hallmark_family_raw_p05_rows": int(hallmark_family["significant_raw_p05"].sum()),
        "n_hallmark_family_bh_q05_rows": int(hallmark_family["significant_bh_q05"].sum()),
    }


def write_kstar_tables() -> dict[str, int]:
    kstar = pd.read_csv(KSTAR_TESTS)
    for col in ["p_median", "q_median_bh_within_sample_contrast", "delta_median"]:
        if col in kstar.columns:
            kstar[col] = pd.to_numeric(kstar[col], errors="coerce")
    if "significant_median_bh_q05" not in kstar.columns:
        kstar["significant_median_bh_q05"] = kstar["q_median_bh_within_sample_contrast"] < 0.05
    else:
        kstar["significant_median_bh_q05"] = kstar["significant_median_bh_q05"].astype(str).str.lower().eq("true")
    if "nominal_median_p05" not in kstar.columns:
        kstar["nominal_median_p05"] = kstar["p_median"] < 0.05
    else:
        kstar["nominal_median_p05"] = kstar["nominal_median_p05"].astype(str).str.lower().eq("true")
    kstar["primary_report_metric"] = "component_median_distance"
    kstar["primary_report_test"] = "two_sided_mannwhitneyu_component_level"
    kstar["primary_bh_family"] = "within sample_label + contrast_id across K* programs"
    kstar["distance_effect_note"] = (
        "delta_median < 0 means the K* hotspot is closer to SNAI1-ac consensus hot "
        "than to the reference group in that sample"
    )
    kstar.to_csv(FINAL_TABLES["kstar_tests"], index=False)
    kstar[kstar["significant_median_bh_q05"]].to_csv(FINAL_TABLES["kstar_significant_bh"], index=False)
    kstar[kstar["nominal_median_p05"]].to_csv(FINAL_TABLES["kstar_nominal"], index=False)
    shutil.copy2(KSTAR_LONG, FINAL_TABLES["kstar_long"])
    return {
        "n_kstar_tests_rows": len(kstar),
        "n_kstar_nominal_p05_rows": int(kstar["nominal_median_p05"].sum()),
        "n_kstar_bh_q05_rows": int(kstar["significant_median_bh_q05"].sum()),
        "n_kstar_samples": int(kstar["sample_label"].nunique()) if "sample_label" in kstar.columns else 0,
    }


def copy_figures() -> dict[str, int]:
    copied = {"gee": 0, "mean_bubble": 0, "batch_bubble": 0, "kstar_component_distance": 0}
    for source in sorted(NATIVE_FIG_ROOT.glob("gee__*.png")):
        shutil.copy2(source, FULL_FIG_ROOT / "gee" / source.name)
        copied["gee"] += 1
    for source in sorted(NATIVE_FIG_ROOT.glob("mean__*.png")):
        shutil.copy2(source, FULL_FIG_ROOT / "mean_bubble" / source.name)
        copied["mean_bubble"] += 1
    for source in sorted(NATIVE_FIG_ROOT.glob("batch__*.png")):
        shutil.copy2(source, FULL_FIG_ROOT / "batch_bubble" / source.name)
        copied["batch_bubble"] += 1
    for source in sorted(KSTAR_FIG_ROOT.glob("*.png")):
        shutil.copy2(source, FULL_FIG_ROOT / "kstar_component_distance" / source.name)
        copied["kstar_component_distance"] += 1

    for source, dest in [
        (
            NATIVE_FIG_ROOT / "README_spottedpy_native_preflight.md",
            FULL_FIG_ROOT / "README_spottedpy_native_figures.md",
        ),
        (
            KSTAR_FIG_ROOT / "README_kstar_component_distance_preflight.md",
            FULL_FIG_ROOT / "README_kstar_component_distance_figures.md",
        ),
    ]:
        if source.exists():
            shutil.copy2(source, dest)
    return copied


def write_input_manifest() -> pd.DataFrame:
    rows = []
    for label, path in [
        ("native_gee_stats_preflight", NATIVE_STATS),
        ("native_long_preflight", NATIVE_LONG),
        ("hallmark_group_map_preflight", HALLMARK_GROUP_MAP),
        ("hallmark_batch_plot_inclusion_preflight", HALLMARK_BATCH_INCLUSION),
        ("kstar_component_tests_preflight", KSTAR_TESTS),
        ("kstar_component_long_preflight", KSTAR_LONG),
        ("component_summary_preflight", COMPONENT_SUMMARY),
        ("sample_target_summary_preflight", SAMPLE_TARGET_SUMMARY),
        ("cross_sample_descriptive_summary_preflight", DESCRIPTIVE_SUMMARY),
    ]:
        rows.append(
            {
                "input_label": label,
                "path": str(path),
                "exists": path.exists(),
                "n_rows": row_count(path),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(FINAL_TABLES["input_manifest"], index=False)
    return out


def write_summary_counts(native_counts: dict[str, int], kstar_counts: dict[str, int], fig_counts: dict[str, int]) -> None:
    rows = []
    for source, counts in [
        ("cohort_gee", native_counts),
        ("kstar_component", kstar_counts),
        ("figures", fig_counts),
    ]:
        for metric, value in counts.items():
            rows.append({"source": source, "metric": metric, "value": value})
    pd.DataFrame(rows).to_csv(FINAL_TABLES["summary_counts"], index=False)


def write_readme(native_counts: dict[str, int], kstar_counts: dict[str, int], fig_counts: dict[str, int]) -> None:
    text = f"""# SpottedPy v2 full distance/GEE branch

Generated: {now_iso()}

This folder promotes the reviewed distance preflight into the report-facing distance branch.
It does not change the biological design or recalculate hotspot distances. The source tables are the vetted preflight outputs in `{TABLE_ROOT}`.

## Design

- Source region: full-slide SNAI1-ac consensus hot.
- Reference regions: full-slide SNAI1-ac consensus cold and full-slide SNAI1-2R hot.
- Distance targets: hotspots only. Target coldspots are excluded.
- Cohort-level GEE layer: MP1-MP8, SpaCET fractions, and Hallmark hotspots.
- K* layer: per-sample component-level tests only; K* programs are not treated as comparable variables for cohort-level GEE.
- Primary K* report metric: component-level median nearest-spot distance.

For cohort GEE and K* component tests, negative deltas/differences mean the target hotspot is closer to SNAI1-ac consensus hot than to the reference region for that contrast.

## Tables

- `tables/distance_full_cohort_gee_stats.csv`: cohort GEE table with raw p-values, BH q-values within contrast/family/scope, and effect direction.
- `tables/distance_full_cohort_gee_significant_raw_p05.csv`: raw p < 0.05 cohort GEE rows.
- `tables/distance_full_cohort_gee_significant_bh_q05.csv`: BH q < 0.05 cohort GEE rows.
- `tables/distance_full_hallmark_shortlist_raw_p05.csv`: Hallmark family-all rows with raw p < 0.05.
- `tables/distance_full_hallmark_shortlist_bh_q05.csv`: Hallmark family-all rows with BH q < 0.05.
- `tables/distance_full_kstar_per_sample_component_tests.csv`: per-sample K* component-level median/mean/min tests.
- `tables/distance_full_kstar_per_sample_component_significant_bh_q05.csv`: K* tests passing BH q < 0.05 within sample and contrast.
- `tables/distance_full_input_table_manifest.csv`: source input table inventory.

## Figures

- `figures/gee`: SpottedPy-style cohort GEE scatter/differential-distance plots.
- `figures/mean_bubble`: descriptive cohort mean-distance bubble plots.
- `figures/batch_bubble`: per-slide bubble diagnostics; black outline denotes raw p < 0.05 and stars denote p < 0.05 / number of slides.
- `figures/kstar_component_distance`: one per-sample K* component-distance box/strip plot.

## Counts

- Cohort GEE rows: {native_counts["n_native_stats_rows"]}
- Cohort raw p < 0.05 rows: {native_counts["n_native_raw_p05_rows"]}
- Cohort BH q < 0.05 rows: {native_counts["n_native_bh_q05_rows"]}
- K* tests: {kstar_counts["n_kstar_tests_rows"]}
- K* nominal p < 0.05 tests: {kstar_counts["n_kstar_nominal_p05_rows"]}
- K* BH q < 0.05 tests: {kstar_counts["n_kstar_bh_q05_rows"]}
- Figures copied: {sum(fig_counts.values())}
"""
    README_OUT.write_text(text, encoding="utf-8")


def write_paper_alignment(native_counts: dict[str, int], kstar_counts: dict[str, int], fig_counts: dict[str, int]) -> None:
    text = f"""# SpottedPy distance/GEE paper-alignment go/no-go

Generated: {now_iso()}

Status: GO for the full distance/GEE branch.

## Paper/tutorial anchors checked

- Figure 1 and the distance-statistics text: SpottedPy measures nearest hotspot distances and compares source/reference hotspot regions.
- Figures 2-4: distance analysis is used as the discovery layer after hotspots are defined.
- Figure 6: relevant to the completed neighborhood source-group comparison logic, not a replacement for distance/GEE.
- Methods/tutorial: Visium hotspot scale k=10 is paper-aligned, k=8 is retained as near-neighbor sensitivity, and GEE is the cross-slide statistical layer for comparable variables.
- Local implementation audit: SpottedPy `calculateDistances` uses nearest distances to hotspot spots; `plot_custom_scatter` summarizes by slide/hotspot and fits GEE for comparable target variables.

Detailed local reread notes: `{PAPER_AUDIT}`

## Our locked design

- Source: full-slide SNAI1-ac consensus hot.
- References: full-slide SNAI1-ac consensus cold and full-slide SNAI1-2R hot.
- Targets: hotspot regions only for MP1-MP8, SpaCET, Hallmark, and K*.
- Target coldspots: excluded from distance analysis.
- MP/SpaCET/Hallmark: comparable across slides, therefore eligible for cohort-level SpottedPy/GEE summary.
- K*: sample-specific program identities, therefore summarized per sample with component-level tests, not cohort-level GEE.
- Primary K* report-facing statistic: component-level median nearest-spot distance; mean and minimum are retained in the table as support/sensitivity.

## Deferred

- All-in-one neighborhood correlation remains a possible validation layer.
- Scale/sensitivity figures are not part of this promotion pass.
- K* cohort-level GEE is not performed because K* variables are not one shared variable universe across slides.

## Promotion outputs

- Full branch: `{FULL_ROOT}`
- Cohort GEE rows: {native_counts["n_native_stats_rows"]}
- Cohort raw p < 0.05 rows: {native_counts["n_native_raw_p05_rows"]}
- Cohort BH q < 0.05 rows: {native_counts["n_native_bh_q05_rows"]}
- K* tests: {kstar_counts["n_kstar_tests_rows"]}
- K* BH q < 0.05 tests: {kstar_counts["n_kstar_bh_q05_rows"]}
- Figures copied: {sum(fig_counts.values())}
"""
    PAPER_ALIGNMENT_OUT.write_text(text, encoding="utf-8")


def copy_scripts() -> None:
    for script_name in [
        "spottedpy_v2_distance_full_promotion.py",
        "spottedpy_v2_distance_spottedpy_native_preflight.py",
        "spottedpy_v2_kstar_component_distance_tests_preflight.py",
        "spottedpy_paper_tutorial_distance_audit.py",
    ]:
        source = CODE_ROOT / "scripts" / script_name
        if source.exists():
            shutil.copy2(source, FULL_SCRIPT_ROOT / script_name)
            shutil.copy2(source, DIST_SCRIPT_ROOT / script_name)


def write_manifest(
    native_counts: dict[str, int],
    kstar_counts: dict[str, int],
    fig_counts: dict[str, int],
    input_manifest: pd.DataFrame,
) -> None:
    manifest = {
        "generated_at": now_iso(),
        "status": "full_distance_gee_promoted_from_reviewed_preflight",
        "full_root": str(FULL_ROOT),
        "paper_alignment_go_no_go": str(PAPER_ALIGNMENT_OUT),
        "readme": str(README_OUT),
        "analysis_policy": {
            "source_group": "full-slide SNAI1-ac consensus hot",
            "reference_groups": [
                "full-slide SNAI1-ac consensus cold",
                "full-slide SNAI1-2R hot",
            ],
            "target_state": "hotspots only",
            "cohort_gee_families": ["mp", "spacet", "hallmark"],
            "kstar_policy": "per-sample component-level tests only; no cohort GEE",
            "primary_kstar_metric": "component_median_distance",
            "kstar_bh_family": "within sample_label + contrast_id across K* programs",
        },
        "counts": {
            **native_counts,
            **kstar_counts,
            **{f"n_figures_{key}": value for key, value in fig_counts.items()},
            "n_input_tables": int(input_manifest.shape[0]),
            "n_missing_input_tables": int((~input_manifest["exists"]).sum()),
        },
        "tables": {key: str(path) for key, path in FINAL_TABLES.items()},
        "figure_dirs": {
            "gee": str(FULL_FIG_ROOT / "gee"),
            "mean_bubble": str(FULL_FIG_ROOT / "mean_bubble"),
            "batch_bubble": str(FULL_FIG_ROOT / "batch_bubble"),
            "kstar_component_distance": str(FULL_FIG_ROOT / "kstar_component_distance"),
        },
    }
    RUN_MANIFEST_OUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (FULL_ROOT / "run_manifest_distance_full_promotion.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> None:
    for path in [
        NATIVE_STATS,
        NATIVE_LONG,
        HALLMARK_GROUP_MAP,
        HALLMARK_BATCH_INCLUSION,
        KSTAR_TESTS,
        KSTAR_LONG,
        COMPONENT_SUMMARY,
        SAMPLE_TARGET_SUMMARY,
        DESCRIPTIVE_SUMMARY,
        NATIVE_FIG_ROOT,
        KSTAR_FIG_ROOT,
    ]:
        require_file(path)
    prepare_dirs()
    input_manifest = write_input_manifest()
    if not bool(input_manifest["exists"].all()):
        missing = input_manifest.loc[~input_manifest["exists"], "path"].tolist()
        raise FileNotFoundError(f"Missing input tables: {missing}")
    native_stats = annotate_native_stats()
    native_counts = write_native_tables(native_stats)
    kstar_counts = write_kstar_tables()
    fig_counts = copy_figures()
    write_summary_counts(native_counts, kstar_counts, fig_counts)
    write_readme(native_counts, kstar_counts, fig_counts)
    write_paper_alignment(native_counts, kstar_counts, fig_counts)
    copy_scripts()
    write_manifest(native_counts, kstar_counts, fig_counts, input_manifest)
    print(
        json.dumps(
            {
                "status": "ok",
                "generated_at": now_iso(),
                "full_root": str(FULL_ROOT),
                "paper_alignment_go_no_go": str(PAPER_ALIGNMENT_OUT),
                "counts": {**native_counts, **kstar_counts, **fig_counts},
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
