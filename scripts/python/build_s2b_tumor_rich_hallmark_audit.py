"""Build report-facing Hallmark summaries for the S2b tumor-rich threshold analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from plot_s2b_step4_hallmark_group_positions import (
    GROUP_TITLES,
    HALLMARK_GROUPS,
    LABELS,
    REMAINING_HALLMARK_GROUPS,
)


DEFAULT_INPUT = Path(
    r"D:\HGSOC_Spatial_Atlas\05_analysis_ready\S2b_CellType_Composition_Correlation"
    r"\step4_threshold_high_low_diagnostics\threshold_10_to_100_positions"
    r"\threshold_high_vs_low_cohens_d_per_sample.csv"
)
DEFAULT_CLINICAL = Path(
    r"D:\HGSOC_Spatial_Atlas\03_metadata\clinical_annotations\clinical annotations.xlsx"
)
CRS_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90]


def holm_adjust(p_values: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().sort_values()
    m = len(valid)
    running_max = 0.0
    for rank, (idx, p_value) in enumerate(valid.items(), start=1):
        running_max = max(running_max, min((m - rank + 1) * float(p_value), 1.0))
        out.loc[idx] = running_max
    return out


def iqr(values: pd.Series) -> float:
    return float(values.quantile(0.75) - values.quantile(0.25))


def normalize_dataset(value: object) -> str:
    text = str(value).strip().lower()
    if "denisenko" in text:
        return "denisenko_2022"
    if "ju" in text:
        return "ju_2024"
    if "yamamoto" in text:
        return "yamamoto_2025"
    return text


def normalize_sample(value: object) -> str:
    return str(value).split(" (", 1)[0].strip()


def load_hallmark_source(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path)
    required = {
        "dataset",
        "sample",
        "variable",
        "variable_type",
        "malignant_threshold",
        "low_mean",
        "high_mean",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise RuntimeError(f"Missing required columns in {path}: {missing}")
    hall = source[source["variable_type"].eq("hallmark")].copy()
    hall["malignant_threshold"] = pd.to_numeric(hall["malignant_threshold"], errors="raise")
    hall["low_mean"] = pd.to_numeric(hall["low_mean"], errors="raise")
    hall["high_mean"] = pd.to_numeric(hall["high_mean"], errors="raise")
    hall["high_minus_low"] = hall["high_mean"] - hall["low_mean"]
    if hall["variable"].nunique() != 50:
        raise RuntimeError(f"Expected 50 Hallmark pathways, found {hall['variable'].nunique()}")
    return hall


def build_threshold_tests(hall: pd.DataFrame) -> pd.DataFrame:
    all_groups = HALLMARK_GROUPS | REMAINING_HALLMARK_GROUPS
    family_lookup = {
        variable: family for family, variables in all_groups.items() for variable in variables
    }
    rows: list[dict[str, object]] = []
    for (variable, threshold), group in hall.groupby(
        ["variable", "malignant_threshold"], sort=True
    ):
        high = group["high_mean"]
        low = group["low_mean"]
        diff = high - low
        p_value = float(stats.wilcoxon(high, low, zero_method="wilcox").pvalue)
        rows.append(
            {
                "pathway": variable,
                "pathway_label": LABELS[variable],
                "pathway_family": family_lookup[variable],
                "pathway_family_label": GROUP_TITLES[family_lookup[variable]],
                "malignant_threshold": float(threshold),
                "n_sections": int(len(group)),
                "high_median": float(high.median()),
                "high_q25": float(high.quantile(0.25)),
                "high_q75": float(high.quantile(0.75)),
                "high_iqr": iqr(high),
                "low_median": float(low.median()),
                "low_q25": float(low.quantile(0.25)),
                "low_q75": float(low.quantile(0.75)),
                "low_iqr": iqr(low),
                "group_median_high_minus_low": float(high.median() - low.median()),
                "paired_median_high_minus_low": float(diff.median()),
                "paired_diff_q25": float(diff.quantile(0.25)),
                "paired_diff_q75": float(diff.quantile(0.75)),
                "direction_consistency_high_gt_low": float((diff > 0).mean()),
                "paired_wilcoxon_p": p_value,
            }
        )
    tests = pd.DataFrame(rows)
    tests["paired_wilcoxon_p_holm_all_500"] = holm_adjust(tests["paired_wilcoxon_p"])
    tests["paired_wilcoxon_p_holm_within_family_50"] = tests.groupby(
        "pathway_family", group_keys=False
    )["paired_wilcoxon_p"].apply(holm_adjust)
    return tests


def build_directional_stability(tests: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (variable, label, family), group in tests.groupby(
        ["pathway", "pathway_label", "pathway_family"], sort=True
    ):
        group = group.sort_values("malignant_threshold")
        group_sign = np.sign(group["group_median_high_minus_low"].to_numpy())
        paired_sign = np.sign(group["paired_median_high_minus_low"].to_numpy())
        group_positive_n = int((group_sign > 0).sum())
        group_negative_n = int((group_sign < 0).sum())
        paired_positive_n = int((paired_sign > 0).sum())
        paired_negative_n = int((paired_sign < 0).sum())
        if group_positive_n == 10:
            direction = 1
            paired_same_n = paired_positive_n
        elif group_negative_n == 10:
            direction = -1
            paired_same_n = paired_negative_n
        else:
            direction = 0
            paired_same_n = 0
        strict = direction != 0 and paired_same_n == 10
        near_stable = direction != 0 and paired_same_n == 9
        rows.append(
            {
                "pathway": variable,
                "pathway_label": label,
                "pathway_family": family,
                "group_median_positive_thresholds": group_positive_n,
                "group_median_negative_thresholds": group_negative_n,
                "paired_median_positive_thresholds": paired_positive_n,
                "paired_median_negative_thresholds": paired_negative_n,
                "shared_direction_thresholds": paired_same_n,
                "direction": "HIGH_gt_LOW" if direction == 1 else "HIGH_lt_LOW" if direction == -1 else "mixed",
                "strict_10_of_10": bool(strict),
                "near_stable_9_of_10": bool(near_stable),
                "screen_class": "strict" if strict else "near_stable" if near_stable else "none",
            }
        )
    return pd.DataFrame(rows)


def build_dispersion_summary(tests: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold, group in tests.groupby("malignant_threshold", sort=True):
        difference = group["high_iqr"] - group["low_iqr"]
        rows.append(
            {
                "malignant_threshold": float(threshold),
                "n_pathways": int(len(group)),
                "median_high_iqr": float(group["high_iqr"].median()),
                "median_low_iqr": float(group["low_iqr"].median()),
                "median_high_minus_low_iqr": float(difference.median()),
                "n_pathways_high_iqr_gt_low": int((difference > 0).sum()),
                "fraction_pathways_high_iqr_gt_low": float((difference > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def load_crs_metadata(path: Path) -> pd.DataFrame:
    clinical = pd.read_excel(path)
    required = {"dataset", "sample", "treatment response (CRS, 3 tier)"}
    missing = sorted(required - set(clinical.columns))
    if missing:
        raise RuntimeError(f"Missing required columns in {path}: {missing}")
    clinical["dataset_norm"] = clinical["dataset"].map(normalize_dataset)
    clinical["sample_norm"] = clinical["sample"].map(normalize_sample)
    clinical["crs_numeric"] = pd.to_numeric(
        clinical["treatment response (CRS, 3 tier)"], errors="coerce"
    )
    return clinical[["dataset_norm", "sample_norm", "crs_numeric"]].drop_duplicates()


def build_crs_screen(hall: pd.DataFrame, clinical_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    clinical = load_crs_metadata(clinical_path)
    den = hall[
        hall["dataset"].eq("denisenko_2022")
        & hall["malignant_threshold"].isin(CRS_THRESHOLDS)
    ].merge(
        clinical,
        left_on=["dataset", "sample"],
        right_on=["dataset_norm", "sample_norm"],
        how="left",
    )
    den = den[den["crs_numeric"].notna()].copy()
    sample_values = (
        den.groupby(["dataset", "sample", "variable", "crs_numeric"], as_index=False)
        .agg(
            high_group_mean_across_thresholds=("high_mean", "mean"),
            high_minus_low_mean_across_thresholds=("high_minus_low", "mean"),
            n_thresholds=("malignant_threshold", "nunique"),
        )
    )
    long_values = sample_values.melt(
        id_vars=["dataset", "sample", "variable", "crs_numeric", "n_thresholds"],
        value_vars=[
            "high_group_mean_across_thresholds",
            "high_minus_low_mean_across_thresholds",
        ],
        var_name="metric",
        value_name="metric_value",
    )
    rows = []
    for (metric, variable), group in long_values.groupby(["metric", "variable"], sort=True):
        rho, p_value = stats.spearmanr(group["crs_numeric"], group["metric_value"])
        rows.append(
            {
                "metric": metric,
                "pathway": variable,
                "pathway_label": LABELS[variable],
                "thresholds_averaged": "0.50;0.60;0.70;0.80;0.90",
                "n_sections": int(len(group)),
                "n_unique_crs_values": int(group["crs_numeric"].nunique()),
                "spearman_rho": float(rho),
                "spearman_p": float(p_value),
            }
        )
    screen = pd.DataFrame(rows)
    screen["spearman_p_holm_all_100"] = holm_adjust(screen["spearman_p"])
    return sample_values, screen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--clinical-xlsx", type=Path, default=DEFAULT_CLINICAL)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hall = load_hallmark_source(args.input_csv)
    tests = build_threshold_tests(hall)
    stability = build_directional_stability(tests)
    dispersion = build_dispersion_summary(tests)
    crs_values, crs_screen = build_crs_screen(hall, args.clinical_xlsx)

    outputs = {
        "tests": args.output_dir / "tumor_rich_hallmark_threshold_tests.csv",
        "stability": args.output_dir / "tumor_rich_hallmark_directional_stability.csv",
        "dispersion": args.output_dir / "tumor_rich_hallmark_dispersion_summary.csv",
        "crs_values": args.output_dir / "tumor_rich_hallmark_crs_sample_values.csv",
        "crs_screen": args.output_dir / "tumor_rich_hallmark_crs_screen.csv",
    }
    tests.to_csv(outputs["tests"], index=False)
    stability.to_csv(outputs["stability"], index=False)
    dispersion.to_csv(outputs["dispersion"], index=False)
    crs_values.to_csv(outputs["crs_values"], index=False)
    crs_screen.to_csv(outputs["crs_screen"], index=False)

    manifest = {
        "inputs": {"threshold_source": str(args.input_csv), "clinical_workbook": str(args.clinical_xlsx)},
        "dimensions": {
            "sections": int(hall[["dataset", "sample"]].drop_duplicates().shape[0]),
            "hallmark_pathways": int(hall["variable"].nunique()),
            "thresholds": sorted(hall["malignant_threshold"].unique().tolist()),
            "primary_tests": int(len(tests)),
        },
        "rules": {
            "primary_multiplicity": "Holm across all 50 pathways x 10 thresholds = 500 tests",
            "strict_direction": "group-median and paired-median contrasts have the same non-zero sign at all 10 thresholds",
            "near_stable_direction": "group-median sign is fixed at all 10 thresholds and paired-median sign agrees at 9 of 10 thresholds",
            "crs_thresholds": CRS_THRESHOLDS,
            "crs_metrics": ["HIGH-group mean", "HIGH-minus-LOW mean"],
            "crs_multiplicity": "Holm across 50 pathways x 2 metrics = 100 tests",
        },
        "results": {
            "strict_pathways": stability.loc[stability["strict_10_of_10"], "pathway_label"].tolist(),
            "near_stable_pathways": stability.loc[stability["near_stable_9_of_10"], "pathway_label"].tolist(),
            "primary_holm_significant_tests": int((tests["paired_wilcoxon_p_holm_all_500"] < 0.05).sum()),
            "high_iqr_gt_low_comparisons": int((tests["high_iqr"] > tests["low_iqr"]).sum()),
            "total_dispersion_comparisons": int(len(tests)),
            "crs_holm_significant_tests": crs_screen.loc[
                crs_screen["spearman_p_holm_all_100"] < 0.05,
                ["metric", "pathway_label", "n_sections", "spearman_rho", "spearman_p", "spearman_p_holm_all_100"],
            ].to_dict("records"),
        },
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    manifest_path = args.output_dir / "tumor_rich_hallmark_audit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), **manifest["results"]}, indent=2))


if __name__ == "__main__":
    main()
