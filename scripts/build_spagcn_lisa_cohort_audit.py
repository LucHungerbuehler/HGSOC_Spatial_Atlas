"""Build report-facing SpaGCN/LISA audit tables for the 23-section thesis cohort."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import anndata as ad
import pandas as pd


DEFAULT_ANALYSIS_ROOT = Path(r"D:\HGSOC_Spatial_Atlas\05_analysis_ready")
DEFAULT_COMPILED_ROOT = DEFAULT_ANALYSIS_ROOT / "cross_sample" / "thesis_23" / "compiled"
DEFAULT_MATCHED_ROOT = (
    DEFAULT_ANALYSIS_ROOT / "cross_sample" / "spagcn_k5_matched_malignancy"
)

THESIS_COHORT = {
    "visium/denisenko_2022": ["SP1", "SP2", "SP3", "SP4", "SP5", "SP6", "SP7", "SP8"],
    "visium/yamamoto_2025": [
        "Pt1-1",
        "Pt1-2",
        "Pt1-3",
        "Pt1-4",
        "Pt2-1",
        "Pt2-2",
        "Pt2-3",
        "Pt2-4",
    ],
    "visium/ju_2024": [
        "CPS_OV1RtOV3",
        "CPS_OV5LtOV4",
        "CPS_OV19_LtOV1",
        "CPS_OV20RtOV4",
        "CPS_OV24RTOV4",
        "CPS_OV34RtOV1",
        "CPS_OV71_1",
    ],
}

COMPILED_FILES = [
    "all_hotspot_comparison.csv",
    "all_lisa_results.csv",
    "all_circularity_check.csv",
    "all_spagcn5_composition.csv",
    "all_spagcn5_snai1ac_stats.csv",
    "all_spagcn5_variance.csv",
    "all_spagcn9_composition.csv",
    "all_spagcn9_snai1ac_stats.csv",
    "all_spagcn9_variance.csv",
]

MATCHED_FILES = [
    "spagcn5_domain_annotation_distribution_summary.csv",
    "spagcn5_within_sample_domain_pairs.csv",
    "spagcn5_matched_pair_summary.csv",
    "spagcn5_matched_pair_theme_direction_summary.csv",
    "spagcn5_matched_malignancy_manifest.json",
]


def require_columns(frame: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise RuntimeError(f"Missing columns in {path}: {missing}")


def load_compiled(compiled_root: Path, name: str) -> pd.DataFrame:
    path = compiled_root / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def cohort_pairs() -> list[tuple[str, str]]:
    return [
        (dataset, sample)
        for dataset, samples in THESIS_COHORT.items()
        for sample in samples
    ]


def build_domain_state_summary(
    analysis_root: Path, compiled_root: Path
) -> pd.DataFrame:
    hotspot = load_compiled(compiled_root, "all_hotspot_comparison.csv")
    require_columns(
        hotspot,
        {"dataset", "sample", "LISA_HH_count", "LISA_LL_count"},
        compiled_root / "all_hotspot_comparison.csv",
    )

    rows: list[dict[str, object]] = []
    for resolution in (5, 9):
        stats = load_compiled(compiled_root, f"all_spagcn{resolution}_snai1ac_stats.csv")
        composition = load_compiled(compiled_root, f"all_spagcn{resolution}_composition.csv")
        variance = load_compiled(compiled_root, f"all_spagcn{resolution}_variance.csv")
        require_columns(stats, {"dataset", "sample", "domain", "mean"}, compiled_root)
        require_columns(
            composition,
            {"dataset", "sample", "domain", "Malignant"},
            compiled_root,
        )
        require_columns(variance, {"dataset", "sample", "ratio"}, compiled_root)

        for dataset, sample in cohort_pairs():
            h5ad_path = analysis_root / dataset / sample / f"{sample}.h5ad"
            if not h5ad_path.exists():
                raise FileNotFoundError(h5ad_path)
            adata = ad.read_h5ad(h5ad_path, backed="r")
            try:
                domain_key = f"spagcn_{resolution}_refined"
                required_obs = {domain_key, "LISA_category"}
                missing_obs = sorted(required_obs - set(adata.obs.columns))
                if missing_obs:
                    raise RuntimeError(f"Missing obs columns in {h5ad_path}: {missing_obs}")
                obs = adata.obs[[domain_key, "LISA_category"]].copy()
            finally:
                adata.file.close()

            obs[domain_key] = obs[domain_key].astype(str)
            counts = (
                obs.groupby([domain_key, "LISA_category"], observed=True)
                .size()
                .unstack(fill_value=0)
            )
            hh = counts.get("High-High", pd.Series(0, index=counts.index))
            ll = counts.get("Low-Low", pd.Series(0, index=counts.index))
            mixed_domains = int(((hh > 0) & (ll > 0)).sum())

            sample_stats = stats[
                stats["dataset"].eq(dataset) & stats["sample"].eq(sample)
            ].copy()
            sample_comp = composition[
                composition["dataset"].eq(dataset) & composition["sample"].eq(sample)
            ].copy()
            sample_var = variance[
                variance["dataset"].eq(dataset) & variance["sample"].eq(sample)
            ]
            sample_hotspot = hotspot[
                hotspot["dataset"].eq(dataset) & hotspot["sample"].eq(sample)
            ]
            if len(sample_stats) != len(sample_comp) or len(sample_stats) != len(counts):
                raise RuntimeError(
                    f"Domain-count mismatch for target k={resolution}, {dataset}/{sample}: "
                    f"h5ad={len(counts)}, score={len(sample_stats)}, "
                    f"composition={len(sample_comp)}"
                )
            if len(sample_var) != 1 or len(sample_hotspot) != 1:
                raise RuntimeError(f"Expected one variance/hotspot row for {dataset}/{sample}")

            top_score_domain = str(
                sample_stats.loc[sample_stats["mean"].astype(float).idxmax(), "domain"]
            )
            top_malignant_domain = str(
                sample_comp.loc[
                    sample_comp["Malignant"].astype(float).idxmax(), "domain"
                ]
            )
            lisa_hh_count = int((obs["LISA_category"] == "High-High").sum())
            lisa_ll_count = int((obs["LISA_category"] == "Low-Low").sum())
            expected_hh = int(sample_hotspot.iloc[0]["LISA_HH_count"])
            expected_ll = int(sample_hotspot.iloc[0]["LISA_LL_count"])
            if (lisa_hh_count, lisa_ll_count) != (expected_hh, expected_ll):
                raise RuntimeError(
                    f"LISA count mismatch for {dataset}/{sample}: "
                    f"h5ad={(lisa_hh_count, lisa_ll_count)}, "
                    f"compiled={(expected_hh, expected_ll)}"
                )

            rows.append(
                {
                    "dataset": dataset,
                    "sample": sample,
                    "resolution_k": resolution,
                    "n_domains": int(len(counts)),
                    "mixed_hh_ll_domains": mixed_domains,
                    "has_mixed_hh_ll_domain": bool(mixed_domains > 0),
                    "lisa_hh_spots": lisa_hh_count,
                    "lisa_ll_spots": lisa_ll_count,
                    "top_mean_snai1ac_domain": top_score_domain,
                    "top_mean_malignant_domain": top_malignant_domain,
                    "top_domains_agree": bool(top_score_domain == top_malignant_domain),
                    "between_to_within_snai1ac_variance_ratio": float(
                        sample_var.iloc[0]["ratio"]
                    ),
                }
            )

    result = pd.DataFrame(rows).sort_values(["resolution_k", "dataset", "sample"])
    if len(result) != 46:
        raise RuntimeError(f"Expected 46 sample-resolution rows, found {len(result)}")
    return result


def build_circularity_summary(compiled_root: Path) -> pd.DataFrame:
    circularity = load_compiled(compiled_root, "all_circularity_check.csv")
    require_columns(
        circularity,
        {
            "comparison",
            "n_signature_genes",
            "n_marker_genes",
            "n_overlap",
            "overlap_pct",
        },
        compiled_root / "all_circularity_check.csv",
    )
    return (
        circularity.groupby("comparison", as_index=False)
        .agg(
            n_sections=("sample", "nunique"),
            signature_genes=("n_signature_genes", "median"),
            median_feature_or_marker_genes=("n_marker_genes", "median"),
            median_overlap_genes=("n_overlap", "median"),
            median_signature_overlap_pct=("overlap_pct", "median"),
        )
        .sort_values("comparison")
    )


def build_cohort_summary(domain_state: pd.DataFrame) -> pd.DataFrame:
    k5 = domain_state[domain_state["resolution_k"].eq(5)]
    k9 = domain_state[domain_state["resolution_k"].eq(9)]
    lisa = k5
    max_k9 = k9.loc[
        k9["between_to_within_snai1ac_variance_ratio"].idxmax()
    ]
    rows = [
        {
            "metric": "thesis_sections",
            "resolution_k": pd.NA,
            "value": int(lisa["sample"].nunique()),
            "q25": pd.NA,
            "q75": pd.NA,
            "numerator": pd.NA,
            "denominator": pd.NA,
            "sample": "",
        },
        {
            "metric": "lisa_hh_spots_per_section",
            "resolution_k": pd.NA,
            "value": float(lisa["lisa_hh_spots"].median()),
            "q25": float(lisa["lisa_hh_spots"].quantile(0.25)),
            "q75": float(lisa["lisa_hh_spots"].quantile(0.75)),
            "numerator": pd.NA,
            "denominator": 23,
            "sample": "",
        },
        {
            "metric": "lisa_ll_spots_per_section",
            "resolution_k": pd.NA,
            "value": float(lisa["lisa_ll_spots"].median()),
            "q25": float(lisa["lisa_ll_spots"].quantile(0.25)),
            "q75": float(lisa["lisa_ll_spots"].quantile(0.75)),
            "numerator": pd.NA,
            "denominator": 23,
            "sample": "",
        },
    ]
    for frame, resolution in ((k5, 5), (k9, 9)):
        rows.extend(
            [
                {
                    "metric": "sections_with_at_least_one_mixed_hh_ll_domain",
                    "resolution_k": resolution,
                    "value": float(frame["has_mixed_hh_ll_domain"].mean()),
                    "q25": pd.NA,
                    "q75": pd.NA,
                    "numerator": int(frame["has_mixed_hh_ll_domain"].sum()),
                    "denominator": int(len(frame)),
                    "sample": "",
                },
                {
                    "metric": "mixed_hh_ll_domains_per_section",
                    "resolution_k": resolution,
                    "value": float(frame["mixed_hh_ll_domains"].median()),
                    "q25": float(frame["mixed_hh_ll_domains"].quantile(0.25)),
                    "q75": float(frame["mixed_hh_ll_domains"].quantile(0.75)),
                    "numerator": pd.NA,
                    "denominator": int(len(frame)),
                    "sample": "",
                },
                {
                    "metric": "top_snai1ac_and_malignant_domain_agreement",
                    "resolution_k": resolution,
                    "value": float(frame["top_domains_agree"].mean()),
                    "q25": pd.NA,
                    "q75": pd.NA,
                    "numerator": int(frame["top_domains_agree"].sum()),
                    "denominator": int(len(frame)),
                    "sample": "",
                },
                {
                    "metric": "between_to_within_snai1ac_variance_ratio",
                    "resolution_k": resolution,
                    "value": float(
                        frame["between_to_within_snai1ac_variance_ratio"].median()
                    ),
                    "q25": float(
                        frame["between_to_within_snai1ac_variance_ratio"].quantile(0.25)
                    ),
                    "q75": float(
                        frame["between_to_within_snai1ac_variance_ratio"].quantile(0.75)
                    ),
                    "numerator": pd.NA,
                    "denominator": int(len(frame)),
                    "sample": "",
                },
            ]
        )
    rows.append(
        {
            "metric": "maximum_between_to_within_snai1ac_variance_ratio",
            "resolution_k": 9,
            "value": float(max_k9["between_to_within_snai1ac_variance_ratio"]),
            "q25": pd.NA,
            "q75": pd.NA,
            "numerator": pd.NA,
            "denominator": 23,
            "sample": str(max_k9["sample"]),
        }
    )
    return pd.DataFrame(rows)


def copy_source_tables(compiled_root: Path, matched_root: Path, output_dir: Path) -> None:
    for name in COMPILED_FILES:
        source = compiled_root / name
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, output_dir / name)
    for name in MATCHED_FILES:
        source = matched_root / name
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, output_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--compiled-root", type=Path, default=DEFAULT_COMPILED_ROOT)
    parser.add_argument("--matched-root", type=Path, default=DEFAULT_MATCHED_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    domain_state = build_domain_state_summary(args.analysis_root, args.compiled_root)
    circularity = build_circularity_summary(args.compiled_root)
    cohort = build_cohort_summary(domain_state)

    domain_state.to_csv(args.output_dir / "spagcn_lisa_domain_state_summary.csv", index=False)
    circularity.to_csv(args.output_dir / "spagcn_lisa_circularity_summary.csv", index=False)
    cohort.to_csv(args.output_dir / "spagcn_lisa_cohort_summary.csv", index=False)
    copy_source_tables(args.compiled_root, args.matched_root, args.output_dir)

    manifest = {
        "cohort": THESIS_COHORT,
        "analysis_root": str(args.analysis_root),
        "compiled_root": str(args.compiled_root),
        "matched_root": str(args.matched_root),
        "lisa_definition": {
            "score_column": "SNAI1-ac_score",
            "coordinate_columns": ["array_row", "array_col"],
            "knn_k": 6,
            "permutations": 999,
            "multiple_testing": "Benjamini-Hochberg FDR within section",
            "alpha": 0.05,
        },
        "outputs": sorted(path.name for path in args.output_dir.iterdir() if path.is_file()),
    }
    (args.output_dir / "spagcn_lisa_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
