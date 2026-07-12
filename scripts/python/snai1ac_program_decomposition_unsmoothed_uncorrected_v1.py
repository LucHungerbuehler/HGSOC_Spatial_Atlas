"""
Decompose unsmoothed, uncorrected SNAI1-ac variation into cNMF tumour programs.

This is a sibling branch to snai1ac_program_decomposition_v1. It keeps the same
per-sample model and the same tumour spots, but replaces the production
smooth/corrected SNAI1-ac target with the GASTON score-alignment column:

    snai1ac_em_unsmoothed_uncorrected

Model:

    unsmoothed_uncorrected_score ~ spatial baseline + malignant fraction + K* cNMF usage
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

import snai1ac_program_decomposition_v1 as base


DATA_ROOT = Path(r"D:\HGSOC_Spatial_Atlas")
CNMF_ROOT = DATA_ROOT / "05_analysis_ready" / "S3_cNMF_Tumor_Programs"
GASTON_ALIGNMENT_ROOT = DATA_ROOT / "05_analysis_ready" / "GASTON_method_v1" / "04_isodepth_score_alignment"
DEFAULT_OUTPUT = CNMF_ROOT / "snai1ac_program_decomposition_unsmoothed_uncorrected_v1"
DEFAULT_SCRIPT = Path(__file__).resolve()

TARGET_SCORE_COL = "snai1ac_em_unsmoothed_uncorrected"
REFERENCE_SCORE_COL = "snai1ac_em_smooth_corrected"
TARGET_LABEL = "unsmoothed_uncorrected"
TARGET_DESCRIPTION = (
    "Weighted EnrichMap SNAI1-ac score recomputed with smoothing=False and "
    "correct_spatial_covariates=False."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-sample cNMF programme model for unsmoothed, uncorrected SNAI1-ac score."
    )
    parser.add_argument("--cnmf-root", type=Path, default=CNMF_ROOT)
    parser.add_argument("--gaston-alignment-root", type=Path, default=GASTON_ALIGNMENT_ROOT)
    parser.add_argument("--signature-weights", type=Path, default=base.SIGNATURE_ROOT / "snai1_ac_weights.json")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--spatial-block-side", type=int, default=4)
    parser.add_argument("--min-spots", type=int, default=40)
    parser.add_argument("--write-predictions", action="store_true", default=True)
    return parser.parse_args()


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    x_arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_arr = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(mask.sum()) < 3 or np.nanstd(x_arr[mask]) <= 1e-12 or np.nanstd(y_arr[mask]) <= 1e-12:
        return math.nan
    return float(spearmanr(x_arr[mask], y_arr[mask]).correlation)


def load_alignment_manifest(root: Path) -> pd.DataFrame:
    manifest = pd.read_csv(root / "sample_alignment_manifest.csv")
    manifest["sample_label"] = manifest["dataset"].astype(str) + "__" + manifest["sample"].astype(str)
    return manifest


def merge_target_score(
    frame: pd.DataFrame,
    sample_label: str,
    alignment_manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    matches = alignment_manifest[alignment_manifest["sample_label"].eq(sample_label)].copy()
    if matches.empty:
        raise FileNotFoundError(f"No score-alignment table found for {sample_label}")
    if len(matches) > 1:
        matches = matches.sort_values(["include_in_primary_cross_sample", "feature_method"], ascending=[False, True])
    row = matches.iloc[0]
    alignment_table = Path(str(row["alignment_table"]))
    alignment = pd.read_csv(alignment_table)
    alignment["spot_id"] = alignment["spot_id"].astype(str)
    keep = ["spot_id", REFERENCE_SCORE_COL, TARGET_SCORE_COL]
    missing = [col for col in keep if col not in alignment.columns]
    if missing:
        raise KeyError(f"{alignment_table} is missing expected columns: {missing}")

    merged = frame.merge(alignment[keep], on="spot_id", how="left", validate="one_to_one")
    have_target = merged[TARGET_SCORE_COL].notna()
    smooth_diff = pd.to_numeric(merged[base.SNAI1_COL], errors="coerce") - pd.to_numeric(
        merged[REFERENCE_SCORE_COL], errors="coerce"
    )
    audit = {
        "dataset": str(frame["dataset"].iloc[0]),
        "sample_id_on_disk": str(frame["sample_id_on_disk"].iloc[0]),
        "sample_label": sample_label,
        "model_tumor_spots": int(len(frame)),
        "alignment_spots_whole_slide": int(len(alignment)),
        "matched_tumor_spots_with_target": int(have_target.sum()),
        "missing_target_score": int((~have_target).sum()),
        "alignment_feature_method": row.get("feature_method", ""),
        "alignment_include_in_primary_cross_sample": row.get("include_in_primary_cross_sample", ""),
        "smooth_corrected_vs_current_target_max_abs_diff": float(smooth_diff.abs().max()),
        "smooth_corrected_vs_unsmoothed_uncorrected_spearman_tumor": safe_spearman(
            merged[REFERENCE_SCORE_COL], merged[TARGET_SCORE_COL]
        ),
        "alignment_table": str(alignment_table),
    }
    return merged.loc[have_target].reset_index(drop=True), audit


def plot_observed_predicted(predictions: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    for sample_label, group in predictions.groupby("sample_label", sort=True):
        fig, ax = plt.subplots(figsize=(5.8, 5.2))
        ax.scatter(
            group[base.SNAI1_COL],
            group["pred_spatial_malignant_usage_raw"],
            s=15,
            alpha=0.55,
            color="#8a4d4d",
            edgecolor="none",
        )
        vals = pd.concat([group[base.SNAI1_COL], group["pred_spatial_malignant_usage_raw"]]).to_numpy(dtype=float)
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        ax.plot([lo, hi], [lo, hi], color="#555555", linewidth=1.0, linestyle="--")
        rho, _ = base.safe_spearman(group[base.SNAI1_COL], group["pred_spatial_malignant_usage_raw"])
        r2 = r2_score(group[base.SNAI1_COL], group["pred_spatial_malignant_usage_raw"])
        ax.set_xlabel("Observed unsmoothed/uncorrected SNAI1-ac score")
        ax.set_ylabel("CV predicted score")
        ax.set_title(sample_label, fontsize=10)
        ax.text(
            0.03,
            0.97,
            f"CV R2={r2:.3f}\nrho={rho:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#dddddd", "boxstyle": "round,pad=0.3"},
        )
        fig.tight_layout()
        path = out_dir / f"{sample_label}__observed_vs_cv_predicted_unsmoothed_uncorrected.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def write_manifest(output_root: Path, args: argparse.Namespace) -> None:
    manifest = {
        "branch": "snai1ac_program_decomposition_unsmoothed_uncorrected_v1",
        "created_by_script": str(DEFAULT_SCRIPT),
        "base_model_script": str(base.DEFAULT_SCRIPT),
        "cnmf_root": str(args.cnmf_root),
        "gaston_alignment_root": str(args.gaston_alignment_root),
        "signature_weights": str(args.signature_weights),
        "target_score_col": TARGET_SCORE_COL,
        "target_score_description": TARGET_DESCRIPTION,
        "reference_score_col_for_audit_only": REFERENCE_SCORE_COL,
        "model": "target_score ~ spatial polynomial baseline + Malignant + raw K* cNMF usage",
        "spatial_baseline_features": ["array_row", "array_col", "array_row2", "array_row_array_col", "array_col2"],
        "tumour_filter": "same cNMF tumour-spot frame as snai1ac_program_decomposition_v1",
    }
    (output_root / "00_manifest_and_provenance" / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def write_readme(
    output_root: Path,
    performance: pd.DataFrame,
    audit: pd.DataFrame,
    concordance: pd.DataFrame,
    label_summary: pd.DataFrame,
) -> None:
    primary = performance[performance["model"].eq("spatial_malignant_usage_raw")].copy()
    baseline = performance[performance["model"].eq("spatial_malignant")].copy()
    merged = primary.merge(
        baseline[["sample_label", "cv_r2"]],
        on="sample_label",
        suffixes=("_full", "_baseline"),
    )
    merged["delta_r2"] = merged["cv_r2_full"] - merged["cv_r2_baseline"]
    lines = [
        "# SNAI1-ac cNMF Program Decomposition, Unsmoothed Uncorrected v1",
        "",
        "This branch fits one model per sample on tumour spots only:",
        "",
        "`snai1ac_em_unsmoothed_uncorrected ~ spatial baseline + malignant fraction + K* cNMF programme usage`",
        "",
        "The only intended target score is the recomputed weighted EnrichMap SNAI1-ac score with smoothing disabled and spatial covariate correction disabled. The production smooth/corrected score is used only as a join audit comparator.",
        "",
        "## Join audit",
        "",
        f"- Samples checked: {audit.shape[0]}",
        f"- Tumour spots checked: {int(audit['model_tumor_spots'].sum())}",
        f"- Tumour spots with target score: {int(audit['matched_tumor_spots_with_target'].sum())}",
        f"- Missing target values after join: {int(audit['missing_target_score'].sum())}",
        f"- Max absolute difference between primary branch target and smooth/corrected alignment score: {audit['smooth_corrected_vs_current_target_max_abs_diff'].max():.6g}",
        f"- Median tumour-spot Spearman, smooth/corrected vs unsmoothed/uncorrected: {audit['smooth_corrected_vs_unsmoothed_uncorrected_spearman_tumor'].median():.3f}",
        "",
        "## Model summary",
        "",
        f"- Samples modelled: {primary.shape[0]}",
        f"- Total valid tumour spots modelled: {int(audit['matched_tumor_spots_with_target'].sum())}",
        f"- Median full-model CV R2: {merged['cv_r2_full'].median():.3f}",
        f"- Mean full-model CV R2: {merged['cv_r2_full'].mean():.3f}",
        f"- Median spatial+malignant baseline CV R2: {merged['cv_r2_baseline'].median():.3f}",
        f"- Median added CV R2 from raw cNMF usage: {merged['delta_r2'].median():.3f}",
        f"- Samples with positive full-model CV R2: {int((merged['cv_r2_full'] > 0).sum())}/{len(merged)}",
        f"- Samples with positive added CV R2: {int((merged['delta_r2'] > 0).sum())}/{len(merged)}",
        "",
        "## Spectrum projection check",
        "",
        f"- Median learned-vs-spectrum Spearman rho: {concordance['learned_vs_spectrum_spearman_absnorm'].median():.3f}",
        "",
        "## Annotation-level weight summary",
        "",
        f"- Annotation labels represented: {label_summary.shape[0]}",
        f"- Strongest median absolute label-level weight: {label_summary['median_abs_standardized_weight'].max():.3f}",
        "",
        "## Main tables",
        "",
        "- `01_score_variant_audit/tables/score_variant_join_audit.csv`",
        "- `02_per_sample_usage_models/tables/per_sample_model_performance.csv`",
        "- `02_per_sample_usage_models/tables/per_sample_program_weights.csv`",
        "- `02_per_sample_usage_models/tables/per_sample_full_model_all_feature_coefficients.csv`",
        "- `02_per_sample_usage_models/tables/per_spot_predictions.csv`",
        "- `03_program_spectrum_projection/tables/program_signature_projection.csv`",
        "- `04_model_projection_concordance/tables/program_weight_projection_joined.csv`",
        "- `04_model_projection_concordance/tables/per_sample_weight_projection_concordance.csv`",
        "- `05_cross_sample_summary/tables/cross_sample_summary.csv`",
        "- `05_cross_sample_summary/tables/program_weight_summary_by_annotation.csv`",
        "- `05_cross_sample_summary/tables/top_positive_negative_programs_per_sample.csv`",
        "",
        "## Interpretation guardrail",
        "",
        "This branch decomposes the unsmoothed/uncorrected score only. It should be compared against the production-score branch before deciding which result belongs in the report narrative.",
    ]
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = ensure(args.output_root)
    dirs = {
        "manifest": ensure(out / "00_manifest_and_provenance"),
        "audit": ensure(out / "01_score_variant_audit" / "tables"),
        "models": ensure(out / "02_per_sample_usage_models" / "tables"),
        "model_figs": ensure(out / "02_per_sample_usage_models" / "figures"),
        "projection": ensure(out / "03_program_spectrum_projection" / "tables"),
        "concordance": ensure(out / "04_model_projection_concordance" / "tables"),
        "summary": ensure(out / "05_cross_sample_summary" / "tables"),
        "figures": ensure(out / "06_figures"),
        "scripts": ensure(out / "scripts_used"),
    }
    shutil.copy2(DEFAULT_SCRIPT, dirs["scripts"] / DEFAULT_SCRIPT.name)
    shutil.copy2(base.DEFAULT_SCRIPT, dirs["scripts"] / base.DEFAULT_SCRIPT.name)

    cnmf_manifest = pd.read_csv(args.cnmf_root / "sample_manifest.csv")
    cnmf_manifest = cnmf_manifest[cnmf_manifest["eligible_for_cnmf"].astype(str).str.lower().eq("true")].copy()
    alignment_manifest = load_alignment_manifest(args.gaston_alignment_root)
    signature_weights = base.load_signature_weights(args.signature_weights)

    audit_rows: list[dict[str, object]] = []
    performance_tables: list[pd.DataFrame] = []
    coef_tables: list[pd.DataFrame] = []
    prediction_tables: list[pd.DataFrame] = []
    projection_tables: list[pd.DataFrame] = []

    for _, row in cnmf_manifest.sort_values(["dataset", "sample_id_on_disk"]).iterrows():
        sample_label = str(row["sample_label"])
        print(f"Analysing {sample_label}", flush=True)
        frame, input_audit = base.load_sample_frame(row, args.cnmf_root)
        if int(input_audit["merged_valid_tumor_spots"]) < args.min_spots:
            continue
        frame, audit = merge_target_score(frame, sample_label, alignment_manifest)
        program_cols = base.program_columns(frame, int(float(row["k_star"])))
        audit.update(
            {
                "k_star": int(float(row["k_star"])),
                "n_programs": int(len(program_cols)),
                "target_score_min": float(frame[TARGET_SCORE_COL].min()),
                "target_score_median": float(frame[TARGET_SCORE_COL].median()),
                "target_score_max": float(frame[TARGET_SCORE_COL].max()),
            }
        )
        audit_rows.append(audit)

        production_score = frame[base.SNAI1_COL].to_numpy(dtype=float)
        frame = frame.copy()
        frame["production_smooth_corrected_score"] = production_score
        frame[base.SNAI1_COL] = pd.to_numeric(frame[TARGET_SCORE_COL], errors="coerce")
        valid = np.isfinite(frame[base.SNAI1_COL].to_numpy(dtype=float))
        frame = frame.loc[valid].reset_index(drop=True)

        perf, coefs, predictions = base.analyze_models_for_sample(
            frame,
            program_cols,
            outer_splits=args.outer_splits,
            inner_splits=args.inner_splits,
            spatial_block_side=args.spatial_block_side,
        )
        predictions["target_score_col"] = TARGET_SCORE_COL
        predictions["target_score"] = predictions[base.SNAI1_COL]
        predictions["production_smooth_corrected_score"] = frame["production_smooth_corrected_score"]
        predictions[TARGET_SCORE_COL] = frame[TARGET_SCORE_COL].to_numpy(dtype=float)
        projection = base.project_signature_onto_spectra(args.cnmf_root, row, signature_weights)

        performance_tables.append(perf)
        coef_tables.append(coefs)
        prediction_tables.append(predictions)
        projection_tables.append(projection)

    audit_df = pd.DataFrame(audit_rows)
    performance = pd.concat(performance_tables, ignore_index=True)
    all_coefs = pd.concat(coef_tables, ignore_index=True)
    weights = base.weight_table_from_coefs(all_coefs, args.cnmf_root)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    projection = pd.concat(projection_tables, ignore_index=True)
    joined, concordance = base.model_projection_concordance(weights, projection)
    label_summary, top_programs = base.summarize_weights_by_annotation(weights)

    audit_df.to_csv(dirs["audit"] / "score_variant_join_audit.csv", index=False)
    performance.to_csv(dirs["models"] / "per_sample_model_performance.csv", index=False)
    weights.to_csv(dirs["models"] / "per_sample_program_weights.csv", index=False)
    all_coefs.to_csv(dirs["models"] / "per_sample_full_model_all_feature_coefficients.csv", index=False)
    if args.write_predictions:
        predictions.to_csv(dirs["models"] / "per_spot_predictions.csv", index=False)
    projection.to_csv(dirs["projection"] / "program_signature_projection.csv", index=False)
    joined.to_csv(dirs["concordance"] / "program_weight_projection_joined.csv", index=False)
    concordance.to_csv(dirs["concordance"] / "per_sample_weight_projection_concordance.csv", index=False)
    label_summary.to_csv(dirs["summary"] / "program_weight_summary_by_annotation.csv", index=False)
    top_programs.to_csv(dirs["summary"] / "top_positive_negative_programs_per_sample.csv", index=False)

    primary = performance[performance["model"].eq("spatial_malignant_usage_raw")].copy()
    baseline = performance[performance["model"].eq("spatial_malignant")].copy()
    summary = primary.merge(
        baseline[["sample_label", "cv_r2", "cv_rmse", "cv_spearman_rho"]],
        on="sample_label",
        suffixes=("_full", "_spatial_malignant"),
    )
    summary["delta_cv_r2_usage_after_spatial_malignant"] = summary["cv_r2_full"] - summary["cv_r2_spatial_malignant"]
    summary = summary.merge(
        concordance[["sample_label", "learned_vs_spectrum_spearman_absnorm"]],
        on="sample_label",
        how="left",
    )
    summary.insert(0, "target_score_col", TARGET_SCORE_COL)
    summary.to_csv(dirs["summary"] / "cross_sample_summary.csv", index=False)

    plot_observed_predicted(predictions, dirs["model_figs"])
    base.plot_model_summary(performance, dirs["figures"])
    base.plot_top_weights(weights, dirs["figures"])
    write_manifest(out, args)
    write_readme(out, performance, audit_df, concordance, label_summary)
    print(f"Wrote branch to {out}", flush=True)


if __name__ == "__main__":
    main()
