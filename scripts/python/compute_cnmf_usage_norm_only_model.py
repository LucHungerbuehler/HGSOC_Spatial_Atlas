"""Compute row-normalized usage-only ridge models for the unsmoothed cNMF branch."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BRANCH = Path(
    r"D:\HGSOC_Spatial_Atlas\05_analysis_ready\S3_cNMF_Tumor_Programs"
) / "snai1ac_program_decomposition_unsmoothed_uncorrected_v1"
SCRIPT_DIR = BRANCH / "scripts_used"
CNMF_ROOT = Path(r"D:\HGSOC_Spatial_Atlas\05_analysis_ready\S3_cNMF_Tumor_Programs")
ALIGNMENT_ROOT = Path(
    r"D:\HGSOC_Spatial_Atlas\05_analysis_ready\GASTON_method_v1\04_isodepth_score_alignment"
)
TABLE_DIR = BRANCH / "07_report_examples" / "tables"

OUTER_SPLITS = 5
INNER_SPLITS = 5
SPATIAL_BLOCK_SIDE = 4
TARGET_SCORE_COL = "snai1ac_em_unsmoothed_uncorrected"


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def row_normalized_usage(frame: pd.DataFrame, program_cols: list[str]) -> np.ndarray:
    usage = frame[program_cols].to_numpy(dtype=float)
    row_sums = usage.sum(axis=1, keepdims=True)
    return usage / np.where(row_sums > 0, row_sums, np.nan)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    base = import_module(SCRIPT_DIR / "snai1ac_program_decomposition_v1.py", "snai1ac_program_decomposition_v1")
    unsmoothed = import_module(
        SCRIPT_DIR / "snai1ac_program_decomposition_unsmoothed_uncorrected_v1.py",
        "snai1ac_program_decomposition_unsmoothed_uncorrected_v1",
    )

    cnmf_manifest = pd.read_csv(CNMF_ROOT / "sample_manifest.csv")
    cnmf_manifest = cnmf_manifest[
        cnmf_manifest["eligible_for_cnmf"].astype(str).str.lower().eq("true")
    ].copy()
    alignment_manifest = unsmoothed.load_alignment_manifest(ALIGNMENT_ROOT)

    performance_rows = []
    prediction_frames = []

    for _, manifest_row in cnmf_manifest.sort_values(["dataset", "sample_id_on_disk"]).iterrows():
        sample_label = str(manifest_row["sample_label"])
        frame, input_audit = base.load_sample_frame(manifest_row, CNMF_ROOT)
        if int(input_audit["merged_valid_tumor_spots"]) < 40:
            continue
        frame, _ = unsmoothed.merge_target_score(frame, sample_label, alignment_manifest)
        frame = frame.copy()
        frame[base.SNAI1_COL] = pd.to_numeric(frame[TARGET_SCORE_COL], errors="coerce")
        frame = frame.loc[np.isfinite(frame[base.SNAI1_COL].to_numpy(dtype=float))].reset_index(drop=True)

        k_star = int(float(manifest_row["k_star"]))
        program_cols = base.program_columns(frame, k_star)
        X = row_normalized_usage(frame, program_cols)
        keep = np.isfinite(X).all(axis=1)
        model_frame = frame.loc[keep].reset_index(drop=True)
        X = X[keep]
        y = model_frame[base.SNAI1_COL].to_numpy(dtype=float)
        norm_feature_names = [f"{col}__row_norm" for col in program_cols]

        pred, alphas, outer = base.nested_predictions(
            model_frame,
            y,
            X,
            outer_splits=OUTER_SPLITS,
            inner_splits=INNER_SPLITS,
            spatial_block_side=SPATIAL_BLOCK_SIDE,
        )
        score = base.metrics(y, pred)
        fit_info, coef_df, _ = base.final_fit(
            X,
            y,
            model_frame,
            feature_names=norm_feature_names,
            inner_splits=INNER_SPLITS,
            spatial_block_side=SPATIAL_BLOCK_SIDE,
        )

        performance_rows.append(
            {
                "dataset": str(model_frame["dataset"].iloc[0]),
                "sample_id_on_disk": str(model_frame["sample_id_on_disk"].iloc[0]),
                "sample_label": sample_label,
                "model": "usage_norm_only",
                "n_spots": int(len(model_frame)),
                "n_features": int(X.shape[1]),
                "n_program_features": int(len(program_cols)),
                "cv_scheme": outer.name,
                "outer_splits": int(outer.n_splits),
                "spatial_group_count": int(outer.n_groups),
                "spatial_block_side": outer.block_side,
                "outer_alpha_mean": float(np.mean(alphas)) if alphas else np.nan,
                "outer_alpha_median": float(np.median(alphas)) if alphas else np.nan,
                "final_alpha": float(fit_info["alpha"]),
                **{f"cv_{k}": v for k, v in score.items()},
            }
        )

        coef_df.insert(0, "sample_label", sample_label)
        coef_df.insert(1, "model", "usage_norm_only")
        coef_df.to_csv(TABLE_DIR / f"{sample_label}_usage_norm_only_coefficients.csv", index=False)

        pred_frame = model_frame[
            ["dataset", "sample_id_on_disk", "sample_label", "spot_id", base.SNAI1_COL]
        ].copy()
        pred_frame["pred_usage_norm_only"] = pred
        pred_frame["target_score_col"] = TARGET_SCORE_COL
        prediction_frames.append(pred_frame)

    performance = pd.DataFrame(performance_rows)
    performance.to_csv(TABLE_DIR / "cohort_usage_norm_only_model_performance.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        TABLE_DIR / "cohort_usage_norm_only_per_spot_predictions.csv",
        index=False,
    )

    print(TABLE_DIR / "cohort_usage_norm_only_model_performance.csv")
    print(TABLE_DIR / "cohort_usage_norm_only_per_spot_predictions.csv")


if __name__ == "__main__":
    main()
