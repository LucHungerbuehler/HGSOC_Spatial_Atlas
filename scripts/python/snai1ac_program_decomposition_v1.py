"""
Decompose continuous SNAI1-ac variation into per-sample cNMF tumour programs.

This branch supersedes the older usage-only ridge attempt without overwriting it.
It fits one model per sample on tumour spots only:

    SNAI1-ac_score ~ spatial baseline + malignant fraction + K* cNMF usage

and compares the learned programme weights with a signature-spectrum projection.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


DATA_ROOT = Path(r"D:\HGSOC_Spatial_Atlas")
CNMF_ROOT = DATA_ROOT / "05_analysis_ready" / "S3_cNMF_Tumor_Programs"
SIGNATURE_ROOT = DATA_ROOT / "05_analysis_ready" / "Signature"
DEFAULT_OUTPUT = CNMF_ROOT / "snai1ac_program_decomposition_v1"
DEFAULT_SCRIPT = Path(__file__).resolve()

SNAI1_COL = "SNAI1-ac_score"
MALIGNANT_COL = "Malignant"
PROGRAM_PATTERN = re.compile(r"__K\d+__P\d+$")
ALPHA_GRID = np.logspace(-4, 4, 25)
RANDOM_STATE = 42


@dataclass
class CvScheme:
    name: str
    splitter: object
    groups: np.ndarray | None
    n_splits: int
    n_groups: int
    block_side: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-sample SNAI1-ac decomposition from K* cNMF tumour-program usage."
    )
    parser.add_argument("--cnmf-root", type=Path, default=CNMF_ROOT)
    parser.add_argument("--signature-weights", type=Path, default=SIGNATURE_ROOT / "snai1_ac_weights.json")
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


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def safe_spearman(x: Iterable[float], y: Iterable[float]) -> tuple[float, float]:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(mask.sum()) < 3 or np.nanstd(x_arr[mask]) <= 1e-12 or np.nanstd(y_arr[mask]) <= 1e-12:
        return math.nan, math.nan
    stat = spearmanr(x_arr[mask], y_arr[mask])
    return float(stat.statistic), float(stat.pvalue)


def safe_pearson(x: Iterable[float], y: Iterable[float]) -> tuple[float, float]:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(mask.sum()) < 3 or np.nanstd(x_arr[mask]) <= 1e-12 or np.nanstd(y_arr[mask]) <= 1e-12:
        return math.nan, math.nan
    stat = pearsonr(x_arr[mask], y_arr[mask])
    return float(stat.statistic), float(stat.pvalue)


def qcut_codes(values: pd.Series, n_bins: int) -> np.ndarray:
    clean = pd.to_numeric(values, errors="coerce")
    n_unique = int(clean.nunique(dropna=True))
    if n_unique < 2:
        return np.zeros(len(clean), dtype=int)
    return pd.qcut(clean, q=min(n_bins, n_unique), labels=False, duplicates="drop").to_numpy()


def spatial_block_groups(frame: pd.DataFrame, preferred_side: int, min_groups: int) -> tuple[np.ndarray | None, int | None]:
    coords = frame[["array_row", "array_col"]].apply(pd.to_numeric, errors="coerce")
    if coords.isna().any().any():
        return None, None
    for side in range(preferred_side, 1, -1):
        row_codes = qcut_codes(coords["array_row"], side)
        col_codes = qcut_codes(coords["array_col"], side)
        labels = pd.Series([f"{r}_{c}" for r, c in zip(row_codes, col_codes, strict=False)])
        groups = labels.astype("category").cat.codes.to_numpy()
        if np.unique(groups).size >= min_groups:
            return groups, side
    return None, None


def build_cv(frame: pd.DataFrame, target_splits: int, spatial_block_side: int) -> CvScheme:
    min_groups = min(target_splits, max(3, target_splits))
    groups, side = spatial_block_groups(frame, spatial_block_side, min_groups)
    if groups is not None:
        n_groups = int(np.unique(groups).size)
        n_splits = min(target_splits, n_groups)
        return CvScheme(
            name="spatial_groupkfold",
            splitter=GroupKFold(n_splits=n_splits),
            groups=groups,
            n_splits=n_splits,
            n_groups=n_groups,
            block_side=side,
        )

    n_splits = min(target_splits, max(3, len(frame) // 25))
    n_splits = max(3, min(n_splits, len(frame)))
    return CvScheme(
        name="random_kfold_fallback",
        splitter=KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE),
        groups=None,
        n_splits=n_splits,
        n_groups=0,
        block_side=None,
    )


def make_pipeline() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])


def fit_ridge_grid(
    X: np.ndarray,
    y: np.ndarray,
    frame: pd.DataFrame,
    target_splits: int,
    spatial_block_side: int,
) -> GridSearchCV:
    scheme = build_cv(frame, target_splits=target_splits, spatial_block_side=spatial_block_side)
    if scheme.groups is not None and scheme.n_splits >= 2:
        cv = scheme.splitter
        groups = scheme.groups
    else:
        cv = KFold(
            n_splits=max(2, min(target_splits, len(frame))),
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        groups = None

    search = GridSearchCV(
        estimator=make_pipeline(),
        param_grid={"ridge__alpha": ALPHA_GRID},
        scoring="r2",
        cv=cv,
        n_jobs=1,
        refit=True,
    )
    if groups is None:
        search.fit(X, y)
    else:
        search.fit(X, y, groups=groups)
    return search


def feature_matrix(frame: pd.DataFrame, program_cols: list[str], mode: str) -> tuple[np.ndarray, list[str], list[str]]:
    spatial = frame[["array_row", "array_col"]].to_numpy(dtype=float)
    spatial_poly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(spatial)
    spatial_names = ["array_row", "array_col", "array_row2", "array_row_array_col", "array_col2"]
    malignant = frame[[MALIGNANT_COL]].to_numpy(dtype=float)
    usage_raw = frame[program_cols].to_numpy(dtype=float)
    row_sums = usage_raw.sum(axis=1, keepdims=True)
    usage_norm = usage_raw / np.where(row_sums > 0, row_sums, np.nan)

    if mode == "intercept_only":
        return np.empty((len(frame), 0)), [], []
    if mode == "usage_raw_only":
        return usage_raw, program_cols, program_cols
    if mode == "spatial":
        return spatial_poly, spatial_names, []
    if mode == "spatial_malignant":
        return np.column_stack([spatial_poly, malignant]), spatial_names + [MALIGNANT_COL], []
    if mode == "spatial_malignant_usage_raw":
        return (
            np.column_stack([spatial_poly, malignant, usage_raw]),
            spatial_names + [MALIGNANT_COL] + program_cols,
            program_cols,
        )
    if mode == "spatial_malignant_usage_norm":
        norm_names = [f"{col}__row_norm" for col in program_cols]
        return (
            np.column_stack([spatial_poly, malignant, usage_norm]),
            spatial_names + [MALIGNANT_COL] + norm_names,
            norm_names,
        )
    raise ValueError(f"Unknown feature mode: {mode}")


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rho, p = safe_spearman(y, pred)
    out = {
        "r2": float(r2_score(y, pred)),
        "rmse": rmse(y, pred),
        "spearman_rho": rho,
        "spearman_p": p,
    }
    return out


def nested_predictions(
    frame: pd.DataFrame,
    y: np.ndarray,
    X: np.ndarray,
    outer_splits: int,
    inner_splits: int,
    spatial_block_side: int,
) -> tuple[np.ndarray, list[float], CvScheme]:
    outer = build_cv(frame, target_splits=outer_splits, spatial_block_side=spatial_block_side)
    pred = np.full(len(y), np.nan, dtype=float)
    alphas: list[float] = []

    if X.shape[1] == 0:
        if outer.groups is None:
            iterator = outer.splitter.split(np.zeros((len(y), 1)), y)
        else:
            iterator = outer.splitter.split(np.zeros((len(y), 1)), y, groups=outer.groups)
        for train_idx, test_idx in iterator:
            pred[test_idx] = float(np.mean(y[train_idx]))
        return pred, alphas, outer

    if outer.groups is None:
        iterator = outer.splitter.split(X, y)
    else:
        iterator = outer.splitter.split(X, y, groups=outer.groups)

    for train_idx, test_idx in iterator:
        train_frame = frame.iloc[train_idx].reset_index(drop=True)
        search = fit_ridge_grid(
            X[train_idx],
            y[train_idx],
            train_frame,
            target_splits=min(inner_splits, max(2, len(train_idx))),
            spatial_block_side=spatial_block_side,
        )
        pred[test_idx] = search.predict(X[test_idx])
        alphas.append(float(search.best_params_["ridge__alpha"]))
    return pred, alphas, outer


def final_fit(
    X: np.ndarray,
    y: np.ndarray,
    frame: pd.DataFrame,
    feature_names: list[str],
    inner_splits: int,
    spatial_block_side: int,
) -> tuple[dict[str, float], pd.DataFrame, Pipeline | None]:
    if X.shape[1] == 0:
        return {"alpha": math.nan, "intercept_original_scale": float(np.mean(y))}, pd.DataFrame(), None
    search = fit_ridge_grid(X, y, frame, target_splits=inner_splits, spatial_block_side=spatial_block_side)
    model = search.best_estimator_
    scaler = model.named_steps["scale"]
    ridge = model.named_steps["ridge"]
    coef_scaled = ridge.coef_.astype(float)
    coef_original = coef_scaled / scaler.scale_
    intercept_original = float(ridge.intercept_ - np.sum(coef_scaled * scaler.mean_ / scaler.scale_))
    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "standardized_coef": coef_scaled,
            "original_scale_coef": coef_original,
        }
    )
    return (
        {
            "alpha": float(search.best_params_["ridge__alpha"]),
            "intercept_original_scale": intercept_original,
        },
        coef_df,
        model,
    )


def standardize_within_group(df: pd.DataFrame, group_cols: list[str], value_col: str, out_col: str) -> pd.DataFrame:
    out = df.copy()
    values = []
    for _, group in out.groupby(group_cols, sort=False):
        x = pd.to_numeric(group[value_col], errors="coerce")
        sd = float(x.std(ddof=0))
        if not np.isfinite(sd) or sd <= 1e-12:
            values.extend([math.nan] * len(group))
        else:
            values.extend(((x - float(x.mean())) / sd).tolist())
    out[out_col] = values
    return out


def program_columns(frame: pd.DataFrame, k_star: int) -> list[str]:
    suffix = f"__K{k_star}__P"
    cols = [c for c in frame.columns if str(c).find(suffix) >= 0 and PROGRAM_PATTERN.search(str(c))]
    return sorted(cols, key=lambda c: int(str(c).split("__P")[-1]))


def load_obs(path: Path) -> pd.DataFrame:
    adata = ad.read_h5ad(path, backed="r")
    obs = adata.obs.copy()
    obs.index = obs.index.astype(str)
    obs.index.name = "spot_id"
    if "spatial" in adata.obsm.keys():
        coords = np.asarray(adata.obsm["spatial"])
        obs["spatial_x"] = coords[:, 0]
        obs["spatial_y"] = coords[:, 1]
    return obs.reset_index()


def load_sample_frame(manifest_row: pd.Series, cnmf_root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    sample_label = str(manifest_row["sample_label"])
    sample_id = str(manifest_row["sample_id_on_disk"])
    dataset = str(manifest_row["dataset"])
    k_star = int(float(manifest_row["k_star"]))

    usage_path = cnmf_root / "per_sample" / sample_id / "representative_usage_kstar.csv"
    minimal_path = cnmf_root / "inputs" / f"{sample_label}__tumor_counts_minimal.h5ad"
    ready_path = Path(str(manifest_row["analysis_ready_h5ad_path"]))

    usage = pd.read_csv(usage_path)
    usage["spot_id"] = usage["spot_id"].astype(str)
    program_cols = program_columns(usage, k_star)
    minimal_obs = load_obs(minimal_path)
    ready_obs = load_obs(ready_path)

    coord_cols = ["spot_id", "array_row", "array_col"]
    if "spatial_x" in ready_obs and "spatial_y" in ready_obs:
        coord_cols += ["spatial_x", "spatial_y"]
    coords = ready_obs[[c for c in coord_cols if c in ready_obs.columns]].copy()

    metadata_cols = [
        "spot_id",
        "dataset",
        "sample_id_on_disk",
        "sample_label",
        "interface",
        MALIGNANT_COL,
        "total_counts",
        "n_genes_by_counts",
        SNAI1_COL,
    ]
    metadata = minimal_obs[[c for c in metadata_cols if c in minimal_obs.columns]].copy()
    frame = usage.merge(metadata, on=["spot_id", "dataset", "sample_id_on_disk", "sample_label"], how="inner")
    frame = frame.merge(coords, on="spot_id", how="left")

    for col in [SNAI1_COL, MALIGNANT_COL, "array_row", "array_col", "total_counts", "n_genes_by_counts", *program_cols]:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "interface" in frame:
        frame = frame[frame["interface"].astype(str).str.lower().eq("tumor")].copy()

    valid_cols = [SNAI1_COL, MALIGNANT_COL, "array_row", "array_col", *program_cols]
    valid_mask = np.ones(len(frame), dtype=bool)
    for col in valid_cols:
        valid_mask &= np.isfinite(pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float))
    frame = frame.loc[valid_mask].reset_index(drop=True)

    audit = {
        "dataset": dataset,
        "sample_id_on_disk": sample_id,
        "sample_label": sample_label,
        "k_star": k_star,
        "usage_path": str(usage_path),
        "minimal_h5ad_path": str(minimal_path),
        "analysis_ready_h5ad_path": str(ready_path),
        "usage_spots": int(len(usage)),
        "minimal_obs_spots": int(len(minimal_obs)),
        "merged_valid_tumor_spots": int(len(frame)),
        "n_programs": int(len(program_cols)),
        "usage_row_sum_min": float(frame[program_cols].sum(axis=1).min()) if len(frame) else math.nan,
        "usage_row_sum_median": float(frame[program_cols].sum(axis=1).median()) if len(frame) else math.nan,
        "usage_row_sum_max": float(frame[program_cols].sum(axis=1).max()) if len(frame) else math.nan,
        "snai1ac_min": float(frame[SNAI1_COL].min()) if len(frame) else math.nan,
        "snai1ac_median": float(frame[SNAI1_COL].median()) if len(frame) else math.nan,
        "snai1ac_max": float(frame[SNAI1_COL].max()) if len(frame) else math.nan,
        "malignant_median": float(frame[MALIGNANT_COL].median()) if len(frame) else math.nan,
    }
    return frame, audit


def add_program_annotation(weights: pd.DataFrame, cnmf_root: Path) -> pd.DataFrame:
    membership_path = cnmf_root / "meta" / "metaprogram_membership.csv"
    annot_path = cnmf_root / "meta" / "annotation" / "metaprogram_annotations.csv"
    out = weights.copy()
    if membership_path.exists():
        membership = pd.read_csv(membership_path)
        out = out.merge(
            membership[["program_id", "candidate_metaprogram_id", "metaprogram_id"]],
            on="program_id",
            how="left",
        )
    if annot_path.exists() and "metaprogram_id" in out:
        annot = pd.read_csv(annot_path)
        keep = [
            "metaprogram_id",
            "annotation_family",
            "provisional_label",
            "annotation_confidence",
            "likely_context",
            "top_consensus_genes",
        ]
        annot = annot[[c for c in keep if c in annot.columns]].drop_duplicates("metaprogram_id")
        out = out.merge(annot, on="metaprogram_id", how="left")
    return out


def analyze_models_for_sample(
    frame: pd.DataFrame,
    program_cols: list[str],
    outer_splits: int,
    inner_splits: int,
    spatial_block_side: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = frame[SNAI1_COL].to_numpy(dtype=float)
    model_rows = []
    prediction_frame = frame[
        ["dataset", "sample_id_on_disk", "sample_label", "spot_id", SNAI1_COL, MALIGNANT_COL, "array_row", "array_col"]
    ].copy()
    full_coef_df = pd.DataFrame()

    for mode in [
        "intercept_only",
        "usage_raw_only",
        "spatial",
        "spatial_malignant",
        "spatial_malignant_usage_raw",
        "spatial_malignant_usage_norm",
    ]:
        X, feature_names, active_program_features = feature_matrix(frame, program_cols, mode)
        pred, alphas, outer = nested_predictions(
            frame,
            y,
            X,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            spatial_block_side=spatial_block_side,
        )
        score = metrics(y, pred)
        fit_info, coef_df, _ = final_fit(
            X,
            y,
            frame,
            feature_names=feature_names,
            inner_splits=inner_splits,
            spatial_block_side=spatial_block_side,
        )
        model_rows.append(
            {
                "dataset": str(frame["dataset"].iloc[0]),
                "sample_id_on_disk": str(frame["sample_id_on_disk"].iloc[0]),
                "sample_label": str(frame["sample_label"].iloc[0]),
                "model": mode,
                "n_spots": int(len(frame)),
                "n_features": int(X.shape[1]),
                "n_program_features": int(len(active_program_features)),
                "cv_scheme": outer.name,
                "outer_splits": int(outer.n_splits),
                "spatial_group_count": int(outer.n_groups),
                "spatial_block_side": outer.block_side,
                "outer_alpha_mean": float(np.mean(alphas)) if alphas else math.nan,
                "outer_alpha_median": float(np.median(alphas)) if alphas else math.nan,
                "final_alpha": safe_float(fit_info.get("alpha")),
                **{f"cv_{k}": v for k, v in score.items()},
            }
        )
        prediction_frame[f"pred_{mode}"] = pred

        if mode == "spatial_malignant_usage_raw":
            full_fit_info = fit_info
            full_coef_df = coef_df.copy()
            full_coef_df["dataset"] = str(frame["dataset"].iloc[0])
            full_coef_df["sample_id_on_disk"] = str(frame["sample_id_on_disk"].iloc[0])
            full_coef_df["sample_label"] = str(frame["sample_label"].iloc[0])
            full_coef_df["is_program_feature"] = full_coef_df["feature"].isin(program_cols)
            full_coef_df["program_id"] = np.where(full_coef_df["is_program_feature"], full_coef_df["feature"], "")
            full_coef_df["intercept_original_scale"] = safe_float(full_fit_info["intercept_original_scale"])

            coef_map = dict(zip(full_coef_df["feature"], full_coef_df["original_scale_coef"], strict=False))
            intercept = safe_float(full_fit_info["intercept_original_scale"])
            baseline_features = [f for f in full_coef_df["feature"] if f not in program_cols]
            X_full, full_names, _ = feature_matrix(frame, program_cols, mode)
            full_feature_df = pd.DataFrame(X_full, columns=full_names)
            baseline_component = intercept + full_feature_df[baseline_features].mul(
                [coef_map[f] for f in baseline_features], axis=1
            ).sum(axis=1)
            usage_component = frame[program_cols].mul([coef_map[p] for p in program_cols], axis=1).sum(axis=1)
            prediction_frame["full_model_baseline_component"] = baseline_component.to_numpy(dtype=float)
            prediction_frame["full_model_usage_component"] = usage_component.to_numpy(dtype=float)
            prediction_frame["full_model_reconstructed_prediction"] = (
                prediction_frame["full_model_baseline_component"] + prediction_frame["full_model_usage_component"]
            )

    model_df = pd.DataFrame(model_rows)
    baseline_r2 = float(
        model_df.loc[model_df["model"].eq("spatial_malignant"), "cv_r2"].iloc[0]
    )
    full_r2 = float(
        model_df.loc[model_df["model"].eq("spatial_malignant_usage_raw"), "cv_r2"].iloc[0]
    )
    model_df["delta_r2_vs_spatial_malignant"] = model_df["cv_r2"] - baseline_r2
    model_df["full_raw_usage_delta_r2"] = full_r2 - baseline_r2
    return model_df, full_coef_df, prediction_frame


def load_signature_weights(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        return {str(k): float(v) for k, v in json.load(handle).items()}


def project_signature_onto_spectra(
    cnmf_root: Path,
    manifest_row: pd.Series,
    signature_weights: dict[str, float],
) -> pd.DataFrame:
    sample_id = str(manifest_row["sample_id_on_disk"])
    sample_label = str(manifest_row["sample_label"])
    k_star = int(float(manifest_row["k_star"]))
    spectra_path = cnmf_root / "per_sample" / sample_id / "extracted_program_spectra.csv"
    spectra = pd.read_csv(spectra_path)
    spectra = spectra[spectra["is_k_star"].astype(str).str.lower().eq("true")].copy()
    if spectra.empty:
        spectra = pd.read_csv(spectra_path)
        spectra = spectra[spectra["source_k"].astype(float).astype(int).eq(k_star)].copy()

    gene_cols = [c for c in spectra.columns if str(c).startswith("__gene__")]
    genes = [c.replace("__gene__", "", 1) for c in gene_cols]
    present_pairs = [(col, gene, signature_weights[gene]) for col, gene in zip(gene_cols, genes, strict=False) if gene in signature_weights]
    abs_sum = sum(abs(weight) for _, _, weight in present_pairs)

    rows = []
    for _, row in spectra.iterrows():
        raw_dot = 0.0
        absnorm_dot = 0.0
        pos_dot = 0.0
        neg_dot = 0.0
        for col, gene, weight in present_pairs:
            value = safe_float(row[col])
            if not np.isfinite(value):
                continue
            raw_dot += weight * value
            if abs_sum > 0:
                absnorm_dot += (weight / abs_sum) * value
            if weight > 0:
                pos_dot += weight * value
            elif weight < 0:
                neg_dot += weight * value
        rows.append(
            {
                "dataset": str(row.get("dataset", manifest_row["dataset"])),
                "sample_id_on_disk": sample_id,
                "sample_label": sample_label,
                "program_id": str(row["program_id"]),
                "source_k": int(float(row["source_k"])),
                "local_program_index": int(float(row["local_program_index"])),
                "n_signature_genes_total": int(len(signature_weights)),
                "n_signature_genes_in_spectra": int(len(present_pairs)),
                "signature_gene_fraction_in_spectra": float(len(present_pairs) / len(signature_weights)),
                "signature_projection_raw_dot": float(raw_dot),
                "signature_projection_absnorm_dot": float(absnorm_dot),
                "signature_projection_positive_weight_dot": float(pos_dot),
                "signature_projection_negative_weight_dot": float(neg_dot),
                "spectra_path": str(spectra_path),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = standardize_within_group(
            out,
            ["sample_label"],
            "signature_projection_absnorm_dot",
            "signature_projection_absnorm_z_within_sample",
        )
        out = standardize_within_group(
            out,
            ["sample_label"],
            "signature_projection_raw_dot",
            "signature_projection_raw_z_within_sample",
        )
    return out


def weight_table_from_coefs(coefs: pd.DataFrame, cnmf_root: Path) -> pd.DataFrame:
    programs = coefs[coefs["is_program_feature"]].copy()
    programs = programs.rename(
        columns={
            "standardized_coef": "program_standardized_weight",
            "original_scale_coef": "program_original_scale_weight",
        }
    )
    programs["abs_standardized_weight"] = programs["program_standardized_weight"].abs()
    programs["weight_direction"] = np.where(
        programs["program_standardized_weight"] > 0,
        "positive",
        np.where(programs["program_standardized_weight"] < 0, "negative", "zero"),
    )
    programs["positive_weight_share"] = 0.0
    programs["negative_weight_share"] = 0.0
    for sample_label, group in programs.groupby("sample_label", sort=False):
        pos = group["program_standardized_weight"].clip(lower=0)
        neg = (-group["program_standardized_weight"].clip(upper=0))
        if float(pos.sum()) > 0:
            programs.loc[group.index, "positive_weight_share"] = pos / float(pos.sum())
        if float(neg.sum()) > 0:
            programs.loc[group.index, "negative_weight_share"] = neg / float(neg.sum())
        programs.loc[group.index, "abs_weight_rank"] = group["abs_standardized_weight"].rank(
            method="first", ascending=False
        )
    programs["abs_weight_rank"] = programs["abs_weight_rank"].astype(int)
    return add_program_annotation(programs, cnmf_root)


def model_projection_concordance(weights: pd.DataFrame, projection: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = weights.merge(
        projection,
        on=["dataset", "sample_id_on_disk", "sample_label", "program_id"],
        how="left",
    )
    rows = []
    for sample_label, group in joined.groupby("sample_label", sort=True):
        rho_abs, p_abs = safe_spearman(
            group["program_standardized_weight"], group["signature_projection_absnorm_dot"]
        )
        pear_abs, pear_p_abs = safe_pearson(
            group["program_standardized_weight"], group["signature_projection_absnorm_dot"]
        )
        rho_raw, p_raw = safe_spearman(
            group["program_standardized_weight"], group["signature_projection_raw_dot"]
        )
        rows.append(
            {
                "dataset": str(group["dataset"].iloc[0]),
                "sample_id_on_disk": str(group["sample_id_on_disk"].iloc[0]),
                "sample_label": sample_label,
                "n_programs": int(len(group)),
                "learned_vs_spectrum_spearman_absnorm": rho_abs,
                "learned_vs_spectrum_spearman_absnorm_p": p_abs,
                "learned_vs_spectrum_pearson_absnorm": pear_abs,
                "learned_vs_spectrum_pearson_absnorm_p": pear_p_abs,
                "learned_vs_spectrum_spearman_raw": rho_raw,
                "learned_vs_spectrum_spearman_raw_p": p_raw,
            }
        )
    return joined, pd.DataFrame(rows)


def bh_fdr(values: pd.Series) -> pd.Series:
    arr = values.astype(float).to_numpy()
    out = np.full(arr.shape, np.nan, dtype=float)
    mask = np.isfinite(arr)
    if not mask.any():
        return pd.Series(out, index=values.index)
    valid = arr[mask]
    order = np.argsort(valid)
    ranks = np.arange(1, len(valid) + 1, dtype=float)
    adjusted = valid[order] * len(valid) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out[np.where(mask)[0][order]] = adjusted
    return pd.Series(out, index=values.index)


def summarize_weights_by_annotation(weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    label_col = "provisional_label" if "provisional_label" in weights.columns else "metaprogram_id"
    family_col = "annotation_family" if "annotation_family" in weights.columns else label_col
    for (family, label), group in weights.groupby([family_col, label_col], dropna=False, sort=True):
        vals = pd.to_numeric(group["program_standardized_weight"], errors="coerce").dropna()
        if vals.empty:
            continue
        n_pos = int((vals > 0).sum())
        n_neg = int((vals < 0).sum())
        majority = max(n_pos, n_neg)
        p = binomtest(majority, len(vals), 0.5, alternative="two-sided").pvalue if len(vals) else math.nan
        rows.append(
            {
                family_col: family,
                label_col: label,
                "n_program_instances": int(len(vals)),
                "n_samples": int(group["sample_label"].nunique()),
                "median_standardized_weight": float(vals.median()),
                "mean_standardized_weight": float(vals.mean()),
                "median_abs_standardized_weight": float(vals.abs().median()),
                "n_positive": n_pos,
                "n_negative": n_neg,
                "positive_fraction": float(n_pos / len(vals)),
                "directional_majority_fraction": float(majority / len(vals)),
                "sign_skew_binom_p": float(p),
                "top_program_ids_by_abs_weight": ";".join(
                    group.sort_values("abs_standardized_weight", ascending=False)["program_id"].head(8).astype(str)
                ),
            }
        )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["sign_skew_fdr_bh"] = bh_fdr(summary["sign_skew_binom_p"])
        summary = summary.sort_values(
            ["median_abs_standardized_weight", "n_program_instances"],
            ascending=[False, False],
        ).reset_index(drop=True)

    top_rows = []
    for sample_label, group in weights.groupby("sample_label", sort=True):
        for direction, subset in [
            ("positive", group[group["program_standardized_weight"] > 0].sort_values("program_standardized_weight", ascending=False)),
            ("negative", group[group["program_standardized_weight"] < 0].sort_values("program_standardized_weight", ascending=True)),
        ]:
            for rank, (_, row) in enumerate(subset.head(3).iterrows(), start=1):
                top_rows.append(
                    {
                        "dataset": row["dataset"],
                        "sample_id_on_disk": row["sample_id_on_disk"],
                        "sample_label": sample_label,
                        "direction": direction,
                        "rank": rank,
                        "program_id": row["program_id"],
                        "program_standardized_weight": row["program_standardized_weight"],
                        "program_original_scale_weight": row["program_original_scale_weight"],
                        "positive_weight_share": row.get("positive_weight_share", math.nan),
                        "negative_weight_share": row.get("negative_weight_share", math.nan),
                        "metaprogram_id": row.get("metaprogram_id", ""),
                        "annotation_family": row.get("annotation_family", ""),
                        "provisional_label": row.get("provisional_label", ""),
                    }
                )
    return summary, pd.DataFrame(top_rows)


def plot_observed_predicted(predictions: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    for sample_label, group in predictions.groupby("sample_label", sort=True):
        fig, ax = plt.subplots(figsize=(5.8, 5.2))
        ax.scatter(
            group[SNAI1_COL],
            group["pred_spatial_malignant_usage_raw"],
            s=15,
            alpha=0.55,
            color="#2b6f83",
            edgecolor="none",
        )
        vals = pd.concat([group[SNAI1_COL], group["pred_spatial_malignant_usage_raw"]]).to_numpy(dtype=float)
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        ax.plot([lo, hi], [lo, hi], color="#555555", linewidth=1.0, linestyle="--")
        rho, _ = safe_spearman(group[SNAI1_COL], group["pred_spatial_malignant_usage_raw"])
        r2 = r2_score(group[SNAI1_COL], group["pred_spatial_malignant_usage_raw"])
        ax.set_xlabel("Observed SNAI1-ac score")
        ax.set_ylabel("CV predicted SNAI1-ac score")
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
        path = out_dir / f"{sample_label}__observed_vs_cv_predicted_full_model.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_model_summary(performance: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    primary = performance[performance["model"].eq("spatial_malignant_usage_raw")].copy()
    baseline = performance[performance["model"].eq("spatial_malignant")].copy()
    merged = primary.merge(
        baseline[["sample_label", "cv_r2"]],
        on="sample_label",
        suffixes=("_full", "_baseline"),
    )
    merged["delta_r2"] = merged["cv_r2_full"] - merged["cv_r2_baseline"]
    merged = merged.sort_values("delta_r2")

    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    ax.barh(merged["sample_label"], merged["delta_r2"], color="#6f8f72")
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("CV R2 gain after adding cNMF usage")
    ax.set_ylabel("")
    ax.set_title("Added explanatory value of tumour-program usage")
    fig.tight_layout()
    path = out_dir / "per_sample_delta_r2_usage_after_spatial_malignant.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    pivot = performance.pivot_table(index="sample_label", columns="model", values="cv_r2")
    wanted = ["intercept_only", "usage_raw_only", "spatial", "spatial_malignant", "spatial_malignant_usage_raw"]
    pivot = pivot[[c for c in wanted if c in pivot]].sort_values("spatial_malignant_usage_raw")
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=-0.6, vmax=0.35)
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title("Cross-validated R2 by nested model (display clipped)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="CV R2")
    fig.tight_layout()
    path = out_dir / "nested_model_cv_r2_heatmap.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)
    return paths


def plot_top_weights(weights: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    top = weights.sort_values(["sample_label", "abs_weight_rank"]).groupby("sample_label").head(3).copy()
    top["label"] = top["sample_id_on_disk"] + " " + top["program_id"].str.extract(r"(__P\d+)$", expand=False).fillna("")
    top = top.sort_values("program_standardized_weight")
    fig, ax = plt.subplots(figsize=(9, max(5, 0.23 * len(top))))
    colors = np.where(top["program_standardized_weight"] >= 0, "#9b4f4f", "#466c8f")
    ax.barh(top["label"], top["program_standardized_weight"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Standardized programme weight")
    ax.set_ylabel("")
    ax.set_title("Top weighted cNMF programmes per sample")
    fig.tight_layout()
    path = out_dir / "top_program_weights_per_sample.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)
    return paths


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
        "# SNAI1-ac cNMF Program Decomposition v1",
        "",
        "This branch fits one model per sample on tumour spots only:",
        "",
        "`SNAI1-ac_score ~ spatial baseline + malignant fraction + K* cNMF programme usage`",
        "",
        "The production SNAI1-ac score is the saved weighted EnrichMap score with smoothing and spatial covariate correction. Therefore the learned usage model is the primary decomposition, while the spectrum projection is an approximate mechanistic cross-check against cNMF programme spectra.",
        "",
        "## Input audit",
        "",
        f"- Samples attempted: {audit.shape[0]}",
        f"- Samples modelled: {primary.shape[0]}",
        f"- Total valid tumour spots modelled: {int(audit['merged_valid_tumor_spots'].sum())}",
        f"- Median K* programmes per sample: {audit['n_programs'].median():.1f}",
        "",
        "## Model summary",
        "",
        f"- Median full-model CV R2: {merged['cv_r2_full'].median():.3f}",
        f"- Median spatial+malignant baseline CV R2: {merged['cv_r2_baseline'].median():.3f}",
        f"- Median added CV R2 from raw cNMF usage: {merged['delta_r2'].median():.3f}",
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
        "- `01_input_audit/tables/input_audit.csv`",
        "- `02_per_sample_usage_models/tables/per_sample_model_performance.csv`",
        "- `02_per_sample_usage_models/tables/per_sample_program_weights.csv`",
        "- `02_per_sample_usage_models/tables/per_spot_predictions.csv`",
        "- `03_program_spectrum_projection/tables/program_signature_projection.csv`",
        "- `04_model_projection_concordance/tables/program_weight_projection_joined.csv`",
        "- `04_model_projection_concordance/tables/per_sample_weight_projection_concordance.csv`",
        "- `05_cross_sample_summary/tables/cross_sample_summary.csv`",
        "- `05_cross_sample_summary/tables/program_weight_summary_by_annotation.csv`",
        "- `05_cross_sample_summary/tables/top_positive_negative_programs_per_sample.csv`",
        "",
        "## Interpretation guardrails",
        "",
        "- This is an internal decomposition of a score derived from the same transcriptome used for cNMF.",
        "- Programme weights are per-sample coefficients, not cross-sample universal effects.",
        "- Raw cNMF usage is the primary exposure. Row-normalized usage is saved as a sensitivity model.",
        "- The spectrum projection is approximate because EnrichMap applies smoothing, spatial covariate correction, and robust scaling that cannot be exactly represented by a simple spectrum dot product.",
    ]
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(output_root: Path, script_path: Path, cnmf_root: Path, signature_weights: Path) -> None:
    manifest = {
        "branch": "snai1ac_program_decomposition_v1",
        "created_by_script": str(script_path),
        "cnmf_root": str(cnmf_root),
        "signature_weights": str(signature_weights),
        "old_regression_branch_status": "left_intact_superseded_not_deleted",
        "primary_model": "SNAI1-ac_score ~ spatial polynomial baseline + Malignant + raw K* cNMF usage",
        "spatial_baseline_features": ["array_row", "array_col", "array_row2", "array_row_array_col", "array_col2"],
        "tumour_filter": "cNMF inputs already restrict to interface == Tumor; script re-checks interface == tumor after merge",
    }
    (output_root / "00_manifest_and_provenance" / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    out = ensure(args.output_root)
    dirs = {
        "manifest": ensure(out / "00_manifest_and_provenance"),
        "audit": ensure(out / "01_input_audit" / "tables"),
        "models": ensure(out / "02_per_sample_usage_models" / "tables"),
        "model_figs": ensure(out / "02_per_sample_usage_models" / "figures"),
        "projection": ensure(out / "03_program_spectrum_projection" / "tables"),
        "concordance": ensure(out / "04_model_projection_concordance" / "tables"),
        "summary": ensure(out / "05_cross_sample_summary" / "tables"),
        "figures": ensure(out / "06_figures"),
        "scripts": ensure(out / "scripts_used"),
    }

    script_copy = dirs["scripts"] / DEFAULT_SCRIPT.name
    shutil.copy2(DEFAULT_SCRIPT, script_copy)

    manifest_path = args.cnmf_root / "sample_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["eligible_for_cnmf"].astype(str).str.lower().eq("true")].copy()
    signature_weights = load_signature_weights(args.signature_weights)

    audit_rows: list[dict[str, object]] = []
    performance_tables: list[pd.DataFrame] = []
    coef_tables: list[pd.DataFrame] = []
    prediction_tables: list[pd.DataFrame] = []
    projection_tables: list[pd.DataFrame] = []

    for _, row in manifest.sort_values(["dataset", "sample_id_on_disk"]).iterrows():
        sample_label = str(row["sample_label"])
        print(f"Analysing {sample_label}", flush=True)
        frame, audit = load_sample_frame(row, args.cnmf_root)
        audit_rows.append(audit)
        if int(audit["merged_valid_tumor_spots"]) < args.min_spots:
            print(f"  skipped: too few valid spots ({audit['merged_valid_tumor_spots']})", flush=True)
            continue
        program_cols = program_columns(frame, int(float(row["k_star"])))
        perf, coefs, predictions = analyze_models_for_sample(
            frame,
            program_cols,
            outer_splits=args.outer_splits,
            inner_splits=args.inner_splits,
            spatial_block_side=args.spatial_block_side,
        )
        projection = project_signature_onto_spectra(args.cnmf_root, row, signature_weights)
        performance_tables.append(perf)
        coef_tables.append(coefs)
        prediction_tables.append(predictions)
        projection_tables.append(projection)

    audit_df = pd.DataFrame(audit_rows)
    performance = pd.concat(performance_tables, ignore_index=True)
    all_coefs = pd.concat(coef_tables, ignore_index=True)
    weights = weight_table_from_coefs(all_coefs, args.cnmf_root)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    projection = pd.concat(projection_tables, ignore_index=True)
    joined, concordance = model_projection_concordance(weights, projection)
    label_summary, top_programs = summarize_weights_by_annotation(weights)

    audit_df.to_csv(dirs["audit"] / "input_audit.csv", index=False)
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
    summary.to_csv(dirs["summary"] / "cross_sample_summary.csv", index=False)

    plot_observed_predicted(predictions, dirs["model_figs"])
    plot_model_summary(performance, dirs["figures"])
    plot_top_weights(weights, dirs["figures"])

    write_manifest(out, script_copy, args.cnmf_root, args.signature_weights)
    write_readme(out, performance, audit_df, concordance, label_summary)
    print(f"Wrote branch to {out}", flush=True)


if __name__ == "__main__":
    main()
