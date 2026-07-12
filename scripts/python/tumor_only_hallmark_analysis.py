"""
Tumor-only Hallmark analysis for SNAI1-ac.

Primary question:
    Within SpaCET-labeled tumor spots, which Hallmark programs track with
    SNAI1-ac, how much of that relationship persists after adjustment for
    malignant fraction and technical factors, and do combinations of Hallmarks
    explain additional variation?

This script adds three complementary layers:
1. Univariate tumor-only associations:
   - Spearman correlation
   - rank-based partial correlation controlling for covariates
2. Nonlinear diagnostics:
   - adjusted quadratic test after residualizing covariates
3. Combined Hallmark model:
   - ridge regression on a prespecified Hallmark focus panel
   - delta R^2 beyond baseline covariates

Primary subset:
    interface == "Tumor"

Sensitivity subset:
    interface == "Tumor" and Malignant >= HIGH_PURITY_THRESHOLD

Outputs:
    D:\HGSOC_Spatial_Atlas\05_analysis_ready\S2b_Tumor_Only_Hallmark_Correlation\
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score


BASE_DIR = Path(r"D:\HGSOC_Spatial_Atlas")
VISIUM_DIR = BASE_DIR / "05_analysis_ready" / "visium"
OUTPUT_DIR = BASE_DIR / "05_analysis_ready" / "S2b_Tumor_Only_Hallmark_Correlation"

SNAI1_COL = "SNAI1-ac_score"
INTERFACE_COL = "interface"
TUMOR_LABEL = "Tumor"
HIGH_PURITY_THRESHOLD = 0.75
MIN_SPOTS = 40
MIN_MODEL_SPOTS = 50
RANDOM_STATE = 42

KNOWN_DATASETS = [
    "denisenko_2022",
    "yamamoto_2025",
    "ju_2024",
    "stur_2021",
    "10X_ov_standard",
    "10X_ov_11mm",
]

FOCUS_PANEL = [
    "EPITHELIAL_MESENCHYMAL_TRANSITION",
    "HYPOXIA",
    "TGF_BETA_SIGNALING",
    "INFLAMMATORY_RESPONSE",
    "GLYCOLYSIS",
    "ANGIOGENESIS",
    "OXIDATIVE_PHOSPHORYLATION",
    "MYC_TARGETS_V1",
    "TNFA_SIGNALING_VIA_NFKB",
    "E2F_TARGETS",
]

METABOLIC_PANEL = [
    "OXIDATIVE_PHOSPHORYLATION",
    "GLYCOLYSIS",
    "FATTY_ACID_METABOLISM",
    "CHOLESTEROL_HOMEOSTASIS",
    "ADIPOGENESIS",
    "PEROXISOME",
    "REACTIVE_OXYGEN_SPECIES_PATHWAY",
    "XENOBIOTIC_METABOLISM",
    "BILE_ACID_METABOLISM",
]

COVARIATE_CANDIDATES = [
    "Malignant",
    "total_counts",
    "n_genes_by_counts",
]

RIDGE_ALPHAS = np.logspace(-3, 3, 13)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tumor-only Hallmark analysis for SNAI1-ac"
    )
    parser.add_argument(
        "--min-spots",
        type=int,
        default=MIN_SPOTS,
        help="Minimum number of tumor spots required for univariate analysis",
    )
    parser.add_argument(
        "--min-model-spots",
        type=int,
        default=MIN_MODEL_SPOTS,
        help="Minimum number of tumor spots required for combined models",
    )
    parser.add_argument(
        "--high-purity-threshold",
        type=float,
        default=HIGH_PURITY_THRESHOLD,
        help="Sensitivity cutoff for Malignant fraction inside tumor spots",
    )
    return parser.parse_args()


def discover_h5ads() -> list[Path]:
    files: list[Path] = []
    for dataset in KNOWN_DATASETS:
        dataset_dir = VISIUM_DIR / dataset
        if not dataset_dir.exists():
            continue
        for sample_dir in sorted(dataset_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            h5ad_path = sample_dir / f"{sample_dir.name}.h5ad"
            if h5ad_path.exists():
                files.append(h5ad_path)
    return files


def get_sample_identity(h5ad_path: Path) -> tuple[str, str, str]:
    sample = h5ad_path.stem
    dataset = h5ad_path.parent.parent.name
    sample_label = f"{dataset}__{sample}"
    return dataset, sample, sample_label


def score_columns(obs_columns: list[str]) -> list[str]:
    return sorted(
        col for col in obs_columns
        if col.startswith("HALLMARK_") and col.endswith("_score")
    )


def pathway_name(score_col: str) -> str:
    return score_col.replace("HALLMARK_", "").replace("_score", "")


def panel_score_columns(obs_columns: list[str], panel: list[str]) -> list[str]:
    cols = []
    colset = set(obs_columns)
    for hallmark in panel:
        col = f"HALLMARK_{hallmark}_score"
        if col in colset:
            cols.append(col)
    return cols


def get_primary_tumor_mask(obs: pd.DataFrame) -> tuple[pd.Series, str]:
    if INTERFACE_COL in obs.columns:
        mask = obs[INTERFACE_COL].astype(str) == TUMOR_LABEL
        if mask.sum() > 0:
            return mask, "spacet_interface_tumor"
    if "Malignant" in obs.columns:
        mask = obs["Malignant"] > 0.5
        return mask, "malignant_gt_0p5_fallback"
    return pd.Series(False, index=obs.index), "missing_tumor_definition"


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


def standardize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    sd = np.std(arr, ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(arr, dtype=float)
    return (arr - np.mean(arr)) / sd


def residualize(y: np.ndarray, covars: np.ndarray | None) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if covars is None or covars.size == 0:
        return y - np.mean(y)
    X = np.asarray(covars, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    mask = finite_mask(x, y)
    x = np.asarray(x)[mask]
    y = np.asarray(y)[mask]
    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, len(x)
    r, p = stats.spearmanr(x, y)
    return float(r), float(p), int(len(x))


def partial_spearman(
    x: np.ndarray,
    y: np.ndarray,
    covars: np.ndarray | None,
) -> tuple[float, float, int]:
    if covars is None or covars.size == 0:
        return np.nan, np.nan, 0

    X = np.asarray(x, dtype=float)
    Y = np.asarray(y, dtype=float)
    C = np.asarray(covars, dtype=float)
    if C.ndim == 1:
        C = C.reshape(-1, 1)

    mask = finite_mask(X, Y, *[C[:, i] for i in range(C.shape[1])])
    X = X[mask]
    Y = Y[mask]
    C = C[mask]

    if len(X) < 6 or np.std(X) == 0 or np.std(Y) == 0:
        return np.nan, np.nan, len(X)

    X_rank = rankdata(X)
    Y_rank = rankdata(Y)
    C_rank = np.column_stack([rankdata(C[:, i]) for i in range(C.shape[1])])

    x_resid = residualize(X_rank, C_rank)
    y_resid = residualize(Y_rank, C_rank)
    if np.std(x_resid) == 0 or np.std(y_resid) == 0:
        return np.nan, np.nan, len(X)

    r, p = stats.pearsonr(x_resid, y_resid)
    return float(r), float(p), int(len(X))


def quadratic_diagnostic(
    x: np.ndarray,
    y: np.ndarray,
    covars: np.ndarray | None,
) -> dict[str, float]:
    X = np.asarray(x, dtype=float)
    Y = np.asarray(y, dtype=float)

    if covars is not None and covars.size != 0:
        C = np.asarray(covars, dtype=float)
        if C.ndim == 1:
            C = C.reshape(-1, 1)
        mask = finite_mask(X, Y, *[C[:, i] for i in range(C.shape[1])])
        X = X[mask]
        Y = Y[mask]
        C = C[mask]
        X = residualize(X, C)
        Y = residualize(Y, C)
    else:
        mask = finite_mask(X, Y)
        X = X[mask]
        Y = Y[mask]

    n = len(X)
    if n < 10 or np.std(X) == 0 or np.std(Y) == 0:
        return {
            "quadratic_linear_r2": np.nan,
            "quadratic_full_r2": np.nan,
            "quadratic_delta_r2": np.nan,
            "quadratic_term": np.nan,
            "quadratic_f": np.nan,
            "quadratic_p": np.nan,
            "quadratic_n": n,
        }

    x_std = standardize(X)
    y_std = standardize(Y)

    design_linear = np.column_stack([np.ones(n), x_std])
    design_full = np.column_stack([np.ones(n), x_std, x_std ** 2])

    beta_linear, *_ = np.linalg.lstsq(design_linear, y_std, rcond=None)
    beta_full, *_ = np.linalg.lstsq(design_full, y_std, rcond=None)

    resid_linear = y_std - design_linear @ beta_linear
    resid_full = y_std - design_full @ beta_full
    rss_linear = float(np.sum(resid_linear ** 2))
    rss_full = float(np.sum(resid_full ** 2))
    tss = float(np.sum((y_std - np.mean(y_std)) ** 2))

    linear_r2 = 1 - rss_linear / tss if tss > 0 else np.nan
    full_r2 = 1 - rss_full / tss if tss > 0 else np.nan
    delta_r2 = full_r2 - linear_r2 if np.isfinite(linear_r2) and np.isfinite(full_r2) else np.nan

    df_num = 1
    df_den = n - design_full.shape[1]
    if df_den <= 0 or rss_full <= 0 or rss_linear < rss_full:
        f_stat = np.nan
        p_value = np.nan
    else:
        f_stat = ((rss_linear - rss_full) / df_num) / (rss_full / df_den)
        p_value = float(stats.f.sf(f_stat, df_num, df_den))

    return {
        "quadratic_linear_r2": linear_r2,
        "quadratic_full_r2": full_r2,
        "quadratic_delta_r2": delta_r2,
        "quadratic_term": float(beta_full[-1]),
        "quadratic_f": float(f_stat) if np.isfinite(f_stat) else np.nan,
        "quadratic_p": p_value,
        "quadratic_n": n,
    }


def fisher_z_meta(r_values: np.ndarray, n_values: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(r_values) & np.isfinite(n_values) & (n_values > 3) & (np.abs(r_values) < 1)
    r_values = r_values[valid]
    n_values = n_values[valid]
    if len(r_values) == 0:
        return np.nan, np.nan
    z_values = np.arctanh(np.clip(r_values, -0.9999, 0.9999))
    weights = n_values - 3
    z_combined = np.average(z_values, weights=weights)
    se_combined = 1.0 / np.sqrt(np.sum(weights))
    z_stat = z_combined / se_combined
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return float(np.tanh(z_combined)), float(p_value)


def cv_splits(n_obs: int) -> int:
    if n_obs >= 100:
        return 5
    if n_obs >= 60:
        return 4
    return 3


def cross_validated_r2(model, X: np.ndarray, y: np.ndarray) -> float:
    n_obs = len(y)
    if n_obs < 10:
        return np.nan
    splitter = KFold(
        n_splits=cv_splits(n_obs),
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    scores = cross_val_score(model, X, y, cv=splitter, scoring="r2")
    return float(np.mean(scores))


def build_covariate_matrix(obs: pd.DataFrame) -> tuple[np.ndarray | None, list[str]]:
    covar_cols = [col for col in COVARIATE_CANDIDATES if col in obs.columns]
    if not covar_cols:
        return None, []
    covars = obs[covar_cols].apply(pd.to_numeric, errors="coerce").values
    return covars, covar_cols


def run_panel_model(
    obs: pd.DataFrame,
    panel_cols: list[str],
    min_model_spots: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    available = []
    for col in panel_cols:
        if col in obs.columns and np.nanstd(obs[col].values) > 0:
            available.append(col)

    result_stub = {
        "n_spots": len(obs),
        "n_hallmarks_used": len(available),
        "alpha_hallmarks_only": np.nan,
        "alpha_combined": np.nan,
        "cv_r2_baseline": np.nan,
        "cv_r2_hallmarks_only": np.nan,
        "cv_r2_combined": np.nan,
        "delta_r2_vs_baseline": np.nan,
    }

    if len(obs) < min_model_spots or len(available) < 3:
        return result_stub, []

    y = obs[SNAI1_COL].astype(float).values
    X_h = obs[available].apply(pd.to_numeric, errors="coerce").values
    covars, covar_cols = build_covariate_matrix(obs)

    blocks = [y, X_h]
    if covars is not None:
        if covars.ndim == 1:
            covars = covars.reshape(-1, 1)
        blocks.append(covars)

    mask = np.ones(len(obs), dtype=bool)
    for block in blocks:
        if block.ndim == 1:
            mask &= np.isfinite(block)
        else:
            mask &= np.all(np.isfinite(block), axis=1)

    y = y[mask]
    X_h = X_h[mask]
    covars = covars[mask] if covars is not None else None

    if len(y) < min_model_spots:
        result_stub["n_spots"] = len(y)
        return result_stub, []

    y_std = standardize(y)
    X_h_std = np.column_stack([standardize(X_h[:, i]) for i in range(X_h.shape[1])])

    if covars is not None and covars.size != 0:
        X_cov_std = np.column_stack([standardize(covars[:, i]) for i in range(covars.shape[1])])
    else:
        X_cov_std = np.empty((len(y_std), 0))
        covar_cols = []

    lr_baseline = LinearRegression()
    best_h = np.nan
    best_c = np.nan

    best_h_model = None
    best_c_model = None
    best_h_score = -np.inf
    best_c_score = -np.inf

    for alpha in RIDGE_ALPHAS:
        model_h = Ridge(alpha=float(alpha))
        score_h = cross_validated_r2(model_h, X_h_std, y_std)
        if np.isfinite(score_h) and score_h > best_h_score:
            best_h_score = score_h
            best_h = float(alpha)
            best_h_model = model_h

        X_comb = np.column_stack([X_cov_std, X_h_std]) if X_cov_std.shape[1] else X_h_std
        model_c = Ridge(alpha=float(alpha))
        score_c = cross_validated_r2(model_c, X_comb, y_std)
        if np.isfinite(score_c) and score_c > best_c_score:
            best_c_score = score_c
            best_c = float(alpha)
            best_c_model = model_c

    if best_h_model is None or best_c_model is None:
        result_stub["n_spots"] = len(y)
        return result_stub, []

    if X_cov_std.shape[1]:
        baseline_r2 = cross_validated_r2(lr_baseline, X_cov_std, y_std)
    else:
        baseline_r2 = np.nan

    hallmarks_r2 = cross_validated_r2(best_h_model, X_h_std, y_std)
    X_combined = np.column_stack([X_cov_std, X_h_std]) if X_cov_std.shape[1] else X_h_std
    combined_r2 = cross_validated_r2(best_c_model, X_combined, y_std)

    best_h_model.fit(X_h_std, y_std)
    best_c_model.fit(X_combined, y_std)

    result = {
        "n_spots": len(y),
        "n_hallmarks_used": len(available),
        "alpha_hallmarks_only": best_h,
        "alpha_combined": best_c,
        "cv_r2_baseline": baseline_r2,
        "cv_r2_hallmarks_only": hallmarks_r2,
        "cv_r2_combined": combined_r2,
        "delta_r2_vs_baseline": (
            combined_r2 - baseline_r2
            if np.isfinite(combined_r2) and np.isfinite(baseline_r2)
            else np.nan
        ),
    }

    coef_rows = []
    offset = X_cov_std.shape[1]
    for idx, col in enumerate(available):
        coef_rows.append(
            {
                "pathway": pathway_name(col),
                "score_col": col,
                "hallmark_only_coef": float(best_h_model.coef_[idx]),
                "combined_coef": float(best_c_model.coef_[offset + idx]),
            }
        )

    return result, coef_rows


def run_panel_pca(
    obs: pd.DataFrame,
    panel_cols: list[str],
    anchor_score_col: str,
    min_model_spots: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    available = []
    for col in panel_cols:
        if col in obs.columns and np.nanstd(obs[col].values) > 0:
            available.append(col)

    result_stub = {
        "n_spots": len(obs),
        "n_hallmarks_used": len(available),
        "pc1_explained_variance": np.nan,
        "pc2_explained_variance": np.nan,
        "pc1_snai1_r": np.nan,
        "pc1_snai1_p": np.nan,
        "pc2_snai1_r": np.nan,
        "pc2_snai1_p": np.nan,
        "pc1_oxphos_loading": np.nan,
    }

    if len(obs) < min_model_spots or len(available) < 3:
        return result_stub, []

    y = obs[SNAI1_COL].astype(float).values
    X_h = obs[available].apply(pd.to_numeric, errors="coerce").values
    covars, _ = build_covariate_matrix(obs)

    blocks = [y, X_h]
    if covars is not None:
        if covars.ndim == 1:
            covars = covars.reshape(-1, 1)
        blocks.append(covars)

    mask = np.ones(len(obs), dtype=bool)
    for block in blocks:
        if block.ndim == 1:
            mask &= np.isfinite(block)
        else:
            mask &= np.all(np.isfinite(block), axis=1)

    y = y[mask]
    X_h = X_h[mask]
    covars = covars[mask] if covars is not None else None

    if len(y) < min_model_spots:
        result_stub["n_spots"] = len(y)
        return result_stub, []

    y_resid = residualize(y, covars)
    X_resid = np.column_stack(
        [standardize(residualize(X_h[:, i], covars)) for i in range(X_h.shape[1])]
    )

    if X_resid.shape[1] < 2:
        return result_stub, []

    pca = PCA(n_components=min(X_resid.shape[1], 3), random_state=RANDOM_STATE)
    pcs = pca.fit_transform(X_resid)
    loadings = pca.components_.copy()

    if anchor_score_col in available:
        anchor_idx = available.index(anchor_score_col)
        if loadings[0, anchor_idx] < 0:
            pcs[:, 0] *= -1
            loadings[0, :] *= -1

    pc1_r, pc1_p, _ = safe_spearman(pcs[:, 0], y_resid)
    if pcs.shape[1] >= 2:
        pc2_r, pc2_p, _ = safe_spearman(pcs[:, 1], y_resid)
        pc2_var = float(pca.explained_variance_ratio_[1])
    else:
        pc2_r, pc2_p, pc2_var = np.nan, np.nan, np.nan

    result = {
        "n_spots": len(y),
        "n_hallmarks_used": len(available),
        "pc1_explained_variance": float(pca.explained_variance_ratio_[0]),
        "pc2_explained_variance": pc2_var,
        "pc1_snai1_r": pc1_r,
        "pc1_snai1_p": pc1_p,
        "pc2_snai1_r": pc2_r,
        "pc2_snai1_p": pc2_p,
        "pc1_oxphos_loading": (
            float(loadings[0, available.index(anchor_score_col)])
            if anchor_score_col in available
            else np.nan
        ),
    }

    loading_rows = []
    for idx, col in enumerate(available):
        loading_rows.append(
            {
                "pathway": pathway_name(col),
                "score_col": col,
                "pc1_loading": float(loadings[0, idx]),
                "pc2_loading": float(loadings[1, idx]) if loadings.shape[0] >= 2 else np.nan,
            }
        )

    return result, loading_rows


def collect_subset_results(
    h5ad_files: list[Path],
    subset_name: str,
    min_spots: int,
    min_model_spots: int,
    high_purity_threshold: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    univariate_rows: list[dict[str, float]] = []
    model_rows: list[dict[str, float]] = []
    coef_rows: list[dict[str, float]] = []

    for h5ad_path in h5ad_files:
        dataset, sample, sample_label = get_sample_identity(h5ad_path)
        print(f"  Loading {dataset}/{sample} [{subset_name}]")

        adata = ad.read_h5ad(h5ad_path, backed="r")
        obs = adata.obs.copy()
        adata.file.close()

        if SNAI1_COL not in obs.columns:
            print("    Skipping: SNAI1-ac score missing")
            continue

        primary_mask, tumor_definition = get_primary_tumor_mask(obs)
        if high_purity_threshold is not None and "Malignant" in obs.columns:
            primary_mask = primary_mask & (pd.to_numeric(obs["Malignant"], errors="coerce") >= high_purity_threshold)

        obs_subset = obs.loc[primary_mask].copy()
        n_spots = len(obs_subset)
        if n_spots < min_spots:
            print(f"    Skipping: only {n_spots} spots")
            continue

        hallmark_cols = score_columns(obs_subset.columns.tolist())
        focus_cols = panel_score_columns(obs_subset.columns.tolist(), FOCUS_PANEL)
        covars, covar_cols = build_covariate_matrix(obs_subset)

        y = pd.to_numeric(obs_subset[SNAI1_COL], errors="coerce").values

        for score_col in hallmark_cols:
            x = pd.to_numeric(obs_subset[score_col], errors="coerce").values
            r, p, n_used = safe_spearman(x, y)
            pr, pp, n_partial = partial_spearman(x, y, covars)
            quad = quadratic_diagnostic(x, y, covars)

            univariate_rows.append(
                {
                    "subset": subset_name,
                    "tumor_definition": tumor_definition,
                    "dataset": dataset,
                    "sample": sample,
                    "sample_label": sample_label,
                    "pathway": pathway_name(score_col),
                    "score_col": score_col,
                    "n_spots": n_spots,
                    "n_used_spearman": n_used,
                    "n_used_partial": n_partial,
                    "n_covariates": len(covar_cols),
                    "covariates": ";".join(covar_cols),
                    "spearman_r": r,
                    "spearman_p": p,
                    "partial_r": pr,
                    "partial_p": pp,
                    **quad,
                }
            )

        model_result, model_coefs = run_panel_model(
            obs_subset,
            panel_cols=focus_cols,
            min_model_spots=min_model_spots,
        )
        model_rows.append(
            {
                "subset": subset_name,
                "tumor_definition": tumor_definition,
                "dataset": dataset,
                "sample": sample,
                "sample_label": sample_label,
                "covariates": ";".join(covar_cols),
                **model_result,
            }
        )
        for row in model_coefs:
            coef_rows.append(
                {
                    "subset": subset_name,
                    "dataset": dataset,
                    "sample": sample,
                    "sample_label": sample_label,
                    **row,
                }
            )

    return (
        pd.DataFrame(univariate_rows),
        pd.DataFrame(model_rows),
        pd.DataFrame(coef_rows),
    )


def collect_panel_results(
    h5ad_files: list[Path],
    subset_name: str,
    panel_name: str,
    panel: list[str],
    min_spots: int,
    min_model_spots: int,
    high_purity_threshold: float | None = None,
    pca_anchor: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_rows: list[dict[str, float]] = []
    coef_rows: list[dict[str, float]] = []
    pca_rows: list[dict[str, float]] = []
    loading_rows: list[dict[str, float]] = []

    for h5ad_path in h5ad_files:
        dataset, sample, sample_label = get_sample_identity(h5ad_path)
        print(f"  Loading {dataset}/{sample} [{subset_name} | {panel_name}]")

        adata = ad.read_h5ad(h5ad_path, backed="r")
        obs = adata.obs.copy()
        adata.file.close()

        if SNAI1_COL not in obs.columns:
            continue

        primary_mask, tumor_definition = get_primary_tumor_mask(obs)
        if high_purity_threshold is not None and "Malignant" in obs.columns:
            primary_mask = primary_mask & (
                pd.to_numeric(obs["Malignant"], errors="coerce") >= high_purity_threshold
            )

        obs_subset = obs.loc[primary_mask].copy()
        n_spots = len(obs_subset)
        if n_spots < min_spots:
            continue

        covars, covar_cols = build_covariate_matrix(obs_subset)
        panel_cols = panel_score_columns(obs_subset.columns.tolist(), panel)

        model_result, model_coefs = run_panel_model(
            obs_subset,
            panel_cols=panel_cols,
            min_model_spots=min_model_spots,
        )
        model_rows.append(
            {
                "panel": panel_name,
                "subset": subset_name,
                "tumor_definition": tumor_definition,
                "dataset": dataset,
                "sample": sample,
                "sample_label": sample_label,
                "covariates": ";".join(covar_cols),
                **model_result,
            }
        )
        for row in model_coefs:
            coef_rows.append(
                {
                    "panel": panel_name,
                    "subset": subset_name,
                    "dataset": dataset,
                    "sample": sample,
                    "sample_label": sample_label,
                    **row,
                }
            )

        if pca_anchor is not None:
            pca_result, pca_loadings = run_panel_pca(
                obs_subset,
                panel_cols=panel_cols,
                anchor_score_col=pca_anchor,
                min_model_spots=min_model_spots,
            )
            pca_rows.append(
                {
                    "panel": panel_name,
                    "subset": subset_name,
                    "tumor_definition": tumor_definition,
                    "dataset": dataset,
                    "sample": sample,
                    "sample_label": sample_label,
                    "covariates": ";".join(covar_cols),
                    **pca_result,
                }
            )
            for row in pca_loadings:
                loading_rows.append(
                    {
                        "panel": panel_name,
                        "subset": subset_name,
                        "dataset": dataset,
                        "sample": sample,
                        "sample_label": sample_label,
                        **row,
                    }
                )

    return (
        pd.DataFrame(model_rows),
        pd.DataFrame(coef_rows),
        pd.DataFrame(pca_rows),
        pd.DataFrame(loading_rows),
    )


def summarize_meta(univariate_df: pd.DataFrame) -> pd.DataFrame:
    meta_rows = []
    if univariate_df.empty:
        return pd.DataFrame()

    for subset in sorted(univariate_df["subset"].unique()):
        subset_df = univariate_df[univariate_df["subset"] == subset]
        for pathway in sorted(subset_df["pathway"].unique()):
            sub = subset_df[subset_df["pathway"] == pathway]
            spearman_r, spearman_p = fisher_z_meta(
                sub["spearman_r"].values.astype(float),
                sub["n_spots"].values.astype(float),
            )
            partial_r, partial_p = fisher_z_meta(
                sub["partial_r"].values.astype(float),
                sub["n_used_partial"].values.astype(float),
            )

            meta_rows.append(
                {
                    "subset": subset,
                    "pathway": pathway,
                    "k_samples": int(sub["sample_label"].nunique()),
                    "spearman_r_combined": spearman_r,
                    "spearman_p_combined": spearman_p,
                    "spearman_mean": float(np.nanmean(sub["spearman_r"])),
                    "spearman_median": float(np.nanmedian(sub["spearman_r"])),
                    "spearman_consistency": float(np.mean(np.sign(sub["spearman_r"].fillna(0)) == np.sign(spearman_r))) if np.isfinite(spearman_r) else np.nan,
                    "partial_r_combined": partial_r,
                    "partial_p_combined": partial_p,
                    "partial_mean": float(np.nanmean(sub["partial_r"])),
                    "partial_median": float(np.nanmedian(sub["partial_r"])),
                    "partial_consistency": float(np.mean(np.sign(sub["partial_r"].fillna(0)) == np.sign(partial_r))) if np.isfinite(partial_r) else np.nan,
                    "quadratic_delta_r2_mean": float(np.nanmean(sub["quadratic_delta_r2"])),
                    "quadratic_delta_r2_median": float(np.nanmedian(sub["quadratic_delta_r2"])),
                    "quadratic_significant_fraction": float(np.mean(sub["quadratic_p"] < 0.05)),
                    "quadratic_positive_term_fraction": float(np.mean(sub["quadratic_term"] > 0)),
                }
            )

    meta_df = pd.DataFrame(meta_rows)
    if not meta_df.empty:
        meta_df = meta_df.sort_values(
            ["subset", "partial_r_combined", "spearman_r_combined"],
            ascending=[True, False, False],
        )
    return meta_df


def summarize_model_coefficients(coef_df: pd.DataFrame) -> pd.DataFrame:
    if coef_df.empty:
        return pd.DataFrame()

    rows = []
    for subset in sorted(coef_df["subset"].unique()):
        sub = coef_df[coef_df["subset"] == subset]
        for pathway in sorted(sub["pathway"].unique()):
            path = sub[sub["pathway"] == pathway]
            rows.append(
                {
                    "subset": subset,
                    "pathway": pathway,
                    "k_samples": int(path["sample_label"].nunique()),
                    "hallmark_only_coef_mean": float(np.nanmean(path["hallmark_only_coef"])),
                    "hallmark_only_coef_median": float(np.nanmedian(path["hallmark_only_coef"])),
                    "combined_coef_mean": float(np.nanmean(path["combined_coef"])),
                    "combined_coef_median": float(np.nanmedian(path["combined_coef"])),
                    "combined_positive_fraction": float(np.mean(path["combined_coef"] > 0)),
                }
            )
    return pd.DataFrame(rows)


def summarize_pca_results(pca_df: pd.DataFrame) -> pd.DataFrame:
    if pca_df.empty:
        return pd.DataFrame()

    rows = []
    for panel in sorted(pca_df["panel"].unique()):
        panel_df = pca_df[pca_df["panel"] == panel]
        for subset in sorted(panel_df["subset"].unique()):
            sub = panel_df[(panel_df["subset"] == subset) & np.isfinite(panel_df["pc1_snai1_r"])]
            if sub.empty:
                continue
            pc1_r, pc1_p = fisher_z_meta(
                sub["pc1_snai1_r"].values.astype(float),
                sub["n_spots"].values.astype(float),
            )
            pc2_r, pc2_p = fisher_z_meta(
                sub["pc2_snai1_r"].values.astype(float),
                sub["n_spots"].values.astype(float),
            )
            rows.append(
                {
                    "panel": panel,
                    "subset": subset,
                    "k_samples": int(sub["sample_label"].nunique()),
                    "pc1_snai1_r_combined": pc1_r,
                    "pc1_snai1_p_combined": pc1_p,
                    "pc1_mean": float(np.nanmean(sub["pc1_snai1_r"])),
                    "pc1_median": float(np.nanmedian(sub["pc1_snai1_r"])),
                    "pc1_consistency": float(np.mean(np.sign(sub["pc1_snai1_r"]) == np.sign(pc1_r))) if np.isfinite(pc1_r) else np.nan,
                    "pc1_explained_variance_mean": float(np.nanmean(sub["pc1_explained_variance"])),
                    "pc1_explained_variance_median": float(np.nanmedian(sub["pc1_explained_variance"])),
                    "pc1_oxphos_loading_mean": float(np.nanmean(sub["pc1_oxphos_loading"])),
                    "pc2_snai1_r_combined": pc2_r,
                    "pc2_snai1_p_combined": pc2_p,
                    "pc2_explained_variance_mean": float(np.nanmean(sub["pc2_explained_variance"])),
                }
            )
    return pd.DataFrame(rows)


def summarize_pca_loadings(loadings_df: pd.DataFrame) -> pd.DataFrame:
    if loadings_df.empty:
        return pd.DataFrame()

    rows = []
    for panel in sorted(loadings_df["panel"].unique()):
        panel_df = loadings_df[loadings_df["panel"] == panel]
        for subset in sorted(panel_df["subset"].unique()):
            sub = panel_df[panel_df["subset"] == subset]
            for pathway in sorted(sub["pathway"].unique()):
                path = sub[sub["pathway"] == pathway]
                rows.append(
                    {
                        "panel": panel,
                        "subset": subset,
                        "pathway": pathway,
                        "k_samples": int(path["sample_label"].nunique()),
                        "pc1_loading_mean": float(np.nanmean(path["pc1_loading"])),
                        "pc1_loading_median": float(np.nanmedian(path["pc1_loading"])),
                        "pc1_positive_fraction": float(np.mean(path["pc1_loading"] > 0)),
                        "pc2_loading_mean": float(np.nanmean(path["pc2_loading"])),
                        "pc2_loading_median": float(np.nanmedian(path["pc2_loading"])),
                    }
                )
    return pd.DataFrame(rows)


def make_focus_panel_meta_plot(meta_df: pd.DataFrame, output_dir: Path) -> None:
    if meta_df.empty:
        return

    subset_df = meta_df[meta_df["subset"] == "primary_tumor"].copy()
    subset_df = subset_df[subset_df["pathway"].isin(FOCUS_PANEL)]
    if subset_df.empty:
        return

    subset_df["pathway"] = pd.Categorical(subset_df["pathway"], categories=FOCUS_PANEL, ordered=True)
    subset_df = subset_df.sort_values("pathway")

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), gridspec_kw={"width_ratios": [1.1, 1.1, 0.9]})

    sns.barplot(
        data=subset_df,
        y="pathway",
        x="spearman_r_combined",
        color="#4C78A8",
        ax=axes[0],
    )
    axes[0].axvline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("Tumor-only Spearman")
    axes[0].set_xlabel("Combined r")
    axes[0].set_ylabel("")

    sns.barplot(
        data=subset_df,
        y="pathway",
        x="partial_r_combined",
        color="#F58518",
        ax=axes[1],
    )
    axes[1].axvline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_title("Adjusted partial correlation")
    axes[1].set_xlabel("Combined partial r")
    axes[1].set_ylabel("")

    sns.barplot(
        data=subset_df,
        y="pathway",
        x="quadratic_significant_fraction",
        color="#54A24B",
        ax=axes[2],
    )
    axes[2].set_title("Nonlinear signal")
    axes[2].set_xlabel("Fraction with quadratic p < 0.05")
    axes[2].set_ylabel("")

    fig.suptitle("Tumor-only Hallmark summary for SNAI1-ac", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "focus_panel_meta_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_primary_heatmap(univariate_df: pd.DataFrame, value_col: str, output_name: str, title: str, output_dir: Path) -> None:
    if univariate_df.empty:
        return

    subset_df = univariate_df[univariate_df["subset"] == "primary_tumor"].copy()
    if subset_df.empty:
        return

    pivot = subset_df.pivot_table(index="pathway", columns="sample_label", values=value_col)
    if pivot.empty:
        return

    order = (
        subset_df.groupby("pathway")[value_col]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    pivot = pivot.loc[order]
    pivot = pivot[sorted(pivot.columns)]

    fig_height = max(10, 0.25 * len(pivot) + 2)
    fig_width = max(12, 0.35 * len(pivot.columns) + 4)
    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(
        pivot,
        cmap="RdBu_r",
        center=0,
        vmin=-0.8,
        vmax=0.8,
        linewidths=0.2,
        linecolor="white",
        cbar_kws={"label": value_col},
    )
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Hallmark pathway")
    plt.tight_layout()
    plt.savefig(output_dir / output_name, dpi=200, bbox_inches="tight")
    plt.close()


def make_model_summary_plot(
    model_df: pd.DataFrame,
    output_dir: Path,
    output_name: str,
    title_prefix: str,
) -> None:
    if model_df.empty:
        return

    subset_df = model_df[model_df["subset"] == "primary_tumor"].copy()
    subset_df = subset_df[np.isfinite(subset_df["cv_r2_combined"])]
    if subset_df.empty:
        return

    plot_df = subset_df.melt(
        id_vars=["sample_label"],
        value_vars=["cv_r2_baseline", "cv_r2_hallmarks_only", "cv_r2_combined"],
        var_name="model",
        value_name="cv_r2",
    )
    label_map = {
        "cv_r2_baseline": "Baseline covariates",
        "cv_r2_hallmarks_only": "Hallmarks only",
        "cv_r2_combined": "Combined",
    }
    plot_df["model"] = plot_df["model"].map(label_map)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(data=plot_df, x="model", y="cv_r2", ax=axes[0], color="#72B7B2")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title(f"{title_prefix}: CV R^2")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("CV R^2")
    axes[0].tick_params(axis="x", rotation=20)

    sns.histplot(subset_df["delta_r2_vs_baseline"].dropna(), bins=20, ax=axes[1], color="#E45756")
    axes[1].axvline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_title(f"{title_prefix}: incremental value")
    axes[1].set_xlabel("Combined R^2 - baseline R^2")
    axes[1].set_ylabel("Samples")

    plt.tight_layout()
    fig.savefig(output_dir / output_name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_coefficient_heatmap(
    coef_df: pd.DataFrame,
    output_dir: Path,
    pathway_order: list[str],
    output_name: str,
    title: str,
) -> None:
    if coef_df.empty:
        return

    subset_df = coef_df[coef_df["subset"] == "primary_tumor"].copy()
    if subset_df.empty:
        return

    pivot = subset_df.pivot_table(index="pathway", columns="sample_label", values="combined_coef")
    if pivot.empty:
        return

    order = [p for p in pathway_order if p in pivot.index]
    pivot = pivot.loc[order]
    pivot = pivot[sorted(pivot.columns)]

    plt.figure(figsize=(max(12, 0.35 * len(pivot.columns) + 4), 5))
    sns.heatmap(
        pivot,
        cmap="RdBu_r",
        center=0,
        linewidths=0.2,
        linecolor="white",
        cbar_kws={"label": "Standardized combined-model coefficient"},
    )
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Hallmark pathway")
    plt.tight_layout()
    plt.savefig(output_dir / output_name, dpi=200, bbox_inches="tight")
    plt.close()


def make_pca_loading_summary_plot(
    loading_summary_df: pd.DataFrame,
    panel_name: str,
    pathway_order: list[str],
    output_dir: Path,
    output_name: str,
    title: str,
) -> None:
    if loading_summary_df.empty:
        return

    sub = loading_summary_df[
        (loading_summary_df["panel"] == panel_name)
        & (loading_summary_df["subset"] == "primary_tumor")
    ].copy()
    if sub.empty:
        return

    sub["pathway"] = pd.Categorical(sub["pathway"], categories=pathway_order, ordered=True)
    sub = sub.sort_values("pathway")

    plt.figure(figsize=(8, max(4, 0.45 * len(sub))))
    sns.barplot(data=sub, y="pathway", x="pc1_loading_mean", color="#4C78A8")
    plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
    plt.title(title)
    plt.xlabel("Mean PC1 loading")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_dir / output_name, dpi=200, bbox_inches="tight")
    plt.close()


def make_pca_summary_plot(
    pca_summary_df: pd.DataFrame,
    panel_name: str,
    output_dir: Path,
    output_name: str,
    title: str,
) -> None:
    if pca_summary_df.empty:
        return

    sub = pca_summary_df[pca_summary_df["panel"] == panel_name].copy()
    if sub.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    sns.barplot(
        data=sub,
        x="subset",
        y="pc1_snai1_r_combined",
        ax=axes[0],
        color="#F58518",
    )
    axes[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("PC1 vs SNAI1-ac")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Combined r")
    axes[0].tick_params(axis="x", rotation=20)

    sns.barplot(
        data=sub,
        x="subset",
        y="pc1_explained_variance_mean",
        ax=axes[1],
        color="#54A24B",
    )
    axes[1].set_title("PC1 explained variance")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Mean variance explained")
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / output_name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_panel_comparison_plot(
    focus_model_df: pd.DataFrame,
    metabolic_model_df: pd.DataFrame,
    output_dir: Path,
    output_name: str,
) -> None:
    focus = focus_model_df[
        (focus_model_df["subset"] == "primary_tumor")
        & np.isfinite(focus_model_df["cv_r2_combined"])
    ][["sample_label", "cv_r2_combined", "delta_r2_vs_baseline"]].copy()
    focus["panel"] = "Broad Hallmark panel"

    metabolic = metabolic_model_df[
        (metabolic_model_df["subset"] == "primary_tumor")
        & np.isfinite(metabolic_model_df["cv_r2_combined"])
    ][["sample_label", "cv_r2_combined", "delta_r2_vs_baseline"]].copy()
    metabolic["panel"] = "Metabolic panel"

    plot_df = pd.concat([focus, metabolic], ignore_index=True)
    if plot_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.boxplot(data=plot_df, x="panel", y="cv_r2_combined", ax=axes[0], color="#72B7B2")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("Combined model performance")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("CV R^2")
    axes[0].tick_params(axis="x", rotation=15)

    sns.boxplot(data=plot_df, x="panel", y="delta_r2_vs_baseline", ax=axes[1], color="#E45756")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_title("Increment beyond covariates")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Delta R^2")
    axes[1].tick_params(axis="x", rotation=15)

    fig.suptitle("Broad vs metabolic Hallmark panels", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / output_name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    h5ad_files = discover_h5ads()
    if not h5ad_files:
        raise SystemExit("No analysis-ready h5ad files found.")

    print("=" * 70)
    print("S2b: Tumor-only Hallmark analysis for SNAI1-ac")
    print("=" * 70)
    print(f"Samples discovered: {len(h5ad_files)}")
    print(f"Primary tumor subset: {INTERFACE_COL} == '{TUMOR_LABEL}'")
    print(f"Sensitivity subset: {INTERFACE_COL} == '{TUMOR_LABEL}' and Malignant >= {args.high_purity_threshold}")

    primary_uni, primary_model, primary_coef = collect_subset_results(
        h5ad_files=h5ad_files,
        subset_name="primary_tumor",
        min_spots=args.min_spots,
        min_model_spots=args.min_model_spots,
        high_purity_threshold=None,
    )

    high_purity_uni, high_purity_model, high_purity_coef = collect_subset_results(
        h5ad_files=h5ad_files,
        subset_name="high_purity_tumor",
        min_spots=args.min_spots,
        min_model_spots=args.min_model_spots,
        high_purity_threshold=args.high_purity_threshold,
    )

    primary_metabolic_model, primary_metabolic_coef, primary_metabolic_pca, primary_metabolic_loadings = collect_panel_results(
        h5ad_files=h5ad_files,
        subset_name="primary_tumor",
        panel_name="metabolic_panel",
        panel=METABOLIC_PANEL,
        min_spots=args.min_spots,
        min_model_spots=args.min_model_spots,
        high_purity_threshold=None,
        pca_anchor="HALLMARK_OXIDATIVE_PHOSPHORYLATION_score",
    )

    high_purity_metabolic_model, high_purity_metabolic_coef, high_purity_metabolic_pca, high_purity_metabolic_loadings = collect_panel_results(
        h5ad_files=h5ad_files,
        subset_name="high_purity_tumor",
        panel_name="metabolic_panel",
        panel=METABOLIC_PANEL,
        min_spots=args.min_spots,
        min_model_spots=args.min_model_spots,
        high_purity_threshold=args.high_purity_threshold,
        pca_anchor="HALLMARK_OXIDATIVE_PHOSPHORYLATION_score",
    )

    univariate_df = pd.concat([primary_uni, high_purity_uni], ignore_index=True)
    model_df = pd.concat([primary_model, high_purity_model], ignore_index=True)
    coef_df = pd.concat([primary_coef, high_purity_coef], ignore_index=True)
    metabolic_model_df = pd.concat([primary_metabolic_model, high_purity_metabolic_model], ignore_index=True)
    metabolic_coef_df = pd.concat([primary_metabolic_coef, high_purity_metabolic_coef], ignore_index=True)
    metabolic_pca_df = pd.concat([primary_metabolic_pca, high_purity_metabolic_pca], ignore_index=True)
    metabolic_loading_df = pd.concat([primary_metabolic_loadings, high_purity_metabolic_loadings], ignore_index=True)

    meta_df = summarize_meta(univariate_df)
    coef_summary_df = summarize_model_coefficients(coef_df)
    metabolic_coef_summary_df = summarize_model_coefficients(metabolic_coef_df)
    metabolic_pca_summary_df = summarize_pca_results(metabolic_pca_df)
    metabolic_loading_summary_df = summarize_pca_loadings(metabolic_loading_df)

    univariate_df.to_csv(OUTPUT_DIR / "tumor_only_per_sample_hallmark_stats.csv", index=False)
    model_df.to_csv(OUTPUT_DIR / "tumor_only_combined_model_summary.csv", index=False)
    coef_df.to_csv(OUTPUT_DIR / "tumor_only_combined_model_coefficients.csv", index=False)
    meta_df.to_csv(OUTPUT_DIR / "tumor_only_hallmark_meta_summary.csv", index=False)
    coef_summary_df.to_csv(OUTPUT_DIR / "tumor_only_combined_model_coefficient_summary.csv", index=False)
    metabolic_model_df.to_csv(OUTPUT_DIR / "tumor_only_metabolic_model_summary.csv", index=False)
    metabolic_coef_df.to_csv(OUTPUT_DIR / "tumor_only_metabolic_model_coefficients.csv", index=False)
    metabolic_coef_summary_df.to_csv(OUTPUT_DIR / "tumor_only_metabolic_model_coefficient_summary.csv", index=False)
    metabolic_pca_df.to_csv(OUTPUT_DIR / "tumor_only_metabolic_pca_summary.csv", index=False)
    metabolic_loading_df.to_csv(OUTPUT_DIR / "tumor_only_metabolic_pca_loadings.csv", index=False)
    metabolic_pca_summary_df.to_csv(OUTPUT_DIR / "tumor_only_metabolic_pca_meta_summary.csv", index=False)
    metabolic_loading_summary_df.to_csv(OUTPUT_DIR / "tumor_only_metabolic_pca_loading_summary.csv", index=False)

    make_primary_heatmap(
        univariate_df,
        value_col="spearman_r",
        output_name="tumor_only_spearman_heatmap.png",
        title="Tumor-only Spearman correlations with SNAI1-ac",
        output_dir=OUTPUT_DIR,
    )
    make_primary_heatmap(
        univariate_df,
        value_col="partial_r",
        output_name="tumor_only_partial_heatmap.png",
        title="Tumor-only adjusted partial correlations with SNAI1-ac",
        output_dir=OUTPUT_DIR,
    )
    make_focus_panel_meta_plot(meta_df, OUTPUT_DIR)
    make_model_summary_plot(
        model_df,
        OUTPUT_DIR,
        output_name="combined_model_summary.png",
        title_prefix="Broad Hallmark panel",
    )
    make_coefficient_heatmap(
        coef_df,
        OUTPUT_DIR,
        pathway_order=FOCUS_PANEL,
        output_name="combined_model_coefficients_heatmap.png",
        title="Tumor-only combined Hallmark model coefficients",
    )
    make_model_summary_plot(
        metabolic_model_df,
        OUTPUT_DIR,
        output_name="metabolic_model_summary.png",
        title_prefix="Metabolic panel",
    )
    make_coefficient_heatmap(
        metabolic_coef_df,
        OUTPUT_DIR,
        pathway_order=METABOLIC_PANEL,
        output_name="metabolic_model_coefficients_heatmap.png",
        title="Tumor-only metabolic model coefficients",
    )
    make_pca_loading_summary_plot(
        metabolic_loading_summary_df,
        panel_name="metabolic_panel",
        pathway_order=METABOLIC_PANEL,
        output_dir=OUTPUT_DIR,
        output_name="metabolic_pc1_loading_summary.png",
        title="Residualized metabolic PC1 loadings",
    )
    make_pca_summary_plot(
        metabolic_pca_summary_df,
        panel_name="metabolic_panel",
        output_dir=OUTPUT_DIR,
        output_name="metabolic_pc1_summary.png",
        title="Residualized metabolic PCA summary",
    )
    make_panel_comparison_plot(
        model_df,
        metabolic_model_df,
        OUTPUT_DIR,
        output_name="broad_vs_metabolic_model_comparison.png",
    )

    print("\nSaved:")
    print(f"  {OUTPUT_DIR / 'tumor_only_per_sample_hallmark_stats.csv'}")
    print(f"  {OUTPUT_DIR / 'tumor_only_hallmark_meta_summary.csv'}")
    print(f"  {OUTPUT_DIR / 'tumor_only_combined_model_summary.csv'}")
    print(f"  {OUTPUT_DIR / 'tumor_only_combined_model_coefficients.csv'}")
    print(f"  {OUTPUT_DIR / 'tumor_only_combined_model_coefficient_summary.csv'}")
    print(f"  {OUTPUT_DIR / 'tumor_only_metabolic_model_summary.csv'}")
    print(f"  {OUTPUT_DIR / 'tumor_only_metabolic_model_coefficients.csv'}")
    print(f"  {OUTPUT_DIR / 'tumor_only_metabolic_model_coefficient_summary.csv'}")
    print(f"  {OUTPUT_DIR / 'tumor_only_metabolic_pca_summary.csv'}")
    print(f"  {OUTPUT_DIR / 'tumor_only_metabolic_pca_loadings.csv'}")
    print(f"  {OUTPUT_DIR / 'tumor_only_metabolic_pca_meta_summary.csv'}")
    print(f"  {OUTPUT_DIR / 'tumor_only_metabolic_pca_loading_summary.csv'}")
    print(f"  {OUTPUT_DIR / 'tumor_only_spearman_heatmap.png'}")
    print(f"  {OUTPUT_DIR / 'tumor_only_partial_heatmap.png'}")
    print(f"  {OUTPUT_DIR / 'focus_panel_meta_summary.png'}")
    print(f"  {OUTPUT_DIR / 'combined_model_summary.png'}")
    print(f"  {OUTPUT_DIR / 'combined_model_coefficients_heatmap.png'}")
    print(f"  {OUTPUT_DIR / 'metabolic_model_summary.png'}")
    print(f"  {OUTPUT_DIR / 'metabolic_model_coefficients_heatmap.png'}")
    print(f"  {OUTPUT_DIR / 'metabolic_pc1_loading_summary.png'}")
    print(f"  {OUTPUT_DIR / 'metabolic_pc1_summary.png'}")
    print(f"  {OUTPUT_DIR / 'broad_vs_metabolic_model_comparison.png'}")


if __name__ == "__main__":
    main()
