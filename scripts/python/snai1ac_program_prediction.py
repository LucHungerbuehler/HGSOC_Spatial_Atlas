"""
Per-sample ridge prediction of continuous SNAI1-ac scores from K* programme usage.

This script answers a deliberately narrow question:

"How well can a weighted combination of programme usages predict the continuous
SNAI1-ac score within each sample?"

It reads the Definition 3b spot-level tables, fits one ridge-regression model
per sample using only the programme-usage columns, evaluates performance with
nested cross-validation, and saves:

1. A per-sample performance table.
2. A per-sample standardized-weight table.
3. Observed-vs-predicted plots based on outer-fold held-out predictions.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_RANDOM_STATE = 42
PROGRAM_PATTERN = re.compile(r"__K\d+__P\d+$")
ALPHA_GRID = np.logspace(-4, 4, 25)


@dataclass
class CvScheme:
    splitter: object
    groups: np.ndarray | None
    scheme_name: str
    n_splits: int
    group_count: int
    block_side: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict SNAI1-ac scores from K* programme usages on a per-sample basis."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Definition 3b / Definition 4 run directory on D: drive.",
    )
    parser.add_argument(
        "--output-subdir",
        default="07_snai1ac_program_prediction",
        help="Name of the output folder to create under the run directory.",
    )
    parser.add_argument(
        "--outer-splits",
        type=int,
        default=5,
        help="Target number of outer cross-validation folds.",
    )
    parser.add_argument(
        "--inner-splits",
        type=int,
        default=5,
        help="Target number of inner cross-validation folds for alpha tuning.",
    )
    parser.add_argument(
        "--spatial-block-side",
        type=int,
        default=4,
        help="Target number of quantile bins per axis when building spatial blocks.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def programme_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if PROGRAM_PATTERN.search(str(column))]


def qcut_codes(values: pd.Series, n_bins: int) -> np.ndarray:
    return pd.qcut(values, q=min(n_bins, values.nunique()), labels=False, duplicates="drop").to_numpy()


def spatial_block_groups(frame: pd.DataFrame, preferred_side: int, min_groups: int) -> tuple[np.ndarray | None, int | None]:
    coords = frame[["array_row", "array_col"]].apply(pd.to_numeric, errors="coerce")
    if coords.isna().any().any():
        return None, None
    for side in range(preferred_side, 1, -1):
        row_codes = qcut_codes(coords["array_row"], side)
        col_codes = qcut_codes(coords["array_col"], side)
        block_labels = pd.Series(
            [f"{row_code}_{col_code}" for row_code, col_code in zip(row_codes, col_codes, strict=False)]
        )
        groups = block_labels.astype("category").cat.codes.to_numpy()
        if np.unique(groups).size >= min_groups:
            return groups, side
    return None, None


def build_cv_scheme(
    frame: pd.DataFrame,
    target_splits: int,
    spatial_block_side: int,
) -> CvScheme:
    n_obs = len(frame)
    min_groups = min(target_splits, max(3, target_splits))
    spatial_groups, used_side = spatial_block_groups(frame, preferred_side=spatial_block_side, min_groups=min_groups)
    if spatial_groups is not None:
        n_groups = int(np.unique(spatial_groups).size)
        n_splits = min(target_splits, n_groups)
        return CvScheme(
            splitter=GroupKFold(n_splits=n_splits),
            groups=spatial_groups,
            scheme_name="spatial_groupkfold",
            n_splits=n_splits,
            group_count=n_groups,
            block_side=used_side,
        )

    n_splits = min(target_splits, max(3, n_obs // 25))
    n_splits = max(3, n_splits)
    return CvScheme(
        splitter=KFold(n_splits=n_splits, shuffle=True, random_state=DEFAULT_RANDOM_STATE),
        groups=None,
        scheme_name="random_kfold_fallback",
        n_splits=n_splits,
        group_count=0,
        block_side=None,
    )


def make_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("ridge", Ridge()),
        ]
    )


def fit_grid_search(
    X: np.ndarray,
    y: np.ndarray,
    cv_scheme: CvScheme,
    target_inner_splits: int,
) -> GridSearchCV:
    if cv_scheme.groups is not None:
        unique_groups = np.unique(cv_scheme.groups)
        inner_splits = min(target_inner_splits, len(unique_groups))
        if inner_splits >= 2:
            inner_cv = GroupKFold(n_splits=inner_splits)
            search = GridSearchCV(
                estimator=make_pipeline(),
                param_grid={"ridge__alpha": ALPHA_GRID},
                scoring="r2",
                cv=inner_cv,
                n_jobs=1,
                refit=True,
            )
            search.fit(X, y, groups=cv_scheme.groups)
            return search

    inner_splits = min(target_inner_splits, max(3, len(y) // 25))
    inner_splits = max(3, inner_splits)
    inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
    search = GridSearchCV(
        estimator=make_pipeline(),
        param_grid={"ridge__alpha": ALPHA_GRID},
        scoring="r2",
        cv=inner_cv,
        n_jobs=1,
        refit=True,
    )
    search.fit(X, y)
    return search


def nested_cv_predictions(
    X: np.ndarray,
    y: np.ndarray,
    frame: pd.DataFrame,
    outer_splits: int,
    inner_splits: int,
    spatial_block_side: int,
) -> tuple[np.ndarray, list[float], CvScheme]:
    outer_scheme = build_cv_scheme(frame, target_splits=outer_splits, spatial_block_side=spatial_block_side)
    preds = np.full(len(y), np.nan, dtype=float)
    chosen_alphas: list[float] = []

    if outer_scheme.groups is not None:
        outer_iter = outer_scheme.splitter.split(X, y, groups=outer_scheme.groups)
    else:
        outer_iter = outer_scheme.splitter.split(X, y)

    for train_idx, test_idx in outer_iter:
        train_frame = frame.iloc[train_idx].reset_index(drop=True)
        inner_scheme = build_cv_scheme(
            train_frame,
            target_splits=min(inner_splits, outer_scheme.n_splits),
            spatial_block_side=spatial_block_side,
        )
        search = fit_grid_search(X[train_idx], y[train_idx], inner_scheme, target_inner_splits=inner_splits)
        preds[test_idx] = search.predict(X[test_idx])
        chosen_alphas.append(float(search.best_params_["ridge__alpha"]))

    return preds, chosen_alphas, outer_scheme


def final_fit_weights(
    X: np.ndarray,
    y: np.ndarray,
    frame: pd.DataFrame,
    inner_splits: int,
    spatial_block_side: int,
    feature_names: list[str],
) -> tuple[float, pd.DataFrame]:
    final_scheme = build_cv_scheme(frame, target_splits=inner_splits, spatial_block_side=spatial_block_side)
    search = fit_grid_search(X, y, final_scheme, target_inner_splits=inner_splits)
    best_model = search.best_estimator_
    weights = best_model.named_steps["ridge"].coef_.astype(float)
    weight_df = pd.DataFrame(
        {
            "program_id": feature_names,
            "standardized_weight": weights,
        }
    )
    weight_df["abs_standardized_weight"] = weight_df["standardized_weight"].abs()
    weight_df["weight_direction"] = np.where(
        weight_df["standardized_weight"] > 0,
        "positive",
        np.where(weight_df["standardized_weight"] < 0, "negative", "zero"),
    )
    weight_df = weight_df.sort_values("abs_standardized_weight", ascending=False).reset_index(drop=True)
    weight_df["abs_weight_rank"] = np.arange(1, len(weight_df) + 1)
    return float(search.best_params_["ridge__alpha"]), weight_df


def plot_predictions(
    observed: np.ndarray,
    predicted: np.ndarray,
    sample_label: str,
    output_path: Path,
    cv_r2: float,
    cv_rmse: float,
    cv_spearman: float,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(observed, predicted, s=18, alpha=0.55, edgecolor="none", color="#1f77b4")
    all_vals = np.concatenate([observed, predicted])
    line_min = float(np.nanmin(all_vals))
    line_max = float(np.nanmax(all_vals))
    ax.plot([line_min, line_max], [line_min, line_max], linestyle="--", linewidth=1.2, color="#444444")
    ax.set_xlabel("Observed SNAI1-ac score")
    ax.set_ylabel("Cross-validated predicted SNAI1-ac score")
    ax.set_title(f"{sample_label}\nProgramme-usage ridge prediction")
    ax.text(
        0.03,
        0.97,
        f"CV R² = {cv_r2:.3f}\nCV RMSE = {cv_rmse:.3f}\nCV Spearman = {cv_spearman:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.35"},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def analyze_sample(
    sample_path: Path,
    outer_splits: int,
    inner_splits: int,
    spatial_block_side: int,
    figures_dir: Path,
) -> tuple[dict[str, object], pd.DataFrame]:
    frame = pd.read_csv(sample_path)
    feature_names = programme_columns(frame)
    if not feature_names:
        raise ValueError(f"No programme columns found in {sample_path}")

    y = pd.to_numeric(frame["SNAI1-ac_score"], errors="coerce").to_numpy(dtype=float)
    X = frame[feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    coord_mask = (
        pd.to_numeric(frame["array_row"], errors="coerce").notna()
        & pd.to_numeric(frame["array_col"], errors="coerce").notna()
    ).to_numpy()
    valid_mask = np.isfinite(y) & np.isfinite(X).all(axis=1) & coord_mask
    model_frame = frame.loc[valid_mask].reset_index(drop=True)
    X_model = X[valid_mask]
    y_model = y[valid_mask]

    if len(y_model) < 30:
        raise ValueError(f"Too few valid spots for modelling in {sample_path}: n={len(y_model)}")

    cv_pred, outer_alphas, outer_scheme = nested_cv_predictions(
        X_model,
        y_model,
        model_frame,
        outer_splits=outer_splits,
        inner_splits=inner_splits,
        spatial_block_side=spatial_block_side,
    )

    cv_r2 = float(r2_score(y_model, cv_pred))
    cv_rmse = rmse(y_model, cv_pred)
    cv_spearman = float(spearmanr(y_model, cv_pred).statistic) if len(y_model) >= 3 else math.nan
    final_alpha, weight_df = final_fit_weights(
        X_model,
        y_model,
        model_frame,
        inner_splits=inner_splits,
        spatial_block_side=spatial_block_side,
        feature_names=feature_names,
    )

    sample_label = str(model_frame["sample_label"].iloc[0])
    plot_predictions(
        observed=y_model,
        predicted=cv_pred,
        sample_label=sample_label,
        output_path=figures_dir / f"{sample_label}__observed_vs_cv_predicted.png",
        cv_r2=cv_r2,
        cv_rmse=cv_rmse,
        cv_spearman=cv_spearman,
    )

    weight_df.insert(0, "sample_label", sample_label)
    weight_df.insert(0, "sample_id_on_disk", str(model_frame["sample_id_on_disk"].iloc[0]))
    weight_df.insert(0, "dataset", str(model_frame["dataset"].iloc[0]))
    weight_df["final_alpha"] = final_alpha

    performance_row = {
        "dataset": str(model_frame["dataset"].iloc[0]),
        "sample_id_on_disk": str(model_frame["sample_id_on_disk"].iloc[0]),
        "sample_label": sample_label,
        "n_spots_modelled": int(len(y_model)),
        "n_programs": int(len(feature_names)),
        "cv_scheme": outer_scheme.scheme_name,
        "outer_n_splits": int(outer_scheme.n_splits),
        "spatial_group_count": int(outer_scheme.group_count),
        "spatial_block_side": outer_scheme.block_side,
        "cv_r2": cv_r2,
        "cv_rmse": cv_rmse,
        "cv_spearman": cv_spearman,
        "outer_alpha_mean": float(np.mean(outer_alphas)),
        "outer_alpha_median": float(np.median(outer_alphas)),
        "final_alpha": final_alpha,
    }
    return performance_row, weight_df


def main(run_dir: Path, output_subdir: str, outer_splits: int, inner_splits: int, spatial_block_side: int) -> None:
    d3b_root = run_dir / "02_definition3b_mixture_programme_niches"
    sample_tables = sorted(d3b_root.glob("*\\tables\\spot_level_table.csv"))
    if not sample_tables:
        raise FileNotFoundError(f"No Definition 3b spot-level tables found under {d3b_root}")

    output_root = ensure_dir(run_dir / output_subdir)
    tables_dir = ensure_dir(output_root / "tables")
    figures_dir = ensure_dir(output_root / "figures")

    performance_rows: list[dict[str, object]] = []
    weight_tables: list[pd.DataFrame] = []

    for sample_path in sample_tables:
        print(f"Analysing {sample_path.parent.parent.name} ...", flush=True)
        performance_row, weight_df = analyze_sample(
            sample_path=sample_path,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            spatial_block_side=spatial_block_side,
            figures_dir=figures_dir,
        )
        performance_rows.append(performance_row)
        weight_tables.append(weight_df)

    performance_df = pd.DataFrame(performance_rows).sort_values("cv_r2", ascending=False).reset_index(drop=True)
    weights_df = pd.concat(weight_tables, ignore_index=True)
    weights_df = weights_df.sort_values(["sample_label", "abs_weight_rank"]).reset_index(drop=True)

    performance_df.to_csv(tables_dir / "per_sample_ridge_cv_performance.csv", index=False)
    weights_df.to_csv(tables_dir / "per_sample_ridge_weights.csv", index=False)
    print(f"Wrote outputs to {output_root}", flush=True)


if __name__ == "__main__":
    args = parse_args()
    main(
        run_dir=Path(args.run_dir),
        output_subdir=str(args.output_subdir),
        outer_splits=int(args.outer_splits),
        inner_splits=int(args.inner_splits),
        spatial_block_side=int(args.spatial_block_side),
    )
