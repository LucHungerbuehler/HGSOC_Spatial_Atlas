"""Build report examples for per-sample cNMF ridge model fits."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"D:\HGSOC_Spatial_Atlas\05_analysis_ready\S3_cNMF_Tumor_Programs")
BRANCH = ROOT / "snai1ac_program_decomposition_unsmoothed_uncorrected_v1"
DIAGNOSTIC_BRANCH = ROOT / "snai1ac_kstar_usage_univariate_and_model_diagnostics_unsmoothed_uncorrected_v1"

PREDICTIONS = BRANCH / "02_per_sample_usage_models" / "tables" / "per_spot_predictions.csv"
COEFFICIENTS = BRANCH / "02_per_sample_usage_models" / "tables" / "per_sample_full_model_all_feature_coefficients.csv"
PERFORMANCE = BRANCH / "02_per_sample_usage_models" / "tables" / "per_sample_model_performance.csv"
EQUATIONS = DIAGNOSTIC_BRANCH / "tables" / "model_equations_original_scale.csv"

OUT_ROOT = BRANCH / "07_report_examples"
FIGURE_DIR = OUT_ROOT / "figures"
TABLE_DIR = OUT_ROOT / "tables"
EQUATION_DIR = OUT_ROOT / "equations"
CV_MODELS = [
    ("spatial_malignant", "pred_spatial_malignant", "Spatial plus malignant"),
    ("usage_raw_only", "pred_usage_raw_only", "Usage only"),
    ("spatial_malignant_usage_raw", "pred_spatial_malignant_usage_raw", "Full model"),
]

TERM_LABELS = {
    "array_row": "row",
    "array_col": "col",
    "array_row2": "row^2",
    "array_row_array_col": "row x col",
    "array_col2": "col^2",
    "Malignant": "malignant",
}


def manual_r2(y: pd.Series, pred: pd.Series) -> float:
    y_arr = y.to_numpy(dtype=float)
    pred_arr = pred.to_numpy(dtype=float)
    ss_res = float(np.sum((y_arr - pred_arr) ** 2))
    ss_tot = float(np.sum((y_arr - float(np.mean(y_arr))) ** 2))
    return 1.0 - ss_res / ss_tot


def label_feature(feature: str) -> str:
    if feature in TERM_LABELS:
        return TERM_LABELS[feature]
    if "__K" in feature and "__P" in feature:
        right = feature.split("__")[-2:]
        return " ".join(part.replace("K", "K").replace("P", "P") for part in right)
    return feature


def save_source_tables(sample: str, pred: pd.DataFrame, coef: pd.DataFrame, equation: str) -> None:
    applied = pred[
        [
            "dataset",
            "sample_id_on_disk",
            "sample_label",
            "spot_id",
            "array_row",
            "array_col",
            "Malignant",
            "SNAI1-ac_score",
            "full_model_reconstructed_prediction",
            "full_model_baseline_component",
            "full_model_usage_component",
        ]
    ].copy()
    applied["final_fit_residual"] = (
        pd.to_numeric(applied["SNAI1-ac_score"], errors="raise")
        - pd.to_numeric(applied["full_model_reconstructed_prediction"], errors="raise")
    )

    coef_out = coef[
        [
            "sample_label",
            "feature",
            "standardized_coef",
            "original_scale_coef",
            "is_program_feature",
            "program_id",
            "intercept_original_scale",
        ]
    ].copy()
    coef_out["display_term"] = coef_out["feature"].map(label_feature)

    applied.to_csv(TABLE_DIR / f"{sample}_final_fit_applied_to_spots.csv", index=False)
    coef_out.to_csv(TABLE_DIR / f"{sample}_final_fit_coefficients.csv", index=False)
    (EQUATION_DIR / f"{sample}_final_fit_equation.txt").write_text(equation + "\n", encoding="utf-8")


def plot_composite(sample: str, pred: pd.DataFrame, coef: pd.DataFrame, performance: pd.DataFrame) -> None:
    observed = pd.to_numeric(pred["SNAI1-ac_score"], errors="raise")
    fitted = pd.to_numeric(pred["full_model_reconstructed_prediction"], errors="raise")
    fit_r2 = manual_r2(observed, fitted)
    fit_rho = float(observed.corr(fitted, method="spearman"))

    coef_plot = coef.copy()
    coef_plot["display_term"] = coef_plot["feature"].map(label_feature)
    coef_plot["standardized_coef"] = pd.to_numeric(coef_plot["standardized_coef"], errors="raise")
    coef_plot["category"] = np.where(
        coef_plot["is_program_feature"].astype(str).str.lower().eq("true"),
        "program usage",
        np.where(coef_plot["feature"].eq("Malignant"), "malignant fraction", "spatial baseline"),
    )
    coef_colors = {
        "spatial baseline": "#7b8794",
        "malignant fraction": "#2f5f8f",
        "program usage": "#0f8b8d",
    }

    final_values = pd.concat([observed, fitted])
    final_pad = float(final_values.max() - final_values.min()) * 0.08
    final_limits = (float(final_values.min()) - final_pad, float(final_values.max()) + final_pad)

    cv_values = [observed]
    for _, pred_col, _ in CV_MODELS:
        cv_values.append(pd.to_numeric(pred[pred_col], errors="raise"))
    cv_values = pd.concat(cv_values)
    cv_pad = float(cv_values.max() - cv_values.min()) * 0.08
    cv_limits = (float(cv_values.min()) - cv_pad, float(cv_values.max()) + cv_pad)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.3,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
        }
    )

    fig_height = max(6.4, 4.8 + 0.12 * len(coef_plot))
    fig = plt.figure(figsize=(9.2, fig_height))
    grid = fig.add_gridspec(2, 6, height_ratios=[1.08, 1.0], hspace=0.38, wspace=0.55)
    ax_coef = fig.add_subplot(grid[0, 0:3])
    ax_final = fig.add_subplot(grid[0, 3:6])
    cv_axes = [
        fig.add_subplot(grid[1, 0:2]),
        fig.add_subplot(grid[1, 2:4]),
        fig.add_subplot(grid[1, 4:6]),
    ]

    coef_rev = coef_plot.iloc[::-1].copy()
    y_pos = np.arange(len(coef_rev))
    ax_coef.barh(
        y_pos,
        coef_rev["standardized_coef"],
        color=[coef_colors[c] for c in coef_rev["category"]],
        height=0.72,
    )
    ax_coef.axvline(0, color="#1f2937", linewidth=0.8)
    ax_coef.set_yticks(y_pos)
    ax_coef.set_yticklabels(coef_rev["display_term"])
    ax_coef.set_xlabel("Standardized coefficient")
    ax_coef.grid(axis="x", color="#e5e7eb", linewidth=0.6)

    ax_final.scatter(fitted, observed, s=10, alpha=0.46, linewidths=0, color="#9f4f63")
    ax_final.plot(final_limits, final_limits, color="#1f2937", linewidth=0.9, linestyle="--")
    ax_final.set_xlim(final_limits)
    ax_final.set_ylim(final_limits)
    ax_final.set_xlabel("Fitted SNAI1-ac")
    ax_final.set_ylabel("Measured SNAI1-ac")
    ax_final.grid(color="#e5e7eb", linewidth=0.6)
    ax_final.text(
        0.04,
        0.96,
        f"final fit\nn = {len(pred)}\nR2 = {fit_r2:.3f}\nrho = {fit_rho:.3f}",
        transform=ax_final.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "boxstyle": "round,pad=0.25", "alpha": 0.92},
    )

    cv_colors = ["#6b7280", "#0f8b8d", "#9f4f63"]
    for ax, (model, pred_col, label), color in zip(cv_axes, CV_MODELS, cv_colors):
        predicted = pd.to_numeric(pred[pred_col], errors="raise")
        perf = performance.loc[performance["model"].eq(model)].iloc[0]
        ax.scatter(predicted, observed, s=9, alpha=0.43, linewidths=0, color=color)
        ax.plot(cv_limits, cv_limits, color="#1f2937", linewidth=0.85, linestyle="--")
        ax.set_xlim(cv_limits)
        ax.set_ylim(cv_limits)
        ax.set_xlabel("Cross-validated predicted SNAI1-ac")
        ax.grid(color="#e5e7eb", linewidth=0.6)
        ax.text(
            0.04,
            0.96,
            f"{label}\nCV R2 = {float(perf['cv_r2']):.3f}\nrho = {float(perf['cv_spearman_rho']):.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            bbox={"facecolor": "white", "edgecolor": "#d1d5db", "boxstyle": "round,pad=0.25", "alpha": 0.92},
        )
    cv_axes[0].set_ylabel("Measured SNAI1-ac")
    for ax in cv_axes[1:]:
        ax.set_yticklabels([])

    fig.savefig(FIGURE_DIR / f"{sample}_model_fit_composite.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_sample(sample: str, predictions: pd.DataFrame, coefficients: pd.DataFrame, performance: pd.DataFrame, equations: pd.DataFrame) -> None:
    pred = predictions.loc[predictions["sample_label"].eq(sample)].copy()
    coef = coefficients.loc[coefficients["sample_label"].eq(sample)].copy()
    perf = performance.loc[performance["sample_label"].eq(sample)].copy()
    equation_rows = equations.loc[equations["sample_label"].eq(sample)].copy()

    if pred.empty:
        raise RuntimeError(f"Missing prediction rows for {sample}")
    if coef.empty:
        raise RuntimeError(f"Missing coefficient rows for {sample}")
    if perf.empty:
        raise RuntimeError(f"Missing performance rows for {sample}")
    if equation_rows.empty:
        raise RuntimeError(f"Missing equation row for {sample}")

    equation = str(equation_rows["original_scale_equation"].iloc[0])
    save_source_tables(sample, pred, coef, equation)
    plot_composite(sample, pred, coef, perf)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    EQUATION_DIR.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(PREDICTIONS)
    coefficients = pd.read_csv(COEFFICIENTS)
    performance = pd.read_csv(PERFORMANCE)
    equations = pd.read_csv(EQUATIONS)

    samples = sorted(
        set(predictions["sample_label"])
        & set(coefficients["sample_label"])
        & set(performance["sample_label"])
        & set(equations["sample_label"])
    )
    if not samples:
        raise RuntimeError("No shared samples found across prediction, coefficient, performance and equation tables")

    for sample in samples:
        build_sample(sample, predictions, coefficients, performance, equations)

    print(OUT_ROOT)
    print(f"generated_samples={len(samples)}")
    print("\n".join(samples))


if __name__ == "__main__":
    main()
