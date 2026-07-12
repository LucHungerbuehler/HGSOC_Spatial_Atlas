"""
Create spot-level SNAI1-ac vs raw local K* cNMF usage plots and per-sample
model diagnostics for the unsmoothed/uncorrected SNAI1-ac decomposition branch.

Inputs are read from the existing fitted branch. This script does not refit the
ridge models; it visualizes the saved CV predictions and programme weights.
"""

from __future__ import annotations

import json
import math
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"D:\HGSOC_Spatial_Atlas\05_analysis_ready\S3_cNMF_Tumor_Programs")
DECOMP = ROOT / "snai1ac_program_decomposition_unsmoothed_uncorrected_v1"
OUT = ROOT / "snai1ac_kstar_usage_univariate_and_model_diagnostics_unsmoothed_uncorrected_v1"

PREDICTIONS = DECOMP / "02_per_sample_usage_models" / "tables" / "per_spot_predictions.csv"
PERFORMANCE = DECOMP / "02_per_sample_usage_models" / "tables" / "per_sample_model_performance.csv"
WEIGHTS = DECOMP / "02_per_sample_usage_models" / "tables" / "per_sample_program_weights.csv"
FULL_COEFFICIENTS = DECOMP / "02_per_sample_usage_models" / "tables" / "per_sample_full_model_all_feature_coefficients.csv"
USAGE_ROOT = ROOT / "per_sample"

TARGET_COL = "snai1ac_em_unsmoothed_uncorrected"
PRED_COL = "pred_spatial_malignant_usage_raw"
BASELINE_COL = "full_model_baseline_component"
USAGE_COMPONENT_COL = "full_model_usage_component"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def program_short(program_id: str) -> str:
    match = re.search(r"__(K\d+)__(P\d+)$", program_id)
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return program_id.split("__")[-1]


def coef_text(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.3g}"


def signed_term(value: float, label: str) -> str:
    if not np.isfinite(value):
        return f"+ NA*{label}"
    return f"{value:+.3g}*{label}"


def is_program_feature(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def build_model_equation_record(
    sample_label: str,
    full_coefficients: pd.DataFrame,
    performance: pd.DataFrame,
) -> dict[str, object]:
    coeff = full_coefficients[full_coefficients["sample_label"].eq(sample_label)].copy()
    perf = performance[
        (performance["sample_label"].eq(sample_label))
        & (performance["model"].eq("spatial_malignant_usage_raw"))
    ]
    if coeff.empty:
        return {
            "sample_label": sample_label,
            "model": "spatial_malignant_usage_raw",
            "model_form": "Missing coefficient rows.",
            "baseline_original_scale_terms": "",
            "program_original_scale_terms": "",
            "original_scale_equation": "",
            "figure_text": "Model equation unavailable: missing full coefficient rows.",
            "final_alpha": math.nan,
        }

    coeff["original_scale_coef"] = pd.to_numeric(coeff["original_scale_coef"], errors="coerce")
    coeff["intercept_original_scale"] = pd.to_numeric(coeff["intercept_original_scale"], errors="coerce")
    intercept = float(coeff["intercept_original_scale"].dropna().iloc[0])
    coeff_map = dict(zip(coeff["feature"], coeff["original_scale_coef"], strict=False))

    baseline_order = ["array_row", "array_col", "array_row2", "array_row_array_col", "array_col2", "Malignant"]
    baseline_terms = [signed_term(float(coeff_map.get(name, math.nan)), name) for name in baseline_order]

    program_coeff = coeff[is_program_feature(coeff["is_program_feature"])].copy()
    program_coeff["short"] = program_coeff["feature"].map(program_short)
    program_coeff = program_coeff.sort_values("feature")
    program_terms = [
        signed_term(float(row.original_scale_coef), str(row.short))
        for row in program_coeff.itertuples(index=False)
    ]

    model_form = (
        "Ridge model: y_hat = intercept + spatial polynomial "
        "(array_row, array_col, array_row2, array_row*array_col, array_col2) "
        "+ Malignant + raw local K* programme usages."
    )
    baseline_text = f"intercept={coef_text(intercept)} " + " ".join(baseline_terms)
    program_text = " ".join(program_terms)
    equation = f"y_hat = {coef_text(intercept)} " + " ".join(baseline_terms + program_terms)
    final_alpha = float(perf["final_alpha"].iloc[0]) if len(perf) else math.nan
    figure_text = "\n".join(
        textwrap.wrap(model_form, width=145)
        + [
            f"Final refit coefficients are original scale; CV prediction panel uses out-of-fold predictions. Ridge alpha={coef_text(final_alpha)}.",
            "Baseline: " + baseline_text,
            "Programme usage term: " + program_text,
        ]
    )

    return {
        "sample_label": sample_label,
        "model": "spatial_malignant_usage_raw",
        "model_form": model_form,
        "baseline_original_scale_terms": baseline_text,
        "program_original_scale_terms": program_text,
        "original_scale_equation": equation,
        "figure_text": figure_text,
        "final_alpha": final_alpha,
    }


def numeric_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    tmp = pd.concat([x, y], axis=1).dropna()
    if len(tmp) < 3:
        return math.nan
    if tmp.iloc[:, 0].nunique() < 2 or tmp.iloc[:, 1].nunique() < 2:
        return math.nan
    return float(tmp.iloc[:, 0].corr(tmp.iloc[:, 1], method=method))


def read_usage(sample_id: str) -> pd.DataFrame:
    path = USAGE_ROOT / sample_id / "representative_usage_kstar.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing usage table for {sample_id}: {path}")
    return pd.read_csv(path)


def program_columns(frame: pd.DataFrame, sample_label: str | None = None) -> list[str]:
    cols = []
    for col in frame.columns:
        if re.search(r"__K\d+__P\d+$", col):
            cols.append(col)
    if sample_label:
        sample_cols = [col for col in cols if col.startswith(sample_label + "__")]
        if sample_cols:
            return sample_cols
    return cols


def row_normalize_usage(frame: pd.DataFrame, program_cols: list[str]) -> pd.DataFrame:
    normed = frame.copy()
    row_sums = normed[program_cols].sum(axis=1)
    denom = row_sums.where(row_sums > 0, np.nan)
    normed[program_cols] = normed[program_cols].div(denom, axis=0)
    return normed


def add_linear_trend(ax, x: np.ndarray, y: np.ndarray) -> None:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 3 or len(np.unique(x)) < 2:
        return
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
    ax.plot(xs, slope * xs + intercept, color="#1f2937", linewidth=1.0, alpha=0.9)


def plot_univariate_sample(
    sample_label: str,
    sample_frame: pd.DataFrame,
    program_cols: list[str],
    out_dir: Path,
    usage_scale: str,
    x_label: str,
    file_tag: str,
) -> list[dict[str, object]]:
    n_programs = len(program_cols)
    ncols = min(4, max(1, n_programs))
    nrows = int(math.ceil(n_programs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 3.4 * nrows), squeeze=False)
    axes_flat = axes.ravel()
    records: list[dict[str, object]] = []

    y = sample_frame[TARGET_COL]
    for idx, program_col in enumerate(program_cols):
        ax = axes_flat[idx]
        x = sample_frame[program_col]
        pearson = numeric_corr(x, y, "pearson")
        spearman = numeric_corr(x, y, "spearman")
        non_missing = pd.concat([x, y], axis=1).dropna()

        ax.scatter(x, y, s=10, alpha=0.45, color="#326b8c", edgecolors="none")
        add_linear_trend(ax, x.to_numpy(dtype=float), y.to_numpy(dtype=float))
        ax.set_title(
            f"{program_short(program_col)}\nPearson r={pearson:.2f}, Spearman r={spearman:.2f}",
            fontsize=9,
        )
        ax.set_xlabel(x_label, fontsize=8)
        ax.set_ylabel("Unsmoothed/uncorrected SNAI1-ac", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, color="#e5e7eb", linewidth=0.6)

        records.append(
            {
                "dataset": sample_frame["dataset"].iloc[0],
                "sample_id_on_disk": sample_frame["sample_id_on_disk"].iloc[0],
                "sample_label": sample_label,
                "usage_scale": usage_scale,
                "program_id": program_col,
                "program_short": program_short(program_col),
                "n_spots": int(len(non_missing)),
                "usage_min": float(non_missing.iloc[:, 0].min()) if len(non_missing) else math.nan,
                "usage_median": float(non_missing.iloc[:, 0].median()) if len(non_missing) else math.nan,
                "usage_max": float(non_missing.iloc[:, 0].max()) if len(non_missing) else math.nan,
                "pearson_r": pearson,
                "spearman_r": spearman,
                "abs_spearman_r": abs(spearman) if np.isfinite(spearman) else math.nan,
            }
        )

    for ax in axes_flat[n_programs:]:
        ax.axis("off")

    fig.suptitle(
        f"{sample_label}: {usage_scale} local K* programme usage vs unsmoothed/uncorrected SNAI1-ac",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = out_dir / f"{safe_name(sample_label)}__{file_tag}_kstar_usage_vs_snai1ac.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return records


def scatter_tissue(
    ax,
    frame: pd.DataFrame,
    value_col: str,
    title: str,
    cmap: str = "coolwarm",
    symmetric: bool = False,
) -> None:
    values = frame[value_col].to_numpy(dtype=float)
    if symmetric:
        vmax = np.nanpercentile(np.abs(values), 98)
        vmin = -vmax
    else:
        vmin = np.nanpercentile(values, 2)
        vmax = np.nanpercentile(values, 98)
    sc = ax.scatter(
        frame["array_col"],
        -frame["array_row"],
        c=values,
        s=12,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="none",
    )
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)


def plot_model_diagnostics_sample(
    sample_label: str,
    sample_frame: pd.DataFrame,
    weights: pd.DataFrame,
    performance: pd.DataFrame,
    model_equation: dict[str, object],
    out_dir: Path,
) -> dict[str, object]:
    sample_frame = sample_frame.copy()
    sample_frame["residual_full_model"] = sample_frame[TARGET_COL] - sample_frame[PRED_COL]

    sample_perf = performance[
        (performance["sample_label"].eq(sample_label))
        & (performance["model"].eq("spatial_malignant_usage_raw"))
    ]
    baseline_perf = performance[
        (performance["sample_label"].eq(sample_label))
        & (performance["model"].eq("spatial_malignant"))
    ]
    cv_r2 = float(sample_perf["cv_r2"].iloc[0]) if len(sample_perf) else math.nan
    cv_spearman = float(sample_perf["cv_spearman_rho"].iloc[0]) if len(sample_perf) else math.nan
    delta_r2 = float(sample_perf["delta_r2_vs_spatial_malignant"].iloc[0]) if len(sample_perf) else math.nan
    baseline_r2 = float(baseline_perf["cv_r2"].iloc[0]) if len(baseline_perf) else math.nan

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.8))

    w = weights[weights["sample_label"].eq(sample_label)].copy()
    if len(w):
        w = w.sort_values("program_standardized_weight")
        colors = np.where(w["program_standardized_weight"] >= 0, "#2f855a", "#c2410c")
        axes[0, 0].barh([program_short(v) for v in w["program_id"]], w["program_standardized_weight"], color=colors)
        axes[0, 0].axvline(0, color="#111827", linewidth=0.8)
        axes[0, 0].set_xlabel("Standardized coefficient", fontsize=8)
        axes[0, 0].set_title("Raw K* programme weights", fontsize=9)
        axes[0, 0].tick_params(labelsize=7)
    else:
        axes[0, 0].text(0.5, 0.5, "No weights found", ha="center", va="center")
        axes[0, 0].axis("off")

    ax = axes[0, 1]
    obs = sample_frame[TARGET_COL].to_numpy(dtype=float)
    pred = sample_frame[PRED_COL].to_numpy(dtype=float)
    ax.scatter(pred, obs, s=10, alpha=0.45, color="#4b5563", edgecolors="none")
    finite = np.isfinite(obs) & np.isfinite(pred)
    if finite.any():
        lo = min(float(obs[finite].min()), float(pred[finite].min()))
        hi = max(float(obs[finite].max()), float(pred[finite].max()))
        ax.plot([lo, hi], [lo, hi], color="#991b1b", linewidth=1.0)
    ax.set_xlabel("CV predicted SNAI1-ac", fontsize=8)
    ax.set_ylabel("Measured unsmoothed/uncorrected SNAI1-ac", fontsize=8)
    ax.set_title(
        f"Measured vs predicted\nCV R2={cv_r2:.2f}, baseline R2={baseline_r2:.2f}, delta={delta_r2:.2f}, rho={cv_spearman:.2f}",
        fontsize=9,
    )
    ax.tick_params(labelsize=7)
    ax.grid(True, color="#e5e7eb", linewidth=0.6)

    scatter_tissue(axes[0, 2], sample_frame, TARGET_COL, "Measured in tissue", symmetric=True)
    scatter_tissue(axes[1, 0], sample_frame, PRED_COL, "CV predicted in tissue", symmetric=True)
    scatter_tissue(axes[1, 1], sample_frame, "residual_full_model", "Residual in tissue", symmetric=True)
    scatter_tissue(axes[1, 2], sample_frame, USAGE_COMPONENT_COL, "Programme usage component", symmetric=True)

    fig.suptitle(
        f"{sample_label}: unsmoothed/uncorrected SNAI1-ac model diagnostics",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.015,
        0.015,
        str(model_equation["figure_text"]),
        ha="left",
        va="bottom",
        fontsize=7.2,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f9fafb", "edgecolor": "#d1d5db", "linewidth": 0.8},
    )
    fig.tight_layout(rect=[0, 0.18, 1, 0.96])
    out_path = out_dir / f"{safe_name(sample_label)}__model_fit_coefficients_tissue.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    return {
        "sample_label": sample_label,
        "n_spots": int(len(sample_frame)),
        "n_program_weights": int(len(w)),
        "cv_r2_full_model": cv_r2,
        "cv_r2_spatial_malignant_baseline": baseline_r2,
        "delta_r2_usage_after_spatial_malignant": delta_r2,
        "cv_spearman_full_model": cv_spearman,
        "figure": str(out_path),
    }


def plot_correlation_heatmap(corr: pd.DataFrame, out_dir: Path, filename: str, title: str) -> Path:
    if corr.empty:
        raise ValueError("No correlation records to plot")
    ordered = corr.sort_values(["dataset", "sample_id_on_disk", "program_short"]).copy()
    ordered["row_label"] = ordered["sample_id_on_disk"] + " " + ordered["program_short"]
    fig_height = max(8, len(ordered) * 0.11)
    fig, ax = plt.subplots(figsize=(6.5, fig_height))
    vals = ordered[["spearman_r"]].to_numpy(dtype=float)
    im = ax.imshow(vals, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_yticks(np.arange(len(ordered)))
    ax.set_yticklabels(ordered["row_label"], fontsize=5)
    ax.set_xticks([0])
    ax.set_xticklabels(["Spearman r"], fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.08, pad=0.03)
    fig.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def write_readme(
    out: Path,
    corr_raw: pd.DataFrame,
    corr_norm: pd.DataFrame,
    join_audit: pd.DataFrame,
    model_manifest: pd.DataFrame,
) -> None:
    lines = [
        "# SNAI1-ac raw K* usage scatter and model diagnostics",
        "",
        "This branch visualizes sample-local K* cNMF programme usage against the",
        "unsmoothed/uncorrected SNAI1-ac score on tumour spots. It includes both the",
        "stored raw usage values and a row-normalized within-spot composition version.",
        "It also visualizes the already-fitted unsmoothed/uncorrected ridge decomposition",
        "branch without refitting the models.",
        "",
        "## Inputs",
        "",
        f"- Decomposition branch: `{DECOMP}`",
        f"- Per-spot predictions: `{PREDICTIONS}`",
        f"- Full model coefficients: `{FULL_COEFFICIENTS}`",
        f"- Raw K* usage source: `{USAGE_ROOT}\\<sample>\\representative_usage_kstar.csv`",
        "",
        "## Outputs",
        "",
        "- `tables/kstar_usage_univariate_correlations.csv`: one row per raw local K* programme.",
        "- `tables/kstar_usage_univariate_correlations_row_normalized.csv`: one row per row-normalized local K* programme.",
        "- `tables/kstar_usage_univariate_correlations_all_scales.csv`: raw and row-normalized correlations stacked together.",
        "- `tables/kstar_usage_join_audit.csv`: join counts for each sample.",
        "- `tables/model_diagnostic_manifest.csv`: per-sample model visualization summary.",
        "- `tables/model_equations_original_scale.csv`: per-sample written model equations using original-scale final refit coefficients.",
        "- `figures/univariate_scatter/`: one raw-usage composite scatter figure per sample.",
        "- `figures/univariate_scatter_row_normalized/`: one row-normalized composite scatter figure per sample.",
        "- `figures/model_diagnostics/`: one model fit / coefficient / tissue panel per sample.",
        "- `figures/overview/kstar_usage_spearman_heatmap_all_programmes.png`: compact raw-usage all-programme correlation overview.",
        "- `figures/overview/kstar_usage_spearman_heatmap_all_programmes_row_normalized.png`: compact row-normalized correlation overview.",
        "",
        "## Scope notes",
        "",
        "- The y-axis is `snai1ac_em_unsmoothed_uncorrected`.",
        "- The raw x-axis is local K* usage exactly as stored in",
        "  `representative_usage_kstar.csv`.",
        "- The row-normalized x-axis divides each programme usage by the spot-level sum",
        "  across all K* programmes from the same sample.",
        "- Programme labels are sample-local and should not be interpreted as harmonized MP1-MP8 identities.",
        "- The model diagnostic panels use saved cross-validated predictions from",
        "  `pred_spatial_malignant_usage_raw`; they do not refit the model.",
        "- The written equations in the model diagnostic panels use final ridge refit",
        "  coefficients on the original feature scale. The measured-vs-predicted",
        "  scatter uses out-of-fold predictions, so its plotted predictions are",
        "  not exactly the same object as the final refit equation.",
        "",
        "## Join audit",
        "",
        f"- Samples joined: {join_audit['sample_label'].nunique()}",
        f"- Total joined tumour spots: {int(join_audit['joined_spots'].sum())}",
        f"- Total local K* programmes plotted: {int(corr_raw['program_id'].nunique())}",
        f"- Minimum joined spots per sample: {int(join_audit['joined_spots'].min())}",
        f"- Maximum joined spots per sample: {int(join_audit['joined_spots'].max())}",
        "",
    ]
    for label, corr in [("Raw usage", corr_raw), ("Row-normalized usage", corr_norm)]:
        strongest_pos = corr.sort_values("spearman_r", ascending=False).head(8)
        strongest_neg = corr.sort_values("spearman_r", ascending=True).head(8)
        lines.extend(["", f"## {label}: strongest positive Spearman correlations", ""])
        for row in strongest_pos.itertuples(index=False):
            lines.append(
                f"- {row.sample_label} {row.program_short}: Spearman r={row.spearman_r:.3f}, n={row.n_spots}"
            )
        lines.extend(["", f"## {label}: strongest negative Spearman correlations", ""])
        for row in strongest_neg.itertuples(index=False):
            lines.append(
                f"- {row.sample_label} {row.program_short}: Spearman r={row.spearman_r:.3f}, n={row.n_spots}"
            )
    lines.extend(
        [
            "",
            "## Model diagnostic availability",
            "",
            f"- Samples with model diagnostic panels: {model_manifest['sample_label'].nunique()}",
            f"- Median full-model CV R2: {model_manifest['cv_r2_full_model'].median():.3f}",
            f"- Median CV R2 gain from raw usage after spatial/malignant baseline: {model_manifest['delta_r2_usage_after_spatial_malignant'].median():.3f}",
            "",
        ]
    )
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    tables_dir = ensure_dir(OUT / "tables")
    univariate_raw_dir = ensure_dir(OUT / "figures" / "univariate_scatter")
    univariate_norm_dir = ensure_dir(OUT / "figures" / "univariate_scatter_row_normalized")
    model_dir = ensure_dir(OUT / "figures" / "model_diagnostics")
    overview_dir = ensure_dir(OUT / "figures" / "overview")

    predictions = pd.read_csv(PREDICTIONS)
    performance = pd.read_csv(PERFORMANCE)
    weights = pd.read_csv(WEIGHTS)
    full_coefficients = pd.read_csv(FULL_COEFFICIENTS)

    missing = [col for col in [TARGET_COL, PRED_COL, "array_row", "array_col"] if col not in predictions.columns]
    if missing:
        raise ValueError(f"Missing required prediction columns: {missing}")

    all_corr_raw: list[dict[str, object]] = []
    all_corr_norm: list[dict[str, object]] = []
    join_records: list[dict[str, object]] = []
    model_records: list[dict[str, object]] = []
    equation_records: list[dict[str, object]] = []

    for sample_label, sample_pred in predictions.groupby("sample_label", sort=True):
        sample_id = str(sample_pred["sample_id_on_disk"].iloc[0])
        usage = read_usage(sample_id)
        program_cols = program_columns(usage, sample_label)
        if not program_cols:
            raise ValueError(f"No raw K* programme columns found for {sample_label}")

        join_cols = ["spot_id", "dataset", "sample_id_on_disk", "sample_label"]
        joined = sample_pred.merge(usage[join_cols + program_cols], on=join_cols, how="inner")

        join_records.append(
            {
                "dataset": sample_pred["dataset"].iloc[0],
                "sample_id_on_disk": sample_id,
                "sample_label": sample_label,
                "prediction_spots": int(len(sample_pred)),
                "usage_spots": int(len(usage)),
                "joined_spots": int(len(joined)),
                "n_program_cols": int(len(program_cols)),
                "missing_from_join": int(len(sample_pred) - len(joined)),
                "usage_row_sum_min": float(joined[program_cols].sum(axis=1).min()),
                "usage_row_sum_median": float(joined[program_cols].sum(axis=1).median()),
                "usage_row_sum_max": float(joined[program_cols].sum(axis=1).max()),
            }
        )

        joined_norm = row_normalize_usage(joined, program_cols)
        all_corr_raw.extend(
            plot_univariate_sample(
                sample_label,
                joined,
                program_cols,
                univariate_raw_dir,
                usage_scale="raw",
                x_label="Raw local K* usage",
                file_tag="raw",
            )
        )
        all_corr_norm.extend(
            plot_univariate_sample(
                sample_label,
                joined_norm,
                program_cols,
                univariate_norm_dir,
                usage_scale="row-normalized",
                x_label="Row-normalized local K* usage",
                file_tag="row_normalized",
            )
        )
        equation_record = build_model_equation_record(sample_label, full_coefficients, performance)
        equation_records.append(equation_record)
        model_records.append(
            plot_model_diagnostics_sample(sample_label, joined, weights, performance, equation_record, model_dir)
        )

    corr_raw = pd.DataFrame(all_corr_raw).sort_values(["dataset", "sample_id_on_disk", "program_id"])
    corr_norm = pd.DataFrame(all_corr_norm).sort_values(["dataset", "sample_id_on_disk", "program_id"])
    corr_all = pd.concat([corr_raw, corr_norm], ignore_index=True)
    join_audit = pd.DataFrame(join_records).sort_values(["dataset", "sample_id_on_disk"])
    model_manifest = pd.DataFrame(model_records).sort_values("sample_label")
    model_equations = pd.DataFrame(equation_records).sort_values("sample_label")

    corr_raw.to_csv(tables_dir / "kstar_usage_univariate_correlations.csv", index=False)
    corr_norm.to_csv(tables_dir / "kstar_usage_univariate_correlations_row_normalized.csv", index=False)
    corr_all.to_csv(tables_dir / "kstar_usage_univariate_correlations_all_scales.csv", index=False)
    join_audit.to_csv(tables_dir / "kstar_usage_join_audit.csv", index=False)
    model_manifest.to_csv(tables_dir / "model_diagnostic_manifest.csv", index=False)
    model_equations.to_csv(tables_dir / "model_equations_original_scale.csv", index=False)
    heatmap_path = plot_correlation_heatmap(
        corr_raw,
        overview_dir,
        "kstar_usage_spearman_heatmap_all_programmes.png",
        "Raw local K* usage vs unsmoothed/uncorrected SNAI1-ac",
    )
    heatmap_norm_path = plot_correlation_heatmap(
        corr_norm,
        overview_dir,
        "kstar_usage_spearman_heatmap_all_programmes_row_normalized.png",
        "Row-normalized local K* usage vs unsmoothed/uncorrected SNAI1-ac",
    )

    manifest = {
        "input_decomposition_branch": str(DECOMP),
        "output_branch": str(OUT),
        "target_col": TARGET_COL,
        "raw_usage_root": str(USAGE_ROOT),
        "n_samples": int(join_audit["sample_label"].nunique()),
        "n_joined_spots": int(join_audit["joined_spots"].sum()),
        "n_programmes": int(corr_raw["program_id"].nunique()),
        "univariate_scatter_raw_dir": str(univariate_raw_dir),
        "univariate_scatter_row_normalized_dir": str(univariate_norm_dir),
        "model_diagnostic_dir": str(model_dir),
        "model_equations": str(tables_dir / "model_equations_original_scale.csv"),
        "raw_heatmap": str(heatmap_path),
        "row_normalized_heatmap": str(heatmap_norm_path),
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_readme(OUT, corr_raw, corr_norm, join_audit, model_manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
