"""
Build report-facing GASTON composite figures from source tables.

This script intentionally regenerates every panel from source data. It does not
crop or paste from existing composite PNGs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import gaston_build_gradient_identity_review as gradient_review
import gaston_sp1_relationship_review as relationship_review


GASTON_ROOT = Path(r"D:\HGSOC_Spatial_Atlas\05_analysis_ready\GASTON_method_v1")
DEFAULT_OUT_DIR = GASTON_ROOT / "08_report_asset"

OBSERVED_SCORE_COLUMNS = [
    "snai1ac_em_smooth_corrected",
    "snai1ac_em_unsmoothed_uncorrected",
]
FITTED_SCORE_COLUMNS = [
    "snai1ac_em_smooth_corrected",
    "snai1ac_em_unsmoothed_uncorrected",
]

SCORE_LABELS = {
    "snai1ac_em_smooth_corrected": "SNAI1-ac smoothed corrected",
    "snai1ac_em_unsmoothed_corrected": "SNAI1-ac unsmoothed corrected",
    "snai1ac_em_unsmoothed_uncorrected": "SNAI1-ac unsmoothed uncorrected",
}
PANEL_LABELS = tuple("ABCDEFGHI")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaston-root", type=Path, default=GASTON_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dataset", default="denisenko_2022")
    parser.add_argument("--sample", default="SP6")
    parser.add_argument("--feature-method", default="glmpca")
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Generate one composite for every row in sample_alignment_manifest.csv.",
    )
    parser.add_argument("--num-bins", type=int, default=15)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--spatial-orientation", choices=["image", "cartesian"], default="image")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def read_manifest(args: argparse.Namespace) -> pd.DataFrame:
    manifest_path = args.gaston_root / "04_isodepth_score_alignment" / "sample_alignment_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    required = ["dataset", "sample", "feature_method", "alignment_table", "h5ad_path"]
    missing = [column for column in required if column not in manifest.columns]
    if missing:
        raise ValueError(f"{manifest_path} is missing required columns: {missing}")
    return manifest


def select_manifest_rows(args: argparse.Namespace) -> pd.DataFrame:
    manifest = read_manifest(args)
    if args.all_samples:
        rows = manifest.copy()
    else:
        rows = manifest[
            (manifest["dataset"].astype(str) == str(args.dataset))
            & (manifest["sample"].astype(str) == str(args.sample))
            & (manifest["feature_method"].astype(str) == str(args.feature_method))
        ].copy()
    if rows.empty:
        if args.all_samples:
            raise ValueError("No rows found in sample_alignment_manifest.csv")
        raise ValueError(
            "No row found in sample_alignment_manifest.csv for "
            f"{args.dataset} / {args.sample} / {args.feature_method}"
        )
    return rows.reset_index(drop=True)


def read_manifest_row(args: argparse.Namespace) -> pd.Series:
    rows = select_manifest_rows(args)
    if len(rows) != 1:
        raise ValueError(f"Expected one manifest row, found {len(rows)}")
    return rows.iloc[0]


def sample_identity(row: pd.Series) -> tuple[str, str, str]:
    return str(row["dataset"]), str(row["sample"]), str(row["feature_method"])


def sample_payload(row: pd.Series) -> dict[str, str]:
    dataset, sample, feature_method = sample_identity(row)
    return {
        "dataset": dataset,
        "sample": sample,
        "feature_method": feature_method,
    }


def shared_zero_centered_score_limits(df: pd.DataFrame) -> tuple[float, float]:
    max_abs = 1.0
    for variant in relationship_review.VARIANTS:
        lo, hi = relationship_review.robust_limits(df[variant["column"]])
        max_abs = max(max_abs, abs(lo), abs(hi))
    return -float(max_abs), float(max_abs)


def symmetric_limits(values: np.ndarray, percentile: float = 98.0) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0
    hi = float(np.nanpercentile(np.abs(values), percentile))
    if not np.isfinite(hi) or hi == 0:
        hi = float(np.nanmax(np.abs(values)))
    if not np.isfinite(hi) or hi == 0:
        hi = 1.0
    return -hi, hi


def shared_displayed_snai1ac_map_limits(
    df: pd.DataFrame,
    fitted_by_column: dict[str, np.ndarray],
) -> tuple[float, float]:
    arrays = []
    for column in OBSERVED_SCORE_COLUMNS:
        arrays.append(pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float))
    for column in FITTED_SCORE_COLUMNS:
        arrays.append(np.asarray(fitted_by_column[column], dtype=float))
    return symmetric_limits(np.concatenate(arrays), percentile=98.0)


def plot_hires_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    h5ad_path: Path,
    spatial_orientation: str,
) -> None:
    relationship_review.plot_hires_or_scaffold(ax, df, h5ad_path, spatial_orientation)
    ax.set_title("H&E", fontsize=10)


def add_panel_labels(fig: plt.Figure, axes: np.ndarray) -> None:
    fig.canvas.draw()
    for label, ax in zip(PANEL_LABELS, axes):
        bbox = ax.get_position()
        fig.text(
            bbox.x0 - 0.018,
            bbox.y1 + 0.006,
            label,
            ha="left",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color="#202124",
        )


def plot_score_trends(ax: plt.Axes, binned: pd.DataFrame, df: pd.DataFrame) -> None:
    x = binned["bin_center"].to_numpy(dtype=float)
    isodepth = df["gaston_isodepth_malignant_oriented"].to_numpy(dtype=float)
    for variant in relationship_review.VARIANTS:
        col = variant["column"]
        rho, _, _ = relationship_review.safe_spearman(
            isodepth,
            df[col].to_numpy(dtype=float),
        )
        rho_label = f"{rho:.2f}" if np.isfinite(rho) else "NA"
        ax.plot(
            x,
            binned[f"{col}_median"].to_numpy(dtype=float),
            marker="o",
            lw=1.8,
            ms=4,
            color=variant["color"],
            label=f"{variant['short_label']}, rho={rho_label}",
        )
        ax.fill_between(
            x,
            binned[f"{col}_q25"].to_numpy(dtype=float),
            binned[f"{col}_q75"].to_numpy(dtype=float),
            color=variant["color"],
            alpha=0.14,
            linewidth=0,
        )
    ax.axhline(0, color="#202124", lw=0.8, alpha=0.45)
    ax.set_ylabel("SNAI1-ac score")
    ax.legend(frameon=False, fontsize=7.5, loc="best")
    ax.tick_params(labelsize=8)
    ax.set_title("SNAI1-ac score trends along isodepth", fontsize=10)
    ax.set_xlabel("GASTON isodepth")


def order_segments(df: pd.DataFrame, mapping_path: Path) -> np.ndarray:
    mapping = pd.read_csv(mapping_path)
    original_to_segment = {
        int(row["original_domain_label"]): int(row["gradient_segment"])
        for _, row in mapping.iterrows()
    }
    return np.asarray(
        [original_to_segment[int(label)] for label in df["gaston_domain_selected"]],
        dtype=int,
    )


def fitted_from_coefficients(
    df: pd.DataFrame,
    fit_table: pd.DataFrame,
    segments: np.ndarray,
    score_column: str,
) -> np.ndarray:
    rows = fit_table[fit_table["score_column"].astype(str) == score_column].copy()
    if rows.empty:
        raise ValueError(f"No fitted-score coefficients found for {score_column}")
    by_segment = rows.set_index("gradient_segment")
    isodepth = df["gaston_isodepth_malignant_oriented"].to_numpy(dtype=float)
    fitted = np.full(len(df), np.nan, dtype=float)
    for segment in sorted(np.unique(segments)):
        if segment not in by_segment.index:
            continue
        coeff = by_segment.loc[segment]
        fitted[segments == segment] = (
            float(coeff["intercept"])
            + float(coeff["slope"]) * isodepth[segments == segment]
        )
    return fitted


def draw_fitted_score_map(
    ax: plt.Axes,
    df: pd.DataFrame,
    fitted: np.ndarray,
    title: str,
    spatial_orientation: str,
    value_limits: tuple[float, float],
    cmap: str = "coolwarm",
) -> None:
    coords = df[["spatial_x", "spatial_y"]].to_numpy(dtype=float)
    vmin, vmax = value_limits
    im = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=fitted,
        cmap=cmap,
        s=float(np.clip(18000.0 / max(coords.shape[0], 1), 4.0, 14.0)),
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
    )
    finite = np.isfinite(fitted)
    if int(finite.sum()) >= 3:
        try:
            contours = ax.tricontour(
                coords[finite, 0],
                coords[finite, 1],
                fitted[finite],
                levels=6,
                linewidths=0.8,
                colors="k",
                linestyles="solid",
            )
            ax.clabel(contours, contours.levels, inline=True, fontsize=6)
        except Exception:
            pass
    ax.set_title(title, fontsize=10)
    relationship_review.format_spatial_axis(ax, spatial_orientation)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def plot_tissue_composition(ax: plt.Axes, binned: pd.DataFrame) -> None:
    x = binned["bin_center"].to_numpy(dtype=float)
    bottom = np.zeros(len(binned), dtype=float)
    legend_handles = {}
    for label in ["tumor", "interface", "stroma", "unknown"]:
        values = binned[f"{label}_fraction"].to_numpy(dtype=float)
        palette_label = label.capitalize() if label != "unknown" else "Unknown"
        fill = ax.fill_between(
            x,
            bottom,
            bottom + values,
            color=gradient_review.INTERFACE_PALETTE[palette_label],
            alpha=0.75,
            label=palette_label,
            linewidth=0,
        )
        if palette_label != "Unknown":
            legend_handles[palette_label] = fill
        bottom += values
    ax.set_ylim(0, 1)
    ax.set_xlabel("GASTON isodepth")
    ax.set_ylabel("Fraction")
    ax.set_title("Tissue-label composition along isodepth", fontsize=10)
    legend_order = ["Stroma", "Interface", "Tumor"]
    legend = ax.legend(
        [legend_handles[label] for label in legend_order if label in legend_handles],
        [label for label in legend_order if label in legend_handles],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        fontsize=7,
        frameon=False,
        ncol=1,
    )
    legend.set_in_layout(False)


def build_composite(args: argparse.Namespace, row: pd.Series | None = None) -> dict[str, Any]:
    if row is None:
        row = read_manifest_row(args)
    dataset, sample, feature_method = sample_identity(row)
    alignment_path = Path(str(row["alignment_table"]))
    h5ad_path = Path(str(row["h5ad_path"]))
    sample_stem = f"{dataset}__{sample}__whole__{feature_method}"
    display_stem = f"{dataset}__{sample}"
    native_root = args.gaston_root / "07_gradient_review" / "02_gaston_native_gradient_identity"
    mapping_path = native_root / "tables" / f"{sample_stem}__domain_order_mapping.csv"
    fit_table_path = native_root / "tables" / f"{sample_stem}__snai1ac_score_piecewise_fits.csv"

    for path in [alignment_path, h5ad_path, mapping_path, fit_table_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    df = pd.read_csv(alignment_path)
    relationship_bins, _ = relationship_review.make_bins(df, args.num_bins)
    gradient_bins, _ = gradient_review.make_bins(df, args.num_bins)
    fit_table = pd.read_csv(fit_table_path)
    segments = order_segments(df, mapping_path)
    fitted_by_column = {
        score_column: fitted_from_coefficients(df, fit_table, segments, score_column)
        for score_column in FITTED_SCORE_COLUMNS
    }
    score_limits = shared_displayed_snai1ac_map_limits(df, fitted_by_column)

    fig, axes = plt.subplots(3, 3, figsize=(13.5, 12.3), constrained_layout=True)
    axes = axes.ravel()

    plot_hires_panel(axes[0], df, h5ad_path, args.spatial_orientation)
    relationship_review.scatter_spatial_continuous(
        axes[1],
        df,
        "snai1ac_em_smooth_corrected",
        "SNAI1-ac smoothed corrected",
        args.spatial_orientation,
        cmap="coolwarm",
        value_limits=score_limits,
    )
    relationship_review.scatter_spatial_continuous(
        axes[2],
        df,
        "snai1ac_em_unsmoothed_uncorrected",
        "SNAI1-ac unsmoothed uncorrected",
        args.spatial_orientation,
        cmap="coolwarm",
        value_limits=score_limits,
    )

    gradient_review.scatter_continuous(
        axes[3],
        df,
        "Malignant",
        "Malignant fraction",
        args.spatial_orientation,
        cmap="magma",
    )
    for ax, score_column in zip(axes[4:6], FITTED_SCORE_COLUMNS):
        draw_fitted_score_map(
            ax,
            df,
            fitted_by_column[score_column],
            f"Fitted {SCORE_LABELS[score_column]}",
            args.spatial_orientation,
            value_limits=score_limits,
    )

    plot_tissue_composition(axes[6], gradient_bins)
    plot_score_trends(axes[7], relationship_bins, df)
    relationship_review.scatter_spatial_continuous(
        axes[8],
        df,
        "gaston_isodepth_malignant_oriented",
        "GASTON isodepth",
        args.spatial_orientation,
        cmap="viridis",
    )
    add_panel_labels(fig, axes)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_png = args.out_dir / f"{display_stem}__gaston_report_composite_3x3.png"
    out_json = args.out_dir / f"{display_stem}__gaston_report_composite_3x3_provenance.json"
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    provenance = {
        "output_png": str(out_png),
        "dataset": dataset,
        "sample": sample,
        "feature_method": feature_method,
        "generated_from_source": True,
        "no_png_cropping_or_pasting": True,
        "sources": {
            "alignment_table": str(alignment_path),
            "h5ad_image_source": str(h5ad_path),
            "domain_order_mapping": str(mapping_path),
            "snai1ac_score_piecewise_fits": str(fit_table_path),
            "relationship_panel_used_as_visual_spec_only": str(
                args.gaston_root
                / "05_relationship_review"
                / f"{display_stem}__relationship_panel.png"
            ),
            "gradient_identity_panel_used_as_visual_spec_only": str(
                args.gaston_root
                / "07_gradient_review"
                / "01_gradient_identity_review"
                / f"{display_stem}__gradient_identity_panel.png"
            ),
        },
        "panel_layout": [
            ["H&E image", "SNAI1-ac smoothed corrected", "SNAI1-ac unsmoothed uncorrected"],
            ["Malignant fraction", "Fitted SNAI1-ac smoothed corrected", "Fitted SNAI1-ac unsmoothed uncorrected"],
            ["Tissue-label composition along isodepth", "SNAI1-ac score trends along isodepth", "GASTON isodepth"],
        ],
        "panel_labels": list(PANEL_LABELS),
        "score_map_limits": {
            "all_displayed_snai1ac_maps": {
                "type": "shared_zero_centered_98th_percentile_absolute_limit",
                "applies_to": [
                    "observed smoothed corrected",
                    "observed unsmoothed uncorrected",
                    "fitted smoothed corrected",
                    "fitted unsmoothed uncorrected",
                ],
                "cmap": "coolwarm",
                "vmin": score_limits[0],
                "vmax": score_limits[1],
            },
        },
    }
    write_json(out_json, provenance)
    return provenance


def main() -> None:
    args = parse_args()
    rows = select_manifest_rows(args)
    provenances: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for _, row in rows.iterrows():
        identity = sample_payload(row)
        label = f"{identity['dataset']} / {identity['sample']} / {identity['feature_method']}"
        try:
            provenance = build_composite(args, row)
            provenances.append(provenance)
            print(f"generated\t{label}\t{provenance['output_png']}", flush=True)
        except Exception as exc:
            failure = {**identity, "error": repr(exc)}
            failures.append(failure)
            print(f"FAILED\t{label}\t{exc}", file=sys.stderr, flush=True)

    batch_manifest = {
        "generated_from_source": True,
        "no_png_cropping_or_pasting": True,
        "num_requested": int(len(rows)),
        "num_generated": int(len(provenances)),
        "num_failed": int(len(failures)),
        "outputs": [
            {
                "dataset": item["dataset"],
                "sample": item["sample"],
                "feature_method": item["feature_method"],
                "output_png": item["output_png"],
            }
            for item in provenances
        ],
        "failures": failures,
    }
    if args.all_samples:
        batch_manifest_path = args.out_dir / "gaston_report_composite_3x3_manifest.json"
        write_json(batch_manifest_path, batch_manifest)
        print(json.dumps({**batch_manifest, "batch_manifest": str(batch_manifest_path)}, indent=2))
    else:
        print(json.dumps(batch_manifest, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
