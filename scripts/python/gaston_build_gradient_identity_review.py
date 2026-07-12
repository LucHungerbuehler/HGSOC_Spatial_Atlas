"""
Build a GASTON gradient-identity evidence pack.

This is a descriptive review layer. It does not assign final biological
interpretations. It summarizes, for each accepted whole-tissue GASTON sample,
what the learned malignant-oriented isodepth is aligned with in the evidence
already generated so far:

- tissue-label composition along isodepth
- QC/depth trends along isodepth
- selected domains along isodepth
- SNAI1-ac score variants as contextual overlays
- GASTON-native topology and class-leader gene evidence generated in the
  gene-gradient layer

This script is a review index and evidence pack. It does not force a
paper-style Type I/II/III taxonomy or assign final biological interpretations.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy, spearmanr


GASTON_ROOT = Path(r"D:\HGSOC_Spatial_Atlas\05_analysis_ready\GASTON_method_v1")
DEFAULT_OUT_DIR = GASTON_ROOT / "07_gradient_review" / "01_gradient_identity_review"
DEFAULT_NATIVE_GENE_DIR = GASTON_ROOT / "07_gradient_review" / "02_gaston_native_gradient_identity"

INTERFACE_ORDER = ["Tumor", "Interface", "Stroma", "Unknown"]
INTERFACE_PALETTE = {
    "Tumor": "#FDE725",
    "Interface": "#21918C",
    "Stroma": "#440154",
    "Unknown": "#BDBDBD",
}

VARIANTS = [
    {
        "column": "snai1ac_em_smooth_corrected",
        "label": "SNAI1-ac smooth+GAM",
        "color": "#4C78A8",
    },
    {
        "column": "snai1ac_em_unsmoothed_corrected",
        "label": "SNAI1-ac unsmoothed+GAM",
        "color": "#F58518",
    },
    {
        "column": "snai1ac_em_unsmoothed_uncorrected",
        "label": "SNAI1-ac unsmoothed no GAM",
        "color": "#54A24B",
    },
]

QC_COLS = ["total_counts", "n_genes_by_counts", "n_genes", "n_counts"]
CORR_COLS = [
    ("Malignant", "Malignant fraction"),
    ("total_counts", "UMI/depth"),
    ("n_genes_by_counts", "Detected genes"),
    ("snai1ac_em_smooth_corrected", "SNAI1-ac smooth+GAM"),
    ("snai1ac_em_unsmoothed_corrected", "SNAI1-ac unsmoothed+GAM"),
    ("snai1ac_em_unsmoothed_uncorrected", "SNAI1-ac unsmoothed no GAM"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaston-root", type=Path, default=GASTON_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--native-gene-dir", type=Path, default=DEFAULT_NATIVE_GENE_DIR)
    parser.add_argument("--samples", nargs="*", default=None)
    parser.add_argument("--all-samples", action="store_true", default=True)
    parser.add_argument("--num-bins", type=int, default=15)
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.20,
        help="Fraction of spots used to summarize low/high isodepth context.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--spatial-orientation",
        choices=["image", "cartesian"],
        default="image",
        help="Use image orientation to match the existing GASTON review packs.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def safe_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def safe_spearman(x: Any, y: Any) -> tuple[float, float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 3:
        return float("nan"), float("nan"), n
    if np.nanstd(x[mask]) == 0 or np.nanstd(y[mask]) == 0:
        return float("nan"), float("nan"), n
    stat = spearmanr(x[mask], y[mask])
    return float(stat.statistic), float(stat.pvalue), n


def strength_bin(rho: float) -> str:
    if not np.isfinite(rho):
        return "not_available"
    a = abs(rho)
    if a < 0.10:
        return "negligible"
    if a < 0.20:
        return "weak"
    if a < 0.35:
        return "moderate"
    return "strong"


def canonical_interface(values: pd.Series) -> pd.Series:
    labels = []
    for value in values.fillna("Unknown"):
        text = str(value).strip().lower()
        if text == "tumor":
            labels.append("Tumor")
        elif text == "interface":
            labels.append("Interface")
        elif text == "stroma":
            labels.append("Stroma")
        else:
            labels.append("Unknown")
    return pd.Series(labels, index=values.index, dtype="object")


def format_spatial_axis(ax: plt.Axes, spatial_orientation: str) -> None:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if spatial_orientation == "image":
        ax.invert_yaxis()


def robust_limits(values: Any) -> tuple[float, float]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    lo, hi = np.nanpercentile(arr, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    return float(lo), float(hi)


def make_bins(df: pd.DataFrame, num_bins: int) -> tuple[pd.DataFrame, np.ndarray]:
    x = df["gaston_isodepth_malignant_oriented"].to_numpy(dtype=float)
    lo = math.floor(float(np.nanmin(x))) - 0.5
    hi = math.ceil(float(np.nanmax(x))) + 0.5
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    bins = np.linspace(lo, hi, num_bins + 1)
    bin_index = np.digitize(x, bins) - 1
    bin_index = np.clip(bin_index, 0, num_bins - 1)
    work = df.copy()
    work["_bin_index"] = bin_index
    work["_bin_center"] = np.asarray([0.5 * (bins[i] + bins[i + 1]) for i in bin_index])
    work["_interface_canonical"] = canonical_interface(work.get("interface", pd.Series(index=work.index)))

    rows: list[dict[str, Any]] = []
    for bin_id, group in work.groupby("_bin_index", sort=True):
        row: dict[str, Any] = {
            "bin_index": int(bin_id),
            "bin_center": float(group["_bin_center"].iloc[0]),
            "bin_left": float(bins[int(bin_id)]),
            "bin_right": float(bins[int(bin_id) + 1]),
            "n_spots": int(len(group)),
            "isodepth_min": float(group["gaston_isodepth_malignant_oriented"].min()),
            "isodepth_max": float(group["gaston_isodepth_malignant_oriented"].max()),
            "domain_mode": int(group["gaston_domain_selected"].mode().iloc[0]),
        }
        labels = group["_interface_canonical"]
        for label in INTERFACE_ORDER:
            row[f"{label.lower()}_fraction"] = float((labels == label).mean())
        for col in ["Malignant", *QC_COLS, *[v["column"] for v in VARIANTS]]:
            if col not in group.columns:
                continue
            values = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_median"] = float(values.median())
            row[f"{col}_q25"] = float(values.quantile(0.25))
            row[f"{col}_q75"] = float(values.quantile(0.75))
        rows.append(row)
    return pd.DataFrame(rows), bins


def domain_context(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["_interface_canonical"] = canonical_interface(work.get("interface", pd.Series(index=work.index)))
    rows: list[dict[str, Any]] = []
    for domain, group in work.groupby("gaston_domain_selected", sort=True):
        labels = group["_interface_canonical"]
        counts = labels.value_counts(normalize=True)
        fractions = {label: float(counts.get(label, 0.0)) for label in INTERFACE_ORDER}
        tissue_entropy = float(
            entropy([fractions[label] for label in INTERFACE_ORDER if fractions[label] > 0.0])
        )
        dominant = max(INTERFACE_ORDER, key=lambda label: fractions[label])
        row: dict[str, Any] = {
            "domain": int(domain),
            "n_spots": int(len(group)),
            "pct_spots": float(100 * len(group) / len(work)),
            "isodepth_min": float(group["gaston_isodepth_malignant_oriented"].min()),
            "isodepth_median": float(group["gaston_isodepth_malignant_oriented"].median()),
            "isodepth_max": float(group["gaston_isodepth_malignant_oriented"].max()),
            "dominant_tissue_label": dominant,
            "dominant_tissue_fraction": fractions[dominant],
            "tissue_entropy": tissue_entropy,
        }
        for label, fraction in fractions.items():
            row[f"{label.lower()}_fraction"] = fraction
        for col in ["Malignant", *QC_COLS, *[v["column"] for v in VARIANTS]]:
            if col in group.columns:
                row[f"{col}_median"] = float(pd.to_numeric(group[col], errors="coerce").median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("isodepth_median").reset_index(drop=True)


def tail_context(df: pd.DataFrame, tail_fraction: float) -> dict[str, Any]:
    work = df.sort_values("gaston_isodepth_malignant_oriented").copy()
    n_tail = max(1, int(round(len(work) * tail_fraction)))
    low = work.head(n_tail).copy()
    high = work.tail(n_tail).copy()
    low_labels = canonical_interface(low.get("interface", pd.Series(index=low.index)))
    high_labels = canonical_interface(high.get("interface", pd.Series(index=high.index)))

    out: dict[str, Any] = {"tail_fraction": float(tail_fraction), "tail_n_spots": int(n_tail)}
    for prefix, labels in [("low", low_labels), ("high", high_labels)]:
        for label in INTERFACE_ORDER:
            out[f"{prefix}_{label.lower()}_fraction"] = float((labels == label).mean())
        out[f"{prefix}_dominant_tissue_label"] = max(
            INTERFACE_ORDER, key=lambda label: out[f"{prefix}_{label.lower()}_fraction"]
        )
    for label in ["tumor", "interface", "stroma", "unknown"]:
        out[f"delta_high_minus_low_{label}_fraction"] = (
            out[f"high_{label}_fraction"] - out[f"low_{label}_fraction"]
        )
    for col in ["Malignant", *QC_COLS, *[v["column"] for v in VARIANTS]]:
        if col in work.columns:
            out[f"low_{col}_median"] = float(pd.to_numeric(low[col], errors="coerce").median())
            out[f"high_{col}_median"] = float(pd.to_numeric(high[col], errors="coerce").median())
            out[f"delta_high_minus_low_{col}_median"] = (
                out[f"high_{col}_median"] - out[f"low_{col}_median"]
            )
    return out


def candidate_identity(summary: dict[str, Any], binned: pd.DataFrame) -> tuple[str, str]:
    rho_malignant = abs(safe_float(summary.get("rho_Malignant")))
    rho_qc_values = [
        abs(safe_float(summary.get("rho_total_counts"))),
        abs(safe_float(summary.get("rho_n_genes_by_counts"))),
        abs(safe_float(summary.get("rho_n_genes"))),
        abs(safe_float(summary.get("rho_n_counts"))),
    ]
    rho_qc = max([v for v in rho_qc_values if np.isfinite(v)] or [float("nan")])
    delta_tumor = abs(safe_float(summary.get("delta_high_minus_low_tumor_fraction")))
    delta_stroma = abs(safe_float(summary.get("delta_high_minus_low_stroma_fraction")))
    delta_interface = abs(safe_float(summary.get("delta_high_minus_low_interface_fraction")))
    composition_shift = max(delta_tumor, delta_stroma, delta_interface)

    interior = binned.iloc[1:-1] if len(binned) > 2 else binned
    interface_peak = safe_float(interior.get("interface_fraction", pd.Series([np.nan])).max())
    edge_interface = max(
        safe_float(binned["interface_fraction"].iloc[0]) if len(binned) else float("nan"),
        safe_float(binned["interface_fraction"].iloc[-1]) if len(binned) else float("nan"),
    )

    low_label = str(summary.get("low_dominant_tissue_label", "Unknown"))
    high_label = str(summary.get("high_dominant_tissue_label", "Unknown"))

    if np.isfinite(rho_qc) and rho_qc >= 0.35 and rho_qc >= max(rho_malignant, composition_shift) + 0.10:
        return (
            "qc_depth_linked_gradient_candidate",
            f"QC/depth association is strongest (max |rho|={rho_qc:.2f}).",
        )
    if {low_label, high_label} == {"Tumor", "Stroma"} and composition_shift >= 0.25:
        return (
            "tumor_stroma_axis_candidate",
            f"Low/high isodepth dominant labels shift {low_label} -> {high_label}.",
        )
    if delta_tumor >= 0.30 or rho_malignant >= 0.35:
        direction = "increases" if safe_float(summary.get("delta_high_minus_low_tumor_fraction")) > 0 else "decreases"
        return (
            "tumor_fraction_axis_candidate",
            f"Tumor fraction {direction} along isodepth; |rho Malignant|={rho_malignant:.2f}.",
        )
    if delta_stroma >= 0.30:
        direction = "increases" if safe_float(summary.get("delta_high_minus_low_stroma_fraction")) > 0 else "decreases"
        return (
            "stromal_fraction_axis_candidate",
            f"Stroma fraction {direction} along isodepth.",
        )
    if np.isfinite(interface_peak) and interface_peak >= 0.30 and interface_peak >= edge_interface + 0.10:
        return (
            "interface_enriched_mid_isodepth_candidate",
            f"Interface fraction peaks in interior bins ({interface_peak:.2f}).",
        )
    if composition_shift >= 0.20:
        return (
            "mixed_tissue_composition_axis_candidate",
            f"Low/high isodepth tissue fractions shift, but not cleanly by one label (max shift={composition_shift:.2f}).",
        )
    if np.isfinite(rho_qc) and rho_qc >= 0.25:
        return (
            "weak_to_moderate_qc_link_candidate",
            f"QC/depth association is present but not dominant (max |rho|={rho_qc:.2f}).",
        )
    return (
        "not_explained_by_basic_tissue_or_qc_candidate",
        "No strong alignment with tissue labels, malignant fraction, or basic QC metrics in this summary.",
    )


def scatter_continuous(
    ax: plt.Axes,
    df: pd.DataFrame,
    col: str,
    title: str,
    spatial_orientation: str,
    cmap: str = "viridis",
) -> None:
    vmin, vmax = robust_limits(df[col])
    sca = ax.scatter(
        df["spatial_x"],
        df["spatial_y"],
        c=pd.to_numeric(df[col], errors="coerce"),
        s=5,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
    )
    ax.set_title(title, fontsize=9)
    format_spatial_axis(ax, spatial_orientation)
    ax.axis("off")
    plt.colorbar(sca, ax=ax, fraction=0.046, pad=0.02)


def scatter_domains(ax: plt.Axes, df: pd.DataFrame, spatial_orientation: str) -> None:
    labels = pd.to_numeric(df["gaston_domain_selected"], errors="coerce").to_numpy(dtype=float)
    max_label = int(np.nanmax(labels)) if labels.size else 0
    cmap = plt.get_cmap("tab20", max_label + 1)
    sca = ax.scatter(
        df["spatial_x"],
        df["spatial_y"],
        c=labels,
        s=5,
        cmap=cmap,
        vmin=-0.5,
        vmax=max_label + 0.5,
        linewidths=0,
    )
    ax.set_title(f"Selected domains (k={max_label + 1})", fontsize=9)
    format_spatial_axis(ax, spatial_orientation)
    ax.axis("off")
    ticks = np.arange(0, max_label + 1)
    cbar = plt.colorbar(sca, ax=ax, fraction=0.046, pad=0.02, ticks=ticks)
    cbar.ax.set_yticklabels([str(t) for t in ticks])


def scatter_interface(ax: plt.Axes, df: pd.DataFrame, spatial_orientation: str) -> None:
    labels = canonical_interface(df.get("interface", pd.Series(index=df.index)))
    for label in INTERFACE_ORDER:
        mask = labels == label
        if not mask.any():
            continue
        ax.scatter(
            df.loc[mask, "spatial_x"],
            df.loc[mask, "spatial_y"],
            s=5,
            c=INTERFACE_PALETTE[label],
            linewidths=0,
            label=label,
        )
    ax.set_title("Tumor / interface / stroma labels", fontsize=9)
    format_spatial_axis(ax, spatial_orientation)
    ax.axis("off")
    ax.legend(loc="lower left", fontsize=7, frameon=False, markerscale=2)


def plot_tissue_bins(ax: plt.Axes, binned: pd.DataFrame) -> None:
    x = binned["bin_center"].to_numpy(dtype=float)
    bottom = np.zeros(len(binned), dtype=float)
    for label in ["tumor", "interface", "stroma", "unknown"]:
        values = binned[f"{label}_fraction"].to_numpy(dtype=float)
        ax.fill_between(
            x,
            bottom,
            bottom + values,
            color=INTERFACE_PALETTE[label.capitalize() if label != "unknown" else "Unknown"],
            alpha=0.75,
            label=label.capitalize(),
            linewidth=0,
        )
        bottom += values
    ax.set_ylim(0, 1)
    ax.set_xlabel("GASTON isodepth")
    ax.set_ylabel("Fraction")
    ax.set_title("Tissue-label composition along isodepth", fontsize=9)
    ax.legend(loc="upper left", fontsize=7, frameon=False, ncol=2)


def plot_qc_bins(ax: plt.Axes, binned: pd.DataFrame) -> None:
    x = binned["bin_center"].to_numpy(dtype=float)
    for col, label, color in [
        ("total_counts_median", "UMI/depth", "#A24F46"),
        ("n_genes_by_counts_median", "Detected genes", "#7B6D8D"),
        ("Malignant_median", "Malignant", "#2A9D8F"),
    ]:
        if col not in binned.columns:
            continue
        y = pd.to_numeric(binned[col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(y).any():
            continue
        lo, hi = np.nanmin(y), np.nanmax(y)
        if hi > lo:
            y_scaled = (y - lo) / (hi - lo)
        else:
            y_scaled = np.zeros_like(y)
        ax.plot(x, y_scaled, marker="o", markersize=3, linewidth=1.3, color=color, label=label)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("GASTON isodepth")
    ax.set_ylabel("Scaled median")
    ax.set_title("QC / malignant trends along isodepth", fontsize=9)
    ax.legend(loc="best", fontsize=7, frameon=False)


def plot_score_bins(ax: plt.Axes, binned: pd.DataFrame) -> None:
    x = binned["bin_center"].to_numpy(dtype=float)
    for variant in VARIANTS:
        col = f"{variant['column']}_median"
        if col not in binned.columns:
            continue
        ax.plot(
            x,
            pd.to_numeric(binned[col], errors="coerce"),
            marker="o",
            markersize=3,
            linewidth=1.2,
            color=variant["color"],
            label=variant["label"],
        )
    ax.axhline(0, color="#777777", linewidth=0.7, alpha=0.7)
    ax.set_xlabel("GASTON isodepth")
    ax.set_ylabel("Median score")
    ax.set_title("SNAI1-ac variants along isodepth", fontsize=9)
    ax.legend(loc="best", fontsize=7, frameon=False)


def plot_corr_bars(ax: plt.Axes, summary: dict[str, Any]) -> None:
    labels = []
    values = []
    colors = []
    for col, label in CORR_COLS:
        key = f"rho_{col}"
        value = safe_float(summary.get(key))
        if not np.isfinite(value):
            continue
        labels.append(label)
        values.append(value)
        if col == "Malignant":
            colors.append("#2A9D8F")
        elif col in QC_COLS:
            colors.append("#A24F46")
        else:
            colors.append("#4C78A8")
    y = np.arange(len(values))
    ax.barh(y, values, color=colors, alpha=0.85)
    for x in [-0.35, -0.20, -0.10, 0.10, 0.20, 0.35]:
        ax.axvline(x, color="#BDBDBD", linewidth=0.5, zorder=0)
    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman rho with isodepth")
    ax.set_title("Association strengths", fontsize=9)


def make_panel(
    df: pd.DataFrame,
    binned: pd.DataFrame,
    summary: dict[str, Any],
    out_png: Path,
    args: argparse.Namespace,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(12, 11), constrained_layout=True)
    axes = axes.ravel()
    scatter_continuous(
        axes[0],
        df,
        "gaston_isodepth_malignant_oriented",
        "GASTON isodepth",
        args.spatial_orientation,
        cmap="viridis",
    )
    scatter_domains(axes[1], df, args.spatial_orientation)
    scatter_interface(axes[2], df, args.spatial_orientation)
    scatter_continuous(axes[3], df, "Malignant", "Malignant fraction", args.spatial_orientation, cmap="magma")
    scatter_continuous(axes[4], df, "total_counts", "UMI/depth", args.spatial_orientation, cmap="cividis")
    scatter_continuous(
        axes[5],
        df,
        "snai1ac_em_smooth_corrected",
        "SNAI1-ac smooth+GAM",
        args.spatial_orientation,
        cmap="RdBu_r",
    )
    plot_tissue_bins(axes[6], binned)
    plot_qc_bins(axes[7], binned)
    plot_corr_bars(axes[8], summary)
    sample_label = f"{summary['dataset']} / {summary['sample']} ({summary['analysis_tier']}, {summary['feature_method']})"
    fig.suptitle(
        f"{sample_label}\nAutomated review flag, not interpretation: {summary['machine_suggested_identity']} | {summary['machine_suggested_reason']}",
        fontsize=11,
        y=1.02,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def make_contact_sheet(panel_paths: list[Path], out_png: Path, title: str, cols: int = 4) -> None:
    existing = [p for p in panel_paths if p.exists()]
    if not existing:
        return
    rows = int(math.ceil(len(existing) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.6), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, path in zip(axes_arr, existing):
        img = plt.imread(path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(path.name.split("__gradient_identity_panel")[0], fontsize=8)
    for ax in axes_arr[len(existing) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def read_optional_table(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def native_source_stem(row: pd.Series | dict[str, Any]) -> str:
    return "__".join(
        [
            str(row.get("dataset", "")),
            str(row.get("sample", "")),
            str(row.get("layer", "whole")),
            str(row.get("feature_method", "")),
        ]
    )


def compact_class_evidence(class_summary: pd.DataFrame, class_type: str, max_rows: int = 4) -> str:
    if class_summary.empty or "class_type" not in class_summary.columns:
        return ""
    sub = class_summary[class_summary["class_type"].astype(str) == class_type].copy()
    if sub.empty:
        return ""
    sub["n_genes_num"] = pd.to_numeric(sub.get("n_genes"), errors="coerce").fillna(0)
    sub = sub.sort_values(["n_genes_num", "class_id"], ascending=[False, True]).head(max_rows)
    pieces: list[str] = []
    for _, row in sub.iterrows():
        genes = str(row.get("top_genes", "")).split(";")
        genes = [gene for gene in genes if gene][:4]
        terms = str(row.get("top_hallmark_terms", "")).split(";")
        terms = [term.replace("HALLMARK_", "") for term in terms if term][:2]
        term_text = f" | Hallmark: {', '.join(terms)}" if terms else ""
        pieces.append(
            f"{row.get('class_id', '')} n={int(row['n_genes_num'])}: {', '.join(genes)}{term_text}"
        )
    return " ; ".join(pieces)


def collect_native_gene_evidence(row: pd.Series, native_dir: Path) -> dict[str, Any]:
    stem = native_source_stem(row)
    tables_dir = native_dir / "tables"
    figures_dir = native_dir / "figures"
    class_summary_path = tables_dir / f"{stem}__gradient_identity_class_summary.csv"
    package_index_path = tables_dir / f"{stem}__package_style_gene_panel_index.csv"

    evidence: dict[str, Any] = {
        "native_gene_source_stem": stem,
        "native_gene_evidence_status": "missing",
        "native_class_summary_csv": str(class_summary_path) if class_summary_path.exists() else "",
        "native_package_panel_index_csv": str(package_index_path) if package_index_path.exists() else "",
        "native_topology_isodepth_contours_png": str(figures_dir / f"{stem}__isodepth_contours_streamlines.png")
        if (figures_dir / f"{stem}__isodepth_contours_streamlines.png").exists()
        else "",
        "native_ordered_domain_boundaries_png": str(figures_dir / f"{stem}__ordered_domain_boundaries.png")
        if (figures_dir / f"{stem}__ordered_domain_boundaries.png").exists()
        else "",
        "native_continuous_leader_curves_png": str(figures_dir / f"{stem}__continuous_class_leader_gene_curves.png")
        if (figures_dir / f"{stem}__continuous_class_leader_gene_curves.png").exists()
        else "",
        "native_discontinuous_leader_curves_png": str(figures_dir / f"{stem}__discontinuous_class_leader_gene_curves.png")
        if (figures_dir / f"{stem}__discontinuous_class_leader_gene_curves.png").exists()
        else "",
        "native_class_size_png": str(figures_dir / f"{stem}__gradient_identity_class_sizes.png")
        if (figures_dir / f"{stem}__gradient_identity_class_sizes.png").exists()
        else "",
        "score_overlay_three_variant_fitted_map_png": str(
            figures_dir / "snai1ac_score_fitted_maps" / f"{stem}__snai1ac_three_variant_fitted_score_maps.png"
        )
        if (figures_dir / "snai1ac_score_fitted_maps" / f"{stem}__snai1ac_three_variant_fitted_score_maps.png").exists()
        else "",
    }

    if class_summary_path.exists():
        class_summary = pd.read_csv(class_summary_path)
        evidence["native_gene_evidence_status"] = "ok"
        evidence["native_n_class_summary_rows"] = int(len(class_summary))
        evidence["native_top_continuous_gene_evidence"] = compact_class_evidence(
            class_summary, "continuous_slope"
        )
        evidence["native_top_discontinuous_gene_evidence"] = compact_class_evidence(
            class_summary, "boundary_discontinuity"
        )
    else:
        evidence["native_n_class_summary_rows"] = 0
        evidence["native_top_continuous_gene_evidence"] = ""
        evidence["native_top_discontinuous_gene_evidence"] = ""

    if package_index_path.exists():
        package_index = pd.read_csv(package_index_path)
        evidence["native_package_panels_n"] = int(len(package_index))
        evidence["native_package_panels_ok_n"] = int((package_index.get("status", "") == "ok").sum())
        status_counts = package_index.get("status", pd.Series(dtype=str)).astype(str).value_counts().to_dict()
        evidence["native_package_panel_status_counts"] = ";".join(
            f"{key}:{value}" for key, value in sorted(status_counts.items())
        )
    else:
        evidence["native_package_panels_n"] = 0
        evidence["native_package_panels_ok_n"] = 0
        evidence["native_package_panel_status_counts"] = ""

    return evidence


def process_sample(row: pd.Series, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    table_path = Path(str(row["alignment_table"]))
    df = pd.read_csv(table_path)
    binned, bins = make_bins(df, args.num_bins)
    domains = domain_context(df)
    tail = tail_context(df, args.tail_fraction)
    x = df["gaston_isodepth_malignant_oriented"].to_numpy(dtype=float)

    summary: dict[str, Any] = {
        "dataset": row["dataset"],
        "sample": row["sample"],
        "layer": row.get("layer", "whole"),
        "feature_method": row["feature_method"],
        "analysis_tier": row["analysis_tier"],
        "include_in_primary_cross_sample": row.get("include_in_primary_cross_sample", ""),
        "selected_k": row.get("selected_k", ""),
        "selection_source": row.get("selection_source", ""),
        "review_decision": row.get("review_decision", ""),
        "review_role": row.get("review_role", ""),
        "attention_flags": row.get("attention_flags", ""),
        "warning_flags": row.get("warning_flags", ""),
        "n_spots": int(len(df)),
        "num_bins": int(args.num_bins),
        "candidate_status": "automated_review_flag_not_interpretation",
        "alignment_table": str(table_path),
    }
    summary.update(tail)

    labels = canonical_interface(df.get("interface", pd.Series(index=df.index)))
    for label in INTERFACE_ORDER:
        summary[f"overall_{label.lower()}_fraction"] = float((labels == label).mean())

    for col in ["Malignant", *QC_COLS, *[v["column"] for v in VARIANTS]]:
        if col not in df.columns:
            continue
        rho, pval, n = safe_spearman(x, df[col].to_numpy(dtype=float))
        summary[f"rho_{col}"] = rho
        summary[f"p_{col}"] = pval
        summary[f"n_{col}"] = n
        summary[f"strength_{col}"] = strength_bin(rho)

    candidate, reason = candidate_identity(summary, binned)
    summary["machine_suggested_identity"] = candidate
    summary["machine_suggested_reason"] = reason
    summary["low_high_tissue_shift"] = (
        f"{summary['low_dominant_tissue_label']} -> {summary['high_dominant_tissue_label']}"
    )
    summary["max_abs_qc_rho"] = max(
        [
            abs(safe_float(summary.get("rho_total_counts"))),
            abs(safe_float(summary.get("rho_n_genes_by_counts"))),
            abs(safe_float(summary.get("rho_n_genes"))),
            abs(safe_float(summary.get("rho_n_counts"))),
        ]
    )
    summary["max_abs_score_rho"] = max(
        [
            abs(safe_float(summary.get("rho_snai1ac_em_smooth_corrected"))),
            abs(safe_float(summary.get("rho_snai1ac_em_unsmoothed_corrected"))),
            abs(safe_float(summary.get("rho_snai1ac_em_unsmoothed_uncorrected"))),
        ]
    )

    stem = f"{row['dataset']}__{row['sample']}"
    panel_png = out_dir / f"{stem}__gradient_identity_panel.png"
    bins_csv = out_dir / f"{stem}__gradient_identity_bins.csv"
    domains_csv = out_dir / f"{stem}__gradient_identity_domains.csv"
    summary_json = out_dir / f"{stem}__gradient_identity_summary.json"

    make_panel(df, binned, summary, panel_png, args)
    binned.to_csv(bins_csv, index=False)
    domains.to_csv(domains_csv, index=False)

    summary["gradient_identity_panel_png"] = str(panel_png)
    summary["gradient_identity_bins_csv"] = str(bins_csv)
    summary["gradient_identity_domains_csv"] = str(domains_csv)
    summary["gradient_identity_summary_json"] = str(summary_json)
    summary["bin_edges"] = [float(v) for v in bins]
    summary.update(collect_native_gene_evidence(row, args.native_gene_dir))
    write_json(summary_json, summary)
    return summary


def write_readme(out_dir: Path, args: argparse.Namespace, overview: pd.DataFrame) -> None:
    counts = overview["machine_suggested_identity"].value_counts().to_dict()
    lines = [
        "# GASTON Gradient Identity Evidence Pack",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Purpose: describe, for each accepted whole-tissue GASTON sample, what the learned malignant-oriented isodepth appears to track before any final sample interpretation is written.",
        "",
        "This pack is intentionally descriptive. `machine_suggested_identity` values are automated review flags, not final biological calls.",
        "",
        "Evidence included:",
        "",
        "- GASTON isodepth map",
        "- selected domain map",
        "- tumor/interface/stroma label map",
        "- malignant fraction map",
        "- UMI/depth map",
        "- production SNAI1-ac score map",
        "- tissue-label fractions along 15 equal-width isodepth bins",
        "- QC/malignant trends along isodepth",
        "- correlation bars using the agreed descriptive strength bins",
        "- pointers to GASTON-native topology plots and class-leader gene evidence",
        "- compact top continuous/discontinuous gene-class evidence from the native GASTON layer",
        "",
        "Evidence deliberately not forced here:",
        "",
        "- paper-style Type I/II/III labels",
        "- final sample-level biological interpretations",
        "- cohort-wide gradient taxonomy",
        "- targeted robust-core/signature gene interpretation",
        "",
        "The goal is to decide, by human review, what each sample's GASTON gradient appears to capture.",
        "",
        "Machine-suggested identity counts:",
    ]
    for key, value in sorted(counts.items()):
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "Main files:",
            "",
            "- `gradient_identity_evidence.csv`: one row per sample.",
            "- `gradient_identity_candidate_counts.csv`: count of machine-suggested review labels.",
            "- `cohort_gradient_identity_contact_sheet.png`: compact visual review sheet.",
            "- `<dataset>__<sample>__gradient_identity_panel.png`: per-sample review panel.",
            "- `<dataset>__<sample>__gradient_identity_bins.csv`: 15-bin evidence table.",
            "- `<dataset>__<sample>__gradient_identity_domains.csv`: selected-domain context table.",
            "- `native_*` columns in `gradient_identity_evidence.csv`: paths and compact summaries from `02_gaston_native_gradient_identity`.",
            "",
            "Primary/secondary rule:",
            "",
            "- `primary_glmpca` samples are the primary set.",
            "- `supplementary_pearson` samples remain secondary/Pearson fallback.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    gaston_root = args.gaston_root
    out_dir = args.out_dir or DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(gaston_root / "04_isodepth_score_alignment" / "sample_alignment_manifest.csv")
    if args.samples:
        manifest = manifest[manifest["sample"].astype(str).isin(args.samples)].copy()
    if manifest.empty:
        raise RuntimeError("No matching samples found in sample_alignment_manifest.csv")

    relationship_overview = read_optional_table(gaston_root / "05_relationship_review" / "cohort_sample_overview.csv")
    topology_manifest = read_optional_table(gaston_root / "06_topography_overlay" / "score_topography_overlay_manifest.csv")

    summaries: list[dict[str, Any]] = []
    panel_paths: list[Path] = []
    for _, row in manifest.sort_values(["dataset", "sample"]).iterrows():
        summary = process_sample(row, args, out_dir)
        summaries.append(summary)
        panel_paths.append(Path(summary["gradient_identity_panel_png"]))

    overview = pd.DataFrame(summaries)
    if not relationship_overview.empty:
        keep = [
            "dataset",
            "sample",
            "main_panel_png",
            "binned_trends_png",
            "scatter_png",
            "domain_context_png",
            "has_hires_image",
            "min_bin_n",
            "min_domain_pct",
        ]
        keep = [col for col in keep if col in relationship_overview.columns]
        overview = overview.merge(relationship_overview[keep], on=["dataset", "sample"], how="left")
    if not topology_manifest.empty:
        keep = ["dataset", "sample", "overlay_png", "gradient_status"]
        keep = [col for col in keep if col in topology_manifest.columns]
        overview = overview.merge(topology_manifest[keep], on=["dataset", "sample"], how="left")

    overview_csv = out_dir / "gradient_identity_evidence.csv"
    overview_json = out_dir / "gradient_identity_evidence.json"
    counts_csv = out_dir / "gradient_identity_candidate_counts.csv"
    contact_png = out_dir / "cohort_gradient_identity_contact_sheet.png"

    overview.to_csv(overview_csv, index=False)
    write_json(overview_json, overview.to_dict(orient="records"))
    overview["machine_suggested_identity"].value_counts().rename_axis("machine_suggested_identity").reset_index(
        name="n_samples"
    ).to_csv(counts_csv, index=False)
    make_contact_sheet(panel_paths, contact_png, "GASTON gradient identity review panels")
    write_readme(out_dir, args, overview)

    provenance = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "gaston_root": str(gaston_root),
        "out_dir": str(out_dir),
        "native_gene_dir": str(args.native_gene_dir),
        "num_bins": args.num_bins,
        "tail_fraction": args.tail_fraction,
        "inputs": {
            "sample_alignment_manifest": str(gaston_root / "04_isodepth_score_alignment" / "sample_alignment_manifest.csv"),
            "relationship_overview": str(gaston_root / "05_relationship_review" / "cohort_sample_overview.csv"),
            "topology_manifest": str(gaston_root / "06_topography_overlay" / "score_topography_overlay_manifest.csv"),
            "native_gene_gradient_layer": str(args.native_gene_dir),
        },
        "outputs": {
            "gradient_identity_evidence_csv": str(overview_csv),
            "gradient_identity_candidate_counts_csv": str(counts_csv),
            "cohort_contact_sheet_png": str(contact_png),
        },
        "notes": [
            "Machine-suggested identities are automated review flags, not final interpretations.",
            "GASTON-native gene/pathway gradient calls are linked from the native layer, not regenerated here.",
            "This review layer does not force paper-style Type I/II/III labels.",
        ],
    }
    write_json(out_dir / "gradient_identity_provenance.json", provenance)
    print(f"Wrote {len(overview)} sample summaries to {overview_csv}")
    print(f"Wrote contact sheet to {contact_png}")


if __name__ == "__main__":
    main()
