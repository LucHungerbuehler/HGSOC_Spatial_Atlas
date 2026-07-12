"""
Method-aligned GASTON all-sample preflight.

This preflight scans the 23-sample cohort used by the original HGSOC GASTON
branch and writes eligibility/QC tables plus visual review panels before the
full rerun. It does not train GASTON models.

Run from the gaston conda environment, for example:
    C:\\Users\\luchu\\anaconda3\\envs\\gaston_env\\python.exe scripts\\gaston_method_aligned_preflight.py
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
from matplotlib.lines import Line2D
from scipy.stats import spearmanr


BASE_DIR = Path(r"D:\HGSOC_Spatial_Atlas\05_analysis_ready\visium")
OUT_ROOT = Path(r"D:\HGSOC_Spatial_Atlas\05_analysis_ready\GASTON_method_v1")

INTERFACE_ORDER = ["Tumor", "Interface", "Stroma"]
INTERFACE_PALETTE = {
    "Tumor": "#FDE725",
    "Interface": "#21918C",
    "Stroma": "#440154",
    "Unknown": "#BDBDBD",
}

SAMPLE_CATALOG = {
    "SP1": ("denisenko_2022", BASE_DIR / "denisenko_2022" / "SP1" / "SP1.h5ad"),
    "SP2": ("denisenko_2022", BASE_DIR / "denisenko_2022" / "SP2" / "SP2.h5ad"),
    "SP3": ("denisenko_2022", BASE_DIR / "denisenko_2022" / "SP3" / "SP3.h5ad"),
    "SP4": ("denisenko_2022", BASE_DIR / "denisenko_2022" / "SP4" / "SP4.h5ad"),
    "SP5": ("denisenko_2022", BASE_DIR / "denisenko_2022" / "SP5" / "SP5.h5ad"),
    "SP6": ("denisenko_2022", BASE_DIR / "denisenko_2022" / "SP6" / "SP6.h5ad"),
    "SP7": ("denisenko_2022", BASE_DIR / "denisenko_2022" / "SP7" / "SP7.h5ad"),
    "SP8": ("denisenko_2022", BASE_DIR / "denisenko_2022" / "SP8" / "SP8.h5ad"),
    "Pt1-1": ("yamamoto_2025", BASE_DIR / "yamamoto_2025" / "Pt1-1" / "Pt1-1.h5ad"),
    "Pt1-2": ("yamamoto_2025", BASE_DIR / "yamamoto_2025" / "Pt1-2" / "Pt1-2.h5ad"),
    "Pt1-3": ("yamamoto_2025", BASE_DIR / "yamamoto_2025" / "Pt1-3" / "Pt1-3.h5ad"),
    "Pt1-4": ("yamamoto_2025", BASE_DIR / "yamamoto_2025" / "Pt1-4" / "Pt1-4.h5ad"),
    "Pt2-1": ("yamamoto_2025", BASE_DIR / "yamamoto_2025" / "Pt2-1" / "Pt2-1.h5ad"),
    "Pt2-2": ("yamamoto_2025", BASE_DIR / "yamamoto_2025" / "Pt2-2" / "Pt2-2.h5ad"),
    "Pt2-3": ("yamamoto_2025", BASE_DIR / "yamamoto_2025" / "Pt2-3" / "Pt2-3.h5ad"),
    "Pt2-4": ("yamamoto_2025", BASE_DIR / "yamamoto_2025" / "Pt2-4" / "Pt2-4.h5ad"),
    "CPS_OV19_LtOV1": ("ju_2024", BASE_DIR / "ju_2024" / "CPS_OV19_LtOV1" / "CPS_OV19_LtOV1.h5ad"),
    "CPS_OV1RtOV3": ("ju_2024", BASE_DIR / "ju_2024" / "CPS_OV1RtOV3" / "CPS_OV1RtOV3.h5ad"),
    "CPS_OV20RtOV4": ("ju_2024", BASE_DIR / "ju_2024" / "CPS_OV20RtOV4" / "CPS_OV20RtOV4.h5ad"),
    "CPS_OV24RTOV4": ("ju_2024", BASE_DIR / "ju_2024" / "CPS_OV24RTOV4" / "CPS_OV24RTOV4.h5ad"),
    "CPS_OV34RtOV1": ("ju_2024", BASE_DIR / "ju_2024" / "CPS_OV34RtOV1" / "CPS_OV34RtOV1.h5ad"),
    "CPS_OV5LtOV4": ("ju_2024", BASE_DIR / "ju_2024" / "CPS_OV5LtOV4" / "CPS_OV5LtOV4.h5ad"),
    "CPS_OV71_1": ("ju_2024", BASE_DIR / "ju_2024" / "CPS_OV71_1" / "CPS_OV71_1.h5ad"),
}


@dataclass
class PreflightRecord:
    sample: str
    dataset: str
    layer: str
    path: str
    n_spots: int
    n_genes: int
    tumor_spots_total: int
    interface_spots_total: int
    stroma_spots_total: int
    tumor_fraction_total: float
    has_counts_layer: bool
    has_spatial: bool
    has_hires_image: bool
    has_sna1ac_score: bool
    has_malignant: bool
    total_counts_median: float
    total_counts_p10: float
    total_counts_p90: float
    total_counts_cv: float
    n_genes_median: float
    n_genes_p10: float
    n_genes_p90: float
    pct_mt_median: float | None
    pct_mt_p90: float | None
    depth_spatial_max_abs_spearman: float | None
    genes_spatial_max_abs_spearman: float | None
    snai1ac_depth_spearman: float | None
    malignant_depth_spearman: float | None
    auto_eligibility: str
    hard_flags: str
    warning_flags: str
    context_flags: str
    review_panel: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--samples", nargs="+", default=list(SAMPLE_CATALOG))
    parser.add_argument("--layers", nargs="+", choices=["whole", "tumor"], default=["whole"])
    parser.add_argument("--min-whole-spots", type=int, default=500)
    parser.add_argument("--min-tumor-spots", type=int, default=150)
    parser.add_argument("--warn-depth-spatial-r", type=float, default=0.60)
    parser.add_argument("--warn-score-depth-r", type=float, default=0.50)
    parser.add_argument("--warn-pct-mt-p90", type=float, default=30.0)
    parser.add_argument("--topographic-map-spot-threshold", type=int, default=1001)
    parser.add_argument("--spatial-orientation", choices=["image", "cartesian"], default="image")
    return parser.parse_args()


def format_spatial_axis(ax: plt.Axes, spatial_orientation: str) -> None:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if spatial_orientation == "image":
        ax.invert_yaxis()


def has_hires_image(adata: AnnData) -> bool:
    if "spatial" not in adata.uns:
        return False
    return any("hires" in lib.get("images", {}) for lib in adata.uns["spatial"].values())


def get_interface_column(adata: AnnData) -> str | None:
    for col in ("interface_label", "interface"):
        if col in adata.obs:
            return col
    return None


def canonical_interface_labels(values: Iterable) -> np.ndarray:
    out = []
    for value in values:
        key = str(value).strip().lower()
        if key == "tumor":
            out.append("Tumor")
        elif key == "interface":
            out.append("Interface")
        elif key == "stroma":
            out.append("Stroma")
        else:
            out.append("Unknown")
    return np.asarray(out, dtype=object)


def get_layer_mask(adata: AnnData, layer: str) -> np.ndarray:
    if layer == "whole":
        return np.ones(adata.n_obs, dtype=bool)
    interface_col = get_interface_column(adata)
    if interface_col is not None:
        return canonical_interface_labels(adata.obs[interface_col].to_numpy()) == "Tumor"
    if "is_tumor" in adata.obs:
        return adata.obs["is_tumor"].astype(bool).to_numpy()
    return np.zeros(adata.n_obs, dtype=bool)


def get_count_matrix(adata: AnnData):
    if "counts" in adata.layers:
        return adata.layers["counts"], True
    return adata.X, False


def vector_from_matrix_sum(matrix, axis: int) -> np.ndarray:
    if sp.issparse(matrix):
        return np.asarray(matrix.sum(axis=axis)).ravel()
    return np.asarray(matrix).sum(axis=axis)


def vector_from_matrix_nnz_rows(matrix) -> np.ndarray:
    if sp.issparse(matrix):
        return np.asarray((matrix > 0).sum(axis=1)).ravel()
    return (np.asarray(matrix) > 0).sum(axis=1)


def qc_vectors(adata: AnnData) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, bool]:
    matrix, has_counts = get_count_matrix(adata)
    if "total_counts" in adata.obs:
        total_counts = adata.obs["total_counts"].to_numpy(dtype=float)
    else:
        total_counts = vector_from_matrix_sum(matrix, axis=1).astype(float)
    if "n_genes_by_counts" in adata.obs:
        n_genes = adata.obs["n_genes_by_counts"].to_numpy(dtype=float)
    else:
        n_genes = vector_from_matrix_nnz_rows(matrix).astype(float)
    pct_mt = None
    mt_cols = [c for c in adata.obs.columns if c.lower() in {"pct_counts_mt", "percent_mt", "pct_mt"}]
    if mt_cols:
        pct_mt = adata.obs[mt_cols[0]].to_numpy(dtype=float)
    else:
        mt_mask = np.asarray([str(g).upper().startswith("MT-") for g in adata.var_names])
        if mt_mask.any():
            mt_counts = vector_from_matrix_sum(matrix[:, mt_mask], axis=1).astype(float)
            pct_mt = np.divide(mt_counts, total_counts, out=np.zeros_like(mt_counts), where=total_counts > 0) * 100
    return total_counts, n_genes, pct_mt, has_counts


def safe_quantile(values: np.ndarray, q: float) -> float:
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return float("nan")
    return float(np.quantile(valid, q))


def safe_cv(values: np.ndarray) -> float:
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return float("nan")
    mean = float(np.mean(valid))
    if mean == 0:
        return float("nan")
    return float(np.std(valid) / mean)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return None
    if np.nanstd(x[mask]) == 0 or np.nanstd(y[mask]) == 0:
        return None
    rho = spearmanr(x[mask], y[mask]).statistic
    if not np.isfinite(rho):
        return None
    return float(rho)


def plot_continuous(
    ax: plt.Axes,
    coords: np.ndarray,
    values: np.ndarray,
    title: str,
    spatial_orientation: str,
    cmap: str = "viridis",
) -> None:
    sca = ax.scatter(coords[:, 0], coords[:, 1], c=np.asarray(values, dtype=float), s=4, cmap=cmap, linewidths=0)
    ax.set_title(title, fontsize=10)
    format_spatial_axis(ax, spatial_orientation)
    ax.axis("off")
    plt.colorbar(sca, ax=ax, fraction=0.046, pad=0.02)


def plot_interface(
    ax: plt.Axes,
    coords: np.ndarray,
    labels: np.ndarray,
    spatial_orientation: str,
) -> None:
    labels = canonical_interface_labels(labels)
    colors = [INTERFACE_PALETTE.get(label, INTERFACE_PALETTE["Unknown"]) for label in labels]
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=4, linewidths=0)
    ax.set_title("interface", fontsize=10)
    format_spatial_axis(ax, spatial_orientation)
    ax.axis("off")
    legend_labels = INTERFACE_ORDER + (["Unknown"] if np.any(labels == "Unknown") else [])
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=INTERFACE_PALETTE[label],
            markeredgecolor="none",
            markersize=6,
            label=label,
        )
        for label in legend_labels
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)


def write_review_panel(
    adata: AnnData,
    mask: np.ndarray,
    coords: np.ndarray,
    total_counts: np.ndarray,
    n_genes: np.ndarray,
    pct_mt: np.ndarray | None,
    out_png: Path,
    title: str,
    spatial_orientation: str,
) -> None:
    panels: list[tuple[str, str, np.ndarray]] = []
    interface_col = get_interface_column(adata)
    if interface_col is not None:
        panels.append(("interface", "interface", adata.obs[interface_col].to_numpy()[mask]))
    if "Malignant" in adata.obs:
        panels.append(("continuous", "Malignant", adata.obs["Malignant"].to_numpy(dtype=float)[mask]))
    if "SNAI1-ac_score" in adata.obs:
        panels.append(("continuous", "SNAI1-ac_score", adata.obs["SNAI1-ac_score"].to_numpy(dtype=float)[mask]))
    panels.append(("continuous", "log1p total_counts", np.log1p(total_counts[mask])))
    panels.append(("continuous", "n_genes_by_counts", n_genes[mask]))
    if pct_mt is not None:
        panels.append(("continuous", "pct_counts_mt", pct_mt[mask]))

    cols = 3
    rows = int(math.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (kind, label, values) in zip(axes.ravel(), panels):
        if kind == "interface":
            plot_interface(ax, coords, values, spatial_orientation)
        else:
            plot_continuous(ax, coords, values, label, spatial_orientation)
    fig.suptitle(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def inspect_one(args: argparse.Namespace, sample: str, layer: str, preflight_dir: Path) -> PreflightRecord:
    dataset, path = SAMPLE_CATALOG[sample]
    if not path.exists():
        return PreflightRecord(
            sample=sample,
            dataset=dataset,
            layer=layer,
            path=str(path),
            n_spots=0,
            n_genes=0,
            tumor_spots_total=0,
            interface_spots_total=0,
            stroma_spots_total=0,
            tumor_fraction_total=0.0,
            has_counts_layer=False,
            has_spatial=False,
            has_hires_image=False,
            has_sna1ac_score=False,
            has_malignant=False,
            total_counts_median=float("nan"),
            total_counts_p10=float("nan"),
            total_counts_p90=float("nan"),
            total_counts_cv=float("nan"),
            n_genes_median=float("nan"),
            n_genes_p10=float("nan"),
            n_genes_p90=float("nan"),
            pct_mt_median=None,
            pct_mt_p90=None,
            depth_spatial_max_abs_spearman=None,
            genes_spatial_max_abs_spearman=None,
            snai1ac_depth_spearman=None,
            malignant_depth_spearman=None,
            auto_eligibility="fail",
            hard_flags="missing h5ad",
            warning_flags="",
            context_flags="",
            review_panel="",
        )

    adata = sc.read_h5ad(path)
    hard_flags = []
    warning_flags = []
    context_flags = []

    has_spatial = "spatial" in adata.obsm
    interface_col = get_interface_column(adata)
    if interface_col is not None:
        interface_all = canonical_interface_labels(adata.obs[interface_col].to_numpy())
    else:
        interface_all = np.asarray(["Unknown"] * adata.n_obs, dtype=object)
        warning_flags.append("missing interface/interface_label")

    tumor_spots_total = int(np.sum(interface_all == "Tumor"))
    interface_spots_total = int(np.sum(interface_all == "Interface"))
    stroma_spots_total = int(np.sum(interface_all == "Stroma"))
    tumor_fraction_total = float(tumor_spots_total / adata.n_obs) if adata.n_obs else 0.0

    total_counts, n_genes, pct_mt, has_counts_layer = qc_vectors(adata)
    mask = get_layer_mask(adata, layer)
    n_spots = int(mask.sum())

    if not has_spatial:
        hard_flags.append("missing spatial coordinates")
        coords = np.zeros((n_spots, 2), dtype=float)
    else:
        coords = np.asarray(adata.obsm["spatial"])[mask, :]
    if not has_counts_layer:
        warning_flags.append("missing counts layer; using adata.X for QC only")
    min_spots = args.min_whole_spots if layer == "whole" else args.min_tumor_spots
    if n_spots < min_spots:
        hard_flags.append(f"too few {layer} spots ({n_spots} < {min_spots})")
    if layer == "tumor" and n_spots < args.topographic_map_spot_threshold:
        warning_flags.append(
            f"tumor spots below old topographic-map threshold ({n_spots} < {args.topographic_map_spot_threshold})"
        )

    layer_total_counts = total_counts[mask]
    layer_n_genes = n_genes[mask]
    if np.nanmedian(layer_total_counts) <= 0:
        hard_flags.append("non-positive median total_counts")

    depth_x = depth_y = genes_x = genes_y = None
    if has_spatial and n_spots > 0:
        depth_x = safe_spearman(layer_total_counts, coords[:, 0])
        depth_y = safe_spearman(layer_total_counts, coords[:, 1])
        genes_x = safe_spearman(layer_n_genes, coords[:, 0])
        genes_y = safe_spearman(layer_n_genes, coords[:, 1])
    depth_spatial = max([abs(v) for v in [depth_x, depth_y] if v is not None], default=None)
    genes_spatial = max([abs(v) for v in [genes_x, genes_y] if v is not None], default=None)
    if depth_spatial is not None and depth_spatial >= args.warn_depth_spatial_r:
        warning_flags.append(f"strong spatial UMI-depth trend (max |rho|={depth_spatial:.2f})")
    if genes_spatial is not None and genes_spatial >= args.warn_depth_spatial_r:
        warning_flags.append(f"strong spatial detected-gene trend (max |rho|={genes_spatial:.2f})")

    snai1ac_depth = None
    if "SNAI1-ac_score" in adata.obs:
        snai1ac_depth = safe_spearman(adata.obs["SNAI1-ac_score"].to_numpy(dtype=float)[mask], layer_total_counts)
        if snai1ac_depth is not None and abs(snai1ac_depth) >= args.warn_score_depth_r:
            warning_flags.append(f"SNAI1-ac associated with UMI depth (rho={snai1ac_depth:.2f})")

    malignant_depth = None
    if "Malignant" in adata.obs:
        malignant_depth = safe_spearman(adata.obs["Malignant"].to_numpy(dtype=float)[mask], layer_total_counts)
        if malignant_depth is not None and abs(malignant_depth) >= args.warn_score_depth_r:
            context_flags.append(f"Malignant associated with UMI depth (rho={malignant_depth:.2f})")

    pct_mt_median = pct_mt_p90 = None
    if pct_mt is not None:
        layer_pct_mt = pct_mt[mask]
        pct_mt_median = safe_quantile(layer_pct_mt, 0.50)
        pct_mt_p90 = safe_quantile(layer_pct_mt, 0.90)
        if pct_mt_p90 >= args.warn_pct_mt_p90:
            warning_flags.append(f"high pct_counts_mt p90 ({pct_mt_p90:.1f})")

    auto_eligibility = "fail" if hard_flags else ("review" if warning_flags else "pass")
    panel_path = preflight_dir / "review_panels" / f"{dataset}__{sample}__{layer}__qc.png"
    if n_spots > 0 and has_spatial:
        write_review_panel(
            adata,
            mask,
            coords,
            total_counts,
            n_genes,
            pct_mt,
            panel_path,
            f"{dataset} / {sample} / {layer}",
            args.spatial_orientation,
        )

    record = PreflightRecord(
        sample=sample,
        dataset=dataset,
        layer=layer,
        path=str(path),
        n_spots=n_spots,
        n_genes=int(adata.n_vars),
        tumor_spots_total=tumor_spots_total,
        interface_spots_total=interface_spots_total,
        stroma_spots_total=stroma_spots_total,
        tumor_fraction_total=tumor_fraction_total,
        has_counts_layer=bool(has_counts_layer),
        has_spatial=bool(has_spatial),
        has_hires_image=bool(has_hires_image(adata)),
        has_sna1ac_score="SNAI1-ac_score" in adata.obs,
        has_malignant="Malignant" in adata.obs,
        total_counts_median=safe_quantile(layer_total_counts, 0.50),
        total_counts_p10=safe_quantile(layer_total_counts, 0.10),
        total_counts_p90=safe_quantile(layer_total_counts, 0.90),
        total_counts_cv=safe_cv(layer_total_counts),
        n_genes_median=safe_quantile(layer_n_genes, 0.50),
        n_genes_p10=safe_quantile(layer_n_genes, 0.10),
        n_genes_p90=safe_quantile(layer_n_genes, 0.90),
        pct_mt_median=pct_mt_median,
        pct_mt_p90=pct_mt_p90,
        depth_spatial_max_abs_spearman=depth_spatial,
        genes_spatial_max_abs_spearman=genes_spatial,
        snai1ac_depth_spearman=snai1ac_depth,
        malignant_depth_spearman=malignant_depth,
        auto_eligibility=auto_eligibility,
        hard_flags="; ".join(hard_flags),
        warning_flags="; ".join(warning_flags),
        context_flags="; ".join(context_flags),
        review_panel=str(panel_path) if panel_path.exists() else "",
    )
    del adata
    gc.collect()
    return record


def write_manual_review_template(records: list[PreflightRecord], out_csv: Path) -> None:
    rows = []
    for rec in records:
        rows.append(
            {
                "sample": rec.sample,
                "dataset": rec.dataset,
                "layer": rec.layer,
                "auto_eligibility": rec.auto_eligibility,
                "warning_flags": rec.warning_flags,
                "context_flags": rec.context_flags,
                "human_include": "",
                "human_exclude_reason": "",
                "orientation_ok": "",
                "interface_palette_ok": "",
                "depth_qc_ok": "",
                "tumor_mask_ok": "",
                "preferred_feature_policy": "glmpca_primary_pearson_fallback",
                "reviewer_notes": "",
            }
        )
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def write_readme(preflight_dir: Path, args: argparse.Namespace, records: list[PreflightRecord]) -> None:
    status_counts = pd.Series([rec.auto_eligibility for rec in records]).value_counts().to_dict()
    lines = [
        "# GASTON method-aligned preflight",
        "",
        "Purpose: screen the original 23-sample HGSOC GASTON cohort before the full method-aligned rerun.",
        "",
        "Root structure is intentionally flat:",
        "",
        "- `00_preflight/sample_layer_preflight.csv`",
        "- `00_preflight/sample_preflight_summary.csv`",
        "- `00_preflight/sample_layer_manual_review.csv`",
        "- `00_preflight/review_panels/{dataset}__{sample}__{layer}__qc.png`",
        "",
        "Feature policy for the future rerun: GLM-PCA primary, analytic Pearson residual PCs as a logged fallback for numerical instability or visually unusable features.",
        "",
        "UMI/depth QC is a guardrail. Depth-driven gradients should be flagged for review, not automatically interpreted as biology.",
        "",
        "`warning_flags` affect auto eligibility. `context_flags` are retained for interpretation/QC context but do not by themselves block a sample.",
        "",
        "Interface palette: Tumor = yellow, Interface = teal, Stroma = purple, Unknown = grey.",
        "",
        f"Auto eligibility counts: `{status_counts}`",
        "",
        "Parameters:",
        "",
        "```json",
        json.dumps(vars(args), indent=2, default=str),
        "```",
        "",
    ]
    (preflight_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    preflight_dir = args.out_root / "00_preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)

    records: list[PreflightRecord] = []
    for sample in args.samples:
        if sample not in SAMPLE_CATALOG:
            raise KeyError(f"Unknown sample: {sample}")
        for layer in args.layers:
            print(f"Inspecting {sample} / {layer}", flush=True)
            records.append(inspect_one(args, sample, layer, preflight_dir))

    df = pd.DataFrame([asdict(rec) for rec in records])
    df.to_csv(preflight_dir / "sample_layer_preflight.csv", index=False)

    summary = (
        df.pivot_table(
            index=["dataset", "sample"],
            columns="layer",
            values=["auto_eligibility", "n_spots", "hard_flags", "warning_flags", "context_flags"],
            aggfunc="first",
        )
        .sort_index()
    )
    summary.columns = ["_".join([str(x) for x in col if x]) for col in summary.columns.to_flat_index()]
    summary.reset_index().to_csv(preflight_dir / "sample_preflight_summary.csv", index=False)

    write_manual_review_template(records, preflight_dir / "sample_layer_manual_review.csv")
    (preflight_dir / "preflight_manifest.json").write_text(
        json.dumps({"records": [asdict(rec) for rec in records], "args": vars(args)}, indent=2, default=str),
        encoding="utf-8",
    )
    write_readme(preflight_dir, args, records)
    print(f"Wrote preflight outputs: {preflight_dir}")


if __name__ == "__main__":
    main()
