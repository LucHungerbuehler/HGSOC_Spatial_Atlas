"""
Method-aligned GASTON rerun driver.

This script runs the replacement HGSOC GASTON workflow in resumable stages.
It uses GLM-PCA as the primary expression feature method and falls back to
analytic Pearson-residual PCs when GLM-PCA fails, times out, or produces
unstable features. Every fallback is logged in a manifest.

Examples:
    # Feature generation for all manually included sample/layer records.
    C:\\Users\\luchu\\anaconda3\\envs\\gaston_env\\python.exe scripts\\gaston_method_aligned_run.py features

    # Full GASTON training from selected features.
    C:\\Users\\luchu\\anaconda3\\envs\\gaston_env\\python.exe scripts\\gaston_method_aligned_run.py train

    # Short smoke test into a separate root.
    C:\\Users\\luchu\\anaconda3\\envs\\gaston_env\\python.exe scripts\\gaston_method_aligned_run.py features --samples Pt1-1 --layers whole --out-root D:\\HGSOC_Spatial_Atlas\\05_analysis_ready\\GASTON_method_v1_smoke
    C:\\Users\\luchu\\anaconda3\\envs\\gaston_env\\python.exe scripts\\gaston_method_aligned_run.py train --samples Pt1-1 --layers whole --epochs 50 --restarts 1 --out-root D:\\HGSOC_Spatial_Atlas\\05_analysis_ready\\GASTON_method_v1_smoke
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import sys
import time
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
import torch
from anndata import AnnData
from glmpca import glmpca
from kneed import KneeLocator
from scipy.stats import spearmanr


GASTON_REPO = Path(r"D:\HGSOC_Spatial_Atlas\git_clones\GASTON")
GASTON_SRC = GASTON_REPO / "src"
if str(GASTON_SRC) not in sys.path:
    sys.path.insert(0, str(GASTON_SRC))

from gaston import dp_related, model_selection, neural_net, parse_adata  # noqa: E402


BASE_DIR = Path(r"D:\HGSOC_Spatial_Atlas\05_analysis_ready\visium")
OUT_ROOT = Path(r"D:\HGSOC_Spatial_Atlas\05_analysis_ready\GASTON_method_v1")
PREFLIGHT_REVIEW = OUT_ROOT / "00_preflight" / "sample_layer_manual_review.csv"

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
class FeatureRecord:
    sample: str
    dataset: str
    layer: str
    selected_method: str
    status: str
    fallback_used: bool
    fallback_reason: str
    feature_npz: str
    feature_png: str
    feature_json: str
    n_spots: int
    n_total_features: int
    n_expression_features: int
    use_rgb: bool
    max_abs_feature: float | None
    glmpca_status: str
    glmpca_reason: str
    pearson_status: str
    seconds: float


@dataclass
class TrainRecord:
    sample: str
    dataset: str
    layer: str
    feature_method: str
    analysis_tier: str
    include_in_primary_cross_sample: bool
    status: str
    n_spots: int
    n_features: int
    epochs: int
    restarts: int
    best_seed: int | None
    best_loss: float | None
    train_seconds: float | None
    auto_k: int | None
    isodepth_depth_spearman: float | None
    isodepth_genes_spearman: float | None
    isodepth_malignant_spearman: float | None
    warning_flags: str
    result_npz: str
    model_pt: str
    ll_png: str
    isodepth_png: str
    metrics_json: str
    error: str


def run_id(dataset: str, sample: str, layer: str) -> str:
    return f"{dataset}__{sample}__{layer}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--out-root", type=Path, default=OUT_ROOT)
        p.add_argument("--preflight-review-csv", type=Path, default=PREFLIGHT_REVIEW)
        p.add_argument("--samples", nargs="+", default=None)
        p.add_argument("--layers", nargs="+", choices=["whole", "tumor"], default=["whole"])
        p.add_argument("--force", action="store_true")

    features = sub.add_parser("features", help="Generate GLM-PCA/Pearson feature inputs with fallback logging.")
    add_common(features)
    features.add_argument("--num-dims", type=int, default=5)
    features.add_argument("--glmpca-penalty", type=float, default=50.0)
    features.add_argument("--glmpca-iters", type=int, default=30)
    features.add_argument("--glmpca-eps", type=float, default=1e-4)
    features.add_argument("--glmpca-num-genes", type=int, default=10000)
    features.add_argument("--pearson-num-genes", type=int, default=5000)
    features.add_argument("--pearson-clip", type=float, default=0.01)
    features.add_argument("--feature-timeout-seconds", type=int, default=1800)
    features.add_argument("--max-abs-feature", type=float, default=100.0)
    features.add_argument("--max-dev-ratio", type=float, default=100.0)
    features.add_argument("--spatial-orientation", choices=["image", "cartesian"], default="image")

    train = sub.add_parser("train", help="Train GASTON models from selected feature manifest rows.")
    add_common(train)
    train.add_argument("--feature-manifest", type=Path, default=None)
    train.add_argument("--epochs", type=int, default=10000)
    train.add_argument("--restarts", type=int, default=30)
    train.add_argument("--checkpoint", type=int, default=500)
    train.add_argument("--train-timeout-seconds", type=int, default=14400)
    train.add_argument("--max-domain-num", type=int, default=8)
    train.add_argument("--num-buckets", type=int, default=100)
    train.add_argument("--device", default="cpu")
    train.add_argument("--torch-threads", type=int, default=0)
    train.add_argument("--warn-isodepth-depth-r", type=float, default=0.50)
    train.add_argument("--spatial-orientation", choices=["image", "cartesian"], default="image")

    worker = sub.add_parser("_feature_worker")
    worker.add_argument("--sample", required=True)
    worker.add_argument("--dataset", required=True)
    worker.add_argument("--layer", required=True)
    worker.add_argument("--path", type=Path, required=True)
    worker.add_argument("--method", choices=["glmpca", "pearson"], required=True)
    worker.add_argument("--out-npz", type=Path, required=True)
    worker.add_argument("--out-json", type=Path, required=True)
    worker.add_argument("--out-png", type=Path, required=True)
    worker.add_argument("--num-dims", type=int, required=True)
    worker.add_argument("--glmpca-penalty", type=float, default=50.0)
    worker.add_argument("--glmpca-iters", type=int, default=30)
    worker.add_argument("--glmpca-eps", type=float, default=1e-4)
    worker.add_argument("--glmpca-num-genes", type=int, default=10000)
    worker.add_argument("--pearson-num-genes", type=int, default=5000)
    worker.add_argument("--pearson-clip", type=float, default=0.01)
    worker.add_argument("--max-abs-feature", type=float, default=100.0)
    worker.add_argument("--max-dev-ratio", type=float, default=100.0)
    worker.add_argument("--spatial-orientation", choices=["image", "cartesian"], default="image")

    train_worker = sub.add_parser("_train_worker")
    train_worker.add_argument("--sample", required=True)
    train_worker.add_argument("--dataset", required=True)
    train_worker.add_argument("--layer", required=True)
    train_worker.add_argument("--path", type=Path, required=True)
    train_worker.add_argument("--feature-npz", type=Path, required=True)
    train_worker.add_argument("--feature-method", required=True)
    train_worker.add_argument("--out-result-npz", type=Path, required=True)
    train_worker.add_argument("--out-model-pt", type=Path, required=True)
    train_worker.add_argument("--out-loss-csv", type=Path, required=True)
    train_worker.add_argument("--out-metrics-json", type=Path, required=True)
    train_worker.add_argument("--out-ll-png", type=Path, required=True)
    train_worker.add_argument("--out-isodepth-png", type=Path, required=True)
    train_worker.add_argument("--epochs", type=int, required=True)
    train_worker.add_argument("--restarts", type=int, required=True)
    train_worker.add_argument("--checkpoint", type=int, default=500)
    train_worker.add_argument("--max-domain-num", type=int, default=8)
    train_worker.add_argument("--num-buckets", type=int, default=100)
    train_worker.add_argument("--device", default="cpu")
    train_worker.add_argument("--torch-threads", type=int, default=0)
    train_worker.add_argument("--warn-isodepth-depth-r", type=float, default=0.50)
    train_worker.add_argument("--spatial-orientation", choices=["image", "cartesian"], default="image")

    return parser.parse_args()


def selected_records(args: argparse.Namespace) -> list[dict]:
    review_csv = args.preflight_review_csv
    if not review_csv.exists():
        raise FileNotFoundError(review_csv)
    df = pd.read_csv(review_csv)
    if "human_include" in df.columns:
        include = df["human_include"].astype(str).str.upper().isin(["TRUE", "YES", "1"])
        df = df[include].copy()
    if args.samples:
        df = df[df["sample"].isin(args.samples)].copy()
    if args.layers:
        df = df[df["layer"].isin(args.layers)].copy()
    records = []
    for row in df.to_dict("records"):
        sample = row["sample"]
        if sample not in SAMPLE_CATALOG:
            raise KeyError(f"Unknown sample in review CSV: {sample}")
        dataset, path = SAMPLE_CATALOG[sample]
        records.append({"sample": sample, "dataset": dataset, "layer": row["layer"], "path": path})
    return records


def format_spatial_axis(ax: plt.Axes, spatial_orientation: str) -> None:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if spatial_orientation == "image":
        ax.invert_yaxis()


def matrix_to_dense(x) -> np.ndarray:
    if sp.issparse(x):
        return x.toarray()
    if hasattr(x, "to_memory"):
        x = x.to_memory()
    return np.asarray(x)


def subset_counts(adata: AnnData, mask: np.ndarray):
    counts = adata.layers["counts"] if "counts" in adata.layers else adata.X
    if sp.issparse(counts):
        return counts[mask, :]
    return np.asarray(counts)[mask, :]


def get_interface_column(adata: AnnData) -> str | None:
    for col in ("interface_label", "interface"):
        if col in adata.obs:
            return col
    return None


def canonical_interface_labels(values: Iterable) -> np.ndarray:
    labels = []
    for value in values:
        key = str(value).strip().lower()
        if key == "tumor":
            labels.append("Tumor")
        elif key == "interface":
            labels.append("Interface")
        elif key == "stroma":
            labels.append("Stroma")
        else:
            labels.append("Unknown")
    return np.asarray(labels, dtype=object)


def get_layer_mask(adata: AnnData, layer: str) -> np.ndarray:
    if layer == "whole":
        return np.ones(adata.n_obs, dtype=bool)
    interface_col = get_interface_column(adata)
    if interface_col is not None:
        return canonical_interface_labels(adata.obs[interface_col].to_numpy()) == "Tumor"
    if "is_tumor" in adata.obs:
        return adata.obs["is_tumor"].astype(bool).to_numpy()
    raise KeyError("Tumor-only layer requested, but no interface/interface_label/is_tumor column exists.")


def has_hires_image(adata: AnnData) -> bool:
    if "spatial" not in adata.uns:
        return False
    return any("hires" in lib.get("images", {}) for lib in adata.uns["spatial"].values())


def compute_rgb(adata: AnnData, mask: np.ndarray) -> tuple[np.ndarray | None, str]:
    if not has_hires_image(adata):
        return None, "no hires image"
    rgb = parse_adata.use_RGB(adata.copy())
    return rgb[mask, :], "hires RGB included"


def compute_glmpca(counts, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    gene_sums = np.asarray(counts.sum(axis=0)).ravel()
    n_keep = min(args.glmpca_num_genes, counts.shape[1])
    top_idx = np.argsort(gene_sums)[-n_keep:]
    counts_top = matrix_to_dense(counts[:, top_idx]).astype(np.float64, copy=False)
    res = glmpca.glmpca(
        counts_top.T,
        args.num_dims,
        fam="poi",
        penalty=args.glmpca_penalty,
        verbose=True,
        ctl={"maxIter": args.glmpca_iters, "eps": args.glmpca_eps, "optimizeTheta": True},
    )
    dev = [float(x) for x in np.asarray(res.get("dev", []), dtype=float)]
    finite_dev = bool(np.all(np.isfinite(dev))) if dev else True
    dev_ratio = float(max(dev) / max(min(dev), np.finfo(float).tiny)) if dev and min(dev) > 0 else None
    return np.asarray(res["factors"]), {
        "n_glmpca_genes": int(n_keep),
        "glmpca_deviance": dev,
        "glmpca_deviance_finite": finite_dev,
        "glmpca_deviance_ratio": dev_ratio,
    }


def compute_pearson(counts, gene_labels: np.ndarray, coords: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    ad = AnnData(X=counts.copy() if sp.issparse(counts) else np.asarray(counts).copy())
    ad.var_names = gene_labels.astype(str)
    ad.obsm["coords"] = coords
    sc.experimental.pp.highly_variable_genes(
        ad,
        flavor="pearson_residuals",
        n_top_genes=min(args.pearson_num_genes, ad.n_vars),
    )
    ad = ad[:, ad.var["highly_variable"]].copy()
    sc.experimental.pp.normalize_pearson_residuals(ad, clip=args.pearson_clip, theta=np.inf)
    sc.pp.pca(ad, n_comps=args.num_dims)
    variance_ratio = ad.uns.get("pca", {}).get("variance_ratio", np.array([]))
    return np.asarray(ad.obsm["X_pca"]), {
        "n_pearson_hvg": int(ad.n_vars),
        "pearson_variance_ratio": [float(x) for x in np.asarray(variance_ratio)[: args.num_dims]],
    }


def feature_labels(method: str, n_expr: int, use_rgb: bool) -> list[str]:
    prefix = "GLM-PC" if method == "glmpca" else "Pearson-PC"
    labels = [f"{prefix}{i + 1}" for i in range(n_expr)]
    if use_rgb:
        labels += ["RGB_R", "RGB_G", "RGB_B"]
    return labels


def summarize_features(A: np.ndarray, meta: dict, args: argparse.Namespace) -> tuple[dict, str, str]:
    finite = bool(np.all(np.isfinite(A)))
    max_abs = float(np.nanmax(np.abs(A))) if A.size else float("nan")
    stats = {
        "feature_finite": finite,
        "feature_min": float(np.nanmin(A)) if A.size else None,
        "feature_max": float(np.nanmax(A)) if A.size else None,
        "feature_max_abs": max_abs,
        "feature_std": [float(x) for x in np.nanstd(A, axis=0)],
    }
    warnings = []
    if not finite:
        warnings.append("non-finite feature values")
    if max_abs > args.max_abs_feature:
        warnings.append(f"extreme feature magnitude > {args.max_abs_feature:g}")
    dev = meta.get("glmpca_deviance")
    if dev:
        if not meta.get("glmpca_deviance_finite", True):
            warnings.append("non-finite GLM-PCA deviance")
        ratio = meta.get("glmpca_deviance_ratio")
        if ratio is not None and ratio > args.max_dev_ratio:
            warnings.append(f"GLM-PCA deviance ratio {ratio:.2e}")
    status = "unstable" if warnings else "ok"
    return stats, status, "; ".join(warnings)


def plot_spatial_features(
    coords: np.ndarray,
    values: np.ndarray,
    labels: list[str],
    out_png: Path,
    title: str,
    spatial_orientation: str,
) -> None:
    n = values.shape[1]
    cols = min(4, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.axis("off")
            continue
        sca = ax.scatter(coords[:, 0], coords[:, 1], c=values[:, i], s=4, cmap="viridis", linewidths=0)
        ax.set_title(labels[i], fontsize=10)
        format_spatial_axis(ax, spatial_orientation)
        ax.axis("off")
        plt.colorbar(sca, ax=ax, fraction=0.046)
    fig.suptitle(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def worker_feature(args: argparse.Namespace) -> int:
    start = time.perf_counter()
    try:
        adata = sc.read_h5ad(args.path)
        mask = get_layer_mask(adata, args.layer)
        if int(mask.sum()) < 10:
            raise ValueError(f"too few spots after mask: {int(mask.sum())}")
        coords = np.asarray(adata.obsm["spatial"])[mask, :]
        counts = subset_counts(adata, mask)
        gene_labels = adata.var_names.to_numpy().astype(str)
        rgb, rgb_note = compute_rgb(adata, mask)
        use_rgb = rgb is not None

        if args.method == "glmpca":
            expr, method_meta = compute_glmpca(counts, args)
        elif args.method == "pearson":
            expr, method_meta = compute_pearson(counts, gene_labels, coords, args)
        else:
            raise ValueError(args.method)

        A = np.hstack([expr, rgb]) if use_rgb else expr
        feature_stats, feature_status, feature_warning = summarize_features(A, method_meta, args)
        labels = feature_labels(args.method, expr.shape[1], use_rgb)

        args.out_npz.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_png.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.out_npz,
            A=A,
            expr=expr,
            coords=coords,
            gene_labels=gene_labels,
            mask_indices=np.flatnonzero(mask),
            use_rgb=np.asarray([bool(use_rgb)]),
        )
        plot_spatial_features(
            coords,
            A,
            labels,
            args.out_png,
            f"{args.dataset} / {args.sample} / {args.layer} / {args.method}",
            args.spatial_orientation,
        )
        metadata = {
            "sample": args.sample,
            "dataset": args.dataset,
            "layer": args.layer,
            "path": str(args.path),
            "method": args.method,
            "n_spots": int(mask.sum()),
            "n_genes": int(adata.n_vars),
            "n_expression_features": int(expr.shape[1]),
            "n_total_features": int(A.shape[1]),
            "use_rgb": bool(use_rgb),
            "rgb_note": rgb_note,
            "feature_stats": feature_stats,
            "feature_status": feature_status,
            "feature_warning": feature_warning,
            "method_metadata": method_meta,
            "seconds": time.perf_counter() - start,
        }
        args.out_json.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        del adata
        gc.collect()
        return 0
    except Exception as exc:
        error_meta = {
            "sample": args.sample,
            "dataset": args.dataset,
            "layer": args.layer,
            "method": args.method,
            "feature_status": "error",
            "feature_warning": repr(exc),
            "seconds": time.perf_counter() - start,
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(error_meta, indent=2, default=str), encoding="utf-8")
        print(f"ERROR in feature worker: {repr(exc)}", file=sys.stderr, flush=True)
        return 2


def run_subprocess(cmd: list[str], timeout: int, stdout_path: Path, stderr_path: Path) -> tuple[str, str]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        stdout_path.write_text(result.stdout or "", encoding="utf-8")
        stderr_path.write_text(result.stderr or "", encoding="utf-8")
        if result.returncode == 0:
            return "ok", ""
        return "error", f"returncode={result.returncode}"
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(exc.stdout or "", encoding="utf-8")
        stderr_path.write_text((exc.stderr or "") + f"\nTIMEOUT after {timeout} seconds\n", encoding="utf-8")
        return "timeout", f"timeout after {timeout} seconds"


def feature_worker_cmd(args: argparse.Namespace, rec: dict, method: str, out_npz: Path, out_json: Path, out_png: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "_feature_worker",
        "--sample",
        rec["sample"],
        "--dataset",
        rec["dataset"],
        "--layer",
        rec["layer"],
        "--path",
        str(rec["path"]),
        "--method",
        method,
        "--out-npz",
        str(out_npz),
        "--out-json",
        str(out_json),
        "--out-png",
        str(out_png),
        "--num-dims",
        str(args.num_dims),
        "--glmpca-penalty",
        str(args.glmpca_penalty),
        "--glmpca-iters",
        str(args.glmpca_iters),
        "--glmpca-eps",
        str(args.glmpca_eps),
        "--glmpca-num-genes",
        str(args.glmpca_num_genes),
        "--pearson-num-genes",
        str(args.pearson_num_genes),
        "--pearson-clip",
        str(args.pearson_clip),
        "--max-abs-feature",
        str(args.max_abs_feature),
        "--max-dev-ratio",
        str(args.max_dev_ratio),
        "--spatial-orientation",
        args.spatial_orientation,
    ]


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_features(args: argparse.Namespace) -> None:
    feature_dir = args.out_root / "01_features"
    arrays_dir = feature_dir / "arrays"
    figures_dir = feature_dir / "figures"
    logs_dir = feature_dir / "logs"
    feature_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for rec in selected_records(args):
        rid = run_id(rec["dataset"], rec["sample"], rec["layer"])
        selected_json = arrays_dir / f"{rid}__selected.json"
        selected_npz = arrays_dir / f"{rid}__selected_features.npz"
        if selected_npz.exists() and selected_json.exists() and not args.force:
            meta = read_json(selected_json)
            records.append(FeatureRecord(**meta["feature_record"]))
            print(f"Skipping existing selected features: {rid}", flush=True)
            continue

        print(f"Feature generation: {rid}", flush=True)
        start = time.perf_counter()

        glmpca_npz = arrays_dir / f"{rid}__glmpca.npz"
        glmpca_json = arrays_dir / f"{rid}__glmpca.json"
        glmpca_png = figures_dir / f"{rid}__glmpca__features.png"
        glmpca_status, glmpca_process_reason = run_subprocess(
            feature_worker_cmd(args, rec, "glmpca", glmpca_npz, glmpca_json, glmpca_png),
            args.feature_timeout_seconds,
            logs_dir / f"{rid}__glmpca.stdout.txt",
            logs_dir / f"{rid}__glmpca.stderr.txt",
        )
        glmpca_meta = read_json(glmpca_json)
        glmpca_feature_status = glmpca_meta.get("feature_status", glmpca_status)
        glmpca_reason = glmpca_meta.get("feature_warning") or glmpca_process_reason

        selected_method = "glmpca"
        selected_npz_source = glmpca_npz
        selected_png = glmpca_png
        selected_json_source = glmpca_json
        fallback_used = False
        fallback_reason = ""
        pearson_status = "not_run"

        if glmpca_status != "ok" or glmpca_feature_status != "ok":
            fallback_used = True
            fallback_reason = f"GLM-PCA {glmpca_feature_status}: {glmpca_reason}".strip()
            print(f"  Falling back to Pearson: {fallback_reason}", flush=True)
            pearson_npz = arrays_dir / f"{rid}__pearson.npz"
            pearson_json = arrays_dir / f"{rid}__pearson.json"
            pearson_png = figures_dir / f"{rid}__pearson__features.png"
            pearson_status, pearson_process_reason = run_subprocess(
                feature_worker_cmd(args, rec, "pearson", pearson_npz, pearson_json, pearson_png),
                args.feature_timeout_seconds,
                logs_dir / f"{rid}__pearson.stdout.txt",
                logs_dir / f"{rid}__pearson.stderr.txt",
            )
            pearson_meta = read_json(pearson_json)
            pearson_feature_status = pearson_meta.get("feature_status", pearson_status)
            if pearson_status == "ok" and pearson_feature_status == "ok":
                selected_method = "pearson"
                selected_npz_source = pearson_npz
                selected_png = pearson_png
                selected_json_source = pearson_json
            else:
                selected_method = "none"
                selected_npz_source = Path("")
                selected_png = Path("")
                selected_json_source = pearson_json
                fallback_reason = f"{fallback_reason}; Pearson {pearson_feature_status}: {pearson_meta.get('feature_warning') or pearson_process_reason}"

        selected_meta = read_json(selected_json_source) if selected_json_source else {}
        if selected_method != "none":
            selected_npz.write_bytes(selected_npz_source.read_bytes())
            selected_payload = {
                "selected_method": selected_method,
                "source_npz": str(selected_npz_source),
                "source_json": str(selected_json_source),
                "source_png": str(selected_png),
                "fallback_used": fallback_used,
                "fallback_reason": fallback_reason,
                "selected_metadata": selected_meta,
            }
            selected_json.write_text(json.dumps(selected_payload, indent=2, default=str), encoding="utf-8")
            status = "ok"
        else:
            selected_payload = {"selected_method": "none", "fallback_reason": fallback_reason, "selected_metadata": selected_meta}
            selected_json.write_text(json.dumps(selected_payload, indent=2, default=str), encoding="utf-8")
            status = "failed"

        feature_record = FeatureRecord(
            sample=rec["sample"],
            dataset=rec["dataset"],
            layer=rec["layer"],
            selected_method=selected_method,
            status=status,
            fallback_used=bool(fallback_used),
            fallback_reason=fallback_reason,
            feature_npz=str(selected_npz) if selected_method != "none" else "",
            feature_png=str(selected_png) if selected_method != "none" else "",
            feature_json=str(selected_json),
            n_spots=int(selected_meta.get("n_spots", 0) or 0),
            n_total_features=int(selected_meta.get("n_total_features", 0) or 0),
            n_expression_features=int(selected_meta.get("n_expression_features", 0) or 0),
            use_rgb=bool(selected_meta.get("use_rgb", False)),
            max_abs_feature=selected_meta.get("feature_stats", {}).get("feature_max_abs"),
            glmpca_status=str(glmpca_feature_status),
            glmpca_reason=str(glmpca_reason),
            pearson_status=str(pearson_status),
            seconds=float(time.perf_counter() - start),
        )
        selected_payload["feature_record"] = asdict(feature_record)
        selected_json.write_text(json.dumps(selected_payload, indent=2, default=str), encoding="utf-8")
        records.append(feature_record)

    df = pd.DataFrame([asdict(r) for r in records])
    df.to_csv(feature_dir / "feature_manifest.csv", index=False)
    (feature_dir / "feature_manifest.json").write_text(json.dumps([asdict(r) for r in records], indent=2), encoding="utf-8")
    write_features_readme(feature_dir, args, records)
    print(f"Wrote feature manifest: {feature_dir / 'feature_manifest.csv'}", flush=True)


def write_features_readme(feature_dir: Path, args: argparse.Namespace, records: list[FeatureRecord]) -> None:
    df = pd.DataFrame([asdict(r) for r in records])
    status_counts = df["status"].value_counts().to_dict() if not df.empty else {}
    method_counts = df["selected_method"].value_counts().to_dict() if not df.empty else {}
    fallback_count = int(df["fallback_used"].sum()) if not df.empty else 0
    lines = [
        "# GASTON method-aligned features",
        "",
        "Flat stage layout:",
        "",
        "- `arrays/{dataset}__{sample}__{layer}__selected_features.npz`",
        "- `arrays/{dataset}__{sample}__{layer}__selected.json`",
        "- `figures/{dataset}__{sample}__{layer}__{method}__features.png`",
        "- `logs/{dataset}__{sample}__{layer}__{method}.stdout.txt`",
        "- `feature_manifest.csv`",
        "",
        "Policy: GLM-PCA primary, Pearson residual PCs fallback when GLM-PCA errors, times out, or is unstable.",
        "",
        f"Status counts: `{status_counts}`",
        f"Selected method counts: `{method_counts}`",
        f"Fallback count: `{fallback_count}`",
        "",
        "Parameters:",
        "",
        "```json",
        json.dumps(vars(args), indent=2, default=str),
        "```",
        "",
    ]
    feature_dir.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")


def get_count_qc_vectors(adata: AnnData) -> tuple[np.ndarray, np.ndarray]:
    counts = adata.layers["counts"] if "counts" in adata.layers else adata.X
    if "total_counts" in adata.obs:
        total_counts = adata.obs["total_counts"].to_numpy(dtype=float)
    else:
        total_counts = np.asarray(counts.sum(axis=1)).ravel() if sp.issparse(counts) else np.asarray(counts).sum(axis=1)
    if "n_genes_by_counts" in adata.obs:
        n_genes = adata.obs["n_genes_by_counts"].to_numpy(dtype=float)
    else:
        n_genes = np.asarray((counts > 0).sum(axis=1)).ravel() if sp.issparse(counts) else (np.asarray(counts) > 0).sum(axis=1)
    return total_counts.astype(float), n_genes.astype(float)


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


def train_canary_like(
    A: np.ndarray,
    coords: np.ndarray,
    epochs: int,
    restarts: int,
    checkpoint: int,
    device: str,
) -> tuple[torch.nn.Module, np.ndarray, np.ndarray, int, float, float, list[dict]]:
    start = time.perf_counter()
    s_torch, a_torch = neural_net.load_rescale_input_data(coords, A)
    best_loss = np.inf
    best_model = None
    best_seed = -1
    loss_records = []
    for seed in range(restarts):
        model, loss_list = neural_net.train(
            s_torch,
            a_torch,
            S_hidden_list=[20, 20],
            A_hidden_list=[20, 20],
            epochs=epochs,
            checkpoint=checkpoint,
            device=device,
            save_dir=None,
            optim="adam",
            seed=seed,
            save_final=False,
        )
        final_loss = float(loss_list[-1])
        min_loss = float(np.min(loss_list))
        loss_records.append({"seed": seed, "final_loss": final_loss, "min_loss": min_loss})
        if final_loss < best_loss:
            best_loss = final_loss
            best_seed = seed
            best_model = model.cpu()
    if best_model is None:
        raise RuntimeError("No GASTON model trained.")
    seconds = time.perf_counter() - start
    return best_model, a_torch.detach().cpu().numpy(), s_torch.detach().cpu().numpy(), best_seed, best_loss, seconds, loss_records


def plot_ll_curve(ll_values: list[float], auto_k: int | None, out_png: Path) -> None:
    x = np.arange(2, len(ll_values) + 1)
    y = np.asarray(ll_values[1:], dtype=float)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, y)
    ax.scatter(x, y)
    if auto_k is not None:
        ax.axvline(auto_k, ls="--", color="grey")
        ax.text(auto_k, float(np.nanmax(y)), f"auto_k={auto_k}", va="top", ha="left")
    else:
        ax.text(0.05, 0.95, "auto_k=None", transform=ax.transAxes, va="top")
    ax.set_xlabel("Number of domains")
    ax.set_ylabel("Negative log-likelihood")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_isodepth_domains(
    coords: np.ndarray,
    isodepth: np.ndarray,
    labels: np.ndarray | None,
    auto_k: int | None,
    out_png: Path,
    title: str,
    spatial_orientation: str,
) -> None:
    cols = 2 if labels is not None else 1
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 4), squeeze=False)
    ax = axes.ravel()[0]
    sca = ax.scatter(coords[:, 0], coords[:, 1], c=isodepth, s=5, cmap="viridis", linewidths=0)
    ax.set_title("isodepth raw")
    format_spatial_axis(ax, spatial_orientation)
    ax.axis("off")
    plt.colorbar(sca, ax=ax, fraction=0.046)
    if labels is not None:
        ax = axes.ravel()[1]
        sca = ax.scatter(coords[:, 0], coords[:, 1], c=labels, s=5, cmap="tab10", linewidths=0)
        ax.set_title(f"domains auto_k={auto_k}")
        format_spatial_axis(ax, spatial_orientation)
        ax.axis("off")
        plt.colorbar(sca, ax=ax, fraction=0.046)
    fig.suptitle(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def worker_train(args: argparse.Namespace) -> int:
    start = time.perf_counter()
    if args.torch_threads and args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    try:
        payload = np.load(args.feature_npz, allow_pickle=True)
        A = np.asarray(payload["A"], dtype=float)
        coords = np.asarray(payload["coords"], dtype=float)
        mask_indices = np.asarray(payload["mask_indices"], dtype=int)

        model, A_scaled, S_scaled, best_seed, best_loss, train_seconds, loss_records = train_canary_like(
            A,
            coords,
            args.epochs,
            args.restarts,
            args.checkpoint,
            args.device,
        )
        args.out_loss_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(loss_records).to_csv(args.out_loss_csv, index=False)
        args.out_model_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model, args.out_model_pt)

        ll_list = model_selection.get_ll_list(model, A_scaled, S_scaled, num_buckets=args.num_buckets, kmax=args.max_domain_num)
        x = np.arange(2, len(ll_list) + 1)
        y = np.asarray(ll_list[1:], dtype=float)
        kneedle = KneeLocator(x, y, curve="convex", direction="decreasing")
        auto_k = int(kneedle.knee) if kneedle.knee is not None else None
        ll_values = [float(v) for v in np.asarray(ll_list, dtype=float)]
        plot_ll_curve(ll_values, auto_k, args.out_ll_png)

        isodepth_raw = model.spatial_embedding(torch.Tensor(S_scaled)).detach().numpy().flatten()
        labels = None
        if auto_k is not None:
            isodepth_raw, labels = dp_related.get_isodepth_labels(model, A_scaled, S_scaled, int(auto_k), num_buckets=args.num_buckets)
        labels_for_save = np.full(isodepth_raw.shape[0], -1, dtype=int) if labels is None else np.asarray(labels, dtype=int)
        plot_isodepth_domains(
            coords,
            isodepth_raw,
            labels if labels is not None else None,
            auto_k,
            args.out_isodepth_png,
            f"{args.dataset} / {args.sample} / {args.layer} / {args.feature_method}",
            args.spatial_orientation,
        )

        adata = sc.read_h5ad(args.path)
        total_counts, n_genes = get_count_qc_vectors(adata)
        total_counts_layer = total_counts[mask_indices]
        n_genes_layer = n_genes[mask_indices]
        depth_rho = safe_spearman(isodepth_raw, total_counts_layer)
        genes_rho = safe_spearman(isodepth_raw, n_genes_layer)
        malignant_rho = None
        if "Malignant" in adata.obs:
            malignant_rho = safe_spearman(isodepth_raw, adata.obs["Malignant"].to_numpy(dtype=float)[mask_indices])
        warning_flags = []
        if depth_rho is not None and abs(depth_rho) >= args.warn_isodepth_depth_r:
            warning_flags.append(f"isodepth associated with UMI depth (rho={depth_rho:.2f})")
        if genes_rho is not None and abs(genes_rho) >= args.warn_isodepth_depth_r:
            warning_flags.append(f"isodepth associated with detected genes (rho={genes_rho:.2f})")

        orientation_sign_malignant = 1
        if args.layer == "whole" and malignant_rho is not None and malignant_rho < 0:
            orientation_sign_malignant = -1
        isodepth_malignant_oriented = isodepth_raw * orientation_sign_malignant

        args.out_result_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.out_result_npz,
            isodepth_raw=isodepth_raw,
            isodepth_malignant_oriented=isodepth_malignant_oriented,
            labels_auto=labels_for_save,
            A_scaled=A_scaled,
            S_scaled=S_scaled,
            auto_k=np.asarray([-1 if auto_k is None else auto_k]),
            mask_indices=mask_indices,
        )
        metrics = {
            "sample": args.sample,
            "dataset": args.dataset,
            "layer": args.layer,
            "feature_method": args.feature_method,
            "n_spots": int(A.shape[0]),
            "n_features": int(A.shape[1]),
            "epochs": int(args.epochs),
            "restarts": int(args.restarts),
            "best_seed": int(best_seed),
            "best_loss": float(best_loss),
            "train_seconds": float(train_seconds),
            "auto_k": auto_k,
            "ll_values": ll_values,
            "isodepth_depth_spearman": depth_rho,
            "isodepth_genes_spearman": genes_rho,
            "isodepth_malignant_spearman": malignant_rho,
            "orientation_sign_malignant": orientation_sign_malignant,
            "warning_flags": "; ".join(warning_flags),
            "total_seconds": time.perf_counter() - start,
        }
        args.out_metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_metrics_json.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
        del adata
        gc.collect()
        return 0
    except Exception as exc:
        args.out_metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_metrics_json.write_text(
            json.dumps(
                {
                    "sample": args.sample,
                    "dataset": args.dataset,
                    "layer": args.layer,
                    "feature_method": args.feature_method,
                    "status": "error",
                    "error": repr(exc),
                    "total_seconds": time.perf_counter() - start,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"ERROR in train worker: {repr(exc)}", file=sys.stderr, flush=True)
        return 2


def train_worker_cmd(args: argparse.Namespace, row: dict, out_paths: dict[str, Path]) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "_train_worker",
        "--sample",
        row["sample"],
        "--dataset",
        row["dataset"],
        "--layer",
        row["layer"],
        "--path",
        str(SAMPLE_CATALOG[row["sample"]][1]),
        "--feature-npz",
        row["feature_npz"],
        "--feature-method",
        row["selected_method"],
        "--out-result-npz",
        str(out_paths["result_npz"]),
        "--out-model-pt",
        str(out_paths["model_pt"]),
        "--out-loss-csv",
        str(out_paths["loss_csv"]),
        "--out-metrics-json",
        str(out_paths["metrics_json"]),
        "--out-ll-png",
        str(out_paths["ll_png"]),
        "--out-isodepth-png",
        str(out_paths["isodepth_png"]),
        "--epochs",
        str(args.epochs),
        "--restarts",
        str(args.restarts),
        "--checkpoint",
        str(args.checkpoint),
        "--max-domain-num",
        str(args.max_domain_num),
        "--num-buckets",
        str(args.num_buckets),
        "--device",
        args.device,
        "--torch-threads",
        str(args.torch_threads),
        "--warn-isodepth-depth-r",
        str(args.warn_isodepth_depth_r),
        "--spatial-orientation",
        args.spatial_orientation,
    ]


def run_train(args: argparse.Namespace) -> None:
    gaston_dir = args.out_root / "02_gaston"
    results_dir = gaston_dir / "results"
    models_dir = gaston_dir / "models"
    figures_dir = gaston_dir / "figures"
    logs_dir = gaston_dir / "logs"
    gaston_dir.mkdir(parents=True, exist_ok=True)

    feature_manifest = args.feature_manifest or (args.out_root / "01_features" / "feature_manifest.csv")
    if not feature_manifest.exists():
        raise FileNotFoundError(feature_manifest)
    df = pd.read_csv(feature_manifest)
    df = df[df["status"] == "ok"].copy()
    if args.samples:
        df = df[df["sample"].isin(args.samples)].copy()
    if args.layers:
        df = df[df["layer"].isin(args.layers)].copy()

    records = []
    for row in df.to_dict("records"):
        rid = run_id(row["dataset"], row["sample"], row["layer"])
        method = row["selected_method"]
        stem = f"{rid}__{method}"
        out_paths = {
            "result_npz": results_dir / f"{stem}__gaston.npz",
            "model_pt": models_dir / f"{stem}__best_model.pt",
            "loss_csv": results_dir / f"{stem}__losses.csv",
            "metrics_json": results_dir / f"{stem}__metrics.json",
            "ll_png": figures_dir / f"{stem}__ll_elbow.png",
            "isodepth_png": figures_dir / f"{stem}__isodepth_domains.png",
        }
        if out_paths["result_npz"].exists() and out_paths["metrics_json"].exists() and not args.force:
            metrics = read_json(out_paths["metrics_json"])
            records.append(train_record_from_metrics(row, metrics, out_paths, args, "ok", ""))
            print(f"Skipping existing GASTON result: {stem}", flush=True)
            continue
        print(f"Training GASTON: {stem}", flush=True)
        status, reason = run_subprocess(
            train_worker_cmd(args, row, out_paths),
            args.train_timeout_seconds,
            logs_dir / f"{stem}.stdout.txt",
            logs_dir / f"{stem}.stderr.txt",
        )
        metrics = read_json(out_paths["metrics_json"])
        if status != "ok":
            records.append(train_record_from_metrics(row, metrics, out_paths, args, status, reason))
        else:
            records.append(train_record_from_metrics(row, metrics, out_paths, args, "ok", ""))

    out_df = pd.DataFrame([asdict(r) for r in records])
    out_df.to_csv(gaston_dir / "gaston_manifest.csv", index=False)
    (gaston_dir / "gaston_manifest.json").write_text(json.dumps([asdict(r) for r in records], indent=2), encoding="utf-8")
    write_train_readme(gaston_dir, args, records)
    print(f"Wrote GASTON manifest: {gaston_dir / 'gaston_manifest.csv'}", flush=True)


def train_record_from_metrics(
    row: dict,
    metrics: dict,
    out_paths: dict[str, Path],
    args: argparse.Namespace,
    status: str,
    error: str,
) -> TrainRecord:
    return TrainRecord(
        sample=row["sample"],
        dataset=row["dataset"],
        layer=row["layer"],
        feature_method=row["selected_method"],
        analysis_tier=str(row.get("analysis_tier") or infer_analysis_tier(row)),
        include_in_primary_cross_sample=parse_bool(
            row.get("include_in_primary_cross_sample", row.get("selected_method") == "glmpca")
        ),
        status=status if metrics.get("status") != "error" else "error",
        n_spots=int(metrics.get("n_spots", row.get("n_spots", 0)) or 0),
        n_features=int(metrics.get("n_features", row.get("n_total_features", 0)) or 0),
        epochs=int(metrics.get("epochs", args.epochs) or args.epochs),
        restarts=int(metrics.get("restarts", args.restarts) or args.restarts),
        best_seed=metrics.get("best_seed"),
        best_loss=metrics.get("best_loss"),
        train_seconds=metrics.get("train_seconds"),
        auto_k=metrics.get("auto_k"),
        isodepth_depth_spearman=metrics.get("isodepth_depth_spearman"),
        isodepth_genes_spearman=metrics.get("isodepth_genes_spearman"),
        isodepth_malignant_spearman=metrics.get("isodepth_malignant_spearman"),
        warning_flags=metrics.get("warning_flags", ""),
        result_npz=str(out_paths["result_npz"]) if out_paths["result_npz"].exists() else "",
        model_pt=str(out_paths["model_pt"]) if out_paths["model_pt"].exists() else "",
        ll_png=str(out_paths["ll_png"]) if out_paths["ll_png"].exists() else "",
        isodepth_png=str(out_paths["isodepth_png"]) if out_paths["isodepth_png"].exists() else "",
        metrics_json=str(out_paths["metrics_json"]) if out_paths["metrics_json"].exists() else "",
        error=metrics.get("error", error),
    )


def infer_analysis_tier(row: dict) -> str:
    return "primary_glmpca" if row.get("selected_method") == "glmpca" else "supplementary_pearson"


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def write_train_readme(gaston_dir: Path, args: argparse.Namespace, records: list[TrainRecord]) -> None:
    df = pd.DataFrame([asdict(r) for r in records])
    status_counts = df["status"].value_counts().to_dict() if not df.empty else {}
    warning_count = int((df.get("warning_flags", pd.Series(dtype=str)).astype(str) != "").sum()) if not df.empty else 0
    lines = [
        "# GASTON method-aligned training",
        "",
        "Flat stage layout:",
        "",
        "- `results/{dataset}__{sample}__{layer}__{method}__gaston.npz`",
        "- `results/{dataset}__{sample}__{layer}__{method}__metrics.json`",
        "- `models/{dataset}__{sample}__{layer}__{method}__best_model.pt`",
        "- `figures/{dataset}__{sample}__{layer}__{method}__ll_elbow.png`",
        "- `figures/{dataset}__{sample}__{layer}__{method}__isodepth_domains.png`",
        "- `gaston_manifest.csv`",
        "",
        "Only the best model is saved for each sample-layer-method to avoid the old branch's large checkpoint explosion.",
        "",
        f"Status counts: `{status_counts}`",
        f"Records with depth/gene warnings: `{warning_count}`",
        "",
        "Parameters:",
        "",
        "```json",
        json.dumps(vars(args), indent=2, default=str),
        "```",
        "",
    ]
    gaston_dir.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.command == "_feature_worker":
        raise SystemExit(worker_feature(args))
    if args.command == "_train_worker":
        raise SystemExit(worker_train(args))
    if args.command == "features":
        run_features(args)
    elif args.command == "train":
        run_train(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
