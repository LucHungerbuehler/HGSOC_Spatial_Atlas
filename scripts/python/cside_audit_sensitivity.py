"""
Audit and sensitivity analyses for the SNAI1-ac C-SIDE branch.

This script reads existing per-sample spacexr C-SIDE outputs and writes
report-facing audit products. By default it targets the Model 2 outputs
(`SNAI1-ac + malignant fraction`), but the result filename and output stem can
be changed for Model 1 or other sensitivity runs.

  * signed C-SIDE ranking audit
  * meta-gene summaries with sample/dataset heterogeneity
  * paper-style mean-Z gene-set permutation tests for Hallmark and KEGG
  * SNAI1-ac signature-gene circularity checks
  * spot-level QC covariate availability derived from the exact RCTD inputs

It deliberately does not overwrite the original RCTD/C-SIDE output branch.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon


DATA_ROOT = Path("D:/HGSOC_Spatial_Atlas")
REPO_ROOT = Path("C:/Users/luchu/Documents/MSc/Master Thesis/Code/HGSOC_Spatial_Atlas")
CSIDE_ROOT = DATA_ROOT / "scRNA_reference" / "rctd_outputs"
QC_CSIDE_ROOT = DATA_ROOT / "scRNA_reference" / "rctd_outputs_qc_sensitivity"
RCTD_INPUT_ROOT = DATA_ROOT / "scRNA_reference" / "rctd_inputs"
QC_INPUT_ROOT = DATA_ROOT / "scRNA_reference" / "rctd_inputs_qc_sensitivity"
OUTPUT_ROOT = DATA_ROOT / "05_analysis_ready" / "S2e_CSIDE_CellTypeSpecific_DE_Audit"
HALLMARK_JSON = DATA_ROOT / "05_analysis_ready" / "Signature" / "hallmark_gene_sets.json"
SIGNATURE_FULL = DATA_ROOT / "05_analysis_ready" / "Signature" / "snai1_acetylation_signature_full.csv"

DEFAULT_DATASETS = ("denisenko_2022", "ju_2024", "yamamoto_2025")
DEFAULT_CELL_TYPES = ("CAF", "Endothelial", "Epithelial", "Fibroblast", "Macrophage")


@dataclass(frozen=True)
class OutputDirs:
    root: Path
    manifest: Path
    signed: Path
    gene_sets: Path
    circularity: Path
    heterogeneity: Path
    qc: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit C-SIDE outputs and run sensitivity analyses.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--cside-root", type=Path, default=CSIDE_ROOT)
    parser.add_argument("--qc-cside-root", type=Path, default=QC_CSIDE_ROOT)
    parser.add_argument("--rctd-input-root", type=Path, default=RCTD_INPUT_ROOT)
    parser.add_argument("--qc-input-root", type=Path, default=QC_INPUT_ROOT)
    parser.add_argument("--hallmark-json", type=Path, default=HALLMARK_JSON)
    parser.add_argument("--signature-full", type=Path, default=SIGNATURE_FULL)
    parser.add_argument("--result-filename", default="cside_2cov_all_results.csv")
    parser.add_argument("--signed-output-name", default="cside_2cov_signed_gene_level_all_samples.csv")
    parser.add_argument("--branch-label", default="S2e_CSIDE_CellTypeSpecific_DE_Audit")
    parser.add_argument(
        "--purpose-label",
        default="C-SIDE signed ranking, gene-set, circularity, heterogeneity, and QC sensitivity audit.",
    )
    parser.add_argument("--n-perm", type=int, default=10000)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--min-set-genes", type=int, default=5)
    parser.add_argument("--abs-logfc-threshold", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--steps",
        default="all",
        help=(
            "Comma-separated subset: manifest,signed,gene_sets,circularity,"
            "heterogeneity,qc,qc_compare. Default: all."
        ),
    )
    return parser.parse_args()


def ensure_dirs(root: Path) -> OutputDirs:
    dirs = OutputDirs(
        root=root,
        manifest=root / "00_manifest",
        signed=root / "01_signed_ranking_audit",
        gene_sets=root / "02_paper_style_meanZ_gene_set_tests",
        circularity=root / "03_signature_circularity",
        heterogeneity=root / "04_sample_dataset_heterogeneity",
        qc=root / "05_qc_sensitivity",
    )
    for path in dirs.__dict__.values():
        path.mkdir(parents=True, exist_ok=True)
    for collection in ("hallmark", "kegg"):
        (dirs.gene_sets / collection).mkdir(parents=True, exist_ok=True)
        (dirs.heterogeneity / collection).mkdir(parents=True, exist_ok=True)
    return dirs


def bh_fdr(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    finite = np.isfinite(p)
    if finite.sum() == 0:
        return pd.Series(out, index=p_values.index)
    p_finite = p[finite]
    order = np.argsort(p_finite)
    ranked = p_finite[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    tmp = np.empty(n, dtype=float)
    tmp[order] = adjusted
    out[np.where(finite)[0]] = tmp
    return pd.Series(out, index=p_values.index)


def safe_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "t", "1", "yes", "y"}


def resolve_rscript() -> str:
    candidates = [
        Path(r"C:\Program Files\R\R-4.4.3\bin\x64\Rscript.exe"),
        Path(r"C:\Program Files\R\R-4.4.3\bin\Rscript.exe"),
        Path(r"C:\Program Files\R\R-4.3.3\bin\x64\Rscript.exe"),
        Path(r"C:\Program Files\R\R-4.3.3\bin\Rscript.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "Rscript"


def sample_label(dataset: str, sample_id: str) -> str:
    return f"{dataset}__{sample_id}"


def discover_cside_results(cside_root: Path, result_filename: str = "cside_2cov_all_results.csv") -> pd.DataFrame:
    rows = []
    for dataset_dir in sorted(p for p in cside_root.iterdir() if p.is_dir() and p.name in DEFAULT_DATASETS):
        for sample_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            result_path = sample_dir / result_filename
            if result_path.exists():
                rows.append(
                    {
                        "dataset": dataset_dir.name,
                        "sample_id_on_disk": sample_dir.name,
                        "sample_label": sample_label(dataset_dir.name, sample_dir.name),
                        "cside_results_path": str(result_path),
                        "cside_2cov_results_path": str(result_path),
                    }
                )
    return pd.DataFrame(rows).sort_values(["dataset", "sample_id_on_disk"]).reset_index(drop=True)


def discover_qc_cside_results(qc_cside_root: Path) -> pd.DataFrame:
    rows = []
    for dataset_dir in sorted(p for p in qc_cside_root.iterdir() if p.is_dir() and p.name in DEFAULT_DATASETS):
        for sample_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            result_path = sample_dir / "cside_qc_all_results.csv"
            if result_path.exists():
                rows.append(
                    {
                        "dataset": dataset_dir.name,
                        "sample_id_on_disk": sample_dir.name,
                        "sample_label": sample_label(dataset_dir.name, sample_dir.name),
                        "cside_results_path": str(result_path),
                        "cside_2cov_results_path": str(result_path),
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["dataset", "sample_id_on_disk"]).reset_index(drop=True)


def read_cside_result(path: Path, dataset: str, sample_id: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Z_score", "log_fc", "se", "conv", "p_val", "cell_type", "gene"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    out = df.copy()
    out["dataset"] = dataset
    out["sample_id_on_disk"] = sample_id
    out["sample_label"] = sample_label(dataset, sample_id)
    out["gene"] = out["gene"].astype(str).str.upper()
    out["cell_type"] = out["cell_type"].astype(str)
    for column in ["Z_score", "log_fc", "se", "p_val", "mean_0", "mean_1"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out["conv_bool"] = out["conv"].map(safe_bool)
    out["signed_z"] = np.sign(out["log_fc"]) * out["Z_score"]
    out["p_from_unsigned_z"] = 2.0 * norm.sf(np.abs(out["Z_score"]))
    out["p_abs_delta"] = (out["p_from_unsigned_z"] - out["p_val"]).abs()
    return out


def load_all_cside(manifest: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for row in manifest.itertuples(index=False):
        result_path = getattr(row, "cside_results_path", None)
        if result_path is None:
            result_path = getattr(row, "cside_2cov_results_path")
        frames.append(
            read_cside_result(
                Path(result_path),
                row.dataset,
                row.sample_id_on_disk,
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_manifest(dirs: OutputDirs, args: argparse.Namespace, manifest: pd.DataFrame) -> None:
    manifest.to_csv(dirs.manifest / "cside_sample_manifest.csv", index=False)
    payload = {
        "branch": args.branch_label,
        "purpose": args.purpose_label,
        "script": str(Path(__file__).resolve()),
        "input_cside_root": str(args.cside_root),
        "result_filename": args.result_filename,
        "input_rctd_root": str(args.rctd_input_root),
        "qc_input_root": str(args.qc_input_root),
        "qc_cside_root": str(args.qc_cside_root),
        "output_root": str(args.output_root),
        "hallmark_json": str(args.hallmark_json),
        "signature_full": str(args.signature_full),
        "n_samples": int(len(manifest)),
        "datasets": sorted(manifest["dataset"].unique().tolist()) if not manifest.empty else [],
        "parameters": {
            "n_perm": args.n_perm,
            "min_samples": args.min_samples,
            "min_set_genes": args.min_set_genes,
            "abs_logfc_threshold": args.abs_logfc_threshold,
            "random_seed": args.random_seed,
            "signed_output_name": args.signed_output_name,
        },
    }
    (dirs.manifest / "run_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def signed_ranking_audit(
    all_cside: pd.DataFrame,
    dirs: OutputDirs,
    min_samples: int,
    signed_output_name: str = "cside_2cov_signed_gene_level_all_samples.csv",
) -> pd.DataFrame:
    conv = all_cside.loc[all_cside["conv_bool"]].copy()
    conv.to_csv(dirs.signed / signed_output_name, index=False)

    audit = (
        all_cside.groupby(["dataset", "sample_id_on_disk", "cell_type"], dropna=False)
        .agg(
            n_rows=("gene", "size"),
            n_converged=("conv_bool", "sum"),
            n_genes=("gene", "nunique"),
            n_signed_positive=("signed_z", lambda x: int((x > 0).sum())),
            n_signed_negative=("signed_z", lambda x: int((x < 0).sum())),
            median_p_abs_delta=("p_abs_delta", "median"),
            max_p_abs_delta=("p_abs_delta", "max"),
        )
        .reset_index()
    )
    audit.to_csv(dirs.signed / "signed_z_reconstruction_audit_by_sample_celltype.csv", index=False)

    meta = build_gene_meta(conv, min_samples=min_samples)
    meta.to_csv(dirs.signed / "meta_gene_effects_signed_stouffer_iv_random_effects.csv", index=False)
    for cell_type, subset in meta.groupby("cell_type", sort=True):
        subset.sort_values("stouffer_z", ascending=False).to_csv(
            dirs.signed / f"meta_gene_effects_{clean_name(cell_type)}.csv", index=False
        )

    compare_existing_meta_rankings(meta, dirs)
    return meta


def build_gene_meta(conv: pd.DataFrame, min_samples: int = 3) -> pd.DataFrame:
    rows = []
    for (cell_type, gene), subset in conv.groupby(["cell_type", "gene"], sort=False):
        subset = subset.dropna(subset=["signed_z", "log_fc", "se"])
        subset = subset.loc[subset["se"] > 0]
        n = len(subset)
        if n < min_samples:
            continue
        z = subset["signed_z"].to_numpy(dtype=float)
        y = subset["log_fc"].to_numpy(dtype=float)
        se = subset["se"].to_numpy(dtype=float)
        w = 1.0 / np.square(se)
        fixed = float(np.sum(w * y) / np.sum(w))
        fixed_se = float(math.sqrt(1.0 / np.sum(w)))
        fixed_z = fixed / fixed_se if fixed_se > 0 else np.nan
        fixed_p = float(2.0 * norm.sf(abs(fixed_z))) if np.isfinite(fixed_z) else np.nan
        q = float(np.sum(w * np.square(y - fixed)))
        df = max(n - 1, 1)
        c = float(np.sum(w) - (np.sum(np.square(w)) / np.sum(w)))
        tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
        w_re = 1.0 / (np.square(se) + tau2)
        re = float(np.sum(w_re * y) / np.sum(w_re))
        re_se = float(math.sqrt(1.0 / np.sum(w_re)))
        re_z = re / re_se if re_se > 0 else np.nan
        re_p = float(2.0 * norm.sf(abs(re_z))) if np.isfinite(re_z) else np.nan
        i2 = max(0.0, (q - df) / q) if q > 0 else 0.0
        stouffer_z = float(np.sum(z) / math.sqrt(n))
        stouffer_p = float(2.0 * norm.sf(abs(stouffer_z)))
        n_pos = int((z > 0).sum())
        n_neg = int((z < 0).sum())
        rows.append(
            {
                "cell_type": cell_type,
                "gene": gene,
                "n_samples": n,
                "n_datasets": int(subset["dataset"].nunique()),
                "mean_signed_z": float(np.mean(z)),
                "median_signed_z": float(np.median(z)),
                "stouffer_z": stouffer_z,
                "stouffer_p": stouffer_p,
                "mean_logfc": float(np.mean(y)),
                "median_logfc": float(np.median(y)),
                "fixed_logfc": fixed,
                "fixed_se": fixed_se,
                "fixed_z": fixed_z,
                "fixed_p": fixed_p,
                "random_logfc": re,
                "random_se": re_se,
                "random_z": re_z,
                "random_p": re_p,
                "q_stat": q,
                "i2": i2,
                "tau2": tau2,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "sign_consistency": max(n_pos, n_neg) / n,
                "datasets": ";".join(sorted(subset["dataset"].unique())),
            }
        )
    meta = pd.DataFrame(rows)
    if meta.empty:
        return meta
    meta["stouffer_q"] = meta.groupby("cell_type", group_keys=False)["stouffer_p"].apply(bh_fdr)
    meta["fixed_q"] = meta.groupby("cell_type", group_keys=False)["fixed_p"].apply(bh_fdr)
    meta["random_q"] = meta.groupby("cell_type", group_keys=False)["random_p"].apply(bh_fdr)
    return meta.sort_values(["cell_type", "stouffer_p", "gene"]).reset_index(drop=True)


def compare_existing_meta_rankings(meta: pd.DataFrame, dirs: OutputDirs) -> None:
    rows = []
    old_dir = CSIDE_ROOT / "gsea_results"
    for cell_type in sorted(meta["cell_type"].unique()):
        old_path = old_dir / f"meta_ranking_{cell_type}.csv"
        if not old_path.exists():
            rows.append({"cell_type": cell_type, "old_path": str(old_path), "status": "missing"})
            continue
        old = read_legacy_meta_ranking(old_path)
        if not {"gene", "z_combined"}.issubset(old.columns):
            rows.append({"cell_type": cell_type, "old_path": str(old_path), "status": "bad_schema"})
            continue
        current = meta.loc[meta["cell_type"] == cell_type, ["gene", "stouffer_z"]]
        merged = current.merge(old[["gene", "z_combined"]], on="gene", how="inner")
        corr = merged["stouffer_z"].corr(merged["z_combined"]) if len(merged) > 2 else np.nan
        rows.append(
            {
                "cell_type": cell_type,
                "old_path": str(old_path),
                "status": "ok",
                "n_overlap": int(len(merged)),
                "pearson_current_vs_old_z": corr,
                "max_abs_delta_z": float((merged["stouffer_z"] - merged["z_combined"]).abs().max())
                if len(merged)
                else np.nan,
            }
        )
    pd.DataFrame(rows).to_csv(dirs.signed / "existing_meta_ranking_comparison.csv", index=False)


def read_legacy_meta_ranking(path: Path) -> pd.DataFrame:
    """Read older meta-ranking CSVs that mix comma and semicolon delimiters."""
    for sep in (None, ";", ","):
        try:
            if sep is None:
                df = pd.read_csv(path, sep=None, engine="python")
            else:
                df = pd.read_csv(path, sep=sep)
        except Exception:
            continue
        if {"gene", "z_combined"}.issubset(df.columns):
            return df
        if len(df.columns) == 1 and ";" in df.columns[0]:
            expanded = df.iloc[:, 0].astype(str).str.split(";", expand=True)
            expanded.columns = str(df.columns[0]).split(";")
            for col in expanded.columns:
                if col != "gene":
                    expanded[col] = pd.to_numeric(expanded[col], errors="ignore")
            if {"gene", "z_combined"}.issubset(expanded.columns):
                return expanded
    return pd.DataFrame()


def clean_name(value: object) -> str:
    text = str(value).strip()
    keep = [ch if ch.isalnum() else "_" for ch in text]
    return "_".join("".join(keep).split("_")).strip("_")


def load_json_gene_sets(path: Path) -> dict[str, set[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): {str(g).upper() for g in v if str(g).strip()} for k, v in data.items()}


def export_kegg_gene_sets(output_json: Path) -> Path:
    if output_json.exists():
        return output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    r_script = output_json.parent / "export_kegg_gene_sets_from_msigdbr.R"
    r_script.write_text(
        """
suppressPackageStartupMessages(library(msigdbr))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
out <- args[1]
kegg <- msigdbr(species = "Homo sapiens", category = "C2", subcategory = "CP:KEGG_LEGACY")
sets <- split(kegg$gene_symbol, kegg$gs_name)
sets <- lapply(sets, unique)
write_json(sets, out, auto_unbox = FALSE)
cat(sprintf("Exported %d KEGG_LEGACY gene sets to %s\\n", length(sets), out))
""".strip()
        + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [resolve_rscript(), str(r_script), str(output_json)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (output_json.parent / "export_kegg_gene_sets_from_msigdbr.log").write_text(
        result.stdout,
        encoding="utf-8",
    )
    if result.returncode != 0 or not output_json.exists():
        raise RuntimeError(f"KEGG gene-set export failed; see {output_json.parent}")
    return output_json


def run_gene_set_tests(
    meta: pd.DataFrame,
    gene_sets: dict[str, set[str]],
    collection: str,
    dirs: OutputDirs,
    n_perm: int,
    min_set_genes: int,
    abs_logfc_threshold: float,
    rng: np.random.Generator,
    output_dir: Path | None = None,
    output_prefix: str | None = None,
    compare_existing: bool = True,
) -> pd.DataFrame:
    out_dir = output_dir if output_dir is not None else dirs.gene_sets / collection
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_prefix if output_prefix is not None else collection
    rows = []
    coverage_rows = []
    for cell_type in sorted(meta["cell_type"].unique()):
        ct = meta.loc[meta["cell_type"] == cell_type].copy()
        ct = ct.loc[ct["stouffer_z"].notna()]
        retained = ct.loc[ct["mean_logfc"].abs() >= abs_logfc_threshold].copy()
        values = retained.set_index("gene")["stouffer_z"].astype(float)
        universe = set(values.index)
        for pathway, genes in gene_sets.items():
            overlap = sorted(universe.intersection(genes))
            total_overlap = sorted(set(ct["gene"]).intersection(genes))
            coverage_rows.append(
                {
                    "collection": collection,
                    "cell_type": cell_type,
                    "pathway": pathway,
                    "gene_set_size": len(genes),
                    "tested_overlap_n": len(total_overlap),
                    "retained_abs_logfc_overlap_n": len(overlap),
                    "tested_overlap_genes": ";".join(total_overlap),
                    "retained_overlap_genes": ";".join(overlap),
                }
            )
            if len(overlap) < min_set_genes or len(values) < len(overlap):
                continue
            observed = float(values.loc[overlap].mean())
            perm = permutation_mean_null(values.to_numpy(dtype=float), len(overlap), n_perm, rng)
            p_perm = (1.0 + float((np.abs(perm) >= abs(observed)).sum())) / (n_perm + 1.0)
            rows.append(
                {
                    "collection": collection,
                    "cell_type": cell_type,
                    "pathway": pathway,
                    "n_genes": len(overlap),
                    "mean_z": observed,
                    "median_z": float(values.loc[overlap].median()),
                    "p_perm_two_sided": p_perm,
                    "direction": "positive" if observed > 0 else "negative" if observed < 0 else "zero",
                    "genes": ";".join(overlap),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["q_perm_bh"] = result.groupby("cell_type", group_keys=False)["p_perm_two_sided"].apply(bh_fdr)
        result = result.sort_values(["cell_type", "q_perm_bh", "p_perm_two_sided", "pathway"])
    result.to_csv(out_dir / f"{prefix}_meanZ_permutation_results.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(out_dir / f"{prefix}_meanZ_gene_set_coverage.csv", index=False)
    if compare_existing:
        compare_to_existing_fgsea(result, collection, out_dir)
    return result


def permutation_mean_null(values: np.ndarray, k: int, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    if k <= 0:
        return np.array([], dtype=float)
    n = len(values)
    means = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        idx = rng.choice(n, size=k, replace=False)
        means[i] = values[idx].mean()
    return means


def compare_to_existing_fgsea(mean_z: pd.DataFrame, collection: str, out_dir: Path) -> None:
    if collection == "hallmark":
        existing_path = CSIDE_ROOT / "gsea_results" / "gsea_hallmark_all_celltypes.csv"
        pathway_col = "pathway"
        padj_col = "padj"
    else:
        existing_path = CSIDE_ROOT / "gsea_results" / "kegg" / "gsea_kegg_all_celltypes.csv"
        pathway_col = "pathway"
        padj_col = "padj"
    if not existing_path.exists() or mean_z.empty:
        return
    old = pd.read_csv(existing_path)
    if pathway_col not in old.columns:
        return
    old = old.rename(columns={"Cell Type": "cell_type", "Pathway": "pathway"})
    old["pathway"] = old["pathway"].astype(str)
    old["cell_type"] = old["cell_type"].astype(str)
    keep_cols = ["cell_type", "pathway"]
    for col in ["NES", "pval", "pvalue", padj_col, "p.adjust"]:
        if col in old.columns:
            keep_cols.append(col)
    merged = mean_z.merge(old[keep_cols], on=["cell_type", "pathway"], how="left")
    merged.to_csv(out_dir / f"{collection}_meanZ_vs_existing_fgsea.csv", index=False)


def pathway_sample_heterogeneity(
    conv: pd.DataFrame,
    gene_sets: dict[str, set[str]],
    collection: str,
    dirs: OutputDirs,
    min_set_genes: int,
    abs_logfc_threshold: float,
) -> pd.DataFrame:
    out_dir = dirs.heterogeneity / collection
    rows = []
    sample_rows = []
    retained = conv.loc[conv["log_fc"].abs() >= abs_logfc_threshold].copy()
    for (cell_type, dataset, sample_id), subset in retained.groupby(
        ["cell_type", "dataset", "sample_id_on_disk"], sort=False
    ):
        values = subset.dropna(subset=["signed_z"]).drop_duplicates("gene").set_index("gene")["signed_z"]
        universe = set(values.index)
        for pathway, genes in gene_sets.items():
            overlap = sorted(universe.intersection(genes))
            if len(overlap) < min_set_genes:
                continue
            sample_rows.append(
                {
                    "collection": collection,
                    "cell_type": cell_type,
                    "dataset": dataset,
                    "sample_id_on_disk": sample_id,
                    "sample_label": sample_label(dataset, sample_id),
                    "pathway": pathway,
                    "n_genes": len(overlap),
                    "sample_mean_z": float(values.loc[overlap].mean()),
                    "sample_median_z": float(values.loc[overlap].median()),
                }
            )
    sample_df = pd.DataFrame(sample_rows)
    sample_df.to_csv(out_dir / f"{collection}_sample_level_pathway_meanZ.csv", index=False)
    if sample_df.empty:
        return sample_df

    for (cell_type, pathway), subset in sample_df.groupby(["cell_type", "pathway"], sort=False):
        vals = subset["sample_mean_z"].to_numpy(dtype=float)
        try:
            p_wilcoxon = float(wilcoxon(vals, zero_method="wilcox").pvalue) if np.any(vals != 0) else 1.0
        except ValueError:
            p_wilcoxon = np.nan
        dataset_means = subset.groupby("dataset")["sample_mean_z"].mean().to_dict()
        loo_dataset = {}
        for dataset in sorted(subset["dataset"].unique()):
            loo = subset.loc[subset["dataset"] != dataset, "sample_mean_z"]
            loo_dataset[f"leave_out_{dataset}_mean_z"] = float(loo.mean()) if len(loo) else np.nan
        n_pos = int((vals > 0).sum())
        n_neg = int((vals < 0).sum())
        rows.append(
            {
                "collection": collection,
                "cell_type": cell_type,
                "pathway": pathway,
                "n_samples": int(len(subset)),
                "n_datasets": int(subset["dataset"].nunique()),
                "mean_sample_mean_z": float(np.mean(vals)),
                "median_sample_mean_z": float(np.median(vals)),
                "sd_sample_mean_z": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                "n_positive_samples": n_pos,
                "n_negative_samples": n_neg,
                "sample_direction_consistency": max(n_pos, n_neg) / len(vals),
                "wilcoxon_p_vs_zero": p_wilcoxon,
                **{f"dataset_mean_z_{k}": v for k, v in dataset_means.items()},
                **loo_dataset,
            }
        )
    summary = pd.DataFrame(rows)
    summary["wilcoxon_q_bh"] = summary.groupby("cell_type", group_keys=False)["wilcoxon_p_vs_zero"].apply(bh_fdr)
    summary = summary.sort_values(["cell_type", "wilcoxon_q_bh", "pathway"])
    summary.to_csv(out_dir / f"{collection}_pathway_sample_dataset_heterogeneity_summary.csv", index=False)
    return summary


def read_signature_genes(path: Path) -> set[str]:
    sig = pd.read_csv(path, sep=";")
    if "Gene" not in sig.columns:
        raise ValueError(f"{path} does not contain a Gene column after semicolon parsing")
    return {str(g).upper() for g in sig["Gene"].dropna() if str(g).strip()}


def circularity_checks(
    meta: pd.DataFrame,
    gene_sets_by_collection: dict[str, dict[str, set[str]]],
    meanz_by_collection: dict[str, pd.DataFrame],
    dirs: OutputDirs,
    signature_path: Path,
    abs_logfc_threshold: float,
) -> None:
    signature = read_signature_genes(signature_path)
    presence_rows = []
    for cell_type, subset in meta.groupby("cell_type", sort=True):
        tested = set(subset["gene"])
        retained = set(subset.loc[subset["mean_logfc"].abs() >= abs_logfc_threshold, "gene"])
        sig_tested = sorted(signature.intersection(tested))
        sig_retained = sorted(signature.intersection(retained))
        sig_sig = sorted(signature.intersection(set(subset.loc[subset["stouffer_q"] < 0.05, "gene"])))
        presence_rows.append(
            {
                "cell_type": cell_type,
                "signature_n": len(signature),
                "tested_signature_n": len(sig_tested),
                "tested_signature_pct": len(sig_tested) / len(signature),
                "retained_abs_logfc_signature_n": len(sig_retained),
                "meta_stouffer_q_lt_0_05_signature_n": len(sig_sig),
                "tested_signature_genes": ";".join(sig_tested),
                "retained_signature_genes": ";".join(sig_retained),
                "significant_signature_genes": ";".join(sig_sig),
            }
        )
    pd.DataFrame(presence_rows).to_csv(
        dirs.circularity / "signature_gene_presence_by_celltype.csv",
        index=False,
    )

    pathway_rows = []
    for collection, gene_sets in gene_sets_by_collection.items():
        tested = meanz_by_collection.get(collection, pd.DataFrame())
        if tested.empty:
            continue
        for row in tested.itertuples(index=False):
            genes = gene_sets.get(row.pathway, set())
            overlap = sorted(signature.intersection(genes))
            result_genes = set(str(row.genes).split(";")) if isinstance(row.genes, str) else set()
            result_overlap = sorted(signature.intersection(result_genes))
            pathway_rows.append(
                {
                    "collection": collection,
                    "cell_type": row.cell_type,
                    "pathway": row.pathway,
                    "mean_z": row.mean_z,
                    "p_perm_two_sided": row.p_perm_two_sided,
                    "q_perm_bh": row.q_perm_bh,
                    "gene_set_signature_overlap_n": len(overlap),
                    "meanZ_test_signature_overlap_n": len(result_overlap),
                    "gene_set_signature_overlap_genes": ";".join(overlap),
                    "meanZ_test_signature_overlap_genes": ";".join(result_overlap),
                }
            )
    pd.DataFrame(pathway_rows).to_csv(
        dirs.circularity / "signature_gene_overlap_with_meanZ_gene_sets.csv",
        index=False,
    )
    leading_edge_circularity(signature, dirs)


def leading_edge_circularity(signature: set[str], dirs: OutputDirs) -> None:
    sources = [
        ("hallmark", CSIDE_ROOT / "gsea_results" / "gsea_hallmark_all_celltypes.csv"),
        ("kegg", CSIDE_ROOT / "gsea_results" / "kegg" / "gsea_kegg_all_celltypes.csv"),
    ]
    rows = []
    for collection, path in sources:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df.rename(columns={"Cell Type": "cell_type", "Pathway": "pathway", "p.adjust": "padj"})
        le_col = "leading_edge_genes" if "leading_edge_genes" in df.columns else "leadingEdge"
        if le_col not in df.columns:
            continue
        for row in df.itertuples(index=False):
            le = getattr(row, le_col)
            genes = {g.strip().upper() for g in str(le).replace(",", ";").split(";") if g.strip()}
            overlap = sorted(signature.intersection(genes))
            rows.append(
                {
                    "collection": collection,
                    "cell_type": getattr(row, "cell_type"),
                    "pathway": getattr(row, "pathway"),
                    "NES": getattr(row, "NES") if hasattr(row, "NES") else np.nan,
                    "padj": getattr(row, "padj") if hasattr(row, "padj") else np.nan,
                    "leading_edge_n": len(genes),
                    "signature_overlap_n": len(overlap),
                    "signature_overlap_genes": ";".join(overlap),
                }
            )
    pd.DataFrame(rows).to_csv(dirs.circularity / "signature_gene_overlap_with_existing_fgsea_leading_edges.csv", index=False)


def qc_availability_audit(
    manifest: pd.DataFrame,
    rctd_input_root: Path,
    qc_input_root: Path,
    dirs: OutputDirs,
) -> None:
    summary_rows = []
    spot_summary_rows = []
    qc_input_root.mkdir(parents=True, exist_ok=True)
    for row in manifest.itertuples(index=False):
        in_dir = rctd_input_root / row.dataset / row.sample_id_on_disk
        meta_path = in_dir / "metadata.csv"
        counts_path = in_dir / "counts.csv.gz"
        if not meta_path.exists() or not counts_path.exists():
            summary_rows.append(
                {
                    "dataset": row.dataset,
                    "sample_id_on_disk": row.sample_id_on_disk,
                    "status": "missing_metadata_or_counts",
                }
            )
            continue
        metadata = pd.read_csv(meta_path, index_col=0)
        counts = pd.read_csv(counts_path, index_col=0)
        total_counts = counts.sum(axis=1)
        n_genes_by_counts = (counts > 0).sum(axis=1)
        mito_cols = [c for c in counts.columns if str(c).upper().startswith("MT-")]
        if mito_cols:
            pct_mito = counts[mito_cols].sum(axis=1) / total_counts.replace(0, np.nan) * 100.0
        else:
            pct_mito = pd.Series(np.nan, index=counts.index)
        qc = metadata.copy()
        qc["total_counts_rctd_input"] = total_counts.reindex(qc.index)
        qc["n_genes_by_counts_rctd_input"] = n_genes_by_counts.reindex(qc.index)
        qc["pct_mito_rctd_input"] = pct_mito.reindex(qc.index)

        out_dir = qc_input_root / row.dataset / row.sample_id_on_disk
        out_dir.mkdir(parents=True, exist_ok=True)
        qc.to_csv(out_dir / "metadata.csv")

        covariates = ["total_counts_rctd_input", "n_genes_by_counts_rctd_input", "pct_mito_rctd_input"]
        corrs = {}
        for cov in covariates:
            valid = qc[["SNAI1-ac_score", "Malignant", cov]].replace([np.inf, -np.inf], np.nan).dropna()
            corrs[f"spearman_{cov}_vs_snai1ac"] = (
                valid[cov].corr(valid["SNAI1-ac_score"], method="spearman") if len(valid) >= 3 else np.nan
            )
            corrs[f"spearman_{cov}_vs_malignant"] = (
                valid[cov].corr(valid["Malignant"], method="spearman") if len(valid) >= 3 else np.nan
            )
        summary_rows.append(
            {
                "dataset": row.dataset,
                "sample_id_on_disk": row.sample_id_on_disk,
                "status": "ok",
                "n_spots": int(len(qc)),
                "n_genes": int(counts.shape[1]),
                "n_mito_genes": int(len(mito_cols)),
                "metadata_columns_original": ";".join(metadata.columns),
                "qc_metadata_path": str(out_dir / "metadata.csv"),
                **corrs,
            }
        )
        for cov in covariates:
            values = qc[cov].dropna()
            spot_summary_rows.append(
                {
                    "dataset": row.dataset,
                    "sample_id_on_disk": row.sample_id_on_disk,
                    "covariate": cov,
                    "n_nonmissing": int(values.size),
                    "mean": float(values.mean()) if values.size else np.nan,
                    "median": float(values.median()) if values.size else np.nan,
                    "sd": float(values.std(ddof=1)) if values.size > 1 else np.nan,
                    "min": float(values.min()) if values.size else np.nan,
                    "max": float(values.max()) if values.size else np.nan,
                }
            )
    pd.DataFrame(summary_rows).to_csv(dirs.qc / "qc_covariate_availability_by_sample.csv", index=False)
    pd.DataFrame(spot_summary_rows).to_csv(dirs.qc / "qc_covariate_distribution_summary.csv", index=False)
    write_qc_rerun_script(dirs.qc)


def write_qc_rerun_script(qc_dir: Path) -> None:
    script_path = REPO_ROOT / "scripts" / "R" / "run_cside_qc_sensitivity.r"
    if script_path.exists():
        return
    script_path.write_text(
        r'''
# Run C-SIDE QC sensitivity from existing clean RCTD objects.
# Model: SNAI1-ac + malignant fraction + available QC covariates.

library(spacexr)

input_meta_base <- "D:/HGSOC_Spatial_Atlas/scRNA_reference/rctd_inputs_qc_sensitivity"
original_rctd_base <- "D:/HGSOC_Spatial_Atlas/scRNA_reference/rctd_outputs"
output_base <- "D:/HGSOC_Spatial_Atlas/scRNA_reference/rctd_outputs_qc_sensitivity"

datasets <- c("denisenko_2022", "ju_2024", "yamamoto_2025")
CELL_TYPE_THRESHOLD <- 50
WEIGHT_THRESHOLD <- 0.8

extract_cside_results <- function(rctd_obj, output_dir, prefix = "cside_qc") {
  all_res <- list()
  for (ct in names(rctd_obj@de_results)) {
    df <- rctd_obj@de_results[[ct]]
    if (is.null(df) || nrow(df) == 0) next
    df$cell_type <- ct
    df$gene <- rownames(df)
    all_res[[ct]] <- df
  }
  combined <- do.call(rbind, all_res)
  write.csv(combined, file.path(output_dir, paste0(prefix, "_all_results.csv")), row.names = FALSE)
}

minmax <- function(x) {
  rng <- range(x, na.rm = TRUE)
  if (!is.finite(rng[1]) || !is.finite(rng[2]) || rng[1] == rng[2]) return(rep(0, length(x)))
  (x - rng[1]) / (rng[2] - rng[1])
}

for (ds in datasets) {
  ds_dir <- file.path(input_meta_base, ds)
  if (!dir.exists(ds_dir)) next
  for (samp in list.files(ds_dir)) {
    meta_path <- file.path(ds_dir, samp, "metadata.csv")
    rctd_path <- file.path(original_rctd_base, ds, samp, "rctd_object.rds")
    if (!file.exists(meta_path) || !file.exists(rctd_path)) next
    out_dir <- file.path(output_base, ds, samp)
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
    out_file <- file.path(out_dir, "cside_qc_all_results.csv")
    if (file.exists(out_file)) next

    metadata <- read.csv(meta_path, row.names = 1, check.names = FALSE)
    rctd <- readRDS(rctd_path)
    barcodes <- rownames(metadata)

    base_required <- c("SNAI1-ac_score", "Malignant")
    if (!all(base_required %in% colnames(metadata))) {
      cat(sprintf("Skipping %s/%s: missing base columns\n", ds, samp))
      next
    }
    qc_candidates <- c(
      "total_counts_rctd_input",
      "n_genes_by_counts_rctd_input",
      "pct_mito_rctd_input"
    )
    qc_available <- c()
    for (covar in qc_candidates) {
      if (!covar %in% colnames(metadata)) next
      x <- metadata[[covar]]
      if (sum(is.finite(x)) < 3) next
      if (sd(x, na.rm = TRUE) == 0) next
      qc_available <- c(qc_available, covar)
    }
    required <- c(base_required, qc_available)
    keep <- complete.cases(metadata[, required, drop = FALSE])
    metadata <- metadata[keep, , drop = FALSE]
    barcodes <- rownames(metadata)
    if (nrow(metadata) < 20) {
      cat(sprintf("Skipping %s/%s: only %d complete spots\n", ds, samp, nrow(metadata)))
      next
    }

    x_parts <- list(
      intercept = rep(1, nrow(metadata)),
      snai1_norm = minmax(metadata[["SNAI1-ac_score"]]),
      mal_norm = minmax(metadata[["Malignant"]])
    )
    if ("total_counts_rctd_input" %in% qc_available) {
      x_parts[["log_total_counts_norm"]] <- minmax(log1p(metadata[["total_counts_rctd_input"]]))
    }
    if ("n_genes_by_counts_rctd_input" %in% qc_available) {
      x_parts[["n_genes_norm"]] <- minmax(metadata[["n_genes_by_counts_rctd_input"]])
    }
    if ("pct_mito_rctd_input" %in% qc_available) {
      x_parts[["pct_mito_norm"]] <- minmax(metadata[["pct_mito_rctd_input"]])
    }
    X <- do.call(cbind, x_parts)
    rownames(X) <- barcodes

    cat(sprintf(
      "Running QC C-SIDE for %s/%s (%d spots; QC covariates: %s)\n",
      ds, samp, nrow(X), paste(qc_available, collapse = ", ")
    ))
    fit <- run.CSIDE(
      rctd,
      X,
      barcodes,
      cell_type_threshold = CELL_TYPE_THRESHOLD,
      doublet_mode = FALSE,
      weight_threshold = WEIGHT_THRESHOLD,
      params_to_test = 2,
      test_mode = "individual"
    )
    extract_cside_results(fit, out_dir, prefix = "cside_qc")
    saveRDS(fit, file.path(out_dir, "rctd_cside_qc_object.rds"))
  }
}
'''.lstrip(),
        encoding="utf-8",
    )
    (qc_dir / "qc_cside_rerun_script_path.txt").write_text(str(script_path), encoding="utf-8")


def qc_adjusted_cside_comparison(
    original_meta: pd.DataFrame,
    qc_cside_root: Path,
    dirs: OutputDirs,
    gene_sets_by_collection: dict[str, dict[str, set[str]]],
    n_perm: int,
    min_samples: int,
    min_set_genes: int,
    abs_logfc_threshold: float,
    random_seed: int,
) -> None:
    qc_dir = dirs.qc / "qc_adjusted_cside_comparison"
    qc_gene_set_root = dirs.qc / "qc_adjusted_gene_set_tests"
    qc_dir.mkdir(parents=True, exist_ok=True)
    qc_gene_set_root.mkdir(parents=True, exist_ok=True)

    qc_manifest = discover_qc_cside_results(qc_cside_root)
    qc_manifest.to_csv(qc_dir / "qc_cside_sample_manifest.csv", index=False)
    if qc_manifest.empty:
        raise RuntimeError(f"No cside_qc_all_results.csv files found under {qc_cside_root}")

    qc_all = load_all_cside(qc_manifest)
    qc_all.to_csv(qc_dir / "qc_cside_signed_gene_level_all_samples.csv", index=False)
    qc_conv = qc_all.loc[qc_all["conv_bool"]].copy()
    qc_meta = build_gene_meta(qc_conv, min_samples=min_samples)
    qc_meta.to_csv(qc_dir / "qc_meta_gene_effects_signed_stouffer_iv_random_effects.csv", index=False)

    sample_celltype = (
        qc_all.groupby(["dataset", "sample_id_on_disk", "cell_type"], dropna=False)
        .agg(
            n_rows=("gene", "size"),
            n_converged=("conv_bool", "sum"),
            n_genes=("gene", "nunique"),
            n_signed_positive=("signed_z", lambda x: int((x > 0).sum())),
            n_signed_negative=("signed_z", lambda x: int((x < 0).sum())),
            median_p_abs_delta=("p_abs_delta", "median"),
            max_p_abs_delta=("p_abs_delta", "max"),
        )
        .reset_index()
    )
    sample_celltype.to_csv(qc_dir / "qc_signed_z_audit_by_sample_celltype.csv", index=False)

    meta_keep = [
        "cell_type",
        "gene",
        "n_samples",
        "n_datasets",
        "stouffer_z",
        "stouffer_p",
        "stouffer_q",
        "mean_logfc",
        "median_logfc",
        "random_logfc",
        "random_z",
        "random_q",
        "i2",
        "sign_consistency",
    ]
    original_keep = [c for c in meta_keep if c in original_meta.columns]
    qc_keep = [c for c in meta_keep if c in qc_meta.columns]
    merged = original_meta[original_keep].merge(
        qc_meta[qc_keep],
        on=["cell_type", "gene"],
        how="outer",
        suffixes=("_original", "_qc"),
        indicator=True,
    )
    if {"stouffer_z_original", "stouffer_z_qc"}.issubset(merged.columns):
        merged["delta_stouffer_z_qc_minus_original"] = merged["stouffer_z_qc"] - merged["stouffer_z_original"]
        merged["same_stouffer_direction"] = np.sign(merged["stouffer_z_qc"]) == np.sign(merged["stouffer_z_original"])
    if {"mean_logfc_original", "mean_logfc_qc"}.issubset(merged.columns):
        merged["delta_mean_logfc_qc_minus_original"] = merged["mean_logfc_qc"] - merged["mean_logfc_original"]
        merged["same_mean_logfc_direction"] = np.sign(merged["mean_logfc_qc"]) == np.sign(merged["mean_logfc_original"])
    merged.to_csv(qc_dir / "qc_vs_original_meta_gene_effects.csv", index=False)

    corr_rows = []
    for cell_type, subset in merged.loc[merged["_merge"] == "both"].groupby("cell_type", sort=True):
        row = {"cell_type": cell_type, "n_overlap_genes": int(len(subset))}
        if {"stouffer_z_original", "stouffer_z_qc"}.issubset(subset.columns):
            valid = subset[["stouffer_z_original", "stouffer_z_qc"]].dropna()
            row["pearson_stouffer_z_original_vs_qc"] = (
                valid["stouffer_z_original"].corr(valid["stouffer_z_qc"]) if len(valid) > 2 else np.nan
            )
            row["median_abs_delta_stouffer_z"] = float(
                (valid["stouffer_z_qc"] - valid["stouffer_z_original"]).abs().median()
            ) if len(valid) else np.nan
            row["direction_agreement_fraction"] = float(
                (np.sign(valid["stouffer_z_original"]) == np.sign(valid["stouffer_z_qc"])).mean()
            ) if len(valid) else np.nan
        corr_rows.append(row)
    pd.DataFrame(corr_rows).to_csv(qc_dir / "qc_vs_original_meta_gene_effect_correlations_by_celltype.csv", index=False)

    rng = np.random.default_rng(random_seed)
    for collection, gene_sets in gene_sets_by_collection.items():
        out_dir = qc_gene_set_root / collection
        qc_result = run_gene_set_tests(
            meta=qc_meta,
            gene_sets=gene_sets,
            collection=collection,
            dirs=dirs,
            n_perm=n_perm,
            min_set_genes=min_set_genes,
            abs_logfc_threshold=abs_logfc_threshold,
            rng=rng,
            output_dir=out_dir,
            output_prefix=f"qc_adjusted_{collection}",
            compare_existing=False,
        )
        original_path = dirs.gene_sets / collection / f"{collection}_meanZ_permutation_results.csv"
        if original_path.exists() and not qc_result.empty:
            original = pd.read_csv(original_path)
            compare = original.merge(
                qc_result,
                on=["collection", "cell_type", "pathway"],
                how="outer",
                suffixes=("_original", "_qc"),
                indicator=True,
            )
            if {"mean_z_original", "mean_z_qc"}.issubset(compare.columns):
                compare["delta_mean_z_qc_minus_original"] = compare["mean_z_qc"] - compare["mean_z_original"]
                compare["same_mean_z_direction"] = np.sign(compare["mean_z_qc"]) == np.sign(compare["mean_z_original"])
            if {"q_perm_bh_original", "q_perm_bh_qc"}.issubset(compare.columns):
                compare["original_significant_q05"] = compare["q_perm_bh_original"] < 0.05
                compare["qc_significant_q05"] = compare["q_perm_bh_qc"] < 0.05
            compare.to_csv(out_dir / f"qc_adjusted_{collection}_meanZ_vs_original.csv", index=False)


def main() -> None:
    args = parse_args()
    selected = {s.strip() for s in args.steps.split(",") if s.strip()}
    if "all" in selected:
        selected = {"manifest", "signed", "gene_sets", "circularity", "heterogeneity", "qc", "qc_compare"}
    dirs = ensure_dirs(args.output_root)
    manifest = discover_cside_results(args.cside_root, result_filename=args.result_filename)
    if manifest.empty:
        raise RuntimeError(f"No {args.result_filename} files found under {args.cside_root}")
    if "manifest" in selected:
        write_manifest(dirs, args, manifest)

    need_cside = bool(selected.intersection({"signed", "gene_sets", "circularity", "heterogeneity", "qc_compare"}))
    all_cside = load_all_cside(manifest) if need_cside else pd.DataFrame()
    conv = all_cside.loc[all_cside["conv_bool"]].copy() if not all_cside.empty else pd.DataFrame()
    meta = pd.DataFrame()
    if "signed" in selected:
        meta = signed_ranking_audit(
            all_cside,
            dirs,
            min_samples=args.min_samples,
            signed_output_name=args.signed_output_name,
        )
    elif need_cside:
        meta_path = dirs.signed / "meta_gene_effects_signed_stouffer_iv_random_effects.csv"
        meta = pd.read_csv(meta_path) if meta_path.exists() else build_gene_meta(conv, args.min_samples)

    gene_sets_by_collection: dict[str, dict[str, set[str]]] = {}
    meanz_by_collection: dict[str, pd.DataFrame] = {}
    if selected.intersection({"gene_sets", "circularity", "heterogeneity", "qc_compare"}):
        gene_sets_by_collection["hallmark"] = load_json_gene_sets(args.hallmark_json)
        kegg_json = dirs.manifest / "kegg_legacy_gene_sets_from_msigdbr.json"
        gene_sets_by_collection["kegg"] = load_json_gene_sets(export_kegg_gene_sets(kegg_json))

    if "gene_sets" in selected:
        rng = np.random.default_rng(args.random_seed)
        for collection, gene_sets in gene_sets_by_collection.items():
            meanz_by_collection[collection] = run_gene_set_tests(
                meta=meta,
                gene_sets=gene_sets,
                collection=collection,
                dirs=dirs,
                n_perm=args.n_perm,
                min_set_genes=args.min_set_genes,
                abs_logfc_threshold=args.abs_logfc_threshold,
                rng=rng,
            )
    else:
        for collection in ("hallmark", "kegg"):
            path = dirs.gene_sets / collection / f"{collection}_meanZ_permutation_results.csv"
            if path.exists():
                meanz_by_collection[collection] = pd.read_csv(path)

    if "heterogeneity" in selected:
        for collection, gene_sets in gene_sets_by_collection.items():
            pathway_sample_heterogeneity(
                conv=conv,
                gene_sets=gene_sets,
                collection=collection,
                dirs=dirs,
                min_set_genes=args.min_set_genes,
                abs_logfc_threshold=args.abs_logfc_threshold,
            )

    if "circularity" in selected:
        circularity_checks(
            meta=meta,
            gene_sets_by_collection=gene_sets_by_collection,
            meanz_by_collection=meanz_by_collection,
            dirs=dirs,
            signature_path=args.signature_full,
            abs_logfc_threshold=args.abs_logfc_threshold,
        )

    if "qc" in selected:
        qc_availability_audit(
            manifest=manifest,
            rctd_input_root=args.rctd_input_root,
            qc_input_root=args.qc_input_root,
            dirs=dirs,
        )

    if "qc_compare" in selected:
        qc_adjusted_cside_comparison(
            original_meta=meta,
            qc_cside_root=args.qc_cside_root,
            dirs=dirs,
            gene_sets_by_collection=gene_sets_by_collection,
            n_perm=args.n_perm,
            min_samples=args.min_samples,
            min_set_genes=args.min_set_genes,
            abs_logfc_threshold=args.abs_logfc_threshold,
            random_seed=args.random_seed,
        )

    print(f"C-SIDE audit complete: {args.output_root}")


if __name__ == "__main__":
    main()
