from __future__ import annotations

from datetime import datetime
import json
import re
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import enrichmap as em


ROOT = Path(r"D:\HGSOC_Spatial_Atlas")
MANUAL_DIR = (
    ROOT
    / "05_analysis_ready"
    / "S3_cNMF_Tumor_Programs"
    / "jaccard_raw_matrices"
    / "inspection_exports_average"
    / "variantB_nonjunk_manual_cut_v2"
)
LEAF_DIR = MANUAL_DIR / "coarse_recluster_inspection"
TOP50_FILE = MANUAL_DIR / "variantB_original_top50_positive_gene_spectra_score_long.csv"
VISIUM_DIR = ROOT / "05_analysis_ready" / "visium"
OUT = MANUAL_DIR / "subcluster_signatures_scoring"

SUBCLUSTERS = {
    "MP1": ("A", 4, 10, "angiogenic-vascular"),
    "MP2": ("A", 12, 15, "iCAF-stress"),
    "MP3": ("A", 16, 21, "complement-CAF"),
    "MP4": ("A", 24, 36, "activated-myCAF"),
    "MP5": ("B", 2, 11, "IFN-TLS-immune"),
    "MP6": ("B", 14, 25, "APC-TAM-myeloid"),
    "MP7": ("C", 10, 16, "malignant-hypoxia"),
    "MP8": ("C", 18, 23, "malignant-acute-phase-secretory"),
}
EXPECTED_COUNTS = {"MP1": 7, "MP2": 4, "MP3": 6, "MP4": 13, "MP5": 10, "MP6": 12, "MP7": 7, "MP8": 6}
EXPECTED_SIGNATURE_SIZES = {"MP1": 47, "MP2": 36, "MP3": 39, "MP4": 38, "MP5": 44, "MP6": 35, "MP7": 37, "MP8": 20}
PROGRAM_RE = re.compile(r"^(denisenko_2022|ju_2024|yamamoto_2025)__(.+)__K\d+__P\d+$")


def stop(message: str) -> None:
    raise SystemExit(f"STOP: {message}")


def require(path: Path) -> None:
    if not path.exists():
        stop(f"Missing input: {path}")


def read_leaf_tables() -> dict[str, pd.DataFrame]:
    tables = {}
    for coarse in ["A", "B", "C"]:
        path = LEAF_DIR / f"variantB_nonjunk_recluster_coarse_{coarse}_leaf_order.csv"
        require(path)
        df = pd.read_csv(path)
        needed = {"position", "program_id"}
        if not needed.issubset(df.columns):
            stop(f"{path} lacks columns: {sorted(needed - set(df.columns))}")
        if df["position"].duplicated().any():
            stop(f"{path} has duplicated leaf positions")
        tables[coarse] = df
    return tables


def map_subclusters(leaf_tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for subcluster_id, (coarse, start, end, label) in SUBCLUSTERS.items():
        leaf = leaf_tables[coarse]
        wanted = set(range(start, end + 1))
        mapped = leaf.loc[leaf["position"].isin(wanted)].sort_values("position")
        got = set(mapped["position"].astype(int))
        if got != wanted:
            stop(f"{subcluster_id} range {coarse}:{start}-{end} did not map cleanly; missing {sorted(wanted - got)}")
        for row in mapped.itertuples(index=False):
            rows.append(
                {
                    "subcluster_id": subcluster_id,
                    "label": label,
                    "coarse": coarse,
                    "position": int(row.position),
                    "program_id": str(row.program_id),
                }
            )

    members = pd.DataFrame(rows)
    counts = members.groupby("subcluster_id")["program_id"].nunique().to_dict()
    if counts != EXPECTED_COUNTS or len(members) != 65:
        stop(f"Program counts mismatch. Observed={counts}, total={len(members)}; expected={EXPECTED_COUNTS}, total=65")

    c_all = set(leaf_tables["C"]["program_id"].astype(str))
    c_mp = set(members.loc[members["coarse"].eq("C"), "program_id"])
    residual = leaf_tables["C"].loc[leaf_tables["C"]["program_id"].astype(str).isin(c_all - c_mp)].copy()
    residual.insert(0, "residual_group", "coarseC_residual_not_scored")
    return members, residual


def extract_signatures(members: pd.DataFrame) -> pd.DataFrame:
    require(TOP50_FILE)
    top50 = pd.read_csv(TOP50_FILE)
    needed = {"program_id", "rank", "gene", "gene_spectra_score"}
    if not needed.issubset(top50.columns):
        stop(f"{TOP50_FILE} lacks columns: {sorted(needed - set(top50.columns))}")
    top50 = top50.loc[top50["rank"].astype(int).le(50)].copy()
    top50["gene_spectra_score"] = pd.to_numeric(top50["gene_spectra_score"], errors="coerce")

    missing = set(members["program_id"]) - set(top50["program_id"])
    if missing:
        stop(f"Top50 table lacks member programs: {sorted(missing)[:10]}")

    signature_rows = []
    for subcluster_id, group in members.groupby("subcluster_id", sort=True):
        label = str(group["label"].iloc[0])
        program_ids = group["program_id"].tolist()
        n_programs = len(program_ids)
        floor = n_programs // 2
        subset = top50.loc[top50["program_id"].isin(program_ids)].copy()
        gene_stats = (
            subset.groupby("gene", as_index=False)
            .agg(
                occurrence=("program_id", "nunique"),
                mean_gene_spectra_score=("gene_spectra_score", "mean"),
            )
            .query("occurrence >= @floor")
            .sort_values(["occurrence", "mean_gene_spectra_score", "gene"], ascending=[False, False, True])
        )
        if len(gene_stats) > 50:
            gene_stats = gene_stats.head(50)
        for row in gene_stats.itertuples(index=False):
            signature_rows.append(
                {
                    "subcluster_id": subcluster_id,
                    "label": label,
                    "gene": str(row.gene),
                    "occurrence": int(row.occurrence),
                    "n_programs": int(n_programs),
                }
            )

    signatures = pd.DataFrame(signature_rows)
    sizes = signatures.groupby("subcluster_id")["gene"].nunique().to_dict()
    if sizes != EXPECTED_SIGNATURE_SIZES:
        stop(f"Signature size mismatch. Observed={sizes}; expected={EXPECTED_SIGNATURE_SIZES}")
    return signatures


def score_key(subcluster_id: str, label: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
    return f"{subcluster_id}_{clean}"


def has_image(adata) -> bool:
    if "spatial" not in adata.uns:
        return False
    for library in adata.uns["spatial"].values():
        if library.get("images"):
            return True
    return False


def preferred_img_key(adata) -> str | None:
    if "spatial" not in adata.uns:
        return None
    for library in adata.uns["spatial"].values():
        images = library.get("images", {})
        if "hires" in images:
            return "hires"
        if "lowres" in images:
            return "lowres"
        if images:
            return next(iter(images))
    return None


def ensure_hires_image_alias(adata) -> None:
    if "spatial" not in adata.uns:
        return
    for library in adata.uns["spatial"].values():
        images = library.get("images", {})
        if "hires" not in images and "lowres" in images:
            images["hires"] = images["lowres"]


def sample_paths() -> list[tuple[str, str, Path]]:
    top50 = pd.read_csv(TOP50_FILE, usecols=["program_id"])
    samples = sorted(
        {
            PROGRAM_RE.match(pid).group(1) + "__" + PROGRAM_RE.match(pid).group(2)
            for pid in top50["program_id"].astype(str).unique()
        }
    )
    paths = []
    for sample_label in samples:
        dataset, sample = sample_label.split("__", 1)
        path = VISIUM_DIR / dataset / sample / f"{sample}.h5ad"
        require(path)
        paths.append((dataset, sample, path))
    return paths


def score_samples(signatures: pd.DataFrame) -> pd.DataFrame:
    score_sets = []
    for (subcluster_id, label), sig in signatures.groupby(["subcluster_id", "label"], sort=True):
        score_sets.append((subcluster_id, label, score_key(subcluster_id, label), sig["gene"].astype(str).tolist()))

    h5ad_dir = OUT / "scored_h5ad"
    score_dir = OUT / "score_tables"
    plot_dir = OUT / "plots"
    h5ad_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset, sample, path in sample_paths():
        print(f"Scoring {dataset}__{sample}")
        adata = sc.read_h5ad(path)
        sample_score_cols = []
        for subcluster_id, label, key, genes in score_sets:
            present = [gene for gene in genes if gene in adata.var_names]
            if not present:
                stop(f"No genes from {subcluster_id} found in {dataset}__{sample}")
            em.tl.score(
                adata=adata,
                gene_set=present,
                score_key=key,
                batch_key=None,
                smoothing=True,
                correct_spatial_covariates=True,
            )
            col = f"{key}_score"
            if col not in adata.obs.columns:
                stop(f"Expected score column was not created: {col} in {dataset}__{sample}")
            sample_score_cols.append(col)
            rows.append(
                {
                    "dataset": dataset,
                    "sample": sample,
                    "subcluster_id": subcluster_id,
                    "label": label,
                    "score_key": key,
                    "score_column": col,
                    "n_signature_genes": len(genes),
                    "n_genes_present": len(present),
                    "score_min": float(adata.obs[col].min()),
                    "score_mean": float(adata.obs[col].mean()),
                    "score_max": float(adata.obs[col].max()),
                }
            )
            if has_image(adata):
                try:
                    ensure_hires_image_alias(adata)
                    fig, ax = plt.subplots(figsize=(8, 8))
                    em.pl.spatial_enrichmap(
                        adata,
                        score_key=col,
                        size=1,
                        img_alpha=1,
                        alpha=0.7,
                        ax=ax,
                    )
                    out_png = plot_dir / dataset / sample / f"spatial_enrichmap_{key}.png"
                    out_png.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(out_png, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                except Exception as exc:
                    plt.close("all")
                    rows[-1]["plot_warning"] = str(exc)

        out_h5ad = h5ad_dir / dataset / f"{sample}.manual_jaccard_MP_scores.h5ad"
        out_h5ad.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(out_h5ad)
        adata.obs[sample_score_cols].to_csv(score_dir / f"{dataset}__{sample}_manual_jaccard_MP_scores.csv")
    return pd.DataFrame(rows)


def write_run_log(sanity: pd.DataFrame, score_summary: pd.DataFrame, manifest: dict) -> None:
    if "plot_warning" in score_summary.columns:
        warning_mask = score_summary["plot_warning"].fillna("").astype(str).str.len() > 0
    else:
        warning_mask = pd.Series(False, index=score_summary.index)
    warning_rows = score_summary.loc[warning_mask]
    warning_summary = (
        warning_rows.groupby(["dataset", "plot_warning"]).size().reset_index(name="n")
        if not warning_rows.empty
        else pd.DataFrame(columns=["dataset", "plot_warning", "n"])
    )
    plot_count = len(list((OUT / "plots").rglob("*.png"))) if (OUT / "plots").exists() else 0
    h5ad_count = len(list((OUT / "scored_h5ad").rglob("*.h5ad"))) if (OUT / "scored_h5ad").exists() else 0
    score_csv_count = len(list((OUT / "score_tables").glob("*.csv"))) if (OUT / "score_tables").exists() else 0

    lines = [
        "Manual Jaccard subcluster EnrichMap scoring log",
        f"Run timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"Script: {Path(__file__)}",
        f"Python: {sys.version.split()[0]}",
        f"Output directory: {OUT}",
        "",
        "Inputs",
        f"- Leaf-order directory: {LEAF_DIR}",
        f"- Top-50 positive gene_spectra_score table: {TOP50_FILE}",
        f"- Visium h5ad root: {VISIUM_DIR}",
        "",
        "Manual cuts",
    ]
    for subcluster_id, (coarse, start, end, label) in SUBCLUSTERS.items():
        lines.append(f"- {subcluster_id} {label}: coarse {coarse}, positions {start}-{end}")
    lines += [
        "- Coarse-C residual programs were recorded but not scored.",
        "",
        "Signature extraction",
        "- Program representation: top 50 positive gene_spectra_score genes.",
        "- Recurrence floor: occurrence >= floor(n_programs / 2).",
        "- Cap: top 50 by occurrence, ties by mean gene_spectra_score.",
        "- No padding; signatures keep their recurrent size.",
        "",
        "Sanity check",
        sanity.to_string(index=False),
        "",
        "EnrichMap scoring settings",
        json.dumps(manifest["scoring"], indent=2),
        "",
        "Sample set",
    ]
    lines.extend(f"- {sample}" for sample in manifest["sample_set"])
    lines += [
        "",
        "Output audit",
        f"- Score summary rows: {len(score_summary)}",
        f"- Unique samples scored: {score_summary['sample'].nunique()}",
        f"- Unique signatures scored: {score_summary['score_key'].nunique()}",
        f"- Scored h5ad files: {h5ad_count}",
        f"- Per-sample score CSVs: {score_csv_count}",
        f"- Spatial EnrichMap PNGs: {plot_count}",
        f"- Plot warning rows: {len(warning_rows)}",
    ]
    if not warning_summary.empty:
        lines += ["", "Plot warnings"]
        lines.extend(f"- {row.dataset}: {row.n} x {row.plot_warning}" for row in warning_summary.itertuples(index=False))
    lines += [
        "",
        "Primary outputs",
        f"- {OUT / 'signatures' / 'manual_subcluster_program_members.csv'}",
        f"- {OUT / 'signatures' / 'coarseC_residual_programs_not_scored.csv'}",
        f"- {OUT / 'signatures' / 'manual_subcluster_recurrent_gene_signatures_long.csv'}",
        f"- {OUT / 'manual_subcluster_sanity_check.csv'}",
        f"- {OUT / 'manual_subcluster_enrichmap_score_summary.csv'}",
        f"- {OUT / 'manual_subcluster_scoring_manifest.json'}",
    ]
    (OUT / "manual_subcluster_scoring_run_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), OUT / Path(__file__).name)
    leaf_tables = read_leaf_tables()
    members, residual = map_subclusters(leaf_tables)
    signatures = extract_signatures(members)

    sig_dir = OUT / "signatures"
    sig_dir.mkdir(parents=True, exist_ok=True)
    members.to_csv(sig_dir / "manual_subcluster_program_members.csv", index=False)
    residual.to_csv(sig_dir / "coarseC_residual_programs_not_scored.csv", index=False)
    signatures.to_csv(sig_dir / "manual_subcluster_recurrent_gene_signatures_long.csv", index=False)
    for (subcluster_id, label), group in signatures.groupby(["subcluster_id", "label"], sort=True):
        key = score_key(subcluster_id, label)
        (sig_dir / f"{key}.genes.txt").write_text("\n".join(group["gene"].astype(str)) + "\n", encoding="utf-8")

    program_counts = members.groupby(["subcluster_id", "label"], as_index=False)["program_id"].nunique()
    program_counts = program_counts.rename(columns={"program_id": "n_programs"})
    signature_sizes = signatures.groupby("subcluster_id", as_index=False)["gene"].nunique().rename(columns={"gene": "signature_size"})
    sanity = program_counts.merge(signature_sizes, on="subcluster_id", how="left")
    sanity["expected_n_programs"] = sanity["subcluster_id"].map(EXPECTED_COUNTS)
    sanity["expected_signature_size"] = sanity["subcluster_id"].map(EXPECTED_SIGNATURE_SIZES)
    sanity.to_csv(OUT / "manual_subcluster_sanity_check.csv", index=False)
    print(sanity.to_string(index=False))

    score_summary = score_samples(signatures)
    score_summary.to_csv(OUT / "manual_subcluster_enrichmap_score_summary.csv", index=False)
    manifest = {
        "input_leaf_dir": str(LEAF_DIR),
        "input_top50_file": str(TOP50_FILE),
        "sample_set": [f"{d}__{s}" for d, s, _ in sample_paths()],
        "scoring": {
            "enrichmap": "em.tl.score",
            "batch_key": None,
            "smoothing": True,
            "correct_spatial_covariates": True,
            "gene_weights": None,
            "weights": "uniform",
        },
    }
    (OUT / "manual_subcluster_scoring_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_run_log(sanity, score_summary, manifest)
    print(f"Done: {OUT}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        stop(str(exc))
