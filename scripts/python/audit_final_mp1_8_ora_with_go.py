from __future__ import annotations

import json
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


PROJECT = Path(r"C:\Users\luchu\Documents\MSc\Master Thesis")
ROOT = Path(r"D:\HGSOC_Spatial_Atlas")
CNMF_ROOT = ROOT / "05_analysis_ready" / "S3_cNMF_Tumor_Programs"
CNMF_RUNS = CNMF_ROOT / "cnmf_runs"
MANUAL_DIR = (
    CNMF_ROOT
    / "jaccard_raw_matrices"
    / "inspection_exports_average"
    / "variantB_nonjunk_manual_cut_v2"
)
SOURCE_OUT = MANUAL_DIR / "recurrence_specificity_diagnostics_final_mp1_8"
AUDIT_OUT = MANUAL_DIR / "recurrence_specificity_diagnostics_final_mp1_8_GO_audit"

LOCKED = SOURCE_OUT / "MP_signatures_locked_long.csv"
FULL = SOURCE_OUT / "MP_gene_recurrence_full_table.csv"
TOP50_FILE = MANUAL_DIR / "variantB_original_top50_positive_gene_spectra_score_long.csv"

MSIGDB_DIR = (
    PROJECT
    / "Code"
    / "HGSOC_Spatial_Atlas"
    / "00_documentation"
    / "kstar_sources_v0_2"
    / "msigdb_2025_1_Hs"
)
HALLMARK_GMT = MSIGDB_DIR / "h.all.v2025.1.Hs.symbols.gmt"
KEGG_GMT = MSIGDB_DIR / "c2.cp.kegg_legacy.v2025.1.Hs.symbols.gmt"
GO_RDS_DIR = (
    ROOT
    / "05_analysis_ready"
    / "S5plus_CSIDE_Alpha_Strengthening"
    / "runs"
    / "20260415_191521_kstar_niches_cside_alpha"
    / "01_kstar_niches"
    / "tmp"
)
GO_BP_RDS = GO_RDS_DIR / "go_bp_pathways.rds"
GO_CC_RDS = GO_RDS_DIR / "go_cc_pathways.rds"
RSCRIPT = Path(r"C:\Program Files\R\R-4.4.3\bin\x64\Rscript.exe")

GROUP_ORDER = [f"MP{i}" for i in range(1, 9)]
REPORT_LABELS = {
    "MP1": "angiogenic/vascular",
    "MP2": "iCAF-stress",
    "MP3": "complement-CAF",
    "MP4": "activated-myCAF",
    "MP5": "IFN/TLS immune",
    "MP6": "APC/TAM myeloid",
    "MP7": "malignant hypoxia",
    "MP8": "malignant acute-phase/secretory",
}


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def file_record(path: Path) -> dict[str, object]:
    require(path)
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "last_write_time": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
    }


def bh_adjust(values: pd.Series | np.ndarray | list[float]) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    out = np.full(vals.shape, np.nan, dtype=float)
    mask = np.isfinite(vals)
    if not mask.any():
        return out
    idx = np.where(mask)[0]
    order = idx[np.argsort(vals[idx])]
    ranked = vals[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out[order] = np.clip(adjusted, 0, 1)
    return out


def parse_program(program_id: str) -> tuple[str, str, int, int]:
    parts = str(program_id).split("__")
    dataset = parts[0]
    sample = parts[1]
    k_value = int(parts[2].replace("K", ""))
    p_value = int(parts[3].replace("P", ""))
    return dataset, sample, k_value, p_value


def spectra_path(program_id: str) -> Path:
    dataset, sample, k_value, _ = parse_program(program_id)
    sample_label = f"{dataset}__{sample}"
    return CNMF_RUNS / sample_label / f"{sample_label}.gene_spectra_score.k_{k_value}.dt_0_5.txt"


def load_gene_spectra_universe(program_ids: list[str]) -> set[str]:
    cache: dict[Path, list[str]] = {}
    universe: set[str] = set()
    for program_id in program_ids:
        path = spectra_path(program_id)
        require(path)
        if path not in cache:
            cache[path] = pd.read_csv(path, sep="\t", index_col=0, nrows=1).columns.astype(str).tolist()
        universe.update(cache[path])
    return universe


def parse_gmt(path: Path, library: str) -> dict[str, dict[str, object]]:
    require(path)
    terms: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            term_id = parts[0]
            genes = {gene.strip() for gene in parts[2:] if gene.strip()}
            terms[f"{library}:{term_id}"] = {
                "library": library,
                "term_id": term_id,
                "term": term_id,
                "genes": genes,
            }
    return terms


def parse_go_rds(path: Path, library: str) -> dict[str, dict[str, object]]:
    require(path)
    require(RSCRIPT)
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "dump_go_pathways.R"
        output = Path(td) / "go_pathways.tsv"
        script.write_text(
            """
args <- commandArgs(trailingOnly=TRUE)
obj <- readRDS(args[[1]])
pathways <- if (!is.null(obj$pathways)) obj$pathways else obj
term_names <- if (!is.null(obj$term_names)) obj$term_names else names(pathways)
pathway_ids <- names(pathways)
if (is.null(pathway_ids)) pathway_ids <- as.character(seq_along(pathways))
if (is.null(term_names)) term_names <- pathway_ids
con <- file(args[[2]], open='wt')
on.exit(close(con))
for (i in seq_along(pathways)) {
  genes <- unique(as.character(pathways[[i]]))
  genes <- genes[!is.na(genes) & nzchar(genes)]
  writeLines(paste(c(pathway_ids[[i]], term_names[[i]], genes), collapse='\\t'), con)
}
""",
            encoding="utf-8",
        )
        subprocess.run(
            [str(RSCRIPT), str(script), str(path), str(output)],
            check=True,
            capture_output=True,
            text=True,
        )
        terms: dict[str, dict[str, object]] = {}
        for line in output.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            term_id, term = parts[0], parts[1]
            genes = {gene for gene in parts[2:] if gene}
            terms[f"{library}:{term_id}"] = {
                "library": library,
                "term_id": term_id,
                "term": term,
                "genes": genes,
            }
        return terms


def run_ora(locked: pd.DataFrame, full: pd.DataFrame, universe: set[str]) -> pd.DataFrame:
    term_sets: dict[str, dict[str, object]] = {}
    term_sets.update(parse_gmt(HALLMARK_GMT, "HALLMARK"))
    term_sets.update(parse_gmt(KEGG_GMT, "KEGG_LEGACY"))
    term_sets.update(parse_go_rds(GO_BP_RDS, "GO_BP"))
    term_sets.update(parse_go_rds(GO_CC_RDS, "GO_CC"))

    specificity = full.set_index(["group", "gene"])[["specificity_delta", "rank_within_group"]].to_dict("index")
    universe = set(universe)
    universe_n = len(universe)
    rows = []

    for group, sig in locked.groupby("group", sort=False):
        genes = set(sig["gene"].astype(str)) & universe
        if not genes:
            continue
        for term_info in term_sets.values():
            term_genes = set(term_info["genes"]) & universe
            if not term_genes:
                continue
            overlap = genes & term_genes
            k = len(overlap)
            if k == 0:
                continue
            n = len(genes)
            K = len(term_genes)
            pvalue = float(hypergeom.sf(k - 1, universe_n, K, n))
            ordered = sorted(
                overlap,
                key=lambda gene: specificity.get((group, gene), {}).get("rank_within_group", 10**9),
            )
            deltas = [specificity.get((group, gene), {}).get("specificity_delta", np.nan) for gene in ordered]
            rows.append(
                {
                    "level": "final_mp",
                    "group": group,
                    "final_label": REPORT_LABELS[group],
                    "set_type": "strict",
                    "library": term_info["library"],
                    "term_id": term_info["term_id"],
                    "term": term_info["term"],
                    "overlap_k": k,
                    "signature_size_n": n,
                    "term_size_K": K,
                    "universe_N": universe_n,
                    "fold_enrichment": (k / n) / (K / universe_n) if n and K and universe_n else np.nan,
                    "pvalue": pvalue,
                    "overlap_genes": ";".join(ordered),
                    "n_overlap_specific": int(np.sum(np.asarray(deltas, dtype=float) >= 0.3)),
                    "n_overlap_shared": int(np.sum(np.asarray(deltas, dtype=float) < 0.1)),
                    "frac_overlap_specific": float(np.mean(np.asarray(deltas, dtype=float) >= 0.3)) if deltas else np.nan,
                    "mean_overlap_specificity_delta": float(np.nanmean(deltas)) if deltas else np.nan,
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["padj"] = np.nan
    for _, idx in result.groupby(["group", "library"]).groups.items():
        result.loc[idx, "padj"] = bh_adjust(result.loc[idx, "pvalue"])
    result["driven_by_shared_only"] = result["n_overlap_specific"].eq(0) & result["n_overlap_shared"].gt(0)
    return result.sort_values(["group", "library", "padj", "fold_enrichment"], ascending=[True, True, True, False])


def main() -> None:
    AUDIT_OUT.mkdir(parents=True, exist_ok=True)
    locked = pd.read_csv(LOCKED)
    full = pd.read_csv(FULL)
    top50 = pd.read_csv(TOP50_FILE, usecols=["program_id"])
    all_programs = sorted(top50["program_id"].astype(str).unique())
    universe = load_gene_spectra_universe(all_programs)
    ora = run_ora(locked, full, universe)

    ora_path = AUDIT_OUT / "MP_signature_ORA_long_with_GO_audit.csv"
    xlsx_path = AUDIT_OUT / "MP_signature_ORA_by_group_with_GO_audit.xlsx"
    counts_path = AUDIT_OUT / "MP_signature_ORA_library_counts_with_GO_audit.csv"
    manifest_path = AUDIT_OUT / "run_manifest.json"
    readme_path = AUDIT_OUT / "README.md"

    ora.to_csv(ora_path, index=False)
    if not ora.empty:
        with pd.ExcelWriter(xlsx_path) as writer:
            for group, sub in ora.groupby("group", sort=False):
                sub.head(50).to_excel(writer, sheet_name=group, index=False)
    counts = ora.groupby("library", as_index=False).size().rename(columns={"size": "n_rows"})
    counts.to_csv(counts_path, index=False)

    manifest = {
        "analysis": "final_mp1_8_ORA_with_GO_audit",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Audit rerun of MP1-MP8 ORA with corrected GO RDS parser. Original June 22 output contained HALLMARK and KEGG_LEGACY rows but no GO rows.",
        "inputs": {
            "locked_signatures": file_record(LOCKED),
            "gene_recurrence_full_table": file_record(FULL),
            "top50_gene_spectra_score": file_record(TOP50_FILE),
            "hallmark_gmt": file_record(HALLMARK_GMT),
            "kegg_gmt": file_record(KEGG_GMT),
            "go_bp_rds": file_record(GO_BP_RDS),
            "go_cc_rds": file_record(GO_CC_RDS),
            "rscript": file_record(RSCRIPT),
        },
        "term_counts_loaded": {
            "HALLMARK": int(sum(1 for item in parse_gmt(HALLMARK_GMT, "HALLMARK").values() if item["genes"])),
            "KEGG_LEGACY": int(sum(1 for item in parse_gmt(KEGG_GMT, "KEGG_LEGACY").values() if item["genes"])),
            "GO_BP": int(sum(1 for item in parse_go_rds(GO_BP_RDS, "GO_BP").values() if item["genes"])),
            "GO_CC": int(sum(1 for item in parse_go_rds(GO_CC_RDS, "GO_CC").values() if item["genes"])),
        },
        "library_rows_in_output": counts.to_dict(orient="records"),
        "outputs": [str(ora_path), str(xlsx_path), str(counts_path), str(manifest_path), str(readme_path)],
        "notes": [
            "This audit does not modify the June 22 MP recurrence outputs.",
            "GO RDS files are wrapper objects with pathways and term_names fields. The audit parser uses pathways as gene sets and term_names as readable labels.",
            "P-values are BH-adjusted separately within each MP and library, matching the original ORA grouping.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    readme_path.write_text(
        "\n".join(
            [
                "# final_mp1_8_ORA_with_GO_audit",
                "",
                "Audit rerun of MP1-MP8 over-representation analysis with a corrected parser for the GO BP and GO CC RDS files.",
                "",
                "The original June 22 ORA output contained HALLMARK and KEGG_LEGACY rows but no GO rows.",
                "This folder leaves the original output untouched and writes a separate supplement-facing audit table.",
                "",
                "Outputs",
                f"- {ora_path.name}",
                f"- {xlsx_path.name}",
                f"- {counts_path.name}",
                f"- {manifest_path.name}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
