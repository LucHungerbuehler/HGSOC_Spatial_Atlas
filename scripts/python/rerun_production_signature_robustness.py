"""Re-run SNAI1-ac signature sensitivities from the production 109-gene set.

This focused audit avoids reconstructing the signature from the bulk workbook.
All variants are strict subsets of the production weights file and are evaluated
in the 23 Visium sections used in the thesis Results chapter.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import enrichmap as em
import libpysal
import numpy as np
import pandas as pd
import scanpy as sc
from esda.moran import Moran
from scipy.stats import pearsonr, spearmanr


BASE_DIR = Path(r"D:\HGSOC_Spatial_Atlas")
ANALYSIS_DIR = BASE_DIR / "05_analysis_ready"
SIGNATURE_DIR = ANALYSIS_DIR / "Signature"
ROBUSTNESS_DIR = SIGNATURE_DIR / "robustness"
OUTPUT_DIR = ROBUSTNESS_DIR / "production_109_rerun"
TABLES_DIR = OUTPUT_DIR / "tables"

SIGNATURE_FILE = SIGNATURE_DIR / "snai1_acetylation_signature_full.csv"
WEIGHTS_FILE = SIGNATURE_DIR / "snai1_ac_weights.json"
COHORT_FILE = ANALYSIS_DIR / "cross_sample" / "compiled" / "all_distribution_stats.csv"
HALLMARK_MORAN_FILE = ANALYSIS_DIR / "cross_sample" / "compiled" / "all_hallmark_morans_I.csv"
CACHE_DIR = ROBUSTNESS_DIR / "h5ad_cache"
LEGACY_TABLES_DIR = ROBUSTNESS_DIR / "tables"

GENE_COL = "Gene"
GENETYPE_COL = "GeneType"
FC_COL = "PEO4-2R_lg2fc (SNAI1-SNAI1)"
PRODUCTION_SCORE_COL = "SNAI1-ac_score"
MORAN_K = 6
REPORT_DATASETS = {
    "visium/denisenko_2022",
    "visium/yamamoto_2025",
    "visium/ju_2024",
}

NULL_MODELS = {
    "smoothed_pc_uniform": (
        "enrichmap_null_summary_per_sample.csv",
        "enrichmap_null_comparison.csv",
    ),
    "smoothed_all_detected_uniform": (
        "enrichmap_null_all_detected_uniform_no_stur_summary_per_sample.csv",
        "enrichmap_null_all_detected_uniform_no_stur_comparison.csv",
    ),
    "smoothed_pc_matched_signed": (
        "enrichmap_null_pc_matched_signed_no_stur_summary_per_sample.csv",
        "enrichmap_null_pc_matched_signed_no_stur_comparison.csv",
    ),
    "unsmoothed_all_detected_uniform": (
        "enrichmap_unsmoothed_null_all_detected_uniform_no_stur_summary_per_sample.csv",
        "enrichmap_unsmoothed_null_all_detected_uniform_no_stur_comparison.csv",
    ),
    "unsmoothed_pc_matched_signed": (
        "enrichmap_unsmoothed_null_pc_matched_signed_no_stur_summary_per_sample.csv",
        "enrichmap_unsmoothed_null_pc_matched_signed_no_stur_comparison.csv",
    ),
}


def load_production_signature() -> tuple[pd.DataFrame, dict[str, float]]:
    signature = pd.read_csv(SIGNATURE_FILE, sep=";")
    signature[GENE_COL] = signature[GENE_COL].astype(str).str.strip()
    signature[GENETYPE_COL] = signature[GENETYPE_COL].astype(str).str.strip().str.lower()
    signature[FC_COL] = pd.to_numeric(signature[FC_COL], errors="raise")

    with WEIGHTS_FILE.open("r", encoding="utf-8") as handle:
        weights = {str(gene): float(weight) for gene, weight in json.load(handle).items()}

    if len(weights) != 109:
        raise ValueError(f"Expected 109 production weights, found {len(weights)}")
    if signature[GENE_COL].duplicated().any():
        raise ValueError("Production signature export contains duplicate gene symbols")

    signature_genes = set(signature[GENE_COL])
    weight_genes = set(weights)
    if signature_genes != weight_genes:
        missing_export = sorted(weight_genes - signature_genes)
        missing_weights = sorted(signature_genes - weight_genes)
        raise ValueError(
            "Production signature and weights disagree: "
            f"missing from export={missing_export}; missing from weights={missing_weights}"
        )

    signature["production_weight"] = signature[GENE_COL].map(weights)
    return signature, weights


def build_variants(
    signature: pd.DataFrame,
    production_weights: dict[str, float],
) -> tuple[dict[str, dict[str, object]], pd.DataFrame]:
    protein_coding = signature.loc[
        signature[GENETYPE_COL].eq("protein_coding")
    ].copy()
    if len(protein_coding) != 70:
        raise ValueError(f"Expected 70 protein-coding production genes, found {len(protein_coding)}")

    wt_higher = protein_coding.loc[protein_coding[FC_COL] < 0].copy()
    ranked_pc = protein_coding.assign(abs_log2fc=protein_coding[FC_COL].abs()).sort_values(
        ["abs_log2fc", GENE_COL], ascending=[False, True], kind="mergesort"
    )
    top10_genes = ranked_pc.head(10)[GENE_COL].tolist()
    drop_top10 = protein_coding.loc[~protein_coding[GENE_COL].isin(top10_genes)].copy()

    subsets = {
        "production_pc70": protein_coding,
        "production_pc_wt_higher": wt_higher,
        "production_pc70_drop_top10": drop_top10,
    }
    score_keys = {
        "production_pc70": "SNAI1_ac_production_pc70",
        "production_pc_wt_higher": "SNAI1_ac_production_pc_wt_higher",
        "production_pc70_drop_top10": "SNAI1_ac_production_pc70_drop_top10",
    }

    variants: dict[str, dict[str, object]] = {}
    membership = signature[[GENE_COL, GENETYPE_COL, FC_COL, "production_weight"]].copy()
    membership["in_production_109"] = True
    membership["removed_in_pc_top10_test"] = membership[GENE_COL].isin(top10_genes)

    for variant_id, subset in subsets.items():
        genes = subset[GENE_COL].tolist()
        variant_weights = {gene: production_weights[gene] for gene in genes}
        variants[variant_id] = {
            "genes": genes,
            "weights": variant_weights,
            "score_key": score_keys[variant_id],
            "score_col": f"{score_keys[variant_id]}_score",
        }
        membership[f"in_{variant_id}"] = membership[GENE_COL].isin(genes)

    membership["absolute_log2fc_rank_among_pc70"] = np.nan
    rank_map = {gene: rank for rank, gene in enumerate(ranked_pc[GENE_COL], start=1)}
    pc_mask = membership[GENE_COL].isin(rank_map)
    membership.loc[pc_mask, "absolute_log2fc_rank_among_pc70"] = (
        membership.loc[pc_mask, GENE_COL].map(rank_map)
    )
    return variants, membership


def load_report_cohort() -> pd.DataFrame:
    cohort = pd.read_csv(COHORT_FILE, usecols=["dataset", "sample"])
    cohort = cohort.loc[cohort["dataset"].isin(REPORT_DATASETS)].drop_duplicates().copy()
    cohort = cohort.sort_values(["dataset", "sample"]).reset_index(drop=True)
    if len(cohort) != 23:
        raise ValueError(f"Expected 23 report sections, found {len(cohort)}")
    return cohort


def cache_path(dataset: str, sample: str) -> Path:
    return CACHE_DIR.joinpath(*dataset.split("/"), f"{sample}.h5ad")


def make_spatial_weights(coords: np.ndarray):
    weights = libpysal.weights.KNN.from_array(coords, k=MORAN_K)
    weights.transform = "R"
    return weights


def finite_correlation(x: np.ndarray, y: np.ndarray, method: str) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return float("nan")
    if method == "spearman":
        return float(spearmanr(x[mask], y[mask]).statistic)
    return float(pearsonr(x[mask], y[mask]).statistic)


def score_report_cohort(
    cohort: pd.DataFrame,
    variants: dict[str, dict[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for record in cohort.itertuples(index=False):
        path = cache_path(record.dataset, record.sample)
        if not path.exists():
            raise FileNotFoundError(f"Missing robustness cache: {path}")

        print(f"Scoring {record.dataset}/{record.sample}", flush=True)
        adata = sc.read_h5ad(path)
        if PRODUCTION_SCORE_COL not in adata.obs:
            raise KeyError(f"{PRODUCTION_SCORE_COL} missing from {path}")
        if "spatial" not in adata.obsm:
            raise KeyError(f"Spatial coordinates missing from {path}")

        production_score = adata.obs[PRODUCTION_SCORE_COL].to_numpy(dtype=float)
        spatial_weights = make_spatial_weights(np.asarray(adata.obsm["spatial"]))
        production_moran = float(Moran(production_score, spatial_weights, permutations=0).I)

        for variant_id, variant in variants.items():
            detected_weights = {
                gene: weight
                for gene, weight in variant["weights"].items()
                if gene in adata.var_names
            }
            if len(detected_weights) < 5:
                raise ValueError(
                    f"Only {len(detected_weights)} genes detected for {variant_id} in "
                    f"{record.dataset}/{record.sample}"
                )

            score_key = str(variant["score_key"])
            score_col = str(variant["score_col"])
            if score_col in adata.obs:
                del adata.obs[score_col]

            em.tl.score(
                adata=adata,
                gene_set=list(detected_weights),
                gene_weights={score_key: detected_weights},
                score_key=score_key,
                smoothing=True,
                correct_spatial_covariates=True,
            )
            variant_score = adata.obs[score_col].to_numpy(dtype=float)
            variant_moran = float(Moran(variant_score, spatial_weights, permutations=0).I)
            rows.append(
                {
                    "dataset": record.dataset,
                    "sample": record.sample,
                    "variant_id": variant_id,
                    "n_variant_genes": len(variant["genes"]),
                    "n_genes_detected": len(detected_weights),
                    "spearman_vs_production109": finite_correlation(
                        production_score, variant_score, "spearman"
                    ),
                    "pearson_vs_production109": finite_correlation(
                        production_score, variant_score, "pearson"
                    ),
                    "production109_morans_I": production_moran,
                    "variant_morans_I": variant_moran,
                    "variant_minus_production_morans_I": variant_moran - production_moran,
                }
            )

        del adata

    return pd.DataFrame(rows)


def summarize(per_section: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant_id, group in per_section.groupby("variant_id", sort=False):
        rows.append(
            {
                "variant_id": variant_id,
                "n_sections": len(group),
                "n_variant_genes": int(group["n_variant_genes"].iloc[0]),
                "min_genes_detected": int(group["n_genes_detected"].min()),
                "max_genes_detected": int(group["n_genes_detected"].max()),
                "median_spearman_vs_production109": float(
                    group["spearman_vs_production109"].median()
                ),
                "min_spearman_vs_production109": float(
                    group["spearman_vs_production109"].min()
                ),
                "max_spearman_vs_production109": float(
                    group["spearman_vs_production109"].max()
                ),
                "mean_production109_morans_I": float(group["production109_morans_I"].mean()),
                "mean_variant_morans_I": float(group["variant_morans_I"].mean()),
                "median_absolute_morans_I_difference": float(
                    group["variant_minus_production_morans_I"].abs().median()
                ),
                "max_absolute_morans_I_difference": float(
                    group["variant_minus_production_morans_I"].abs().max()
                ),
            }
        )
    return pd.DataFrame(rows)


def cohort_filter(data: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    keys = cohort[["dataset", "sample"]].drop_duplicates()
    return data.merge(keys, on=["dataset", "sample"], how="inner", validate="many_to_one")


def export_supporting_tables(cohort: pd.DataFrame) -> dict[str, pd.DataFrame]:
    distribution = cohort_filter(pd.read_csv(COHORT_FILE), cohort)
    detected = cohort_filter(
        pd.read_csv(LEGACY_TABLES_DIR / NULL_MODELS["smoothed_pc_uniform"][0]),
        cohort,
    )[["dataset", "sample", "n_real_genes_detected"]]
    section_metrics = distribution.merge(
        detected, on=["dataset", "sample"], how="left", validate="one_to_one"
    )
    section_metrics.to_csv(TABLES_DIR / "production_score_metrics_23_sections.csv", index=False)

    null_section_frames = []
    null_draw_frames = []
    null_summary_rows = []
    for model_id, (summary_name, draws_name) in NULL_MODELS.items():
        section = cohort_filter(pd.read_csv(LEGACY_TABLES_DIR / summary_name), cohort)
        section.insert(2, "null_model", model_id)
        if len(section) != 23:
            raise ValueError(f"Expected 23 rows for {model_id}, found {len(section)}")
        null_section_frames.append(section)

        draws = cohort_filter(pd.read_csv(LEGACY_TABLES_DIR / draws_name), cohort)
        draws.insert(2, "null_model", model_id)
        if len(draws) != 2300:
            raise ValueError(f"Expected 2,300 null draws for {model_id}, found {len(draws)}")
        null_draw_frames.append(draws)

        null_summary_rows.append(
            {
                "null_model": model_id,
                "n_sections": len(section),
                "n_null_draws_per_section": int(section["n_null_iterations"].min()),
                "mean_real_morans_I": float(section["real_morans_I"].mean()),
                "mean_section_null_mean_morans_I": float(section["null_mean_morans_I"].mean()),
                "mean_real_minus_null_mean_morans_I": float(
                    section["real_minus_null_mean_morans_I"].mean()
                ),
                "median_real_minus_null_mean_morans_I": float(
                    section["real_minus_null_mean_morans_I"].median()
                ),
                "median_real_moran_percentile": float(section["real_moran_percentile"].median()),
                "n_sections_at_or_above_95th_percentile": int(
                    (section["real_moran_percentile"] >= 0.95).sum()
                ),
                "n_sections_empirical_p_lt_0_05": int(
                    (section["null_empirical_p_morans"] < 0.05).sum()
                ),
                "n_sections_real_above_null_mean": int(
                    (section["real_minus_null_mean_morans_I"] > 0).sum()
                ),
            }
        )

    null_per_section = pd.concat(null_section_frames, ignore_index=True)
    null_draws = pd.concat(null_draw_frames, ignore_index=True)
    null_summary = pd.DataFrame(null_summary_rows)
    null_per_section.to_csv(TABLES_DIR / "null_model_results_23_sections.csv", index=False)
    null_draws.to_csv(TABLES_DIR / "null_model_draws_23_sections.csv", index=False)
    null_summary.to_csv(TABLES_DIR / "null_model_cohort_summary_23_sections.csv", index=False)

    hallmark_morans = cohort_filter(pd.read_csv(HALLMARK_MORAN_FILE), cohort)
    production_morans = section_metrics[["dataset", "sample", "morans_I"]].rename(
        columns={"morans_I": "production109_morans_I"}
    )
    hallmark_morans = hallmark_morans.merge(
        production_morans, on=["dataset", "sample"], how="left", validate="many_to_one"
    )
    hallmark_morans["production109_exceeds_hallmark"] = (
        hallmark_morans["production109_morans_I"] > hallmark_morans["morans_I"]
    )
    hallmark_morans.to_csv(TABLES_DIR / "hallmark_morans_23_sections.csv", index=False)

    hallmark_reference = (
        hallmark_morans.groupby(["dataset", "sample"], as_index=False)
        .agg(
            production109_morans_I=("production109_morans_I", "first"),
            n_hallmark_scores=("pathway", "size"),
            n_hallmarks_below_production=("production109_exceeds_hallmark", "sum"),
        )
    )
    if not hallmark_reference["n_hallmark_scores"].eq(50).all():
        raise ValueError("Each report section must have exactly 50 Hallmark Moran values")
    hallmark_reference["fraction_hallmarks_below_production"] = (
        hallmark_reference["n_hallmarks_below_production"]
        / hallmark_reference["n_hallmark_scores"]
    )
    hallmark_reference.to_csv(TABLES_DIR / "hallmark_reference_23_sections.csv", index=False)

    return {
        "section_metrics": section_metrics,
        "null_per_section": null_per_section,
        "null_draws": null_draws,
        "null_summary": null_summary,
        "hallmark_morans": hallmark_morans,
        "hallmark_reference": hallmark_reference,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run clean production-signature variants and package the 23-section evidence."
    )
    parser.add_argument(
        "--summaries-only",
        action="store_true",
        help="Reuse existing clean variant results and rebuild only supporting tables/workbook.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    signature, production_weights = load_production_signature()
    variants, membership = build_variants(signature, production_weights)
    cohort = load_report_cohort()

    definition_rows = [
        {
            "variant_id": "production_109",
            "n_genes": len(signature),
            "definition": "authoritative production SNAI1-ac weights",
        }
    ]
    definitions = {
        "production_pc70": "protein-coding members of production 109",
        "production_pc_wt_higher": (
            "protein-coding production genes with negative 2R-versus-WT log2FC"
        ),
        "production_pc70_drop_top10": (
            "production protein-coding set after removing ten largest absolute bulk log2FC values"
        ),
    }
    for variant_id, variant in variants.items():
        definition_rows.append(
            {
                "variant_id": variant_id,
                "n_genes": len(variant["genes"]),
                "definition": definitions[variant_id],
            }
        )

    definitions_table = pd.DataFrame(definition_rows)
    per_section_path = TABLES_DIR / "production_signature_variant_results_per_section.csv"
    if args.summaries_only:
        per_section = pd.read_csv(per_section_path)
    else:
        per_section = score_report_cohort(cohort, variants)
    cohort_summary = summarize(per_section)
    export_supporting_tables(cohort)

    definitions_table.to_csv(
        TABLES_DIR / "production_signature_variant_definitions.csv", index=False
    )
    membership.to_csv(TABLES_DIR / "production_signature_variant_membership.csv", index=False)
    cohort.to_csv(TABLES_DIR / "report_cohort_23_sections.csv", index=False)
    per_section.to_csv(TABLES_DIR / "production_signature_variant_results_per_section.csv", index=False)
    cohort_summary.to_csv(
        TABLES_DIR / "production_signature_variant_results_cohort_summary.csv", index=False
    )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "production_signature_file": str(SIGNATURE_FILE),
        "production_weights_file": str(WEIGHTS_FILE),
        "cohort_source_file": str(COHORT_FILE),
        "n_production_genes": len(signature),
        "n_production_protein_coding_genes": int(
            signature[GENETYPE_COL].eq("protein_coding").sum()
        ),
        "n_report_sections": len(cohort),
        "moran_k": MORAN_K,
        "scoring": {
            "smoothing": True,
            "correct_spatial_covariates": True,
            "reference_score_column": PRODUCTION_SCORE_COL,
        },
        "variant_gene_counts": {
            variant_id: len(variant["genes"]) for variant_id, variant in variants.items()
        },
        "supplementary_workbook": str(OUTPUT_DIR / "snai1ac_spatial_robustness_data.xlsx"),
        "null_models": list(NULL_MODELS),
    }
    with (OUTPUT_DIR / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(cohort_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
