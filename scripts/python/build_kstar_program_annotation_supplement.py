"""Build a merged K* cNMF program annotation supplement workbook.

The workbook is report-facing and uses the manually reviewed annotation
workbook as the authoritative source for alignment_category_draft. It then
joins family assignments, top-gene/QC summaries, top signature provenance,
K-selection provenance, and top-100 gene loadings.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pandas as pd


TABLES = Path(
    r"D:\HGSOC_Spatial_Atlas\05_analysis_ready"
    r"\20260424_definition3b_definition4_raw_geneNMF"
    r"\11_research_synthesis\tables"
)
DOCS = Path(r"C:\Users\luchu\Documents\MSc\Master Thesis\Code\HGSOC_Spatial_Atlas\00_documentation")
CNMF = Path(r"D:\HGSOC_Spatial_Atlas\05_analysis_ready\S3_cNMF_Tumor_Programs")

AUTHORITATIVE_XLSX = TABLES / "kstar_program_annotations_v0_2_authoritative.xlsx"
FAMILY_SNAPSHOT = TABLES / "program_family_annotation_snapshot.csv"
TOP_GENE_SUMMARY = DOCS / "kstar_evidence_v0_2" / "kstar_top_gene_summary.csv"
EVIDENCE_FIRST = (
    DOCS
    / "kstar_annotations_v0_2_evidence_first"
    / "kstar_program_annotations_v0_2_evidence_first_draft.csv"
)
SIGNATURE_COMPENDIUM = DOCS / "kstar_signature_compendium_v0_2.csv"
SOURCE_REGISTRY = DOCS / "Kstar_signature_source_registry.csv"
RANKINGS = DOCS / "kstar_evidence_v0_2" / "kstar_rankings_v0_2.csv"
MANUAL_K = CNMF / "k_selection" / "parsed_manual_k_decisions.csv"
K_REVIEW = CNMF / "k_selection" / "k_selection_review_summary.csv"
K_VALIDATION = CNMF / "k_selection" / "manual_k_validation_report.csv"

OUTPUT = TABLES / "kstar_program_annotation_evidence_supplement_v0_1.xlsx"

ILLEGAL_EXCEL_CHARS = re.compile(r"[\000-\010]|[\013-\014]|[\016-\037]")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def clean_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """Remove control characters that openpyxl refuses to write."""
    out = df.copy()
    object_cols = out.select_dtypes(include=["object"]).columns
    for col in object_cols:
        out[col] = out[col].map(
            lambda x: ILLEGAL_EXCEL_CHARS.sub("", x) if isinstance(x, str) else x
        )
    return out


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def split_genes(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [g.strip() for g in str(value).split(";") if g.strip()]


def build_top100_long(top_gene_summary: pd.DataFrame) -> pd.DataFrame:
    top100_rows: list[dict[str, object]] = []
    for _, row in top_gene_summary.iterrows():
        for rank, gene in enumerate(split_genes(row["top100_genes"]), start=1):
            top100_rows.append(
                {
                    "program_id": row["program_id"],
                    "gene": gene,
                    "top100_rank": rank,
                }
            )
    top100_keys = pd.DataFrame(top100_rows)

    chunks = pd.read_csv(
        RANKINGS,
        encoding="utf-8-sig",
        chunksize=500_000,
        usecols=[
            "dataset",
            "sample_id_on_disk",
            "sample_label",
            "program_id",
            "source_k",
            "local_program_index",
            "gene",
            "loading",
            "is_mitochondrial",
            "is_ribosomal",
            "is_hemoglobin",
        ],
    )

    matched_chunks: list[pd.DataFrame] = []
    for chunk in chunks:
        matched = chunk.merge(top100_keys, on=["program_id", "gene"], how="inner")
        if not matched.empty:
            matched_chunks.append(matched)

    if not matched_chunks:
        raise RuntimeError("No top-100 genes matched the rankings table.")

    top100_long = pd.concat(matched_chunks, ignore_index=True)
    top100_long = top100_long.sort_values(["program_id", "top100_rank"]).reset_index(drop=True)
    return top100_long


def compact_top100_loadings(top100_long: pd.DataFrame) -> pd.DataFrame:
    def collapse(group: pd.DataFrame) -> str:
        ordered = group.sort_values("top100_rank")
        return ";".join(
            f"{int(row.top100_rank)}:{row.gene}={float(row.loading):.6g}"
            for row in ordered.itertuples(index=False)
        )

    return (
        top100_long.groupby("program_id", as_index=False)
        .apply(lambda g: collapse(g), include_groups=False)
        .rename(columns={None: "top100_gene_loadings"})
    )


def add_signature_provenance(main: pd.DataFrame, compendium: pd.DataFrame) -> pd.DataFrame:
    sig_cols = [
        "signature_id",
        "signature_name",
        "source_id",
        "citation_key",
        "full_citation",
        "doi_or_url",
        "source_table_or_figure",
        "database_version_or_release",
        "signature_class",
        "compartment",
        "biological_family",
        "alignment_category",
        "evidence_tier",
        "gene_derivation_method",
        "genes",
        "review_status",
    ]
    comp = compendium[sig_cols].copy()
    for n, id_col in [
        (1, "top_overall_signature_id"),
        (2, "top_overall_signature_2_id"),
        (3, "top_overall_signature_3_id"),
    ]:
        prefix = f"top_signature_{n}_"
        renamed = comp.rename(columns={c: f"{prefix}{c}" for c in sig_cols})
        main = main.merge(
            renamed,
            left_on=id_col,
            right_on=f"{prefix}signature_id",
            how="left",
        )
    return main


def main() -> None:
    for path in [
        AUTHORITATIVE_XLSX,
        FAMILY_SNAPSHOT,
        TOP_GENE_SUMMARY,
        EVIDENCE_FIRST,
        SIGNATURE_COMPENDIUM,
        SOURCE_REGISTRY,
        RANKINGS,
        MANUAL_K,
        K_REVIEW,
        K_VALIDATION,
    ]:
        require_file(path)

    authoritative = pd.read_excel(
        AUTHORITATIVE_XLSX,
        sheet_name="kstar_program_annotations_v0_2_",
        engine="openpyxl",
    )
    family = read_csv(FAMILY_SNAPSHOT)
    top_genes = read_csv(TOP_GENE_SUMMARY)
    evidence = read_csv(EVIDENCE_FIRST)
    compendium = read_csv(SIGNATURE_COMPENDIUM)
    source_registry = read_csv(SOURCE_REGISTRY)
    manual_k = read_csv(MANUAL_K)
    k_review = read_csv(K_REVIEW)
    k_validation = read_csv(K_VALIDATION)

    if authoritative["program_id"].duplicated().any():
        raise RuntimeError("Authoritative annotation workbook contains duplicated program_id values.")

    main_df = authoritative.copy()

    family_cols = [
        "program_id",
        "dataset",
        "sample_id_on_disk",
        "sample_label",
        "component_id",
        "family_id",
        "family_label",
        "analysis_role",
        "include_primary_snaI1_tme",
        "family_assignment_reason",
        "program_identity_draft",
        "candidate_alignment_category",
        "biological_label_draft",
        "best_signature_id",
        "best_signature_alignment_category",
        "highest_mean_spacet_compartment",
        "manual_family_override",
        "manual_notes",
        "include_primary_snaI1_tme_bool",
        "primary_family_bool",
    ]
    main_df = main_df.merge(family[family_cols], on="program_id", how="left")

    top_gene_cols = [
        "program_id",
        "source_k",
        "local_program_index",
        "top20_genes",
        "top50_genes",
        "top100_genes",
        "top20_ribosomal_fraction",
        "top50_ribosomal_fraction",
        "top100_ribosomal_fraction",
        "top20_mitochondrial_fraction",
        "top50_mitochondrial_fraction",
        "top100_mitochondrial_fraction",
        "top50_hemoglobin_fraction",
        "top50_malat1_fraction",
        "top50_heatshock_fraction",
    ]
    main_df = main_df.merge(top_genes[top_gene_cols], on="program_id", how="left")

    evidence_cols = [
        "program_id",
        "primary_evidence_domains",
        "primary_score",
        "runner_up_category",
        "runner_up_score",
        "review_reasons",
        "top_overall_signature_relation_to_identity",
        "top_signature_hits_v0_2",
        "evidence_summary",
    ]
    main_df = main_df.merge(evidence[evidence_cols], on="program_id", how="left")

    k_cols = [
        "sample_label",
        "k_star",
        "k_window_values",
        "rationale_short",
        "tumor_spots_primary",
        "tumor_spots_high_purity",
        "k_window_n",
        "manual_k_source_path",
        "manual_k_source_format",
    ]
    main_df = main_df.merge(manual_k[k_cols], on="sample_label", how="left")

    k_review_cols = [
        "sample_label",
        "k_best_silhouette",
        "max_silhouette",
        "k_smallest_within_95pct_max_silhouette",
        "k_smallest_within_98pct_max_silhouette",
        "k_error_elbow_log_curve",
        "k_min_prediction_error",
        "prediction_error_drop_pct_k4_to_k12",
        "review_shortlist",
        "plot_filename",
        "template_filename",
    ]
    main_df = main_df.merge(k_review[k_review_cols], on="sample_label", how="left")

    main_df["cnmf_candidate_k_range"] = "4-12"
    main_df["cnmf_n_iter_per_k"] = 100
    main_df["cnmf_numgenes"] = 2000
    main_df["cnmf_seed"] = 42
    main_df["cnmf_consensus_local_density_threshold"] = 0.5
    main_df["cnmf_consensus_local_neighborhood_size"] = 0.3

    main_df = add_signature_provenance(main_df, compendium)

    top100_long = build_top100_long(top_genes)
    top100_compact = compact_top100_loadings(top100_long)
    main_df = main_df.merge(top100_compact, on="program_id", how="left")

    family_alignment_audit = authoritative[["program_id", "alignment_category_draft"]].merge(
        family[["program_id", "alignment_category_draft"]].rename(
            columns={"alignment_category_draft": "family_snapshot_alignment_category_draft"}
        ),
        on="program_id",
        how="left",
    )
    family_alignment_audit["matches_authoritative"] = (
        family_alignment_audit["alignment_category_draft"].fillna("")
        == family_alignment_audit["family_snapshot_alignment_category_draft"].fillna("")
    )

    doi_rows = [
        {
            "citation_key": "Kotliar2019",
            "source_id": "cnmf_method",
            "source_name": "Consensus NMF method",
            "doi_or_url": "https://doi.org/10.1101/gr.244749.118",
            "role_in_supplement": "cNMF decomposition method",
        }
    ]
    for _, row in source_registry.iterrows():
        doi_rows.append(
            {
                "citation_key": row.get("citation_key", ""),
                "source_id": row.get("source_id", ""),
                "source_name": row.get("source_name", ""),
                "doi_or_url": row.get("doi_or_url", ""),
                "role_in_supplement": row.get("admissible_use", ""),
            }
        )
    doi_checklist = pd.DataFrame(doi_rows)

    readme = pd.DataFrame(
        [
            ("generated_on", datetime.now().isoformat(timespec="seconds")),
            ("output_file", str(OUTPUT)),
            ("authoritative_annotation_workbook", str(AUTHORITATIVE_XLSX)),
            ("family_assignment_snapshot", str(FAMILY_SNAPSHOT)),
            ("top_gene_summary", str(TOP_GENE_SUMMARY)),
            ("evidence_first_annotation_table", str(EVIDENCE_FIRST)),
            ("signature_compendium", str(SIGNATURE_COMPENDIUM)),
            ("signature_source_registry", str(SOURCE_REGISTRY)),
            ("full_gene_level_rankings_source", str(RANKINGS)),
            ("manual_k_decisions", str(MANUAL_K)),
            ("k_selection_review_summary", str(K_REVIEW)),
            ("k_validation_report", str(K_VALIDATION)),
            ("n_programs", int(main_df["program_id"].nunique())),
            ("n_rows_main_sheet", len(main_df)),
            ("n_top100_loading_rows", len(top100_long)),
            ("expected_top100_loading_rows", int(main_df["program_id"].nunique()) * 100),
            (
                "full_spectrum_note",
                "The full gene-level ranked spectra are too large for a practical Excel sheet. "
                "This workbook embeds top-100 gene loadings per program and records the full "
                "ranking source path for complete spectrum provenance.",
            ),
            (
                "authoritative_annotation_note",
                "alignment_category_draft is taken from the manually reviewed authoritative "
                "workbook copied from Mappe1.xlsx.",
            ),
        ],
        columns=["field", "value"],
    )

    merge_qc = pd.DataFrame(
        [
            ("main_rows", len(main_df)),
            ("unique_program_ids_main", main_df["program_id"].nunique()),
            ("authoritative_rows", len(authoritative)),
            ("family_snapshot_rows", len(family)),
            ("top_gene_summary_rows", len(top_genes)),
            ("evidence_first_rows", len(evidence)),
            ("signature_compendium_rows", len(compendium)),
            ("source_registry_rows", len(source_registry)),
            ("manual_k_rows", len(manual_k)),
            ("k_review_rows", len(k_review)),
            ("k_validation_rows", len(k_validation)),
            (
                "alignment_category_mismatches_vs_family_snapshot",
                int((~family_alignment_audit["matches_authoritative"]).sum()),
            ),
            ("programs_missing_top100_loadings", int(main_df["top100_gene_loadings"].isna().sum())),
        ],
        columns=["check", "value"],
    )

    sheet_map = {
        "README": readme,
        "program_annotation_evidence": main_df,
        "top100_gene_loadings": top100_long,
        "signature_compendium": compendium,
        "source_registry": source_registry,
        "citation_doi_checklist": doi_checklist,
        "k_manual_decisions": manual_k,
        "k_selection_review": k_review,
        "k_validation": k_validation,
        "alignment_audit": family_alignment_audit,
        "merge_qc": merge_qc,
    }

    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        for sheet_name, df in sheet_map.items():
            clean_for_excel(df).to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Wrote {OUTPUT}")
    print(merge_qc.to_string(index=False))


if __name__ == "__main__":
    main()
