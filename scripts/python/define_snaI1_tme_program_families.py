"""
Define SNAI1-ac tumor-spot program families from K* program annotations.

This is a deliberately small, thesis-question-oriented family system. It is
not a standalone HGSOC metaprogram atlas. The goal is to define defensible
families for downstream spatial association with SNAI1-ac while preserving all
evidence needed for review. The source cNMF was run on SpaCET Tumor spots, so
CAF/immune/vascular labels describe tumor-spot context or admixture programs,
not purified non-malignant-cell programs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


RUN_DIR = Path(r"D:\HGSOC_Spatial_Atlas\05_analysis_ready\20260424_definition3b_definition4_raw_geneNMF")
AUDIT_TABLES = RUN_DIR / "07_metaprogram_robustness_audit" / "tables"
ANNOTATION_XLSX = RUN_DIR / "final_program_annotation.xlsx"
ANNOTATION_SHEET = "kstar_program_annotations_v0_2_"
EVIDENCE_DIR = Path(r"C:\Users\luchu\Documents\MSc\Master Thesis\Code\HGSOC_Spatial_Atlas\00_documentation\kstar_evidence_v0_2")
OUT_DIR = RUN_DIR / "08_snaI1_tme_family_definitions"


FAMILY_DEFINITIONS = [
    {
        "family_id": "F01_ECM_MYCAF",
        "family_label": "ECM-remodelling myCAF",
        "analysis_role": "primary_tumor_spot_context",
        "include_primary_snaI1_tme": True,
        "description": "Tumor-spot ECM/myCAF context or admixture: collagen/MMP/TGF/contractile/stromal matrix variants.",
    },
    {
        "family_id": "F02_INFLAMMATORY_HYPOXIC_CAF_STRESS",
        "family_label": "Inflammatory/hypoxic CAF-stress",
        "analysis_role": "primary_tumor_spot_context",
        "include_primary_snaI1_tme": True,
        "description": "Tumor-spot inflammatory/hypoxic CAF-adjacent context: AP-1/IEG, iCAF-like or stress-adapted stromal signal.",
    },
    {
        "family_id": "F03_VASCULAR_ANGIO_PERICYTE",
        "family_label": "Angiogenic vascular/pericyte",
        "analysis_role": "primary_tumor_spot_context",
        "include_primary_snaI1_tme": True,
        "description": "Tumor-spot angiogenic/vascular context or admixture: endothelial, pericyte and vascular support signal.",
    },
    {
        "family_id": "F04_IFN_TLS_CHEMOKINE",
        "family_label": "IFN/TLS chemokine immune",
        "analysis_role": "primary_tumor_spot_context",
        "include_primary_snaI1_tme": True,
        "description": "Tumor-spot IFN/chemokine/lymphoid context: type I/II IFN, CXCL9/10/11/13, antigen-processing and TLS-like signal.",
    },
    {
        "family_id": "F05_APC_TAM_MYELOID",
        "family_label": "APC/TAM myeloid",
        "analysis_role": "primary_tumor_spot_context",
        "include_primary_snaI1_tme": True,
        "description": "Tumor-spot APC/TAM/myeloid context or admixture: antigen-presenting, phagocytic, LAM-like or tolerogenic macrophage signal.",
    },
    {
        "family_id": "F06_PLASMA_BCELL_IG",
        "family_label": "Plasma/B-cell immunoglobulin",
        "analysis_role": "primary_tumor_spot_context",
        "include_primary_snaI1_tme": True,
        "description": "Tumor-spot plasma/B-cell context or admixture: immunoglobulin-secreting and humoral immune signal.",
    },
    {
        "family_id": "F07_CILIATED_EPITHELIAL",
        "family_label": "Ciliated epithelial context",
        "analysis_role": "context_primary_or_sensitivity",
        "include_primary_snaI1_tme": True,
        "description": "Motile ciliated epithelial identity/context programs; not stromal but spatially relevant.",
    },
    {
        "family_id": "F08_ACUTE_PHASE_NEUTROPHIL_EPITHELIAL",
        "family_label": "Acute-phase/neutrophil epithelial inflammation",
        "analysis_role": "context_primary_or_sensitivity",
        "include_primary_snaI1_tme": True,
        "description": "Acute-phase, antimicrobial, neutrophil-recruiting inflammatory epithelial programs.",
    },
    {
        "family_id": "F09_MALIGNANT_HYPOXIA_STRESS",
        "family_label": "Malignant hypoxia/stress",
        "analysis_role": "primary_tumor_intrinsic",
        "include_primary_snaI1_tme": True,
        "description": "Tumor-intrinsic hypoxia, glycolysis, ER/autophagy/AP-1 stress and survival programs.",
    },
    {
        "family_id": "F10_MALIGNANT_OXPHOS_METABOLIC",
        "family_label": "Malignant OXPHOS/metabolic",
        "analysis_role": "primary_tumor_intrinsic",
        "include_primary_snaI1_tme": True,
        "description": "Tumor-intrinsic OXPHOS, metabolic, proteasome/spliceosome and biosynthetic variants.",
    },
    {
        "family_id": "F11_MALIGNANT_PROLIFERATION_BIOSYNTHESIS",
        "family_label": "Malignant proliferation/biosynthesis",
        "analysis_role": "primary_tumor_intrinsic",
        "include_primary_snaI1_tme": True,
        "description": "G2/M, G1/S, MYC, ribosome-biogenesis and proliferative malignant programs.",
    },
    {
        "family_id": "F12_MALIGNANT_EMT_INTERFACE_SECRETORY",
        "family_label": "Malignant EMT/interface/secretory",
        "analysis_role": "primary_tumor_intrinsic",
        "include_primary_snaI1_tme": True,
        "description": "Partial EMT, invasion-front, reactive epithelial, secretory/luminal and interface malignant programs.",
    },
    {
        "family_id": "F13_MESOTHELIAL_ADIPOSE_CONTEXT",
        "family_label": "Mesothelial/adipose/peritoneal context",
        "analysis_role": "context_sensitivity",
        "include_primary_snaI1_tme": False,
        "description": "Mesothelial, omental/adipose/resting stromal or peritoneal context programs.",
    },
    {
        "family_id": "F90_TECHNICAL_LOW_QUALITY",
        "family_label": "Technical/low-quality",
        "analysis_role": "exclude_qc",
        "include_primary_snaI1_tme": False,
        "description": "MT/ribosomal/hemoglobin/lncRNA or low-quality mixed technical programs.",
    },
    {
        "family_id": "F99_UNRESOLVED_TAIL",
        "family_label": "Unresolved/sample-specific tail",
        "analysis_role": "exclude_or_descriptive_tail",
        "include_primary_snaI1_tme": False,
        "description": "Programs not confidently assigned to a biologically coherent family for primary SNAI1-ac analysis.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Define SNAI1-ac TME program families.")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    return parser.parse_args()


def norm(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.lower().replace("–", "-").replace("—", "-").split())


def split_genes(value: object, limit: int | None = None) -> list[str]:
    if not isinstance(value, str) or not value:
        return []
    genes = [x.strip() for x in value.split(";") if x.strip()]
    return genes if limit is None else genes[:limit]


def coalesce_duplicate_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Prefer manual-review values, then fill from draft evidence duplicates."""
    for column in columns:
        candidates = [c for c in [column, f"{column}_x", f"{column}_y"] if c in df.columns]
        if not candidates:
            continue
        series = df[candidates[0]]
        for candidate in candidates[1:]:
            series = series.combine_first(df[candidate])
        df[column] = series
        drop_cols = [c for c in candidates if c != column]
        if drop_cols:
            df = df.drop(columns=drop_cols)
    return df


def load_program_table() -> pd.DataFrame:
    manual = pd.read_excel(ANNOTATION_XLSX, sheet_name=ANNOTATION_SHEET)
    manual["program_id"] = manual["program_id"].astype(str)
    draft = pd.read_csv(EVIDENCE_DIR / "kstar_program_annotations_v0_2_draft.csv")
    draft["program_id"] = draft["program_id"].astype(str)
    top = pd.read_csv(EVIDENCE_DIR / "kstar_top_gene_summary.csv")
    top["program_id"] = top["program_id"].astype(str)
    df = manual.merge(
        draft[
            [
                "program_id",
                "dataset",
                "sample_id_on_disk",
                "biological_label_draft",
                "candidate_alignment_category",
                "confidence_tier_draft",
                "use_tier_draft",
                "technical_review_status",
                "best_signature_id",
                "best_signature_alignment_category",
                "highest_mean_spacet_compartment",
                "top_cell_fraction_associations",
                "top_functional_terms_v0_2",
                "evidence_summary",
            ]
        ],
        on="program_id",
        how="left",
    ).merge(
        top[
            [
                "program_id",
                "top20_genes",
                "top50_genes",
                "top100_genes",
                "top50_ribosomal_fraction",
                "top50_mitochondrial_fraction",
                "top50_hemoglobin_fraction",
                "top50_malat1_fraction",
                "top50_heatshock_fraction",
                "technical_flag",
            ]
        ],
        on="program_id",
        how="left",
        suffixes=("", "_top"),
    )
    components_path = AUDIT_TABLES / "robust_backbone_components_ge_0_75_members_for_review.csv"
    if components_path.exists():
        comps = pd.read_csv(components_path)[["program_id", "component_id"]].drop_duplicates()
        df = df.merge(comps, on="program_id", how="left")
    else:
        df["component_id"] = ""
    df = coalesce_duplicate_columns(
        df,
        [
            "top10_genes",
            "top_cell_fraction_associations",
            "top_functional_terms_v0_2",
        ],
    )
    return df


def is_technical(row: pd.Series, text: str) -> bool:
    review = norm(row.get("technical_review_status", ""))
    use_tier = norm(row.get("use_tier_draft", ""))
    ribo = pd.to_numeric(row.get("top50_ribosomal_fraction"), errors="coerce")
    mito = pd.to_numeric(row.get("top50_mitochondrial_fraction"), errors="coerce")
    hemo = pd.to_numeric(row.get("top50_hemoglobin_fraction"), errors="coerce")
    return bool(
        "low-quality" in text
        or "low quality" in text
        or "technical_mitochondrial_respiration" in text
        or "mitochondrial/respiration technical" in text
        or "ribosomal only" in text
        or "translation-dominated technical" in text
        or "do_not_interpret" in use_tier
        or ("technical_flag_consistent" in review)
        or (pd.notna(ribo) and ribo >= 0.50)
        or (pd.notna(mito) and mito >= 0.35)
        or (pd.notna(hemo) and hemo >= 0.10)
    )


def assign_family(row: pd.Series) -> tuple[str, str]:
    program_id = str(row.get("program_id", ""))
    label_pieces = [
        row.get("alignment_category_draft", ""),
        row.get("program_identity_draft", ""),
        row.get("biological_label_draft", ""),
        row.get("candidate_alignment_category", ""),
        row.get("best_signature_alignment_category", ""),
        row.get("top_functional_terms_v0_2", ""),
    ]
    text = norm(" ".join(str(x) for x in label_pieces if pd.notna(x)))
    alignment = norm(row.get("alignment_category_draft", ""))
    identity = norm(row.get("program_identity_draft", ""))
    candidate = norm(row.get("candidate_alignment_category", ""))
    label_core = norm(" ".join([alignment, identity, candidate]))
    gene_tokens = {g.upper() for g in split_genes(row.get("top20_genes", ""), 20)}
    plasma_markers = {"IGKC", "IGHG1", "IGHG3", "IGHA1", "IGHG4", "MZB1", "JCHAIN", "IGLC2", "IGLC3"}
    ciliated_markers = {"TPPP3", "CAPS", "C9ORF24", "CFAP73", "FAM183A", "C20ORF85", "CCDC17"}

    if program_id == "ju_2024__CPS_OV5LtOV4__K8__P4":
        return (
            "F09_MALIGNANT_HYPOXIA_STRESS",
            "manual correction: tumor-compartment Gavish hypoxia/glycolytic stress program",
        )
    if program_id == "yamamoto_2025__Pt1-2__K6__P4":
        return (
            "F05_APC_TAM_MYELOID",
            "manual correction: M2-like TAM/myeloid program with macrophage markers and Olbrecht CCL18 signature",
        )
    if program_id == "yamamoto_2025__Pt2-2__K7__P3":
        return (
            "F90_TECHNICAL_LOW_QUALITY",
            "manual correction: unresolved low-coherence mixed program excluded from biological families",
        )
    if program_id == "ju_2024__CPS_OV34RtOV1__K5__P3":
        return (
            "F01_ECM_MYCAF",
            "manual correction: dissolved F06; reassigned to ECM-remodelling myCAF",
        )
    if program_id in {"ju_2024__CPS_OV19_LtOV1__K10__P10", "denisenko_2022__SP2__K5__P5"}:
        return (
            "F05_APC_TAM_MYELOID",
            "manual correction: dissolved F06; folded immunoglobulin/plasma-adjacent program into APC/TAM myeloid family",
        )
    if program_id == "yamamoto_2025__Pt1-2__K6__P5":
        return (
            "F09_MALIGNANT_HYPOXIA_STRESS",
            "manual correction: reassigned to malignant hypoxia/stress",
        )
    if program_id == "yamamoto_2025__Pt2-3__K5__P4":
        return (
            "F09_MALIGNANT_HYPOXIA_STRESS",
            "manual correction: reassigned to malignant hypoxia/stress",
        )
    if program_id in {"ju_2024__CPS_OV71_1__K4__P2", "yamamoto_2025__Pt2-2__K7__P6"}:
        return (
            "F09_MALIGNANT_HYPOXIA_STRESS",
            "manual correction: dissolved F08; folded acute-phase/neutrophil epithelial program into malignant hypoxia/stress",
        )
    if program_id == "yamamoto_2025__Pt2-1__K5__P1":
        return (
            "F12_MALIGNANT_EMT_INTERFACE_SECRETORY",
            "manual correction: tumor epithelial polarity/secretory-interface program removed from low-quality family",
        )
    if program_id == "yamamoto_2025__Pt2-4__K5__P3":
        return (
            "F11_MALIGNANT_PROLIFERATION_BIOSYNTHESIS",
            "manual correction: MYC/translation/ribosome-biogenesis malignant biosynthesis program removed from low-quality family",
        )
    if program_id in {"denisenko_2022__SP4__K6__P1", "ju_2024__CPS_OV5LtOV4__K8__P1"}:
        return (
            "F10_MALIGNANT_OXPHOS_METABOLIC",
            "manual correction: OXPHOS/splicing/metabolic malignant program assigned after individual review",
        )
    if program_id in {"yamamoto_2025__Pt2-1__K5__P2", "denisenko_2022__SP3__K7__P4"}:
        return (
            "F11_MALIGNANT_PROLIFERATION_BIOSYNTHESIS",
            "manual correction: G2/M-E2F proliferative malignant program assigned after individual review",
        )

    if any(x in label_core for x in ["ciliated", "motile cilium", "motile ciliated"]) or len(gene_tokens & ciliated_markers) >= 2:
        return "F07_CILIATED_EPITHELIAL", "ciliated epithelial context"
    if "plasma cell" in alignment or "plasma-cell" in identity or "plasma-cell" in candidate or len(gene_tokens & plasma_markers) >= 2:
        return "F06_PLASMA_BCELL_IG", "plasma/B-cell/immunoglobulin evidence"
    if is_technical(row, text):
        return "F90_TECHNICAL_LOW_QUALITY", "technical flag, low-quality annotation, or high MT/ribosomal/hemoglobin fraction"

    if any(x in label_core for x in ["angiogenesis", "angiogenic", "endothelial", "pericyte", "vascular", "tip cell", "smooth muscle"]):
        if "mycaf" in alignment or "caf" in alignment or "collagen-rich mycaf" in alignment:
            return "F01_ECM_MYCAF", "ECM/myCAF with angiogenic/vascular admixture"
        return "F03_VASCULAR_ANGIO_PERICYTE", "angiogenesis/endothelial/pericyte evidence"
    if any(x in label_core for x in ["mycaf", "caf", "collagen", "ecm", "matrix", "mmp", "loxl", "prrx", "snai2", "tgf", "contractile", "serpine1", "postn"]):
        if any(x in alignment for x in ["hypoxic icaf", "ap-1", "ieg", "inflammatory hypoxic", "stress-activated", "ccn1", "junb", "egr1", "cxcl1", "cxcl2", "cxcl3", "il6"]):
            return "F02_INFLAMMATORY_HYPOXIC_CAF_STRESS", "inflammatory/hypoxic/stress CAF-adjacent evidence"
        return "F01_ECM_MYCAF", "ECM/myCAF evidence"
    if any(x in label_core for x in ["tam", "macrophage", "myeloid", "antigen-presenting", "phagocytic", "lam-like"]):
        return "F05_APC_TAM_MYELOID", "TAM/APC/myeloid annotation"
    if any(x in label_core for x in ["ifn", "interferon", "cxcl9", "cxcl10", "cxcl11", "cxcl13", "tls", "lymphoid", "t-cell", "t cell", "chemokine", "b-cell", "b cell"]):
        return "F04_IFN_TLS_CHEMOKINE", "IFN/chemokine/TLS/lymphoid evidence"
    if any(x in label_core for x in ["acute-phase", "neutrophil", "antimicrobial", "granulocyte"]):
        return "F08_ACUTE_PHASE_NEUTROPHIL_EPITHELIAL", "acute-phase/neutrophil inflammatory epithelial evidence"
    if any(x in label_core for x in ["partial-emt", "emt", "invasion-front", "reactive epithelial", "secretory", "luminal", "muc1", "wfcd2", "wfdc2"]):
        return "F12_MALIGNANT_EMT_INTERFACE_SECRETORY", "malignant epithelial interface/EMT/secretory evidence"
    if any(x in label_core for x in ["hypoxia", "hif", "glycolytic", "autophagy", "er stress", "stress response", "ap1", "ap-1"]):
        return "F09_MALIGNANT_HYPOXIA_STRESS", "malignant hypoxia/stress evidence"
    if any(x in label_core for x in ["oxphos", "metabolic", "metabolism", "proteasome", "spliceosome", "splicing", "mesothelial-associated"]):
        return "F10_MALIGNANT_OXPHOS_METABOLIC", "malignant OXPHOS/metabolic evidence"
    if any(x in label_core for x in ["g2/m", "g1/s", "proliferation", "cell cycle", "myc", "biosynthesis", "ribosome biogenesis"]):
        return "F11_MALIGNANT_PROLIFERATION_BIOSYNTHESIS", "malignant proliferation/biosynthesis evidence"
    if any(x in label_core for x in ["mesothelial", "omentum", "adipose", "resting stromal", "peritoneal"]):
        return "F13_MESOTHELIAL_ADIPOSE_CONTEXT", "mesothelial/adipose/peritoneal context evidence"
    if "unresolved" in text or "mixed" in text:
        return "F99_UNRESOLVED_TAIL", "unresolved/mixed annotation"
    return "F99_UNRESOLVED_TAIL", "no confident primary family assignment"


def family_gene_sets(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family_id, group in assignments.groupby("family_id", sort=True):
        if family_id.startswith("F90") or family_id.startswith("F99"):
            continue
        gene_counter = Counter()
        rank_sum = defaultdict(float)
        n_programs = len(group)
        for row in group.itertuples(index=False):
            genes = split_genes(getattr(row, "top100_genes", ""), 100)
            for rank, gene in enumerate(genes, start=1):
                gene_counter[gene] += 1
                rank_sum[gene] += rank
        gene_rows = []
        for gene, count in gene_counter.items():
            gene_rows.append(
                {
                    "family_id": family_id,
                    "gene": gene,
                    "n_programs_with_gene": count,
                    "program_fraction": count / n_programs if n_programs else 0,
                    "mean_rank_when_present": rank_sum[gene] / count,
                    "consensus_score": count / (rank_sum[gene] / count),
                }
            )
        ranked = pd.DataFrame(gene_rows).sort_values(
            ["n_programs_with_gene", "consensus_score", "mean_rank_when_present"],
            ascending=[False, False, True],
        )
        ranked["gene_rank_in_family"] = range(1, len(ranked) + 1)
        rows.append(ranked)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    programs = load_program_table()
    assigned = programs.copy()
    assignments = assigned.apply(assign_family, axis=1, result_type="expand")
    assigned["family_id"] = assignments[0]
    assigned["family_assignment_reason"] = assignments[1]
    family_df = pd.DataFrame(FAMILY_DEFINITIONS)
    assigned = assigned.merge(family_df, on="family_id", how="left")
    assigned["sample_label"] = assigned["dataset"].astype(str) + "__" + assigned["sample_id_on_disk"].astype(str)

    keep_cols = [
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
        "alignment_category_draft",
        "program_identity_draft",
        "candidate_alignment_category",
        "biological_label_draft",
        "best_signature_id",
        "best_signature_alignment_category",
        "confidence_tier_draft",
        "use_tier_draft",
        "technical_review_status",
        "technical_flag",
        "top10_genes",
        "top20_genes",
        "top50_genes",
        "top100_genes",
        "top_functional_terms_v0_2",
        "top_cell_fraction_associations",
        "highest_mean_spacet_compartment",
        "top50_ribosomal_fraction",
        "top50_mitochondrial_fraction",
        "top50_hemoglobin_fraction",
        "manual_family_override",
        "manual_notes",
    ]
    assigned["manual_family_override"] = ""
    assigned["manual_notes"] = ""
    assigned[keep_cols].sort_values(["family_id", "dataset", "sample_id_on_disk", "program_id"]).to_csv(
        out_dir / "program_to_snaI1_tme_family_v1.csv", index=False
    )

    summary = (
        assigned.groupby(["family_id", "family_label", "analysis_role", "include_primary_snaI1_tme"], dropna=False)
        .agg(
            n_programs=("program_id", "size"),
            n_samples=("sample_label", "nunique"),
            n_datasets=("dataset", "nunique"),
            n_robust_backbone_components=("component_id", lambda s: s.dropna().nunique()),
            n_technical_flagged=("technical_flag", lambda s: int(s.astype(str).str.lower().eq("true").sum())),
            example_programs=("program_id", lambda s: ";".join(list(s.astype(str).head(6)))),
        )
        .reset_index()
        .sort_values(["analysis_role", "family_id"])
    )
    family_df.merge(summary, on=["family_id", "family_label", "analysis_role", "include_primary_snaI1_tme"], how="left").to_csv(
        out_dir / "snaI1_tme_family_definitions_v1.csv", index=False
    )

    genes = family_gene_sets(assigned.loc[assigned["analysis_role"].ne("exclude_qc")].copy())
    genes.to_csv(out_dir / "snaI1_tme_family_consensus_genes_v1.csv", index=False)
    wide_gene_sets = (
        genes.loc[genes["gene_rank_in_family"].le(100)]
        .groupby("family_id")["gene"]
        .apply(lambda s: ";".join(s.astype(str)))
        .reset_index(name="top100_consensus_genes")
    )
    wide_gene_sets.to_csv(out_dir / "snaI1_tme_family_top100_gene_sets_v1.csv", index=False)

    summary.to_csv(out_dir / "snaI1_tme_family_assignment_summary_v1.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
