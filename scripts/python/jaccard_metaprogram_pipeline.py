from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch, Rectangle
from scipy.cluster.hierarchy import cophenet, leaves_list, linkage
from scipy.stats import binomtest, hypergeom, norm
from scipy.spatial.distance import squareform


SEED = 0
ROOT = Path(r"D:\HGSOC_Spatial_Atlas")
CNMF_ROOT = ROOT / "05_analysis_ready" / "S3_cNMF_Tumor_Programs"
CNMF_RUNS = CNMF_ROOT / "cnmf_runs"
MATRIX_DIR = CNMF_ROOT / "jaccard_raw_matrices"
POS_NEG_DIR = MATRIX_DIR / "pos_neg"
NEG_DIR = MATRIX_DIR / "neg"
INSPECT_DIR = MATRIX_DIR / "inspection_exports_average"
MANUAL_DIR = INSPECT_DIR / "variantB_nonjunk_manual_cut_v2"
ASSIGNMENT_FILE = MANUAL_DIR / "program_cluster_assignment_v2_variantB_nonjunk_manual.csv"
MANUAL_HEATMAP_FILE = MANUAL_DIR / "variantB_nonjunk_manual_cut_v2_heatmap.png"
MANUAL_RHO_ORDERED_HEATMAP_FILE = MANUAL_DIR / "variantB_nonjunk_manual_cut_v2_heatmap_ordered_by_snai1ac_rho.png"
MANUAL_RHO_ORDERED_TABLE = MANUAL_DIR / "variantB_nonjunk_manual_cut_v2_ordered_by_snai1ac_rho.csv"
COARSE_RECLUSTER_DIR = MANUAL_DIR / "coarse_recluster_inspection"
FAMILY_RECLUSTER_DIR = MANUAL_DIR / "family_recluster"
FAMILY_SNAPSHOT_FILE = (
    ROOT
    / "05_analysis_ready"
    / "20260424_definition3b_definition4_raw_geneNMF"
    / "11_research_synthesis"
    / "tables"
    / "program_family_annotation_snapshot.csv"
)
FAMILY_RECLUSTER_LABELS = [
    "Malignant EMT/interface/secretory",
    "Malignant proliferation/biosynthesis",
    "IFN/TLS chemokine immune",
    "Malignant hypoxia/stress",
    "Malignant OXPHOS/metabolic",
]
ANNOTATION_XLSX = ROOT / "05_analysis_ready" / "20260424_definition3b_definition4_raw_geneNMF" / "final_program_annotation.xlsx"
ANNOTATION_SHEET = "kstar_program_annotations_v0_2_"
REFERENCE_FILE = (
    ROOT
    / "05_analysis_ready"
    / "20260424_definition3b_definition4_raw_geneNMF"
    / "08_snaI1_tme_family_definitions"
    / "program_to_snaI1_tme_family_v1.csv"
)
PROGRAMME_SNAI1AC_CORR_FILE = (
    ROOT
    / "05_analysis_ready"
    / "20260424_definition3b_definition4_raw_geneNMF"
    / "06_summary"
    / "programme_snai1ac_correlations_all_samples.csv"
)

VARIANT_A_MATRIX = MATRIX_DIR / "jaccard_variantA_cohortintersect.csv"
VARIANT_B_MATRIX = MATRIX_DIR / "jaccard_variantB_keep_everything.csv"
JUNK_FILE = INSPECT_DIR / "objective_junk_programs_original_top50_ribo20_mito20.csv"

OUT = MANUAL_DIR / "recurrence_specificity_diagnostics"
REPRO_OUT = MANUAL_DIR / "reproducibility_core_exports"
USAGE_OUT = MANUAL_DIR / "native_usage_hh_meta"
LOADING_DIAGNOSTIC_DIR = MANUAL_DIR / "program_loading_elbow_diagnostics"
REPORT_FILE = OUT / "metaprogram_pipeline_report.md"
FINAL_LABELS_FILE = OUT / "MP_final_labels.csv"
LABELLED_CLUSTER_SUMMARY_FILE = OUT / "MP_cluster_summary_labelled.csv"
MSIGDB_DIR = Path(
    r"C:\Users\luchu\Documents\MSc\Master Thesis\Code\HGSOC_Spatial_Atlas"
    r"\00_documentation\kstar_sources_v0_2\msigdb_2025_1_Hs"
)
HALLMARK_GMT = MSIGDB_DIR / "h.all.v2025.1.Hs.symbols.gmt"
KEGG_LEGACY_GMT = MSIGDB_DIR / "c2.cp.kegg_legacy.v2025.1.Hs.symbols.gmt"
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
HH_USAGE_TABLE_DIR = (
    ROOT
    / "05_analysis_ready"
    / "20260424_definition3b_definition4_raw_geneNMF"
    / "08_hh_programme_characterization"
    / "tables"
)
HH_SPOT_CONTRASTS = HH_USAGE_TABLE_DIR / "hh_spot_level_programme_contrasts.csv"
HH_NEIGHBORHOOD_CONTRASTS = HH_USAGE_TABLE_DIR / "hh_neighborhood_programme_contrasts.csv"
MP_TRUE_DIR = MATRIX_DIR / "mp_score_true_enrichment"
MP_TRUE_TABLE_DIR = MP_TRUE_DIR / "tables"
MP_TRUE_FIG_DIR = MP_TRUE_DIR / "figures"
MP_TRUE_MANIFEST = MP_TRUE_DIR / "mp_score_true_enrichment_manifest.json"
MP_TRUE_META = MP_TRUE_TABLE_DIR / "matched_hh_nonhh_mp_meta_analysis.csv"

PROGRAM_RE = re.compile(r"^(denisenko_2022|ju_2024|yamamoto_2025)__(.+)__K(\d+)__P(\d+)$")
RIBO_PREFIXES = ("RPL", "RPS")
MITO_PREFIX = "MT-"
PSEUDOCOUNT = 1e-6
STRICT_FRACTION = 1 / 3
STRICT_CAP = 50
RELAXED_ORA_SIZE = 20

MANUAL_RANGES = {
    "A": range(11, 47),
    "B": range(49, 74),
    "C": range(74, 125),
    "1": range(12, 22),
    "2": range(23, 47),
    "3": range(49, 63),
    "4": range(63, 74),
    "C1": range(76, 96),
    "6": range(98, 103),
    "C2": range(103, 125),
    "5": range(76, 83),
    "7": range(104, 119),
}

FINAL_LABEL_ROWS = [
    {
        "group": "MP1",
        "final_label": "Endothelial/vascular (+pericyte)",
        "label_type": "functional",
        "confidence": "firm",
        "key_evidence": "GO_BP blood vessel / vasculature development",
        "anchor_genes": "CLEC14A,CD93,VWF,PECAM1,CDH5,ENG,CLDN5,PDGFRB,RGS5",
        "caution": "basement-membrane ECM (COL4A1/A2,LAMB1,NID1,HSPG2) is vascular, not fibrillar — distinct from MP2",
    },
    {
        "group": "MP2",
        "final_label": "ECM-remodelling myCAF",
        "label_type": "functional",
        "confidence": "firm",
        "key_evidence": "GO_CC collagen-containing extracellular matrix",
        "anchor_genes": "COL1A1,COL1A2,COL3A1,DCN,LUM,FN1,MMP2,POSTN,VCAN,BGN",
        "caution": "HALLMARK_EMT (top term, FE=94) is matrisome/stromal NOT epithelial EMT — do not read as tumour EMT",
    },
    {
        "group": "MP3",
        "final_label": "Antigen-presenting TAM / myeloid",
        "label_type": "functional",
        "confidence": "firm",
        "key_evidence": "GO_BP antigen processing & presentation via MHC-II (FE=283)",
        "anchor_genes": "HLA-DRA,HLA-DPA1,HLA-DQA1,CD74,CTSS,IFI30,B2M,C1QB,CD68,LYZ,TYROBP",
        "caution": "APC call carried by MHC-II lineage markers",
    },
    {
        "group": "MP4",
        "final_label": "Type-I/II IFN response (ISG)",
        "label_type": "functional",
        "confidence": "firm",
        "key_evidence": "HALLMARK_INTERFERON_GAMMA_RESPONSE / INTERFERON_ALPHA_RESPONSE",
        "anchor_genes": "ISG15,OAS1,OAS2,OAS3,MX1,MX2,IFIT1,IFIT3,RSAD2,STAT1,IRF7",
        "caution": "—",
    },
    {
        "group": "MP5",
        "final_label": "Hypoxia / glycolysis",
        "label_type": "functional",
        "confidence": "firm",
        "key_evidence": "HALLMARK_HYPOXIA / GO_BP response to hypoxia",
        "anchor_genes": "SLC2A1,VEGFA,NDRG1,EGLN3,ADM,ENO1,ENO2,HK2,PGK1,CA9",
        "caution": "—",
    },
    {
        "group": "MP6",
        "final_label": "Proliferation / cell-cycle",
        "label_type": "functional",
        "confidence": "firm biology, provisional robustness",
        "key_evidence": "HALLMARK_E2F_TARGETS / G2M_CHECKPOINT / chromosome segregation",
        "anchor_genes": "UBE2C,TPX2,CDC20,MYBL2,TOP2A,MKI67",
        "caution": "provisional flag is small-n robustness, not biology",
    },
    {
        "group": "C1",
        "final_label": "Stressed/secretory malignant (hypoxic + acute-phase)",
        "label_type": "functional",
        "confidence": "firm",
        "key_evidence": "functional terms (hypoxia + acute-phase secretory + TNFA/IFN), NOT the top generic extracellular-localization terms",
        "anchor_genes": "S100A10,LCN2,SLPI,SAA1,NDRG1,SLC2A1,VEGFA,WFDC2",
        "caution": "top ORA terms are large-K extracellular-localization noise; broader stress compartment than MP5",
    },
    {
        "group": "MP7",
        "final_label": "Tumour lineage identity (residual, non-stressed)",
        "label_type": "descriptive",
        "confidence": "provisional, low-content",
        "key_evidence": "weak ORA (best padj ~5e-4); top hits are WT1/PAX8-driven nephron-development GO artefact",
        "anchor_genes": "MSLN,MUC16,WT1,PAX8,FOLR1,LAMA5",
        "caution": "NOT a functional programme; role = non-stressed contrast to MP5/C1",
    },
    {
        "group": "C2",
        "final_label": "Broad malignant epithelium, no coherent programme",
        "label_type": "descriptive",
        "confidence": "provisional, low-content",
        "key_evidence": "only generic extracellular-localization terms, frac_overlap_specific=0.00",
        "anchor_genes": "MSLN,MUC16",
        "caution": "nothing specific to label",
    },
]


def spectra_path(program_id: str) -> tuple[Path, int]:
    match = PROGRAM_RE.match(program_id)
    if not match:
        raise ValueError(f"Cannot parse program_id: {program_id}")
    dataset, sample, k_value, p_value = match.group(1), match.group(2), int(match.group(3)), int(match.group(4))
    sample_label = f"{dataset}__{sample}"
    return CNMF_RUNS / sample_label / f"{sample_label}.gene_spectra_score.k_{k_value}.dt_0_5.txt", p_value


def spectra_tpm_path(program_id: str) -> tuple[Path, int]:
    match = PROGRAM_RE.match(program_id)
    if not match:
        raise ValueError(f"Cannot parse program_id: {program_id}")
    dataset, sample, k_value, p_value = match.group(1), match.group(2), int(match.group(3)), int(match.group(4))
    sample_label = f"{dataset}__{sample}"
    return CNMF_RUNS / sample_label / f"{sample_label}.gene_spectra_tpm.k_{k_value}.dt_0_5.txt", p_value


def load_program_scores(program_ids: list[str], top_n: int = 100) -> tuple[dict[str, list[str]], dict[str, dict[str, float]]]:
    cache: dict[Path, pd.DataFrame] = {}
    top_genes: dict[str, list[str]] = {}
    top_scores: dict[str, dict[str, float]] = {}
    for program_id in program_ids:
        path, p_value = spectra_path(program_id)
        if path not in cache:
            df = pd.read_csv(path, sep="\t", index_col=0)
            df.index = [int(str(idx).replace("GEP", "").replace(".0", "")) for idx in df.index]
            cache[path] = df
        scores = cache[path].loc[p_value].astype(float).sort_values(ascending=False)
        genes = [str(gene) for gene in scores.index[:top_n]]
        top_genes[program_id] = genes
        top_scores[program_id] = {str(gene): float(score) for gene, score in scores.iloc[:top_n].items()}
    return top_genes, top_scores


def load_gene_spectra_universe(program_ids: list[str]) -> set[str]:
    cache: dict[Path, pd.DataFrame] = {}
    universe: set[str] = set()
    for program_id in program_ids:
        path, _ = spectra_path(program_id)
        if path not in cache:
            df = pd.read_csv(path, sep="\t", index_col=0, nrows=1)
            cache[path] = df
        universe.update(str(gene) for gene in cache[path].columns)
    return universe


def jaccard_matrix(program_ids: list[str], top_genes: dict[str, list[str]], n: int = 50) -> pd.DataFrame:
    sets = {pid: set(top_genes[pid][:n]) for pid in program_ids}
    matrix = np.zeros((len(program_ids), len(program_ids)), dtype=float)
    for i, pid_i in enumerate(program_ids):
        for j, pid_j in enumerate(program_ids):
            union = sets[pid_i] | sets[pid_j]
            matrix[i, j] = len(sets[pid_i] & sets[pid_j]) / len(union) if union else 0.0
    return pd.DataFrame(matrix, index=program_ids, columns=program_ids)


def jaccard_matrix_from_feature_sets(feature_sets: dict[str, set[str]]) -> pd.DataFrame:
    program_ids = list(feature_sets)
    matrix = np.zeros((len(program_ids), len(program_ids)), dtype=float)
    for i, pid_i in enumerate(program_ids):
        for j, pid_j in enumerate(program_ids):
            union = feature_sets[pid_i] | feature_sets[pid_j]
            matrix[i, j] = len(feature_sets[pid_i] & feature_sets[pid_j]) / len(union) if union else 0.0
    return pd.DataFrame(matrix, index=program_ids, columns=program_ids)


def average_linkage_from_jaccard(matrix: pd.DataFrame) -> tuple[np.ndarray, list[str], float]:
    dist = 1 - matrix.to_numpy()
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    z = linkage(condensed, method="average")
    coph_corr, _ = cophenet(z, condensed)
    order = matrix.index[leaves_list(z)].tolist()
    return z, order, float(coph_corr)


def linkage_for_visible_dendrogram(z: np.ndarray) -> np.ndarray:
    """Display-only transform: shorten terminal stems without changing leaf order."""
    plot_z = z.copy()
    if plot_z.size:
        plot_z[:, 2] = plot_z[:, 2] - float(np.nanmin(plot_z[:, 2]))
    return plot_z


def content_junk(top_genes: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for program_id, genes in top_genes.items():
        genes_50 = genes[:50]
        ribo = sum(gene.startswith(RIBO_PREFIXES) for gene in genes_50)
        mito = sum(gene.startswith(MITO_PREFIX) for gene in genes_50)
        rows.append(
            {
                "program_id": program_id,
                "ribosomal_fraction": ribo / len(genes_50),
                "mito_fraction": mito / len(genes_50),
                "objective_junk": ribo / len(genes_50) >= 0.20 or mito / len(genes_50) >= 0.20,
            }
        )
    return pd.DataFrame(rows)


def manual_cut_from_position(position: int) -> tuple[str, str, str, str]:
    coarse = "unassigned"
    if position in MANUAL_RANGES["A"]:
        coarse = "A"
    elif position in MANUAL_RANGES["B"]:
        coarse = "B"
    elif position in MANUAL_RANGES["C"]:
        coarse = "C"

    mid = "none"
    for label in ["1", "2", "3", "4", "C1", "6", "C2"]:
        if position in MANUAL_RANGES[label]:
            mid = label
            break

    fine = "none"
    if position in MANUAL_RANGES["5"]:
        fine = "5"
    elif position in MANUAL_RANGES["7"]:
        fine = "7"

    if coarse == "unassigned":
        path = "unassigned"
    elif coarse == "A" and mid in {"1", "2"}:
        path = f"A/{mid}"
    elif coarse == "B" and mid in {"3", "4"}:
        path = f"B/{mid}"
    elif coarse == "C" and mid == "C1" and fine == "5":
        path = "C/C1/5"
    elif coarse == "C" and mid == "C1":
        path = "C/C1"
    elif coarse == "C" and mid == "6":
        path = "C/6"
    elif coarse == "C" and mid == "C2" and fine == "7":
        path = "C/C2/7"
    elif coarse == "C" and mid == "C2":
        path = "C/C2"
    else:
        path = coarse
    return coarse, mid, fine, path


def fine_from_path(path: str) -> str:
    return {
        "A/1": "1",
        "A/2": "2",
        "B/3": "3",
        "B/4": "4",
        "C/C1/5": "5",
        "C/6": "6",
        "C/C2/7": "7",
    }.get(str(path), "none")


def read_family_snapshot() -> pd.DataFrame:
    snapshot = pd.read_csv(FAMILY_SNAPSHOT_FILE)
    snapshot = snapshot.drop_duplicates("program_id").copy()
    snapshot["family_label"] = snapshot["family_label"].fillna("missing").astype(str)
    return snapshot


def family_label_map_from_snapshot() -> dict[str, str]:
    snapshot = read_family_snapshot()
    return snapshot.set_index("program_id")["family_label"].to_dict()


def family_palette_from_snapshot() -> dict[str, str]:
    snapshot = read_family_snapshot()
    labels = sorted(set(snapshot["family_label"]) | {"missing"})
    colors = dict(zip(labels, sns.color_palette("tab20", n_colors=len(labels)).as_hex(), strict=False))
    colors["missing"] = "#D9D9D9"
    return colors


def c1c2_from_path(path: str) -> str:
    path = str(path)
    if path.startswith("C/C1"):
        return "C1"
    if path.startswith("C/C2"):
        return "C2"
    return "other"


def read_manual_cut_annotation_columns() -> pd.DataFrame:
    approved_cols = ["program_id", "alignment_category_draft", "top_cell_fraction_associations"]
    if ANNOTATION_XLSX.exists():
        annot = pd.read_excel(ANNOTATION_XLSX, sheet_name=ANNOTATION_SHEET)
    else:
        annot = pd.read_csv(REFERENCE_FILE)
    missing = [col for col in approved_cols if col not in annot.columns]
    if missing:
        raise KeyError(f"Missing approved annotation columns: {missing}")
    return annot[approved_cols].drop_duplicates("program_id")


def variant_b_nonjunk_matrix_and_order() -> tuple[pd.DataFrame, np.ndarray, list[str], float]:
    matrix = pd.read_csv(VARIANT_B_MATRIX, index_col=0)
    junk = set(pd.read_csv(JUNK_FILE)["program_id"].astype(str))
    nonjunk = [program_id for program_id in matrix.index.astype(str) if program_id not in junk]
    matrix = matrix.loc[nonjunk, nonjunk]
    z, order, coph = average_linkage_from_jaccard(matrix)
    return matrix, z, order, coph


def recreate_manual_cut_assignment(order: list[str]) -> pd.DataFrame:
    annot = read_manual_cut_annotation_columns()
    rows = []
    for position, program_id in enumerate(order, start=1):
        coarse, mid, fine, path = manual_cut_from_position(position)
        rows.append(
            {
                "position": position,
                "program_id": program_id,
                "cluster_coarse": coarse,
                "cluster_mid": mid,
                "cluster_fine": fine,
                "cluster_path": path,
            }
        )
    assignment = pd.DataFrame(rows)
    assignment = assignment.merge(annot, on="program_id", how="left")
    assignment = assignment[
        [
            "position",
            "program_id",
            "alignment_category_draft",
            "top_cell_fraction_associations",
            "cluster_coarse",
            "cluster_mid",
            "cluster_fine",
            "cluster_path",
        ]
    ]
    return assignment


def draw_manual_cut_heatmap(matrix: pd.DataFrame, z: np.ndarray, assignment: pd.DataFrame, output_path: Path) -> None:
    ref_map = family_label_map_from_snapshot()
    corr = pd.read_csv(PROGRAMME_SNAI1AC_CORR_FILE, usecols=["program_id", "spearman_rho"]).drop_duplicates("program_id")
    rho_map = corr.set_index("program_id")["spearman_rho"].astype(float).to_dict()
    assignment = assignment.set_index("program_id").reindex(matrix.index)
    assignment.index.name = "program_id"
    assignment = assignment.reset_index()

    coarse_palette = {"A": "#4C78A8", "B": "#54A24B", "C": "#E45756", "unassigned": "#B9B9B9"}
    fine_palette = {
        "1": "#8DD3C7",
        "2": "#FFFFB3",
        "3": "#BEBADA",
        "4": "#FB8072",
        "5": "#80B1D3",
        "6": "#FDB462",
        "7": "#B3DE69",
        "none": "#D9D9D9",
    }
    family_labels = sorted({ref_map.get(pid, "missing") for pid in matrix.index})
    family_colors = family_palette_from_snapshot()
    rho_norm = mcolors.TwoSlopeNorm(vmin=-0.4, vcenter=0.0, vmax=0.4)
    rho_cmap = plt.get_cmap("vlag")

    def rho_to_color(program_id: str) -> str:
        rho = rho_map.get(program_id)
        if pd.isna(rho):
            return "#D9D9D9"
        return mcolors.to_hex(rho_cmap(rho_norm(float(np.clip(rho, -0.4, 0.4)))))

    row_colors = pd.DataFrame(
        {
            "coarse": assignment["cluster_coarse"].map(coarse_palette).fillna("#D9D9D9").to_numpy(),
            "fine 1-7": assignment["cluster_path"].map(fine_from_path).map(fine_palette).fillna("#D9D9D9").to_numpy(),
        },
        index=assignment["program_id"],
    )
    col_colors = pd.DataFrame(
        {
            "reference family": [family_colors.get(ref_map.get(pid, "missing"), "#D9D9D9") for pid in matrix.index],
            "SNAI1-ac rho": [rho_to_color(pid) for pid in matrix.index],
        },
        index=matrix.index,
    )

    plot_matrix = matrix.copy()
    np.fill_diagonal(plot_matrix.values, np.nan)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    sns.set_theme(style="white")
    plot_z = linkage_for_visible_dendrogram(z)
    g = sns.clustermap(
        plot_matrix,
        row_linkage=plot_z,
        col_linkage=plot_z,
        row_colors=row_colors,
        col_colors=col_colors,
        cmap=cmap,
        vmin=0,
        vmax=0.2,
        xticklabels=False,
        yticklabels=False,
        linewidths=0,
        figsize=(12.5, 12.5),
        dendrogram_ratio=(0.18, 0.18),
        tree_kws={"linewidths": 0.8},
        cbar_kws={"label": "Jaccard similarity"},
    )
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")
    g.ax_col_colors.set_title("reference family / SNAI1-ac rho", fontsize=9, pad=6)

    for label, positions in {"C1": MANUAL_RANGES["C1"], "C2": MANUAL_RANGES["C2"]}.items():
        start = min(positions) - 1
        width = max(positions) - min(positions) + 1
        g.ax_heatmap.add_patch(Rectangle((start, start), width, width, fill=False, lw=1.3, ec="black"))
        g.ax_heatmap.text(start + 1, start + 2, label, ha="left", va="top", fontsize=9, weight="bold", color="black")

    coarse_handles = [Patch(facecolor=color, label=label) for label, color in coarse_palette.items()]
    fine_handles = [Patch(facecolor=fine_palette[label], label=label) for label in ["1", "2", "3", "4", "5", "6", "7", "none"]]
    family_handles = [Patch(facecolor=family_colors[label], label=label) for label in family_labels]
    family_legend = g.fig.legend(
        handles=family_handles,
        title="reference family",
        loc="upper left",
        bbox_to_anchor=(0.93, 0.74),
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )
    rho_mappable = plt.cm.ScalarMappable(norm=rho_norm, cmap=rho_cmap)
    rho_mappable.set_array([])
    rho_cax = g.fig.add_axes([1.13, 0.61, 0.018, 0.13])
    rho_cbar = g.fig.colorbar(rho_mappable, cax=rho_cax)
    rho_cbar.set_label("Spearman rho vs SNAI1-ac", fontsize=8)
    rho_cbar.set_ticks([-0.4, 0.0, 0.4])
    rho_cbar.ax.tick_params(labelsize=8)
    coarse_legend = g.fig.legend(
        handles=coarse_handles,
        title="coarse",
        loc="upper left",
        bbox_to_anchor=(0.93, 0.44),
        frameon=False,
    )
    fine_legend = g.fig.legend(
        handles=fine_handles,
        title="fine",
        loc="upper left",
        bbox_to_anchor=(1.13, 0.44),
        frameon=False,
    )
    g.fig.add_artist(family_legend)
    g.fig.add_artist(coarse_legend)
    g.fig.add_artist(fine_legend)
    g.fig.suptitle("Variant B non-junk manual metaprogram cut", y=1.02, fontsize=13)
    g.fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(g.fig)


def export_manual_cut_assignment_and_heatmap() -> tuple[Path, Path, float]:
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    matrix, z, order, coph = variant_b_nonjunk_matrix_and_order()
    assignment = recreate_manual_cut_assignment(order)
    assignment.to_csv(ASSIGNMENT_FILE, index=False)
    draw_manual_cut_heatmap(matrix, z, assignment, MANUAL_HEATMAP_FILE)
    return ASSIGNMENT_FILE, MANUAL_HEATMAP_FILE, coph


def draw_manual_cut_heatmap_ordered_by_rho(matrix: pd.DataFrame, assignment: pd.DataFrame) -> pd.DataFrame:
    ref_map = family_label_map_from_snapshot()
    corr = pd.read_csv(PROGRAMME_SNAI1AC_CORR_FILE, usecols=["program_id", "spearman_rho"]).drop_duplicates("program_id")
    rho_map = corr.set_index("program_id")["spearman_rho"].astype(float).to_dict()
    assignment = assignment.set_index("program_id").reindex(matrix.index)

    order = sorted(matrix.index, key=lambda pid: (pd.isna(rho_map.get(pid, np.nan)), -rho_map.get(pid, -np.inf)))
    ordered = matrix.loc[order, order].copy()
    np.fill_diagonal(ordered.values, np.nan)

    coarse_palette = {"A": "#4C78A8", "B": "#54A24B", "C": "#E45756", "unassigned": "#B9B9B9"}
    fine_palette = {
        "1": "#8DD3C7",
        "2": "#FFFFB3",
        "3": "#BEBADA",
        "4": "#FB8072",
        "5": "#80B1D3",
        "6": "#FDB462",
        "7": "#B3DE69",
        "none": "#D9D9D9",
    }
    family_colors = family_palette_from_snapshot()
    family_labels = sorted({ref_map.get(pid, "missing") for pid in order})
    rho_norm = mcolors.TwoSlopeNorm(vmin=-0.4, vcenter=0.0, vmax=0.4)
    rho_cmap = plt.get_cmap("vlag")

    def color_row(colors: list[str]) -> np.ndarray:
        return np.array([[mcolors.to_rgba(color) for color in colors]])

    def rho_to_color(pid: str) -> str:
        rho = rho_map.get(pid)
        if pd.isna(rho):
            return "#D9D9D9"
        return mcolors.to_hex(rho_cmap(rho_norm(float(np.clip(rho, -0.4, 0.4)))))

    top_family_colors = [family_colors.get(ref_map.get(pid, "missing"), "#D9D9D9") for pid in order]
    top_rho_colors = [rho_to_color(pid) for pid in order]
    left_coarse_colors = [coarse_palette.get(str(assignment.loc[pid, "cluster_coarse"]), "#D9D9D9") for pid in order]
    left_fine_colors = [
        fine_palette.get(fine_from_path(str(assignment.loc[pid, "cluster_path"])), "#D9D9D9") for pid in order
    ]

    sns.set_theme(style="white")
    fig = plt.figure(figsize=(13.5, 12.5))
    gs = fig.add_gridspec(
        nrows=4,
        ncols=4,
        width_ratios=[0.22, 0.22, 10.0, 2.25],
        height_ratios=[0.22, 0.22, 10.0, 0.25],
        wspace=0.03,
        hspace=0.03,
    )
    ax_top_family = fig.add_subplot(gs[0, 2])
    ax_top_rho = fig.add_subplot(gs[1, 2])
    ax_left_coarse = fig.add_subplot(gs[2, 0])
    ax_left_fine = fig.add_subplot(gs[2, 1])
    ax_heatmap = fig.add_subplot(gs[2, 2])
    ax_legend = fig.add_subplot(gs[2, 3])
    ax_legend.axis("off")

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    im = ax_heatmap.imshow(ordered.values, cmap=cmap, vmin=0, vmax=0.2, interpolation="nearest", aspect="equal")
    ax_heatmap.set_xticks([])
    ax_heatmap.set_yticks([])
    ax_heatmap.set_title("Variant B non-junk Jaccard, ordered by SNAI1-ac Spearman rho", fontsize=12, pad=10)

    ax_top_family.imshow(color_row(top_family_colors), aspect="auto")
    ax_top_rho.imshow(color_row(top_rho_colors), aspect="auto")
    for ax, label in [(ax_top_family, "reference family"), (ax_top_rho, "SNAI1-ac rho")]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8, labelpad=58)
        for spine in ax.spines.values():
            spine.set_visible(False)

    ax_left_coarse.imshow(color_row(left_coarse_colors).transpose((1, 0, 2)), aspect="auto")
    ax_left_fine.imshow(color_row(left_fine_colors).transpose((1, 0, 2)), aspect="auto")
    for ax, label in [(ax_left_coarse, "coarse"), (ax_left_fine, "fine")]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(label, rotation=90, fontsize=8, labelpad=6)
        for spine in ax.spines.values():
            spine.set_visible(False)

    jac_cax = ax_legend.inset_axes([0.70, 0.04, 0.09, 0.56])
    cbar = fig.colorbar(im, cax=jac_cax)
    cbar.set_label("Jaccard similarity", fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    rho_mappable = plt.cm.ScalarMappable(norm=rho_norm, cmap=rho_cmap)
    rho_mappable.set_array([])
    rho_cax = ax_legend.inset_axes([0.0, 0.61, 0.58, 0.035])
    rho_cbar = fig.colorbar(rho_mappable, cax=rho_cax, orientation="horizontal")
    rho_cbar.set_label("Spearman rho vs SNAI1-ac", fontsize=8)
    rho_cbar.set_ticks([-0.4, 0.0, 0.4])
    rho_cbar.ax.tick_params(labelsize=8)

    family_handles = [Patch(facecolor=family_colors[label], label=label) for label in family_labels]
    coarse_handles = [Patch(facecolor=color, label=label) for label, color in coarse_palette.items()]
    fine_handles = [Patch(facecolor=fine_palette[label], label=label) for label in ["1", "2", "3", "4", "5", "6", "7", "none"]]
    leg1 = ax_legend.legend(handles=family_handles, title="reference family", loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=8, title_fontsize=9)
    ax_legend.add_artist(leg1)
    leg2 = ax_legend.legend(handles=coarse_handles, title="coarse", loc="upper left", bbox_to_anchor=(0.0, 0.48), frameon=False, fontsize=8, title_fontsize=9)
    ax_legend.add_artist(leg2)
    ax_legend.legend(handles=fine_handles, title="fine", loc="upper left", bbox_to_anchor=(0.45, 0.48), frameon=False, fontsize=8, title_fontsize=9)

    order_table = pd.DataFrame(
        {
            "rho_order_position": range(1, len(order) + 1),
            "program_id": order,
            "spearman_rho": [rho_map.get(pid, np.nan) for pid in order],
            "cluster_path": [assignment.loc[pid, "cluster_path"] for pid in order],
            "cluster_coarse": [assignment.loc[pid, "cluster_coarse"] for pid in order],
            "fine_1_7": [fine_from_path(str(assignment.loc[pid, "cluster_path"])) for pid in order],
            "family_label": [ref_map.get(pid, "missing") for pid in order],
        }
    )
    fig.savefig(MANUAL_RHO_ORDERED_HEATMAP_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return order_table


def export_manual_cut_rho_ordered_heatmap() -> tuple[Path, Path]:
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    matrix_full = pd.read_csv(VARIANT_B_MATRIX, index_col=0)
    if ASSIGNMENT_FILE.exists():
        assignment = pd.read_csv(ASSIGNMENT_FILE)
    else:
        matrix, _, order, _ = variant_b_nonjunk_matrix_and_order()
        assignment = recreate_manual_cut_assignment(order)
        assignment.to_csv(ASSIGNMENT_FILE, index=False)
        matrix_full = matrix_full.loc[assignment["program_id"], assignment["program_id"]]
    program_ids = assignment["program_id"].tolist()
    matrix = matrix_full.loc[program_ids, program_ids]
    order_table = draw_manual_cut_heatmap_ordered_by_rho(matrix, assignment)
    order_table.to_csv(MANUAL_RHO_ORDERED_TABLE, index=False)
    return MANUAL_RHO_ORDERED_HEATMAP_FILE, MANUAL_RHO_ORDERED_TABLE


def program_parts(program_id: str) -> dict[str, str | int]:
    match = PROGRAM_RE.match(program_id)
    if not match:
        return {"cohort": "", "sample": "", "k": -1, "p": -1}
    return {
        "cohort": match.group(1),
        "sample": match.group(2),
        "k": int(match.group(3)),
        "p": int(match.group(4)),
    }


def build_pos_neg_feature_sets(program_ids: list[str], n_each: int = 50) -> tuple[dict[str, set[str]], pd.DataFrame, pd.DataFrame]:
    spectra = load_full_program_spectra(program_ids)
    feature_sets: dict[str, set[str]] = {}
    long_rows = []
    wide_rows = []
    for program_id in program_ids:
        parts = program_parts(program_id)
        scores = spectra[program_id]
        positive = scores.sort_values(ascending=False).head(n_each)
        negative = scores.sort_values(ascending=True).head(n_each)
        features = {f"{gene}__pos" for gene in positive.index.astype(str)} | {f"{gene}__neg" for gene in negative.index.astype(str)}
        feature_sets[program_id] = features
        wide_row = {"program_id": program_id, "cohort": parts["cohort"], "sample": f"{parts['cohort']}__{parts['sample']}"}
        for direction, series in [("positive", positive), ("negative", negative)]:
            suffix = "pos" if direction == "positive" else "neg"
            for rank, (gene, score) in enumerate(series.items(), start=1):
                gene = str(gene)
                signed_feature = f"{gene}__{suffix}"
                long_rows.append(
                    {
                        "program_id": program_id,
                        "cohort": parts["cohort"],
                        "sample": f"{parts['cohort']}__{parts['sample']}",
                        "direction": direction,
                        "rank_within_direction": rank,
                        "gene": gene,
                        "gene_spectra_score": float(score),
                        "signed_feature": signed_feature,
                    }
                )
                wide_row[f"{suffix}_gene_{rank}"] = gene
                wide_row[f"{suffix}_score_{rank}"] = float(score)
        wide_rows.append(wide_row)
    return feature_sets, pd.DataFrame(long_rows), pd.DataFrame(wide_rows)


def build_negative_feature_sets(program_ids: list[str], n: int = 50) -> tuple[dict[str, set[str]], pd.DataFrame, pd.DataFrame]:
    spectra = load_full_program_spectra(program_ids)
    feature_sets: dict[str, set[str]] = {}
    long_rows = []
    wide_rows = []
    for program_id in program_ids:
        parts = program_parts(program_id)
        negative = spectra[program_id].sort_values(ascending=True).head(n)
        features = {f"{gene}__neg" for gene in negative.index.astype(str)}
        feature_sets[program_id] = features
        wide_row = {"program_id": program_id, "cohort": parts["cohort"], "sample": f"{parts['cohort']}__{parts['sample']}"}
        for rank, (gene, score) in enumerate(negative.items(), start=1):
            gene = str(gene)
            signed_feature = f"{gene}__neg"
            long_rows.append(
                {
                    "program_id": program_id,
                    "cohort": parts["cohort"],
                    "sample": f"{parts['cohort']}__{parts['sample']}",
                    "direction": "negative",
                    "rank_within_direction": rank,
                    "gene": gene,
                    "gene_spectra_score": float(score),
                    "signed_feature": signed_feature,
                }
            )
            wide_row[f"neg_gene_{rank}"] = gene
            wide_row[f"neg_score_{rank}"] = float(score)
        wide_rows.append(wide_row)
    return feature_sets, pd.DataFrame(long_rows), pd.DataFrame(wide_rows)


def pos_neg_sort_key(program_id: str) -> tuple[str, str, int, int]:
    parts = program_parts(program_id)
    return (str(parts["cohort"]), str(parts["sample"]), int(parts["k"]), int(parts["p"]))


def offdiag_values(matrix: pd.DataFrame) -> np.ndarray:
    values = matrix.to_numpy(dtype=float)
    return values[np.triu_indices_from(values, k=1)]


def summarize_jaccard_distribution(matrix: pd.DataFrame, variant: str) -> pd.DataFrame:
    values = offdiag_values(matrix)
    return pd.DataFrame(
        [
            {
                "variant": variant,
                "n_programs": matrix.shape[0],
                "n_pairs": len(values),
                "min": float(np.min(values)),
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "max": float(np.max(values)),
                "pct_zero_pairs": float(np.mean(values == 0) * 100),
            }
        ]
    )


def summarize_relationship_tiers(matrix: pd.DataFrame, variant: str) -> pd.DataFrame:
    rows = []
    program_ids = matrix.index.tolist()
    for i, pid_i in enumerate(program_ids):
        pi = program_parts(pid_i)
        sample_i = f"{pi['cohort']}__{pi['sample']}"
        for pid_j in program_ids[i + 1 :]:
            pj = program_parts(pid_j)
            sample_j = f"{pj['cohort']}__{pj['sample']}"
            if sample_i == sample_j:
                tier = "same sample"
            elif pi["cohort"] == pj["cohort"]:
                tier = "same cohort different sample"
            else:
                tier = "different cohort"
            rows.append({"tier": tier, "jaccard": float(matrix.loc[pid_i, pid_j])})
    out = pd.DataFrame(rows).groupby("tier", as_index=False)["jaccard"].agg(["mean", "median", "count"]).reset_index()
    out.insert(0, "variant", variant)
    return out


def matrix_colors(program_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[Patch], mcolors.TwoSlopeNorm, mcolors.Colormap]:
    ref_map = family_label_map_from_snapshot()
    family_colors = family_palette_from_snapshot()
    corr = pd.read_csv(PROGRAMME_SNAI1AC_CORR_FILE, usecols=["program_id", "spearman_rho"]).drop_duplicates("program_id")
    rho_map = corr.set_index("program_id")["spearman_rho"].astype(float).to_dict()
    rho_norm = mcolors.TwoSlopeNorm(vmin=-0.4, vcenter=0.0, vmax=0.4)
    rho_cmap = plt.get_cmap("vlag")

    def rho_to_color(program_id: str) -> str:
        rho = rho_map.get(program_id)
        if pd.isna(rho):
            return "#D9D9D9"
        return mcolors.to_hex(rho_cmap(rho_norm(float(np.clip(rho, -0.4, 0.4)))))

    colors = pd.DataFrame(
        {
            "reference family": [family_colors.get(ref_map.get(pid, "missing"), "#D9D9D9") for pid in program_ids],
            "SNAI1-ac rho": [rho_to_color(pid) for pid in program_ids],
        },
        index=program_ids,
    )
    labels = sorted({ref_map.get(pid, "missing") for pid in program_ids})
    handles = [Patch(facecolor=family_colors[label], label=label) for label in labels]
    return colors, colors.copy(), handles, rho_norm, rho_cmap


def save_jaccard_heatmap(
    matrix: pd.DataFrame,
    output_path: Path,
    title: str,
    z: np.ndarray | None = None,
    ordered_ids: list[str] | None = None,
) -> None:
    if ordered_ids is not None:
        matrix = matrix.loc[ordered_ids, ordered_ids]
    plot_matrix = matrix.copy()
    np.fill_diagonal(plot_matrix.values, np.nan)
    row_colors, col_colors, family_handles, rho_norm, rho_cmap = matrix_colors(matrix.index.tolist())
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    vmax = min(0.2, float(np.nanquantile(plot_matrix.to_numpy(dtype=float), 0.99)))
    vmax = max(vmax, 0.01)
    sns.set_theme(style="white")
    kwargs = {}
    if z is None:
        kwargs.update({"row_cluster": False, "col_cluster": False})
    else:
        plot_z = linkage_for_visible_dendrogram(z)
        kwargs.update({"row_linkage": plot_z, "col_linkage": plot_z})
    g = sns.clustermap(
        plot_matrix,
        row_colors=row_colors,
        col_colors=col_colors,
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        linewidths=0,
        figsize=(12, 12),
        dendrogram_ratio=(0.18, 0.18),
        tree_kws={"linewidths": 0.8},
        cbar_kws={"label": "signed-feature Jaccard similarity"},
        **kwargs,
    )
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")
    g.ax_col_colors.set_title("reference family / SNAI1-ac rho", fontsize=9, pad=6)
    g.fig.suptitle(title, y=1.02, fontsize=13)
    family_legend = g.fig.legend(
        handles=family_handles,
        title="reference family",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.86),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    rho_mappable = plt.cm.ScalarMappable(norm=rho_norm, cmap=rho_cmap)
    rho_mappable.set_array([])
    rho_cax = g.fig.add_axes([1.03, 0.62, 0.02, 0.13])
    rho_cbar = g.fig.colorbar(rho_mappable, cax=rho_cax)
    rho_cbar.set_label("Spearman rho vs SNAI1-ac", fontsize=8)
    rho_cbar.set_ticks([-0.4, 0.0, 0.4])
    rho_cbar.ax.tick_params(labelsize=8)
    g.fig.add_artist(family_legend)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    g.fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(g.fig)


def save_original_style_jaccard_clustermap(
    matrix: pd.DataFrame,
    z: np.ndarray,
    output_path: Path,
    title: str,
    familybar: bool = False,
) -> None:
    plot_matrix = matrix.copy()
    np.fill_diagonal(plot_matrix.values, np.nan)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    sns.set_theme(style="white")

    kwargs = {}
    if familybar:
        ref_map = family_label_map_from_snapshot()
        family_colors = family_palette_from_snapshot()
        row_colors = pd.Series(
            [family_colors.get(ref_map.get(pid, "missing"), "#D9D9D9") for pid in matrix.index],
            index=matrix.index,
            name="Reference family_label",
        )
        kwargs["row_colors"] = row_colors

    g = sns.clustermap(
        plot_matrix,
        row_linkage=z,
        col_linkage=z,
        cmap=cmap,
        vmin=0,
        vmax=0.2,
        xticklabels=False,
        yticklabels=False,
        linewidths=0,
        figsize=(12, 12),
        cbar_kws={"label": "Jaccard"},
        **kwargs,
    )
    g.ax_heatmap.set_xlabel("Programs, clustered leaf order")
    g.ax_heatmap.set_ylabel("Programs, clustered leaf order")
    g.fig.suptitle(title, y=1.02)

    if familybar:
        ref_map = family_label_map_from_snapshot()
        family_colors = family_palette_from_snapshot()
        labels = sorted({ref_map.get(pid, "missing") for pid in matrix.index})
        handles = [Patch(facecolor=family_colors[label], label=label) for label in labels]
        g.fig.legend(
            handles=handles,
            title="Reference family_label",
            loc="center left",
            bbox_to_anchor=(1.04, 0.5),
            frameon=False,
            fontsize=7,
            title_fontsize=8,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    g.fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(g.fig)


def save_original_style_ordered_heatmap(matrix: pd.DataFrame, ordered_ids: list[str], output_path: Path, title: str) -> None:
    plot_matrix = matrix.loc[ordered_ids, ordered_ids].copy()
    np.fill_diagonal(plot_matrix.values, np.nan)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(
        plot_matrix,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=0.2,
        xticklabels=False,
        yticklabels=False,
        linewidths=0,
        cbar_kws={"label": "Jaccard"},
    )
    ax.set_xlabel("Programs, cohort/sample order")
    ax.set_ylabel("Programs, cohort/sample order")
    ax.set_title(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_position_indexed_clustermap(matrix: pd.DataFrame, z: np.ndarray, output_path: Path, title: str) -> pd.DataFrame:
    order = matrix.index[leaves_list(z)].tolist()
    leaf_table = []
    for position, program_id in enumerate(order, start=1):
        next_pid = order[position] if position < len(order) else None
        parts = program_parts(program_id)
        leaf_table.append(
            {
                "position": position,
                "program_id": program_id,
                "cohort": parts["cohort"],
                "sample": f"{parts['cohort']}__{parts['sample']}",
                "jaccard_to_next_leaf": float(matrix.loc[program_id, next_pid]) if next_pid else np.nan,
            }
        )
    plot_matrix = matrix.copy()
    np.fill_diagonal(plot_matrix.values, np.nan)
    ref_map = family_label_map_from_snapshot()
    family_colors = family_palette_from_snapshot()
    row_colors = pd.Series(
        [family_colors.get(ref_map.get(pid, "missing"), "#D9D9D9") for pid in matrix.index],
        index=matrix.index,
        name="Reference family_label",
    )
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    sns.set_theme(style="white")
    g = sns.clustermap(
        plot_matrix,
        row_linkage=z,
        col_linkage=z,
        row_colors=row_colors,
        cmap=cmap,
        vmin=0,
        vmax=0.2,
        xticklabels=False,
        yticklabels=False,
        linewidths=0,
        figsize=(12, 12),
        cbar_kws={"label": "Jaccard"},
    )
    n = len(order)
    ticks = np.arange(4, n, 5)
    labels = [str(i + 1) for i in ticks]
    g.ax_heatmap.set_xticks(ticks)
    g.ax_heatmap.set_yticks(ticks)
    g.ax_heatmap.set_xticklabels(labels, fontsize=6, rotation=90)
    g.ax_heatmap.set_yticklabels(labels, fontsize=6)
    for pos in ticks:
        g.ax_heatmap.axhline(pos + 0.5, color="white", lw=0.25, alpha=0.35)
        g.ax_heatmap.axvline(pos + 0.5, color="white", lw=0.25, alpha=0.35)
    g.ax_heatmap.set_xlabel("Dendrogram leaf position")
    g.ax_heatmap.set_ylabel("Dendrogram leaf position")
    g.fig.suptitle(title, y=1.02, fontsize=13)
    labels = sorted({ref_map.get(pid, "missing") for pid in matrix.index})
    handles = [Patch(facecolor=family_colors[label], label=label) for label in labels]
    g.fig.legend(
        handles=handles,
        title="Reference family_label",
        loc="upper left",
        bbox_to_anchor=(1.04, 0.5),
        frameon=False,
        fontsize=7,
        title_fontsize=8,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    g.fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(g.fig)
    return pd.DataFrame(leaf_table)


def export_pos_neg_variant_b() -> dict[str, Path]:
    POS_NEG_DIR.mkdir(parents=True, exist_ok=True)
    substrate_dir = POS_NEG_DIR / "00_substrate"
    raw_dir = POS_NEG_DIR / "01_raw_matrix"
    raw_heatmap_dir = POS_NEG_DIR / "02_raw_heatmaps"
    clustered_dir = POS_NEG_DIR / "03_average_clustering"
    nonjunk_dir = POS_NEG_DIR / "04_nonjunk_average_clustering"
    manual_input_dir = POS_NEG_DIR / "05_manual_cut_inputs"
    for path in [substrate_dir, raw_dir, raw_heatmap_dir, clustered_dir, nonjunk_dir, manual_input_dir]:
        path.mkdir(parents=True, exist_ok=True)

    program_ids = pd.read_csv(VARIANT_B_MATRIX, index_col=0).index.astype(str).tolist()
    feature_sets, long_features, wide_features = build_pos_neg_feature_sets(program_ids, n_each=50)
    long_path = substrate_dir / "extracted_program_top50pos_top50neg_signedfeatures_variantB_long.csv"
    wide_path = substrate_dir / "extracted_program_top50pos_top50neg_signedfeatures_variantB_wide.csv"
    long_features.to_csv(long_path, index=False)
    wide_features.to_csv(wide_path, index=False)

    hygiene = (
        long_features.groupby(["program_id", "direction"])
        .agg(n_genes=("gene", "size"), n_unique_genes=("gene", "nunique"), n_unique_features=("signed_feature", "nunique"))
        .reset_index()
    )
    both_tails = (
        long_features.groupby(["program_id", "gene"])["direction"]
        .nunique()
        .reset_index(name="n_directions")
        .query("n_directions > 1")
    )
    hygiene_path = substrate_dir / "signed_top50pos_top50neg_hygiene.csv"
    hygiene.to_csv(hygiene_path, index=False)
    both_tails_path = substrate_dir / "genes_appearing_in_both_pos_and_neg_top50_same_program.csv"
    both_tails.to_csv(both_tails_path, index=False)

    matrix = jaccard_matrix_from_feature_sets(feature_sets)
    matrix_path = raw_dir / "jaccard_variantB_top50pos_top50neg_signedfeatures.csv"
    matrix.to_csv(matrix_path)
    distribution = summarize_jaccard_distribution(matrix, "variantB_posneg_signedfeatures")
    tiers = summarize_relationship_tiers(matrix, "variantB_posneg_signedfeatures")
    distribution_path = raw_dir / "jaccard_variantB_posneg_distribution_summary.csv"
    tiers_path = raw_dir / "jaccard_variantB_posneg_relationship_tiers.csv"
    distribution.to_csv(distribution_path, index=False)
    tiers.to_csv(tiers_path, index=False)

    raw_order = sorted(program_ids, key=pos_neg_sort_key)
    raw_heatmap = raw_heatmap_dir / "jaccard_variantB_posneg_raw_ordered_cohort_sample.png"
    save_original_style_ordered_heatmap(
        matrix,
        raw_order,
        raw_heatmap,
        "Variant B pos+neg signed-feature Jaccard, ordered by cohort/sample",
    )

    z, order, coph = average_linkage_from_jaccard(matrix)
    linkage_path = clustered_dir / "variantB_posneg_average_linkage_Z.csv"
    leaf_path = clustered_dir / "variantB_posneg_average_linkage_leaf_order.csv"
    pd.DataFrame(z, columns=["child_1", "child_2", "distance", "n_leaves"]).to_csv(linkage_path, index=False)
    pd.DataFrame({"position": range(1, len(order) + 1), "program_id": order}).to_csv(leaf_path, index=False)
    clustered_heatmap = clustered_dir / "jaccard_variantB_posneg_clustermap_average.png"
    clustered_familybar_heatmap = clustered_dir / "jaccard_variantB_posneg_clustermap_average_familybar.png"
    save_original_style_jaccard_clustermap(
        matrix,
        z,
        clustered_heatmap,
        "Variant B pos+neg: average-linkage clustered signed-feature Jaccard (diagonal masked, vmax=0.2000)",
        familybar=False,
    )
    save_original_style_jaccard_clustermap(
        matrix,
        z,
        clustered_familybar_heatmap,
        "Variant B pos+neg: average-linkage signed-feature Jaccard clustermap with reference family labels (vmax=0.2000)",
        familybar=True,
    )

    junk = set(pd.read_csv(JUNK_FILE)["program_id"].astype(str))
    nonjunk = [pid for pid in program_ids if pid not in junk]
    nonjunk_matrix = matrix.loc[nonjunk, nonjunk]
    nonjunk_matrix_path = nonjunk_dir / "jaccard_variantB_top50pos_top50neg_signedfeatures_nonjunk.csv"
    nonjunk_matrix.to_csv(nonjunk_matrix_path)
    nonjunk_distribution = summarize_jaccard_distribution(nonjunk_matrix, "variantB_posneg_signedfeatures_nonjunk")
    nonjunk_tiers = summarize_relationship_tiers(nonjunk_matrix, "variantB_posneg_signedfeatures_nonjunk")
    nonjunk_distribution_path = nonjunk_dir / "jaccard_variantB_posneg_nonjunk_distribution_summary.csv"
    nonjunk_tiers_path = nonjunk_dir / "jaccard_variantB_posneg_nonjunk_relationship_tiers.csv"
    nonjunk_distribution.to_csv(nonjunk_distribution_path, index=False)
    nonjunk_tiers.to_csv(nonjunk_tiers_path, index=False)

    nonjunk_z, nonjunk_order, nonjunk_coph = average_linkage_from_jaccard(nonjunk_matrix)
    nonjunk_linkage_path = nonjunk_dir / "variantB_posneg_nonjunk_average_linkage_Z.csv"
    nonjunk_leaf_path = nonjunk_dir / "variantB_posneg_nonjunk_average_linkage_leaf_order.csv"
    pd.DataFrame(nonjunk_z, columns=["child_1", "child_2", "distance", "n_leaves"]).to_csv(nonjunk_linkage_path, index=False)
    pd.DataFrame({"position": range(1, len(nonjunk_order) + 1), "program_id": nonjunk_order}).to_csv(nonjunk_leaf_path, index=False)
    nonjunk_heatmap = nonjunk_dir / "jaccard_variantB_posneg_nonjunk_clustermap_average.png"
    nonjunk_familybar_heatmap = nonjunk_dir / "jaccard_variantB_posneg_nonjunk_clustermap_average_familybar.png"
    save_original_style_jaccard_clustermap(
        nonjunk_matrix,
        nonjunk_z,
        nonjunk_heatmap,
        "Variant B pos+neg non-junk: average-linkage clustered signed-feature Jaccard (diagonal masked, vmax=0.2000)",
        familybar=False,
    )
    save_original_style_jaccard_clustermap(
        nonjunk_matrix,
        nonjunk_z,
        nonjunk_familybar_heatmap,
        "Variant B pos+neg non-junk: average-linkage signed-feature Jaccard clustermap with reference family labels (vmax=0.2000)",
        familybar=True,
    )

    manual_position_heatmap = manual_input_dir / "variantB_posneg_nonjunk_position_indexed_heatmap.png"
    manual_leaf_table = manual_input_dir / "variantB_posneg_nonjunk_ordered_leaf_table.csv"
    leaf_table = save_position_indexed_clustermap(
        nonjunk_matrix,
        nonjunk_z,
        manual_position_heatmap,
        "Variant B pos+neg non-junk, manual-cut input leaf positions",
    )
    ref_map = family_label_map_from_snapshot()
    leaf_table["reference_family"] = leaf_table["program_id"].map(ref_map).fillna("missing")
    leaf_table.to_csv(manual_leaf_table, index=False)

    run_summary = pd.DataFrame(
        [
            {"metric": "n_programs_full", "value": len(program_ids)},
            {"metric": "n_programs_nonjunk", "value": len(nonjunk)},
            {"metric": "n_signed_features_per_program", "value": 100},
            {"metric": "full_average_cophenetic_correlation", "value": coph},
            {"metric": "nonjunk_average_cophenetic_correlation", "value": nonjunk_coph},
            {"metric": "same_gene_in_pos_and_neg_top50_same_program_rows", "value": len(both_tails)},
        ]
    )
    summary_path = POS_NEG_DIR / "variantB_posneg_run_summary.csv"
    run_summary.to_csv(summary_path, index=False)

    return {
        "long_features": long_path,
        "wide_features": wide_path,
        "hygiene": hygiene_path,
        "both_tails": both_tails_path,
        "matrix": matrix_path,
        "nonjunk_matrix": nonjunk_matrix_path,
        "distribution": distribution_path,
        "tiers": tiers_path,
        "nonjunk_distribution": nonjunk_distribution_path,
        "nonjunk_tiers": nonjunk_tiers_path,
        "linkage": linkage_path,
        "leaf_order": leaf_path,
        "nonjunk_linkage": nonjunk_linkage_path,
        "nonjunk_leaf_order": nonjunk_leaf_path,
        "raw_heatmap": raw_heatmap,
        "clustered_heatmap": clustered_heatmap,
        "clustered_familybar_heatmap": clustered_familybar_heatmap,
        "nonjunk_clustered_heatmap": nonjunk_heatmap,
        "nonjunk_familybar_heatmap": nonjunk_familybar_heatmap,
        "manual_position_heatmap": manual_position_heatmap,
        "manual_leaf_table": manual_leaf_table,
        "run_summary": summary_path,
    }


def export_neg_variant_b() -> dict[str, Path]:
    NEG_DIR.mkdir(parents=True, exist_ok=True)
    substrate_dir = NEG_DIR / "00_substrate"
    raw_dir = NEG_DIR / "01_raw_matrix"
    raw_heatmap_dir = NEG_DIR / "02_raw_heatmaps"
    clustered_dir = NEG_DIR / "03_average_clustering"
    nonjunk_dir = NEG_DIR / "04_nonjunk_average_clustering"
    manual_input_dir = NEG_DIR / "05_manual_cut_inputs"
    for path in [substrate_dir, raw_dir, raw_heatmap_dir, clustered_dir, nonjunk_dir, manual_input_dir]:
        path.mkdir(parents=True, exist_ok=True)

    program_ids = pd.read_csv(VARIANT_B_MATRIX, index_col=0).index.astype(str).tolist()
    feature_sets, long_features, wide_features = build_negative_feature_sets(program_ids, n=50)
    long_path = substrate_dir / "extracted_program_top50neg_signedfeatures_variantB_long.csv"
    wide_path = substrate_dir / "extracted_program_top50neg_signedfeatures_variantB_wide.csv"
    long_features.to_csv(long_path, index=False)
    wide_features.to_csv(wide_path, index=False)

    hygiene = (
        long_features.groupby(["program_id", "direction"])
        .agg(n_genes=("gene", "size"), n_unique_genes=("gene", "nunique"), n_unique_features=("signed_feature", "nunique"))
        .reset_index()
    )
    hygiene_path = substrate_dir / "signed_top50neg_hygiene.csv"
    hygiene.to_csv(hygiene_path, index=False)

    matrix = jaccard_matrix_from_feature_sets(feature_sets)
    matrix_path = raw_dir / "jaccard_variantB_top50neg_signedfeatures.csv"
    matrix.to_csv(matrix_path)
    distribution = summarize_jaccard_distribution(matrix, "variantB_neg_signedfeatures")
    tiers = summarize_relationship_tiers(matrix, "variantB_neg_signedfeatures")
    distribution_path = raw_dir / "jaccard_variantB_neg_distribution_summary.csv"
    tiers_path = raw_dir / "jaccard_variantB_neg_relationship_tiers.csv"
    distribution.to_csv(distribution_path, index=False)
    tiers.to_csv(tiers_path, index=False)

    raw_order = sorted(program_ids, key=pos_neg_sort_key)
    raw_heatmap = raw_heatmap_dir / "jaccard_variantB_neg_raw_ordered_cohort_sample.png"
    save_original_style_ordered_heatmap(
        matrix,
        raw_order,
        raw_heatmap,
        "Variant B negative-tail signed-feature Jaccard, ordered by cohort/sample",
    )

    z, order, coph = average_linkage_from_jaccard(matrix)
    linkage_path = clustered_dir / "variantB_neg_average_linkage_Z.csv"
    leaf_path = clustered_dir / "variantB_neg_average_linkage_leaf_order.csv"
    pd.DataFrame(z, columns=["child_1", "child_2", "distance", "n_leaves"]).to_csv(linkage_path, index=False)
    pd.DataFrame({"position": range(1, len(order) + 1), "program_id": order}).to_csv(leaf_path, index=False)
    clustered_heatmap = clustered_dir / "jaccard_variantB_neg_clustermap_average.png"
    clustered_familybar_heatmap = clustered_dir / "jaccard_variantB_neg_clustermap_average_familybar.png"
    save_original_style_jaccard_clustermap(
        matrix,
        z,
        clustered_heatmap,
        "Variant B negative-tail: average-linkage clustered signed-feature Jaccard (diagonal masked, vmax=0.2000)",
        familybar=False,
    )
    save_original_style_jaccard_clustermap(
        matrix,
        z,
        clustered_familybar_heatmap,
        "Variant B negative-tail: average-linkage signed-feature Jaccard clustermap with reference family labels (vmax=0.2000)",
        familybar=True,
    )

    junk = set(pd.read_csv(JUNK_FILE)["program_id"].astype(str))
    nonjunk = [pid for pid in program_ids if pid not in junk]
    nonjunk_matrix = matrix.loc[nonjunk, nonjunk]
    nonjunk_matrix_path = nonjunk_dir / "jaccard_variantB_top50neg_signedfeatures_nonjunk.csv"
    nonjunk_matrix.to_csv(nonjunk_matrix_path)
    nonjunk_distribution = summarize_jaccard_distribution(nonjunk_matrix, "variantB_neg_signedfeatures_nonjunk")
    nonjunk_tiers = summarize_relationship_tiers(nonjunk_matrix, "variantB_neg_signedfeatures_nonjunk")
    nonjunk_distribution_path = nonjunk_dir / "jaccard_variantB_neg_nonjunk_distribution_summary.csv"
    nonjunk_tiers_path = nonjunk_dir / "jaccard_variantB_neg_nonjunk_relationship_tiers.csv"
    nonjunk_distribution.to_csv(nonjunk_distribution_path, index=False)
    nonjunk_tiers.to_csv(nonjunk_tiers_path, index=False)

    nonjunk_z, nonjunk_order, nonjunk_coph = average_linkage_from_jaccard(nonjunk_matrix)
    nonjunk_linkage_path = nonjunk_dir / "variantB_neg_nonjunk_average_linkage_Z.csv"
    nonjunk_leaf_path = nonjunk_dir / "variantB_neg_nonjunk_average_linkage_leaf_order.csv"
    pd.DataFrame(nonjunk_z, columns=["child_1", "child_2", "distance", "n_leaves"]).to_csv(nonjunk_linkage_path, index=False)
    pd.DataFrame({"position": range(1, len(nonjunk_order) + 1), "program_id": nonjunk_order}).to_csv(nonjunk_leaf_path, index=False)
    nonjunk_heatmap = nonjunk_dir / "jaccard_variantB_neg_nonjunk_clustermap_average.png"
    nonjunk_familybar_heatmap = nonjunk_dir / "jaccard_variantB_neg_nonjunk_clustermap_average_familybar.png"
    save_original_style_jaccard_clustermap(
        nonjunk_matrix,
        nonjunk_z,
        nonjunk_heatmap,
        "Variant B negative-tail non-junk: average-linkage clustered signed-feature Jaccard (diagonal masked, vmax=0.2000)",
        familybar=False,
    )
    save_original_style_jaccard_clustermap(
        nonjunk_matrix,
        nonjunk_z,
        nonjunk_familybar_heatmap,
        "Variant B negative-tail non-junk: average-linkage signed-feature Jaccard clustermap with reference family labels (vmax=0.2000)",
        familybar=True,
    )

    manual_position_heatmap = manual_input_dir / "variantB_neg_nonjunk_position_indexed_heatmap.png"
    manual_leaf_table = manual_input_dir / "variantB_neg_nonjunk_ordered_leaf_table.csv"
    leaf_table = save_position_indexed_clustermap(
        nonjunk_matrix,
        nonjunk_z,
        manual_position_heatmap,
        "Variant B negative-tail non-junk, manual-cut input leaf positions",
    )
    ref_map = family_label_map_from_snapshot()
    leaf_table["reference_family"] = leaf_table["program_id"].map(ref_map).fillna("missing")
    leaf_table.to_csv(manual_leaf_table, index=False)

    run_summary = pd.DataFrame(
        [
            {"metric": "n_programs_full", "value": len(program_ids)},
            {"metric": "n_programs_nonjunk", "value": len(nonjunk)},
            {"metric": "n_signed_features_per_program", "value": 50},
            {"metric": "full_average_cophenetic_correlation", "value": coph},
            {"metric": "nonjunk_average_cophenetic_correlation", "value": nonjunk_coph},
        ]
    )
    summary_path = NEG_DIR / "variantB_neg_run_summary.csv"
    run_summary.to_csv(summary_path, index=False)

    return {
        "long_features": long_path,
        "wide_features": wide_path,
        "hygiene": hygiene_path,
        "matrix": matrix_path,
        "nonjunk_matrix": nonjunk_matrix_path,
        "distribution": distribution_path,
        "tiers": tiers_path,
        "nonjunk_distribution": nonjunk_distribution_path,
        "nonjunk_tiers": nonjunk_tiers_path,
        "linkage": linkage_path,
        "leaf_order": leaf_path,
        "nonjunk_linkage": nonjunk_linkage_path,
        "nonjunk_leaf_order": nonjunk_leaf_path,
        "raw_heatmap": raw_heatmap,
        "clustered_heatmap": clustered_heatmap,
        "clustered_familybar_heatmap": clustered_familybar_heatmap,
        "nonjunk_clustered_heatmap": nonjunk_heatmap,
        "nonjunk_familybar_heatmap": nonjunk_familybar_heatmap,
        "manual_position_heatmap": manual_position_heatmap,
        "manual_leaf_table": manual_leaf_table,
        "run_summary": summary_path,
    }


def positive_loading_elbow_rank(values: np.ndarray) -> int:
    if len(values) <= 2:
        return int(len(values))
    x = np.linspace(0, 1, len(values))
    y = values / values[0] if values[0] else values
    start = np.array([x[0], y[0]])
    end = np.array([x[-1], y[-1]])
    line = end - start
    line_norm = np.linalg.norm(line)
    if line_norm == 0:
        return 1
    points = np.column_stack([x, y])
    distances = np.abs(line[0] * (start[1] - points[:, 1]) - line[1] * (start[0] - points[:, 0])) / line_norm
    return int(np.argmax(distances) + 1)


def load_full_program_spectra(program_ids: list[str]) -> dict[str, pd.Series]:
    cache: dict[Path, pd.DataFrame] = {}
    spectra: dict[str, pd.Series] = {}
    for program_id in program_ids:
        path, p_value = spectra_path(program_id)
        if path not in cache:
            df = pd.read_csv(path, sep="\t", index_col=0)
            df.index = [int(str(idx).replace("GEP", "").replace(".0", "")) for idx in df.index]
            cache[path] = df
        spectra[program_id] = cache[path].loc[p_value].astype(float).sort_values(ascending=False)
    return spectra


def load_full_program_spectra_tpm(program_ids: list[str]) -> dict[str, pd.Series]:
    cache: dict[Path, pd.DataFrame] = {}
    spectra: dict[str, pd.Series] = {}
    for program_id in program_ids:
        path, p_value = spectra_tpm_path(program_id)
        if path not in cache:
            df = pd.read_csv(path, sep="\t", index_col=0)
            df.index = [int(str(idx).replace("GEP", "").replace(".0", "")) for idx in df.index]
            numeric = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
            if (numeric < 0).any().any():
                raise ValueError(f"Raw cNMF gene_spectra_tpm contains negative values: {path}")
            cache[path] = numeric
        spectra[program_id] = cache[path].loc[p_value].astype(float).sort_values(ascending=False)
    return spectra


def export_program_loading_elbow_diagnostics() -> dict[str, Path]:
    LOADING_DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    assignment = pd.read_csv(ASSIGNMENT_FILE)
    program_ids = pd.read_csv(VARIANT_B_MATRIX, index_col=0).index.astype(str).tolist()
    junk_ids = set(pd.read_csv(JUNK_FILE)["program_id"].astype(str))
    spectra = load_full_program_spectra_tpm(program_ids)

    summary_rows = []
    curve_rows = []
    cumulative_rows = []
    weight_values = []
    thresholds = [0.50, 0.80, 0.90, 0.95]
    for program_id, scores in spectra.items():
        values = scores.to_numpy(dtype=float)
        weights = scores.loc[scores > 0].to_numpy(dtype=float)
        total_weight = float(weights.sum())
        cumulative = np.cumsum(weights) / total_weight if total_weight > 0 else np.array([])
        threshold_counts = {
            f"n_genes_for_{int(threshold * 100)}pct_total_weight": int(np.searchsorted(cumulative, threshold) + 1)
            if len(cumulative)
            else 0
            for threshold in thresholds
        }
        row = {
            "program_id": program_id,
            "sample": program_sample_from_id(program_id),
            "n_genes_total": int(len(values)),
            "n_positive_weight_genes": int(len(weights)),
            "n_zero_genes": int(np.sum(values == 0)),
            "fraction_positive_weight_genes": float(np.mean(values > 0)),
            "min_gene_weight": float(np.min(values)),
            "p01_gene_weight": float(np.quantile(values, 0.01)),
            "p05_gene_weight": float(np.quantile(values, 0.05)),
            "p25_gene_weight": float(np.quantile(values, 0.25)),
            "median_gene_weight": float(np.median(values)),
            "p75_gene_weight": float(np.quantile(values, 0.75)),
            "p95_gene_weight": float(np.quantile(values, 0.95)),
            "p99_gene_weight": float(np.quantile(values, 0.99)),
            "max_gene_weight": float(np.max(values)),
            "total_gene_weight": total_weight,
            "elbow_rank_gene_weight": positive_loading_elbow_rank(weights),
            "top10_fraction_total_weight": float(weights[:10].sum() / total_weight) if total_weight > 0 else np.nan,
            "top50_fraction_total_weight": float(weights[:50].sum() / total_weight) if total_weight > 0 else np.nan,
            "top100_fraction_total_weight": float(weights[:100].sum() / total_weight) if total_weight > 0 else np.nan,
        }
        row.update(threshold_counts)
        summary_rows.append(row)
        weight_values.append(
            pd.DataFrame(
                {
                    "program_id": program_id,
                    "gene_weight": values,
                }
            )
        )

        keep = min(500, len(weights))
        for rank in range(1, keep + 1):
            curve_rows.append(
                {
                    "program_id": program_id,
                    "rank": rank,
                    "gene_weight": float(weights[rank - 1]),
                    "weight_fraction_of_max": float(weights[rank - 1] / weights[0]) if weights[0] else np.nan,
                    "cumulative_weight_fraction": float(cumulative[rank - 1]),
                }
            )
        for rank in [10, 25, 50, 75, 100, 150, 200, 300, 500, 1000]:
            if rank <= len(cumulative):
                cumulative_rows.append(
                    {
                        "program_id": program_id,
                        "rank": rank,
                        "cumulative_weight_fraction": float(cumulative[rank - 1]),
                    }
                )

    summary = pd.DataFrame(summary_rows)
    curves = pd.DataFrame(curve_rows)
    cumulative_checkpoints = pd.DataFrame(cumulative_rows)
    weight_all = pd.concat(weight_values, ignore_index=True)
    summary = summary.merge(assignment[["program_id", "cluster_path"]], on="program_id", how="left")
    summary["is_junk_by_original_top50"] = summary["program_id"].isin(junk_ids)

    summary_path = LOADING_DIAGNOSTIC_DIR / "raw_cnmf_gene_weight_distribution_summary.csv"
    curves_path = LOADING_DIAGNOSTIC_DIR / "raw_cnmf_gene_weight_ranked_curves_top500.csv"
    checkpoints_path = LOADING_DIAGNOSTIC_DIR / "raw_cnmf_gene_weight_cumulative_checkpoints.csv"
    signed_histogram_path = LOADING_DIAGNOSTIC_DIR / "raw_cnmf_gene_weight_distribution_histogram.csv"
    summary.to_csv(summary_path, index=False)
    curves.to_csv(curves_path, index=False)
    cumulative_checkpoints.to_csv(checkpoints_path, index=False)
    hist_counts, hist_edges = np.histogram(weight_all["gene_weight"], bins=400)
    pd.DataFrame(
        {
            "bin_left": hist_edges[:-1],
            "bin_right": hist_edges[1:],
            "count": hist_counts,
        }
    ).to_csv(signed_histogram_path, index=False)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for _, sub in curves.groupby("program_id"):
        axes[0].plot(sub["rank"], sub["weight_fraction_of_max"], color="#4C78A8", alpha=0.10, lw=0.8)
        axes[1].plot(sub["rank"], sub["cumulative_weight_fraction"], color="#54A24B", alpha=0.10, lw=0.8)
    median_curve = curves.groupby("rank", as_index=False).median(numeric_only=True)
    axes[0].plot(median_curve["rank"], median_curve["weight_fraction_of_max"], color="#1F4E79", lw=2.5, label="median")
    axes[1].plot(median_curve["rank"], median_curve["cumulative_weight_fraction"], color="#2E7D32", lw=2.5, label="median")
    for ax in axes:
        ax.set_xlabel("ranked raw cNMF gene-weight rank")
        ax.legend(frameon=False)
    axes[0].set_ylabel("gene weight / top gene weight")
    axes[0].set_title("Ranked raw cNMF gene-weight decay")
    axes[1].set_ylabel("cumulative total gene-weight fraction")
    axes[1].set_title("Cumulative raw cNMF gene-weight concentration")
    for rank in [50, 100]:
        axes[0].axvline(rank, color="black", ls="--", lw=0.8, alpha=0.45)
        axes[1].axvline(rank, color="black", ls="--", lw=0.8, alpha=0.45)
    fig.tight_layout()
    curves_fig = LOADING_DIAGNOSTIC_DIR / "raw_cnmf_gene_weight_elbow_curves.png"
    fig.savefig(curves_fig, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, title in [
        (axes[0], "elbow_rank_gene_weight", "Elbow rank"),
        (axes[1], "n_genes_for_80pct_total_weight", "Genes for 80% total weight"),
        (axes[2], "top50_fraction_total_weight", "Top-50 weight mass"),
    ]:
        sns.histplot(summary[col], bins=20, ax=ax, color="#4C78A8")
        ax.axvline(summary[col].median(), color="black", ls="--", lw=1)
        ax.set_title(title)
    fig.tight_layout()
    histogram_fig = LOADING_DIAGNOSTIC_DIR / "raw_cnmf_gene_weight_cutpoint_histograms.png"
    fig.savefig(histogram_fig, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    lo, hi = np.quantile(weight_all["gene_weight"], [0.001, 0.999])
    sns.histplot(weight_all["gene_weight"], bins=250, ax=axes[0], color="#4C78A8")
    axes[0].set_xlim(lo, hi)
    axes[0].set_yscale("log")
    axes[0].set_title("Pooled raw cNMF gene_spectra_tpm weight distribution")
    axes[0].set_xlabel("raw cNMF gene weight")
    axes[0].set_ylabel("gene-program entries, log scale")
    for _, sub in curves.groupby("program_id"):
        axes[1].plot(sub["rank"], sub["weight_fraction_of_max"], color="#4C78A8", alpha=0.10, lw=0.8)
    median_curve = curves.groupby("rank", as_index=False).median(numeric_only=True)
    axes[1].plot(median_curve["rank"], median_curve["weight_fraction_of_max"], color="#1F4E79", lw=2.5, label="median")
    axes[1].set_title("Ranked raw cNMF gene-weight decay")
    axes[1].set_xlabel("rank")
    axes[1].set_ylabel("gene weight / top gene weight")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    signed_distribution_fig = LOADING_DIAGNOSTIC_DIR / "raw_cnmf_gene_weight_distribution.png"
    fig.savefig(signed_distribution_fig, dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for _, sub in curves.groupby("program_id"):
        ax.plot(sub["rank"], sub["cumulative_weight_fraction"], color="#54A24B", alpha=0.10, lw=0.8)
    median_curve = curves.groupby("rank", as_index=False).median(numeric_only=True)
    ax.plot(median_curve["rank"], median_curve["cumulative_weight_fraction"], color="#2E7D32", lw=2.5, label="median")
    ax.set_xlabel("rank")
    ax.set_ylabel("cumulative total gene-weight fraction")
    ax.set_title("Raw cNMF gene-weight cumulative concentration")
    ax.legend(frameon=False)
    for rank in [50, 100]:
        ax.axvline(rank, color="black", ls="--", lw=0.8, alpha=0.45)
    fig.tight_layout()
    signed_cumulative_fig = LOADING_DIAGNOSTIC_DIR / "raw_cnmf_gene_weight_cumulative_curves.png"
    fig.savefig(signed_cumulative_fig, dpi=300)
    plt.close(fig)

    ordered_programs = summary.sort_values(["sample", "program_id"])["program_id"].tolist()
    global_lo, global_hi = np.quantile(weight_all["gene_weight"], [0.001, 0.999])
    ncols = 12
    nrows = math.ceil(len(ordered_programs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 2.0 * nrows), sharex=True, sharey=False)
    axes_flat = np.asarray(axes).ravel()
    for ax, program_id in zip(axes_flat, ordered_programs):
        vals = spectra[program_id].to_numpy(dtype=float)
        ax.hist(vals, bins=60, range=(global_lo, global_hi), color="#4C78A8", alpha=0.85)
        title = program_id
        if program_id in junk_ids:
            title += " [junk]"
        ax.set_title(title, fontsize=5)
        ax.tick_params(axis="both", labelsize=5, length=1.5)
    for ax in axes_flat[len(ordered_programs) :]:
        ax.axis("off")
    fig.suptitle("Per-program raw cNMF gene_spectra_tpm weight distributions", fontsize=14, y=0.995)
    fig.supxlabel("raw cNMF gene weight", fontsize=10)
    fig.supylabel("gene count", fontsize=10)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.985))
    per_program_distribution_fig = LOADING_DIAGNOSTIC_DIR / "raw_cnmf_gene_weight_distribution_by_program.png"
    fig.savefig(per_program_distribution_fig, dpi=300)
    plt.close(fig)

    return {
        "summary": summary_path,
        "curves": curves_path,
        "checkpoints": checkpoints_path,
        "signed_histogram": signed_histogram_path,
        "curves_figure": curves_fig,
        "histogram_figure": histogram_fig,
        "signed_distribution_figure": signed_distribution_fig,
        "signed_cumulative_figure": signed_cumulative_fig,
        "per_program_distribution_figure": per_program_distribution_fig,
    }


def program_sample_from_id(program_id: str) -> str:
    match = PROGRAM_RE.match(program_id)
    if not match:
        return ""
    return f"{match.group(1)}__{match.group(2)}"


def write_coarse_recluster_leaf_table(
    matrix: pd.DataFrame,
    order: list[str],
    assignment: pd.DataFrame,
    ref_map: dict[str, str],
    rho_map: dict[str, float],
    path: Path,
) -> pd.DataFrame:
    assigned = assignment.set_index("program_id")
    rows = []
    for i, program_id in enumerate(order):
        next_pid = order[i + 1] if i + 1 < len(order) else None
        rows.append(
            {
                "position": i + 1,
                "program_id": program_id,
                "sample": program_sample_from_id(program_id),
                "cluster_path_current": assigned.loc[program_id, "cluster_path"],
                "fine_current": fine_from_path(assigned.loc[program_id, "cluster_path"]),
                "reference_family": ref_map.get(program_id, "missing"),
                "spearman_rho": rho_map.get(program_id, np.nan),
                "jaccard_to_next_leaf": matrix.loc[program_id, next_pid] if next_pid else np.nan,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(path, index=False)
    return table


def draw_coarse_recluster_heatmap(
    coarse: str,
    matrix: pd.DataFrame,
    z: np.ndarray,
    assignment: pd.DataFrame,
    ref_map: dict[str, str],
    rho_map: dict[str, float],
    output_path: Path,
) -> None:
    assignment = assignment.set_index("program_id").reindex(matrix.index)
    assignment.index.name = "program_id"
    assignment = assignment.reset_index()

    fine_palette = {
        "1": "#8DD3C7",
        "2": "#FFFFB3",
        "3": "#BEBADA",
        "4": "#FB8072",
        "5": "#80B1D3",
        "6": "#FDB462",
        "7": "#B3DE69",
        "none": "#D9D9D9",
    }
    family_labels = sorted({ref_map.get(pid, "missing") for pid in matrix.index})
    family_colors = family_palette_from_snapshot()
    rho_norm = mcolors.TwoSlopeNorm(vmin=-0.4, vcenter=0.0, vmax=0.4)
    rho_cmap = plt.get_cmap("vlag")

    def rho_to_color(program_id: str) -> str:
        rho = rho_map.get(program_id)
        if pd.isna(rho):
            return "#D9D9D9"
        return mcolors.to_hex(rho_cmap(rho_norm(float(np.clip(rho, -0.4, 0.4)))))

    row_colors = pd.DataFrame(
        {"current fine": assignment["cluster_path"].map(fine_from_path).map(fine_palette).fillna("#D9D9D9").to_numpy()},
        index=assignment["program_id"],
    )
    col_colors = pd.DataFrame(
        {
            "reference family": [family_colors.get(ref_map.get(pid, "missing"), "#D9D9D9") for pid in matrix.index],
            "SNAI1-ac rho": [rho_to_color(pid) for pid in matrix.index],
        },
        index=matrix.index,
    )

    plot_matrix = matrix.copy()
    np.fill_diagonal(plot_matrix.values, np.nan)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    sns.set_theme(style="white")
    size = 7.5 if len(matrix) <= 25 else 9.5
    plot_z = linkage_for_visible_dendrogram(z)
    g = sns.clustermap(
        plot_matrix,
        row_linkage=plot_z,
        col_linkage=plot_z,
        row_colors=row_colors,
        col_colors=col_colors,
        cmap=cmap,
        vmin=0,
        vmax=0.2,
        xticklabels=False,
        yticklabels=False,
        linewidths=0,
        figsize=(size, size),
        dendrogram_ratio=(0.18, 0.18),
        tree_kws={"linewidths": 0.8},
        cbar_kws={"label": "Jaccard similarity"},
    )
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")
    g.ax_col_colors.set_title("reference family / SNAI1-ac rho", fontsize=9, pad=5)
    g.fig.suptitle(f"Variant B non-junk recluster within coarse {coarse}", y=1.02, fontsize=12)

    fine_labels = ["1", "2", "3", "4", "5", "6", "7", "none"]
    fine_handles = [Patch(facecolor=fine_palette[label], label=label) for label in fine_labels]
    family_handles = [Patch(facecolor=family_colors[label], label=label) for label in family_labels]
    family_legend = g.fig.legend(
        handles=family_handles,
        title="reference family",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.78),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    fine_legend = g.fig.legend(
        handles=fine_handles,
        title="current fine",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.43),
        frameon=False,
    )
    rho_mappable = plt.cm.ScalarMappable(norm=rho_norm, cmap=rho_cmap)
    rho_mappable.set_array([])
    rho_cax = g.fig.add_axes([1.30, 0.65, 0.02, 0.13])
    rho_cbar = g.fig.colorbar(rho_mappable, cax=rho_cax)
    rho_cbar.set_label("Spearman rho vs SNAI1-ac", fontsize=8)
    rho_cbar.set_ticks([-0.4, 0.0, 0.4])
    rho_cbar.ax.tick_params(labelsize=8)
    g.fig.add_artist(family_legend)
    g.fig.add_artist(fine_legend)
    g.fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(g.fig)


def export_coarse_recluster_inspection() -> pd.DataFrame:
    COARSE_RECLUSTER_DIR.mkdir(parents=True, exist_ok=True)
    assignment = pd.read_csv(ASSIGNMENT_FILE)
    matrix_full = pd.read_csv(VARIANT_B_MATRIX, index_col=0)
    ref_map = family_label_map_from_snapshot()
    corr = pd.read_csv(PROGRAMME_SNAI1AC_CORR_FILE, usecols=["program_id", "spearman_rho"]).drop_duplicates("program_id")
    rho_map = corr.set_index("program_id")["spearman_rho"].astype(float).to_dict()

    rows = []
    for coarse in ["A", "B", "C"]:
        program_ids = assignment.loc[assignment["cluster_coarse"] == coarse, "program_id"].tolist()
        submatrix = matrix_full.loc[program_ids, program_ids]
        z, order, coph = average_linkage_from_jaccard(submatrix)
        heatmap_path = COARSE_RECLUSTER_DIR / f"variantB_nonjunk_recluster_coarse_{coarse}.png"
        leaf_path = COARSE_RECLUSTER_DIR / f"variantB_nonjunk_recluster_coarse_{coarse}_leaf_order.csv"
        draw_coarse_recluster_heatmap(coarse, submatrix, z, assignment, ref_map, rho_map, heatmap_path)
        leaf_table = write_coarse_recluster_leaf_table(submatrix, order, assignment, ref_map, rho_map, leaf_path)
        rows.append(
            {
                "coarse": coarse,
                "n_programs": len(program_ids),
                "cophenetic_correlation": coph,
                "heatmap_path": str(heatmap_path),
                "leaf_order_path": str(leaf_path),
                "mean_jaccard_to_next_leaf": float(leaf_table["jaccard_to_next_leaf"].mean(skipna=True)),
                "min_jaccard_to_next_leaf": float(leaf_table["jaccard_to_next_leaf"].min(skipna=True)),
            }
        )
    summary = pd.DataFrame(rows)
    summary_path = COARSE_RECLUSTER_DIR / "variantB_nonjunk_coarse_recluster_summary.csv"
    summary.to_csv(summary_path, index=False)
    return summary


def safe_filename(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()


def write_family_recluster_leaf_table(
    matrix: pd.DataFrame,
    order: list[str],
    snapshot: pd.DataFrame,
    rho_map: dict[str, float],
    path: Path,
) -> pd.DataFrame:
    snap = snapshot.set_index("program_id")
    rows = []
    for i, program_id in enumerate(order):
        next_pid = order[i + 1] if i + 1 < len(order) else None
        rows.append(
            {
                "position": i + 1,
                "program_id": program_id,
                "sample": program_sample_from_id(program_id),
                "family_id": snap.loc[program_id, "family_id"],
                "family_label": snap.loc[program_id, "family_label"],
                "analysis_role": snap.loc[program_id, "analysis_role"],
                "alignment_category_draft": snap.loc[program_id, "alignment_category_draft"],
                "program_identity_draft": snap.loc[program_id, "program_identity_draft"],
                "spearman_rho": rho_map.get(program_id, np.nan),
                "jaccard_to_next_leaf": matrix.loc[program_id, next_pid] if next_pid else np.nan,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(path, index=False)
    return table


def draw_family_recluster_heatmap(
    family_label: str,
    matrix: pd.DataFrame,
    z: np.ndarray,
    snapshot: pd.DataFrame,
    family_colors: dict[str, str],
    rho_map: dict[str, float],
    output_path: Path,
) -> None:
    snapshot = snapshot.set_index("program_id").reindex(matrix.index)
    snapshot.index.name = "program_id"
    snapshot = snapshot.reset_index()
    rho_norm = mcolors.TwoSlopeNorm(vmin=-0.4, vcenter=0.0, vmax=0.4)
    rho_cmap = plt.get_cmap("vlag")

    def rho_to_color(program_id: str) -> str:
        rho = rho_map.get(program_id)
        if pd.isna(rho):
            return "#D9D9D9"
        return mcolors.to_hex(rho_cmap(rho_norm(float(np.clip(rho, -0.4, 0.4)))))

    family_color = family_colors.get(family_label, "#D9D9D9")
    row_colors = pd.DataFrame(
        {"family": [family_color for _ in matrix.index]},
        index=matrix.index,
    )
    col_colors = pd.DataFrame(
        {
            "family": [family_color for _ in matrix.index],
            "SNAI1-ac rho": [rho_to_color(pid) for pid in matrix.index],
        },
        index=matrix.index,
    )

    plot_matrix = matrix.copy()
    np.fill_diagonal(plot_matrix.values, np.nan)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    sns.set_theme(style="white")
    size = 6.8 if len(matrix) <= 12 else 8.2
    plot_z = linkage_for_visible_dendrogram(z)
    g = sns.clustermap(
        plot_matrix,
        row_linkage=plot_z,
        col_linkage=plot_z,
        row_colors=row_colors,
        col_colors=col_colors,
        cmap=cmap,
        vmin=0,
        vmax=0.2,
        xticklabels=False,
        yticklabels=False,
        linewidths=0,
        figsize=(size, size),
        dendrogram_ratio=(0.18, 0.18),
        tree_kws={"linewidths": 0.8},
        cbar_kws={"label": "Jaccard similarity"},
    )
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")
    g.ax_col_colors.set_title("family / SNAI1-ac rho", fontsize=9, pad=5)
    g.fig.suptitle(f"Variant B recluster within {family_label}", y=1.02, fontsize=12)

    family_handle = [Patch(facecolor=family_color, label=family_label)]
    family_legend = g.fig.legend(
        handles=family_handle,
        title="family",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.78),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    rho_mappable = plt.cm.ScalarMappable(norm=rho_norm, cmap=rho_cmap)
    rho_mappable.set_array([])
    rho_cax = g.fig.add_axes([1.22, 0.65, 0.022, 0.13])
    rho_cbar = g.fig.colorbar(rho_mappable, cax=rho_cax)
    rho_cbar.set_label("Spearman rho vs SNAI1-ac", fontsize=8)
    rho_cbar.set_ticks([-0.4, 0.0, 0.4])
    rho_cbar.ax.tick_params(labelsize=8)
    g.fig.add_artist(family_legend)
    g.fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(g.fig)


def draw_selected_family_pool_heatmap(
    matrix: pd.DataFrame,
    z: np.ndarray,
    snapshot: pd.DataFrame,
    family_colors: dict[str, str],
    rho_map: dict[str, float],
    output_path: Path,
) -> None:
    snapshot = snapshot.set_index("program_id").reindex(matrix.index)
    snapshot.index.name = "program_id"
    snapshot = snapshot.reset_index()
    rho_norm = mcolors.TwoSlopeNorm(vmin=-0.4, vcenter=0.0, vmax=0.4)
    rho_cmap = plt.get_cmap("vlag")

    def rho_to_color(program_id: str) -> str:
        rho = rho_map.get(program_id)
        if pd.isna(rho):
            return "#D9D9D9"
        return mcolors.to_hex(rho_cmap(rho_norm(float(np.clip(rho, -0.4, 0.4)))))

    row_colors = pd.DataFrame(
        {"family": snapshot["family_label"].map(family_colors).fillna("#D9D9D9").to_numpy()},
        index=snapshot["program_id"],
    )
    col_colors = pd.DataFrame(
        {
            "family": [family_colors.get(str(label), "#D9D9D9") for label in snapshot["family_label"]],
            "SNAI1-ac rho": [rho_to_color(pid) for pid in matrix.index],
        },
        index=matrix.index,
    )

    plot_matrix = matrix.copy()
    np.fill_diagonal(plot_matrix.values, np.nan)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    sns.set_theme(style="white")
    plot_z = linkage_for_visible_dendrogram(z)
    g = sns.clustermap(
        plot_matrix,
        row_linkage=plot_z,
        col_linkage=plot_z,
        row_colors=row_colors,
        col_colors=col_colors,
        cmap=cmap,
        vmin=0,
        vmax=0.2,
        xticklabels=False,
        yticklabels=False,
        linewidths=0,
        figsize=(11, 11),
        dendrogram_ratio=(0.18, 0.18),
        tree_kws={"linewidths": 0.8},
        cbar_kws={"label": "Jaccard similarity"},
    )
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")
    g.ax_col_colors.set_title("family / SNAI1-ac rho", fontsize=9, pad=5)
    g.fig.suptitle("Variant B recluster across selected reference families", y=1.02, fontsize=13)

    family_handles = [Patch(facecolor=family_colors[label], label=label) for label in FAMILY_RECLUSTER_LABELS]
    family_legend = g.fig.legend(
        handles=family_handles,
        title="family",
        loc="upper left",
        bbox_to_anchor=(1.00, 0.78),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    rho_mappable = plt.cm.ScalarMappable(norm=rho_norm, cmap=rho_cmap)
    rho_mappable.set_array([])
    rho_cax = g.fig.add_axes([1.25, 0.65, 0.02, 0.13])
    rho_cbar = g.fig.colorbar(rho_mappable, cax=rho_cax)
    rho_cbar.set_label("Spearman rho vs SNAI1-ac", fontsize=8)
    rho_cbar.set_ticks([-0.4, 0.0, 0.4])
    rho_cbar.ax.tick_params(labelsize=8)
    g.fig.add_artist(family_legend)
    g.fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(g.fig)


def export_family_recluster_inspection() -> pd.DataFrame:
    FAMILY_RECLUSTER_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = read_family_snapshot()
    matrix_full = pd.read_csv(VARIANT_B_MATRIX, index_col=0)
    corr = pd.read_csv(PROGRAMME_SNAI1AC_CORR_FILE, usecols=["program_id", "spearman_rho"]).drop_duplicates("program_id")
    rho_map = corr.set_index("program_id")["spearman_rho"].astype(float).to_dict()

    family_colors = family_palette_from_snapshot()
    selected_snapshot = snapshot.loc[snapshot["family_label"].astype(str).isin(FAMILY_RECLUSTER_LABELS)].copy()
    selected_program_ids = [pid for pid in selected_snapshot["program_id"].tolist() if pid in matrix_full.index]
    selected_matrix = matrix_full.loc[selected_program_ids, selected_program_ids]
    selected_z, selected_order, selected_coph = average_linkage_from_jaccard(selected_matrix)
    pooled_heatmap_path = FAMILY_RECLUSTER_DIR / "variantB_recluster_selected_families_pooled.png"
    pooled_leaf_path = FAMILY_RECLUSTER_DIR / "variantB_recluster_selected_families_pooled_leaf_order.csv"
    draw_selected_family_pool_heatmap(selected_matrix, selected_z, selected_snapshot, family_colors, rho_map, pooled_heatmap_path)
    pooled_leaf_table = write_family_recluster_leaf_table(
        selected_matrix,
        selected_order,
        selected_snapshot,
        rho_map,
        pooled_leaf_path,
    )
    removed_first12 = selected_order[:12]
    removed_first12_path = FAMILY_RECLUSTER_DIR / "variantB_recluster_selected_families_pooled_removed_first12.csv"
    selected_snapshot.set_index("program_id").reindex(removed_first12).reset_index().to_csv(removed_first12_path, index=False)
    remaining_after_first12 = [pid for pid in selected_program_ids if pid not in set(removed_first12)]
    trimmed_matrix = matrix_full.loc[remaining_after_first12, remaining_after_first12]
    trimmed_z, trimmed_order, trimmed_coph = average_linkage_from_jaccard(trimmed_matrix)
    trimmed_heatmap_path = FAMILY_RECLUSTER_DIR / "variantB_recluster_selected_families_pooled_without_first12.png"
    trimmed_leaf_path = FAMILY_RECLUSTER_DIR / "variantB_recluster_selected_families_pooled_without_first12_leaf_order.csv"
    trimmed_snapshot = selected_snapshot.loc[selected_snapshot["program_id"].isin(remaining_after_first12)].copy()
    draw_selected_family_pool_heatmap(trimmed_matrix, trimmed_z, trimmed_snapshot, family_colors, rho_map, trimmed_heatmap_path)
    trimmed_leaf_table = write_family_recluster_leaf_table(
        trimmed_matrix,
        trimmed_order,
        trimmed_snapshot,
        rho_map,
        trimmed_leaf_path,
    )
    removed_second_stage = trimmed_order[12:19]
    removed_second_stage_path = (
        FAMILY_RECLUSTER_DIR / "variantB_recluster_selected_families_pooled_without_first12_removed_leaf13_19.csv"
    )
    trimmed_snapshot.set_index("program_id").reindex(removed_second_stage).reset_index().to_csv(
        removed_second_stage_path,
        index=False,
    )
    remaining_after_second_stage = [pid for pid in remaining_after_first12 if pid not in set(removed_second_stage)]
    second_trimmed_matrix = matrix_full.loc[remaining_after_second_stage, remaining_after_second_stage]
    second_trimmed_z, second_trimmed_order, second_trimmed_coph = average_linkage_from_jaccard(second_trimmed_matrix)
    second_trimmed_heatmap_path = (
        FAMILY_RECLUSTER_DIR / "variantB_recluster_selected_families_pooled_without_first12_leaf13_19.png"
    )
    second_trimmed_leaf_path = (
        FAMILY_RECLUSTER_DIR / "variantB_recluster_selected_families_pooled_without_first12_leaf13_19_leaf_order.csv"
    )
    second_trimmed_snapshot = selected_snapshot.loc[selected_snapshot["program_id"].isin(remaining_after_second_stage)].copy()
    draw_selected_family_pool_heatmap(
        second_trimmed_matrix,
        second_trimmed_z,
        second_trimmed_snapshot,
        family_colors,
        rho_map,
        second_trimmed_heatmap_path,
    )
    second_trimmed_leaf_table = write_family_recluster_leaf_table(
        second_trimmed_matrix,
        second_trimmed_order,
        second_trimmed_snapshot,
        rho_map,
        second_trimmed_leaf_path,
    )

    rows = []
    for family_label in FAMILY_RECLUSTER_LABELS:
        subset = snapshot.loc[snapshot["family_label"].astype(str) == family_label].copy()
        program_ids = [pid for pid in subset["program_id"].tolist() if pid in matrix_full.index]
        missing = sorted(set(subset["program_id"]) - set(program_ids))
        submatrix = matrix_full.loc[program_ids, program_ids]
        z, order, coph = average_linkage_from_jaccard(submatrix)
        stem = safe_filename(family_label)
        heatmap_path = FAMILY_RECLUSTER_DIR / f"variantB_recluster_family_{stem}.png"
        leaf_path = FAMILY_RECLUSTER_DIR / f"variantB_recluster_family_{stem}_leaf_order.csv"
        draw_family_recluster_heatmap(family_label, submatrix, z, subset, family_colors, rho_map, heatmap_path)
        leaf_table = write_family_recluster_leaf_table(submatrix, order, subset, rho_map, leaf_path)
        rows.append(
            {
                "family_label": family_label,
                "n_programs": len(program_ids),
                "n_missing_from_matrix": len(missing),
                "missing_program_ids": ";".join(missing),
                "cophenetic_correlation": coph,
                "heatmap_path": str(heatmap_path),
                "leaf_order_path": str(leaf_path),
                "mean_jaccard_to_next_leaf": float(leaf_table["jaccard_to_next_leaf"].mean(skipna=True)),
                "min_jaccard_to_next_leaf": float(leaf_table["jaccard_to_next_leaf"].min(skipna=True)),
            }
        )
    summary = pd.DataFrame(rows)
    pooled_row = pd.DataFrame(
        [
            {
                "family_label": "POOLED_SELECTED_FAMILIES",
                "n_programs": len(selected_program_ids),
                "n_missing_from_matrix": int(len(selected_snapshot) - len(selected_program_ids)),
                "missing_program_ids": ";".join(sorted(set(selected_snapshot["program_id"]) - set(selected_program_ids))),
                "cophenetic_correlation": selected_coph,
                "heatmap_path": str(pooled_heatmap_path),
                "leaf_order_path": str(pooled_leaf_path),
                "mean_jaccard_to_next_leaf": float(pooled_leaf_table["jaccard_to_next_leaf"].mean(skipna=True)),
                "min_jaccard_to_next_leaf": float(pooled_leaf_table["jaccard_to_next_leaf"].min(skipna=True)),
            },
            {
                "family_label": "POOLED_SELECTED_FAMILIES_WITHOUT_FIRST12",
                "n_programs": len(remaining_after_first12),
                "n_missing_from_matrix": 0,
                "missing_program_ids": "",
                "cophenetic_correlation": trimmed_coph,
                "heatmap_path": str(trimmed_heatmap_path),
                "leaf_order_path": str(trimmed_leaf_path),
                "mean_jaccard_to_next_leaf": float(trimmed_leaf_table["jaccard_to_next_leaf"].mean(skipna=True)),
                "min_jaccard_to_next_leaf": float(trimmed_leaf_table["jaccard_to_next_leaf"].min(skipna=True)),
            },
            {
                "family_label": "POOLED_SELECTED_FAMILIES_WITHOUT_FIRST12_AND_TRIMMED_LEAF13_19",
                "n_programs": len(remaining_after_second_stage),
                "n_missing_from_matrix": 0,
                "missing_program_ids": "",
                "cophenetic_correlation": second_trimmed_coph,
                "heatmap_path": str(second_trimmed_heatmap_path),
                "leaf_order_path": str(second_trimmed_leaf_path),
                "mean_jaccard_to_next_leaf": float(second_trimmed_leaf_table["jaccard_to_next_leaf"].mean(skipna=True)),
                "min_jaccard_to_next_leaf": float(second_trimmed_leaf_table["jaccard_to_next_leaf"].min(skipna=True)),
            },
        ]
    )
    summary = pd.concat([pooled_row, summary], ignore_index=True)
    summary_path = FAMILY_RECLUSTER_DIR / "variantB_family_recluster_summary.csv"
    summary.to_csv(summary_path, index=False)
    return summary


def build_groups(assignment: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    assignment = assignment.copy()
    assignment["fine"] = assignment["cluster_path"].map(fine_from_path)
    assignment["c1c2"] = assignment["cluster_path"].map(c1c2_from_path)
    groups: dict[str, dict[str, list[str]]] = {"coarse": {}, "intermediate": {}, "fine": {}}
    for label in ["A", "B", "C"]:
        groups["coarse"][label] = assignment.loc[assignment["cluster_coarse"].eq(label), "program_id"].tolist()
    for label in ["C1", "C2"]:
        groups["intermediate"][label] = assignment.loc[assignment["c1c2"].eq(label), "program_id"].tolist()
    for label in [str(i) for i in range(1, 8)]:
        groups["fine"][label] = assignment.loc[assignment["fine"].eq(label), "program_id"].tolist()
    return groups


def group_gene_stats(
    level: str,
    group: str,
    members: list[str],
    level_groups: dict[str, list[str]],
    all_programs: list[str],
    top_genes: dict[str, list[str]],
    top_scores: dict[str, dict[str, float]],
) -> pd.DataFrame:
    member_sets = {pid: set(top_genes[pid][:50]) for pid in members}
    outside = [pid for pid in all_programs if pid not in set(members)]
    outside_sets = {pid: set(top_genes[pid][:50]) for pid in outside}
    genes = sorted(set().union(*member_sets.values())) if member_sets else []
    rows = []
    for gene in genes:
        count_in = sum(gene in geneset for geneset in member_sets.values())
        frac_in = count_in / len(members) if members else np.nan
        loadings = [top_scores[pid][gene] for pid in members if gene in top_scores[pid]]
        count_outside = sum(gene in geneset for geneset in outside_sets.values())
        other_fracs = []
        for other_group, other_members in level_groups.items():
            if other_group == group or not other_members:
                continue
            other_count = sum(gene in top_genes[pid][:50] for pid in other_members)
            other_fracs.append(other_count / len(other_members))
        max_other = max(other_fracs) if other_fracs else 0.0
        rows.append(
            {
                "level": level,
                "group": group,
                "n_programs_in_group": len(members),
                "gene": gene,
                "recurrence_count_in_MP": count_in,
                "recurrence_fraction_in_MP": frac_in,
                "mean_loading_in_MP": float(np.mean(loadings)) if loadings else np.nan,
                "median_loading_in_MP": float(np.median(loadings)) if loadings else np.nan,
                "recurrence_count_outside_MP": count_outside,
                "max_recurrence_fraction_other_MP": max_other,
                "specificity_delta": frac_in - max_other,
                "specificity_ratio": frac_in / (max_other + PSEUDOCOUNT),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["level", "group", "recurrence_count_in_MP", "recurrence_fraction_in_MP", "mean_loading_in_MP"],
            ascending=[True, True, False, False, False],
        ).reset_index(drop=True)
        out["rank_within_group"] = out.groupby(["level", "group"]).cumcount() + 1
    return out


def threshold_grid(full: pd.DataFrame) -> pd.DataFrame:
    rows = []
    thresholds = [("frac_ge_1_5", 1 / 5), ("frac_ge_1_4", 1 / 4), ("frac_ge_1_3", 1 / 3), ("frac_ge_1_2", 1 / 2)]
    for (level, group), sub in full.groupby(["level", "group"], sort=False):
        n_members = int(sub["n_programs_in_group"].iloc[0])
        for label, fraction in thresholds:
            passed = sub.loc[sub["recurrence_fraction_in_MP"] >= fraction].copy()
            for _, row in passed.iterrows():
                rows.append(
                    {
                        "threshold_label": label,
                        "threshold_fraction": fraction,
                        "threshold_count": math.ceil(fraction * n_members),
                        "candidate_count_for_group_threshold": len(passed),
                        **row.to_dict(),
                    }
                )
        discrete = math.ceil(n_members / 3)
        passed = sub.loc[sub["recurrence_count_in_MP"] >= discrete].copy()
        for _, row in passed.iterrows():
            rows.append(
                {
                    "threshold_label": "discrete_count_ge_ceil_n_over_3",
                    "threshold_fraction": np.nan,
                    "threshold_count": discrete,
                    "candidate_count_for_group_threshold": len(passed),
                    **row.to_dict(),
                }
            )
    return pd.DataFrame(rows)


def plot_recurrence_curves(full: pd.DataFrame, path: Path) -> None:
    sns.set_theme(style="whitegrid")
    groups = list(full[["level", "group"]].drop_duplicates().itertuples(index=False, name=None))
    fig, axes = plt.subplots(4, 3, figsize=(13, 12), sharey=True)
    axes = axes.ravel()
    for ax, (level, group) in zip(axes, groups):
        sub = full.loc[(full["level"].eq(level)) & (full["group"].eq(group))].copy()
        sub = sub.sort_values(["recurrence_fraction_in_MP", "mean_loading_in_MP"], ascending=False)
        ax.plot(np.arange(1, len(sub) + 1), sub["recurrence_fraction_in_MP"], color="#39568C", lw=1.8)
        ax.set_title(f"{level}: {group} (n={int(sub['n_programs_in_group'].iloc[0])})", fontsize=10)
        ax.set_xlabel("gene rank")
        ax.set_ylabel("member fraction")
        ax.set_ylim(-0.02, 1.02)
    for ax in axes[len(groups) :]:
        ax.axis("off")
    fig.suptitle("Per-group recurrence-fraction curves; no threshold chosen", y=0.995, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_threshold_sizes(grid: pd.DataFrame, path: Path) -> None:
    keep = ["frac_ge_1_5", "frac_ge_1_4", "frac_ge_1_3", "frac_ge_1_2"]
    size = grid.drop_duplicates(["level", "group", "threshold_label"]).loc[lambda x: x["threshold_label"].isin(keep)].copy()
    size["threshold_label"] = pd.Categorical(size["threshold_label"], keep, ordered=True)
    size["group_label"] = size["level"] + ":" + size["group"].astype(str)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=size, x="threshold_label", y="candidate_count_for_group_threshold", hue="group_label", marker="o", ax=ax)
    ax.set_xlabel("candidate threshold")
    ax.set_ylabel("genes passing threshold")
    ax.set_title("Signature-size sensitivity grid; inspection only")
    ax.legend(title="group", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_specificity(full: pd.DataFrame, path: Path) -> None:
    top = full.loc[full["rank_within_group"] <= 80].copy()
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.scatterplot(
        data=top,
        x="recurrence_fraction_in_MP",
        y="max_recurrence_fraction_other_MP",
        hue="level",
        style="level",
        alpha=0.75,
        s=30,
        ax=ax,
    )
    ax.plot([0, 1], [0, 1], color="0.45", lw=1, ls="--")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Specificity diagnostic for top recurrent genes; no filter applied")
    ax.set_xlabel("recurrence fraction within group")
    ax.set_ylabel("max recurrence fraction in another same-level group")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def export_core_reproduction(assignment: pd.DataFrame, top_genes: dict[str, list[str]]) -> pd.DataFrame:
    REPRO_OUT.mkdir(parents=True, exist_ok=True)
    program_ids = assignment["program_id"].tolist()
    junk = content_junk(top_genes)
    junk.to_csv(REPRO_OUT / "recomputed_objective_junk_from_signed_top50.csv", index=False)

    matrix_b_saved = pd.read_csv(VARIANT_B_MATRIX, index_col=0).loc[program_ids, program_ids]
    matrix_b_recomputed = jaccard_matrix(program_ids, top_genes, n=50)
    max_abs_diff = float((matrix_b_saved - matrix_b_recomputed).abs().to_numpy().max())
    z_b, order_b, coph_b = average_linkage_from_jaccard(matrix_b_saved)

    pd.DataFrame(z_b, columns=["child_1", "child_2", "distance", "n_leaves"]).to_csv(
        REPRO_OUT / "recomputed_variantB_nonjunk_average_linkage_Z.csv", index=False
    )
    pd.DataFrame({"position": range(1, len(order_b) + 1), "program_id": order_b}).to_csv(
        REPRO_OUT / "recomputed_variantB_nonjunk_leaf_order.csv", index=False
    )
    manual_rows = []
    for position, program_id in enumerate(order_b, start=1):
        coarse, mid, fine, path = manual_cut_from_position(position)
        manual_rows.append(
            {
                "position": position,
                "program_id": program_id,
                "cluster_coarse": coarse,
                "cluster_mid": mid,
                "cluster_fine": fine,
                "cluster_path": path,
            }
        )
    pd.DataFrame(manual_rows).to_csv(REPRO_OUT / "recomputed_manual_cut_from_leaf_positions.csv", index=False)

    matrix_a_saved = pd.read_csv(VARIANT_A_MATRIX, index_col=0).loc[program_ids, program_ids]
    z_a, order_a, coph_a = average_linkage_from_jaccard(matrix_a_saved)
    pd.DataFrame(z_a, columns=["child_1", "child_2", "distance", "n_leaves"]).to_csv(
        REPRO_OUT / "recomputed_variantA_nonjunk_average_linkage_Z.csv", index=False
    )
    pd.DataFrame({"position": range(1, len(order_a) + 1), "program_id": order_a}).to_csv(
        REPRO_OUT / "recomputed_variantA_nonjunk_leaf_order.csv", index=False
    )

    summary = pd.DataFrame(
        [
            {"check": "variantB_top50_recomputed_vs_saved_matrix_max_abs_diff", "value": max_abs_diff},
            {"check": "variantB_average_cophenetic", "value": coph_b},
            {"check": "variantA_average_cophenetic", "value": coph_a},
            {"check": "nonjunk_programs", "value": len(program_ids)},
            {"check": "objective_junk_rows_recomputed_for_nonjunk_subset", "value": int(junk["objective_junk"].sum())},
        ]
    )
    summary.to_csv(REPRO_OUT / "core_reproduction_summary.csv", index=False)
    return summary


def run_recurrence_diagnostics(assignment: pd.DataFrame, top_genes: dict[str, list[str]], top_scores: dict[str, dict[str, float]]) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    program_ids = assignment["program_id"].tolist()
    groups = build_groups(assignment)
    full_tables = []
    for level, level_groups in groups.items():
        for group, members in level_groups.items():
            full_tables.append(group_gene_stats(level, group, members, level_groups, program_ids, top_genes, top_scores))
    full = pd.concat(full_tables, ignore_index=True)
    grid = threshold_grid(full)

    full.to_csv(OUT / "MP_gene_recurrence_full_table.csv", index=False)
    grid.to_csv(OUT / "MP_signature_candidates_threshold_grid.csv", index=False)
    plot_recurrence_curves(full, OUT / "MP_recurrence_distribution_plots.png")
    plot_threshold_sizes(grid, OUT / "MP_signature_size_vs_threshold.png")
    plot_specificity(full, OUT / "MP_specificity_diagnostic.png")

    summary = (
        full.groupby(["level", "group"], as_index=False)
        .agg(
            n_programs=("n_programs_in_group", "first"),
            n_genes_seen=("gene", "nunique"),
            max_recurrence_fraction=("recurrence_fraction_in_MP", "max"),
            n_genes_frac_ge_1_3=("recurrence_fraction_in_MP", lambda x: int((x >= 1 / 3).sum())),
            n_genes_frac_ge_1_2=("recurrence_fraction_in_MP", lambda x: int((x >= 1 / 2).sum())),
        )
        .sort_values(["level", "group"])
    )
    summary.to_csv(OUT / "MP_recurrence_group_summary.csv", index=False)
    return summary


def locked_group_name(level: str, group: str) -> str:
    if level == "fine":
        return f"MP{group}"
    return group


def confidence_flag(group_name: str) -> str:
    if group_name in {"MP1", "MP2", "MP3", "MP4", "MP5", "C1"}:
        return "firm"
    if group_name == "MP6":
        return "provisional"
    if group_name in {"MP7", "C2"}:
        return "provisional, low-content"
    return ""


def locked_signature_rows(full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = [("fine", str(i)) for i in range(1, 8)] + [("intermediate", "C1"), ("intermediate", "C2")]
    rows = []
    checks = []
    for level, group in targets:
        group_name = locked_group_name(level, group)
        sub = full.loc[(full["level"].eq(level)) & (full["group"].astype(str).eq(group))].copy()
        sub = sub.sort_values(
            ["recurrence_count_in_MP", "recurrence_fraction_in_MP", "mean_loading_in_MP"],
            ascending=[False, False, False],
        )
        strict_all = sub.loc[sub["recurrence_fraction_in_MP"] >= STRICT_FRACTION].copy()
        strict = strict_all.head(STRICT_CAP).copy()
        checks.append(
            {
                "level": level,
                "group": group_name,
                "n_programs_in_group": int(sub["n_programs_in_group"].iloc[0]),
                "strict_uncapped_n": len(strict_all),
                "strict_locked_n": len(strict),
                "capped_at_50": len(strict_all) > STRICT_CAP,
            }
        )
        for rank, (_, row) in enumerate(strict.iterrows(), start=1):
            rows.append(signature_row(row, level, group_name, "strict", rank))

        if group_name in {"MP7", "C2"}:
            relaxed = sub.head(RELAXED_ORA_SIZE).copy()
            for rank, (_, row) in enumerate(relaxed.iterrows(), start=1):
                rows.append(signature_row(row, level, group_name, "relaxed_ora", rank))

    locked = pd.DataFrame(rows)
    checks_df = pd.DataFrame(checks)
    return locked, checks_df


def signature_row(row: pd.Series, level: str, group_name: str, set_type: str, rank: int) -> dict[str, object]:
    return {
        "level": level,
        "group": group_name,
        "n_programs_in_group": int(row["n_programs_in_group"]),
        "set_type": set_type,
        "rank_within_set": rank,
        "gene": row["gene"],
        "recurrence_count_in_MP": int(row["recurrence_count_in_MP"]),
        "recurrence_fraction_in_MP": float(row["recurrence_fraction_in_MP"]),
        "mean_loading_in_MP": float(row["mean_loading_in_MP"]),
        "median_loading_in_MP": float(row["median_loading_in_MP"]),
        "recurrence_count_outside_MP": int(row["recurrence_count_outside_MP"]),
        "max_recurrence_fraction_other_MP": float(row["max_recurrence_fraction_other_MP"]),
        "specificity_delta": float(row["specificity_delta"]),
        "specificity_ratio": float(row["specificity_ratio"]),
        "confidence_flag": confidence_flag(group_name),
    }


def cluster_member_ids(assignment: pd.DataFrame, label: str) -> list[str]:
    if label in {"1", "2", "3", "4"}:
        path = {"1": "A/1", "2": "A/2", "3": "B/3", "4": "B/4"}[label]
        return assignment.loc[assignment["cluster_path"].eq(path), "program_id"].tolist()
    if label == "5":
        return assignment.loc[assignment["cluster_path"].eq("C/C1/5"), "program_id"].tolist()
    if label == "6":
        return assignment.loc[assignment["cluster_path"].eq("C/6"), "program_id"].tolist()
    if label == "7":
        return assignment.loc[assignment["cluster_path"].eq("C/C2/7"), "program_id"].tolist()
    if label == "C1":
        return assignment.loc[assignment["cluster_path"].astype(str).str.startswith("C/C1"), "program_id"].tolist()
    if label == "C2":
        return assignment.loc[assignment["cluster_path"].astype(str).str.startswith("C/C2"), "program_id"].tolist()
    raise ValueError(f"Unknown cluster label: {label}")


def split_functional_terms(value: object) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    out = []
    for item in value.split("|"):
        term = item.strip()
        if not term:
            continue
        term = term.split(":NES=", 1)[0].strip()
        out.append(term)
    return out


def split_cell_associations(value: object) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    out = []
    for item in value.split("|"):
        cell = item.strip().split(":rho=", 1)[0].strip()
        if cell:
            out.append(cell)
    return out


def counted_digest(values: list[str], top_n: int = 8) -> str:
    if not values:
        return ""
    counts = pd.Series(values).value_counts().head(top_n)
    return "; ".join(f"{idx} (n={int(val)})" for idx, val in counts.items())


def build_cluster_summary(assignment: pd.DataFrame, locked: pd.DataFrame) -> pd.DataFrame:
    ref = pd.read_csv(REFERENCE_FILE, usecols=["program_id", "family_label", "top_functional_terms_v0_2"])
    ann = assignment.merge(ref, on="program_id", how="left")
    rows = []
    for label in ["1", "2", "3", "4", "5", "6", "7", "C1", "C2"]:
        group_name = f"MP{label}" if label.isdigit() else label
        member_ids = cluster_member_ids(ann, label)
        sub = ann.loc[ann["program_id"].isin(member_ids)].copy()
        strict = locked.loc[(locked["group"].eq(group_name)) & (locked["set_type"].eq("strict"))].sort_values("rank_within_set")
        display = group_name
        flag = confidence_flag(group_name)
        if flag:
            display = f"{display} [{flag}]"
        top_recurrent = "; ".join(strict["gene"].head(15).tolist())
        sub = sub.set_index("program_id").loc[member_ids].reset_index()
        for _, row in sub.iterrows():
            rows.append(
                {
                    "Jaccard MP": display,
                    "member programs": row["program_id"],
                    "program name": row.get("alignment_category_draft", ""),
                    "old manual family labels": row.get("family_label", ""),
                    "top recurrent genes": top_recurrent,
                    "GSEA terms": row.get("top_functional_terms_v0_2", ""),
                    "cell-fraction associations": row.get("top_cell_fraction_associations", ""),
                }
            )
    return pd.DataFrame(rows)


def export_locked_signatures(assignment: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full = pd.read_csv(OUT / "MP_gene_recurrence_full_table.csv")
    locked, checks = locked_signature_rows(full)
    locked.to_csv(OUT / "MP_signatures_locked_long.csv", index=False)
    checks.to_csv(OUT / "MP_signatures_locked_size_check.csv", index=False)

    with pd.ExcelWriter(OUT / "MP_signatures_locked_by_group.xlsx") as writer:
        for group, sub in locked.groupby("group", sort=False):
            sheet = str(group).replace("/", "_")[:31]
            sub.to_excel(writer, sheet_name=sheet, index=False)

    summary = build_cluster_summary(assignment, locked)
    summary.to_csv(OUT / "MP_cluster_summary.csv", index=False)
    return locked, checks, summary


def update_report_signature_section(checks: pd.DataFrame) -> None:
    if not REPORT_FILE.exists():
        return
    text = REPORT_FILE.read_text(encoding="utf-8")
    section_title = "\n## Signature Lock-In\n"
    if section_title in text:
        text = text.split(section_title)[0].rstrip() + "\n"

    strict_sizes = ", ".join(
        f"{row.group}: {int(row.strict_locked_n)}"
        + (f" (uncapped {int(row.strict_uncapped_n)})" if bool(row.capped_at_50) else "")
        for row in checks.itertuples(index=False)
    )
    section = f"""
## Signature Lock-In

Strict metaprogram signatures were locked from the Variant B non-junk recurrence table using recurrence_fraction_in_MP >= 1/3, equivalently at least ceil(n/3) member programs. Genes were ranked by recurrence_count_in_MP descending, then recurrence_fraction_in_MP descending, then mean_loading_in_MP descending. Strict signatures were capped at the top 50 genes; this cap applies to MP3 and MP4.

Strict locked sizes: {strict_sizes}.

Confidence flags are carried as metadata and are not filters. Firm groups are MP1, MP2, MP3, MP4, MP5, and C1. Provisional groups are MP6, MP7, and C2; MP7 and C2 are also marked low-content. MP6 is provisional because n=5, despite a coherent cell-cycle/histone core.

Relaxed ORA input sets were created only for MP7 and C2 because their strict signatures are too short for enrichment. The relaxed_ora set is the top 20 genes by the same recurrence ranking. It is explicitly an ORA input only, not the biological signature; the strict set remains the biological claim.

Specificity is reported, never filtered. Each locked-gene row carries recurrence_count_in_MP, recurrence_fraction_in_MP, mean_loading_in_MP, median_loading_in_MP, recurrence_count_outside_MP, max_recurrence_fraction_other_MP, specificity_delta, and specificity_ratio.

Signature outputs:

- D:\\HGSOC_Spatial_Atlas\\05_analysis_ready\\S3_cNMF_Tumor_Programs\\jaccard_raw_matrices\\inspection_exports_average\\variantB_nonjunk_manual_cut_v2\\recurrence_specificity_diagnostics\\MP_signatures_locked_long.csv
- D:\\HGSOC_Spatial_Atlas\\05_analysis_ready\\S3_cNMF_Tumor_Programs\\jaccard_raw_matrices\\inspection_exports_average\\variantB_nonjunk_manual_cut_v2\\recurrence_specificity_diagnostics\\MP_signatures_locked_by_group.xlsx
- D:\\HGSOC_Spatial_Atlas\\05_analysis_ready\\S3_cNMF_Tumor_Programs\\jaccard_raw_matrices\\inspection_exports_average\\variantB_nonjunk_manual_cut_v2\\recurrence_specificity_diagnostics\\MP_cluster_summary.csv
"""
    REPORT_FILE.write_text(text.rstrip() + "\n" + section, encoding="utf-8")


def read_gmt(path: Path, library: str) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            term_id = parts[0]
            genes = sorted(set(gene for gene in parts[2:] if gene))
            rows.append({"library": library, "term_id": term_id, "term": term_id, "genes": genes})
    return pd.DataFrame(rows)


def export_go_rds_to_terms(path: Path, library: str) -> pd.DataFrame:
    r_code = r'''
args <- commandArgs(trailingOnly = TRUE)
in_path <- args[[1]]
out_path <- args[[2]]
x <- readRDS(in_path)
ids <- names(x$pathways)
term_names <- unname(x$term_names[ids])
term_names[is.na(term_names)] <- ids[is.na(term_names)]
genes <- vapply(x$pathways[ids], function(g) paste(unique(g[!is.na(g) & g != ""]), collapse = ";"), character(1))
out <- data.frame(term_id = ids, term = term_names, genes = genes, stringsAsFactors = FALSE)
write.csv(out, out_path, row.names = FALSE)
'''
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        r_path = tmp_dir / "export_go_terms.R"
        out_path = tmp_dir / f"{library}.csv"
        r_path.write_text(r_code, encoding="utf-8")
        subprocess.run([str(RSCRIPT), str(r_path), str(path), str(out_path)], check=True)
        df = pd.read_csv(out_path)
    df["library"] = library
    df["genes"] = df["genes"].fillna("").map(lambda value: sorted(set(str(value).split(";")) - {""}))
    return df[["library", "term_id", "term", "genes"]]


def load_ora_libraries() -> pd.DataFrame:
    frames = [
        read_gmt(HALLMARK_GMT, "HALLMARK"),
        read_gmt(KEGG_LEGACY_GMT, "KEGG_LEGACY"),
        export_go_rds_to_terms(GO_BP_RDS, "GO_BP"),
        export_go_rds_to_terms(GO_CC_RDS, "GO_CC"),
    ]
    return pd.concat(frames, ignore_index=True)


def bh_adjust(pvalues: pd.Series) -> pd.Series:
    values = pvalues.astype(float).to_numpy()
    n = len(values)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for i in range(n - 1, -1, -1):
        running = min(running, ranked[i] * n / (i + 1))
        adjusted[order[i]] = running
    return pd.Series(np.minimum(adjusted, 1.0), index=pvalues.index)


def run_signature_ora(program_ids: list[str]) -> tuple[pd.DataFrame, int]:
    locked = pd.read_csv(OUT / "MP_signatures_locked_long.csv")
    universe = load_gene_spectra_universe(program_ids)
    universe_n = len(universe)
    libraries = load_ora_libraries()

    target = pd.concat(
        [
            locked.loc[(locked["set_type"].eq("strict")) & (locked["group"].isin(["MP1", "MP2", "MP3", "MP4", "MP5", "MP6", "C1"]))],
            locked.loc[(locked["set_type"].eq("relaxed_ora")) & (locked["group"].isin(["MP7", "C2"]))],
        ],
        ignore_index=True,
    )
    rows = []
    for (level, group, set_type), sig in target.groupby(["level", "group", "set_type"], sort=False):
        sig = sig.sort_values("rank_within_set")
        sig_genes = [gene for gene in sig["gene"].astype(str).tolist() if gene in universe]
        sig_set = set(sig_genes)
        sig_rank = {gene: i for i, gene in enumerate(sig_genes)}
        spec = sig.set_index("gene")["specificity_delta"].astype(float).to_dict()
        n = len(sig_set)
        if n == 0:
            continue
        for _, term in libraries.iterrows():
            term_genes = set(term["genes"]) & universe
            k = len(sig_set & term_genes)
            if k == 0:
                continue
            term_size = len(term_genes)
            overlap = sorted(sig_set & term_genes, key=lambda gene: sig_rank.get(gene, 10**9))
            deltas = [float(spec.get(gene, np.nan)) for gene in overlap]
            rows.append(
                {
                    "level": level,
                    "group": group,
                    "set_type": set_type,
                    "confidence_flag": confidence_flag(group),
                    "descriptive_only": bool(set_type == "relaxed_ora"),
                    "library": term["library"],
                    "term_id": term["term_id"],
                    "term": term["term"],
                    "overlap_k": k,
                    "signature_size_n": n,
                    "term_size_K": term_size,
                    "universe_N": universe_n,
                    "fold_enrichment": (k / n) / (term_size / universe_n) if term_size else np.nan,
                    "pvalue": float(hypergeom.sf(k - 1, universe_n, term_size, n)),
                    "overlap_genes": ";".join(overlap),
                    "n_overlap_specific": int(sum(delta >= 0.3 for delta in deltas if np.isfinite(delta))),
                    "n_overlap_shared": int(sum(delta < 0.1 for delta in deltas if np.isfinite(delta))),
                    "frac_overlap_specific": float(sum(delta >= 0.3 for delta in deltas if np.isfinite(delta)) / k),
                    "mean_overlap_specificity_delta": float(np.nanmean(deltas)),
                    "driven_by_shared_only": bool(all(delta < 0.1 for delta in deltas if np.isfinite(delta))),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        result.to_csv(OUT / "MP_signature_ORA_long.csv", index=False)
        return result, universe_n
    result["padj"] = np.nan
    for _, idx in result.groupby(["group", "set_type", "library"]).groups.items():
        result.loc[idx, "padj"] = bh_adjust(result.loc[idx, "pvalue"]).values
    result = result.loc[result["padj"] < 0.05].copy()
    result = result.sort_values(["group", "set_type", "library", "padj", "fold_enrichment"], ascending=[True, True, True, True, False])
    ordered_cols = [
        "level",
        "group",
        "set_type",
        "confidence_flag",
        "descriptive_only",
        "library",
        "term_id",
        "term",
        "overlap_k",
        "signature_size_n",
        "term_size_K",
        "universe_N",
        "fold_enrichment",
        "pvalue",
        "padj",
        "overlap_genes",
        "n_overlap_specific",
        "n_overlap_shared",
        "frac_overlap_specific",
        "mean_overlap_specificity_delta",
        "driven_by_shared_only",
    ]
    result = result[ordered_cols]
    result.to_csv(OUT / "MP_signature_ORA_long.csv", index=False)

    with pd.ExcelWriter(OUT / "MP_signature_ORA_by_group.xlsx") as writer:
        for group, sub in result.groupby("group", sort=False):
            display_cols = ["term", "library", "fold_enrichment", "padj", "overlap_genes", "driven_by_shared_only"]
            sub.sort_values(["padj", "fold_enrichment"], ascending=[True, False]).head(20)[display_cols].to_excel(
                writer, sheet_name=str(group)[:31], index=False
            )
    return result, universe_n


def update_report_ora_section(universe_n: int) -> None:
    if not REPORT_FILE.exists():
        return
    text = REPORT_FILE.read_text(encoding="utf-8")
    section_title = "\n## Signature ORA Method\n"
    if section_title in text:
        text = text.split(section_title)[0].rstrip() + "\n"
    section = f"""
## Signature ORA Method

Locked-signature functional annotation evidence was generated with an offline over-representation analysis. No metaprogram labels were assigned, proposed, ranked, or interpreted in this step.

The input signatures were pulled verbatim from MP_signatures_locked_long.csv. ORA was run for strict MP1, MP2, MP3, MP4, MP5, MP6, and C1 signatures, plus relaxed_ora MP7 and C2 input sets. The relaxed_ora sets are descriptive ORA inputs only and are not biological signatures.

The four libraries were matched to the member-program GSEA layer: HALLMARK and KEGG_LEGACY from MSigDB v2025.1.Hs GMT files, and GO_BP and GO_CC from the cached local Bioconductor org.Hs.eg.db/GO.db pathway RDS files used for the member-program fgsea runs.

The ORA universe was the explicit custom Variant-B cNMF gene_spectra_score universe across the 124 non-junk programs, deduplicated in the analysis namespace. Universe size N = {universe_n}. Enrichr's online whole-genome background was avoided because it would not match the genes eligible to enter these cNMF signatures and can inflate significance.

For each term, the script reports overlap_k, signature_size_n, term_size_K, universe_N, fold_enrichment = (k/n)/(K/N), raw hypergeometric p value, and Benjamini-Hochberg adjusted p value within each group/set/library test family. All terms with padj < 0.05 are retained; no additional filters are applied.

The passenger screen joins locked-gene specificity_delta onto each term's overlap genes and reports overlap_genes ordered by recurrence rank, n_overlap_specific (specificity_delta >= 0.3), n_overlap_shared (specificity_delta < 0.1), frac_overlap_specific, mean_overlap_specificity_delta, and driven_by_shared_only. These columns are for inspection only.

ORA outputs:

- D:\\HGSOC_Spatial_Atlas\\05_analysis_ready\\S3_cNMF_Tumor_Programs\\jaccard_raw_matrices\\inspection_exports_average\\variantB_nonjunk_manual_cut_v2\\recurrence_specificity_diagnostics\\MP_signature_ORA_long.csv
- D:\\HGSOC_Spatial_Atlas\\05_analysis_ready\\S3_cNMF_Tumor_Programs\\jaccard_raw_matrices\\inspection_exports_average\\variantB_nonjunk_manual_cut_v2\\recurrence_specificity_diagnostics\\MP_signature_ORA_by_group.xlsx
"""
    REPORT_FILE.write_text(text.rstrip() + "\n" + section, encoding="utf-8")


def export_final_labels_only() -> tuple[Path, Path, Path]:
    labels = pd.DataFrame(FINAL_LABEL_ROWS)
    labels.to_csv(FINAL_LABELS_FILE, index=False)

    summary = pd.read_csv(OUT / "MP_cluster_summary.csv")
    summary = summary.copy()
    summary["_group_key"] = summary["Jaccard MP"].astype(str).str.split(" ", n=1).str[0]
    labelled = summary.merge(labels[["group", "final_label", "label_type", "confidence"]], left_on="_group_key", right_on="group", how="left")
    labelled = labelled.drop(columns=["_group_key", "group"])
    cols = ["Jaccard MP", "final_label", "label_type", "confidence"] + [
        col for col in labelled.columns if col not in {"Jaccard MP", "final_label", "label_type", "confidence"}
    ]
    labelled = labelled[cols]
    labelled.to_csv(LABELLED_CLUSTER_SUMMARY_FILE, index=False)

    update_report_final_labels_section(labels)
    return FINAL_LABELS_FILE, LABELLED_CLUSTER_SUMMARY_FILE, REPORT_FILE


def update_report_final_labels_section(labels: pd.DataFrame) -> None:
    if not REPORT_FILE.exists():
        return
    text = REPORT_FILE.read_text(encoding="utf-8")
    section_title = "\n## Final Labels\n"
    if section_title in text:
        text = text.split(section_title)[0].rstrip() + "\n"
    table = labels.to_markdown(index=False)
    section = f"""
## Final Labels

The final metaprogram labels below are recorded verbatim from the human labelling checkpoint. This section does not re-run ORA, re-derive labels, reinterpret evidence, or link metaprograms to SNAI1-ac.

{table}

Standing reading rules used in the labelling:

1. HALLMARK_EMT in stromal/CAF programmes is matrisome/ECM, not epithelial EMT.
2. Generic extracellular-localization GO terms (extracellular space/region/exosome/vesicle/organelle) are large-K, low-information; read past them to functional terms when labelling.

Final-label outputs:

- D:\\HGSOC_Spatial_Atlas\\05_analysis_ready\\S3_cNMF_Tumor_Programs\\jaccard_raw_matrices\\inspection_exports_average\\variantB_nonjunk_manual_cut_v2\\recurrence_specificity_diagnostics\\MP_final_labels.csv
- D:\\HGSOC_Spatial_Atlas\\05_analysis_ready\\S3_cNMF_Tumor_Programs\\jaccard_raw_matrices\\inspection_exports_average\\variantB_nonjunk_manual_cut_v2\\recurrence_specificity_diagnostics\\MP_cluster_summary_labelled.csv
"""
    REPORT_FILE.write_text(text.rstrip() + "\n" + section, encoding="utf-8")


def bh_adjust_values(values: pd.Series) -> pd.Series:
    return bh_adjust(pd.to_numeric(values, errors="coerce"))


def final_label_lookup() -> pd.DataFrame:
    labels = pd.read_csv(FINAL_LABELS_FILE)
    return labels[["group", "final_label"]].copy()


def group_membership_table(assignment: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in ["1", "2", "3", "4", "5", "6", "7", "C1", "C2"]:
        group = f"MP{label}" if label.isdigit() else label
        for program_id in cluster_member_ids(assignment, label):
            rows.append({"group": group, "program_id": program_id})
    return pd.DataFrame(rows)


def load_native_usage_member_effects(assignment: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    membership = group_membership_table(assignment)
    labels = final_label_lookup()
    context_frames = []
    for context, path in [("centre", HH_SPOT_CONTRASTS), ("neighborhood", HH_NEIGHBORHOOD_CONTRASTS)]:
        df = pd.read_csv(path)
        df = df.merge(membership, on="program_id", how="inner")
        df["context"] = context
        df = df.rename(columns={"cohens_dz_hh_minus_nonhh": "dz"})
        context_frames.append(df)
    member = pd.concat(context_frames, ignore_index=True)
    member = member.merge(labels, on="group", how="left")
    member = member[
        [
            "source",
            "context",
            "dataset",
            "sample_id_on_disk",
            "sample_label",
            "group",
            "final_label",
            "program_id",
            "n_pairs",
            "dz",
            "hh_mean",
            "matched_nonhh_mean",
            "mean_difference_hh_minus_nonhh",
            "median_difference_hh_minus_nonhh",
            "direction",
            "p_value",
            "fdr_bh",
        ]
    ] if "source" in member.columns else member[
        [
            "context",
            "dataset",
            "sample_id_on_disk",
            "sample_label",
            "group",
            "final_label",
            "program_id",
            "n_pairs",
            "dz",
            "hh_mean",
            "matched_nonhh_mean",
            "mean_difference_hh_minus_nonhh",
            "median_difference_hh_minus_nonhh",
            "direction",
            "p_value",
            "fdr_bh",
        ]
    ]
    member.insert(0, "source", "cnmf_usage_native")

    assigned_unique = set(membership["program_id"])
    all_kstar = set(pd.read_csv(HH_SPOT_CONTRASTS, usecols=["program_id"])["program_id"])
    coverage = {
        "assigned_to_9_groups_unique": len(assigned_unique),
        "not_assigned_to_9_groups": len(all_kstar - assigned_unique),
        "all_kstar_programs": len(all_kstar),
    }
    return member, coverage


def collapse_native_usage_per_sample(member: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["context", "dataset", "sample_id_on_disk", "sample_label", "group", "final_label"]
    for keys, sub in member.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, keys, strict=False))
        row["source"] = "cnmf_usage_native"
        row["dz"] = float(sub["dz"].mean())
        row["n_pairs"] = int(sub["n_pairs"].max())
        row["n_member_programs_in_sample"] = int(sub["program_id"].nunique())
        row["member_programs"] = ";".join(sub["program_id"].astype(str).tolist())
        rows.append(row)
    out = pd.DataFrame(rows)
    return out[
        [
            "source",
            "context",
            "dataset",
            "sample_id_on_disk",
            "sample_label",
            "group",
            "final_label",
            "n_pairs",
            "dz",
            "n_member_programs_in_sample",
            "member_programs",
        ]
    ]


def meta_analyse_native_usage(per_sample: pd.DataFrame, assignment: pd.DataFrame) -> pd.DataFrame:
    membership = group_membership_table(assignment)
    member_counts = membership.groupby("group")["program_id"].nunique().to_dict()
    rows = []
    for (context, group), sub in per_sample.groupby(["context", "group"], sort=False):
        weights = sub["n_pairs"].astype(float).to_numpy()
        dz = sub["dz"].astype(float).to_numpy()
        total_pairs = float(weights.sum())
        weighted_mean = float(np.sum(weights * dz) / total_pairs) if total_pairs else np.nan
        meta_z = float(weighted_mean * np.sqrt(total_pairs)) if total_pairs else np.nan
        meta_p = float(2 * norm.sf(abs(meta_z))) if np.isfinite(meta_z) else np.nan
        n_enriched = int(np.sum(dz > 0))
        n_depleted = int(np.sum(dz < 0))
        dominant_direction = "HH_enriched" if n_enriched >= n_depleted else "HH_depleted"
        dominant_n = max(n_enriched, n_depleted)
        directional_n = n_enriched + n_depleted
        binom_p = float(binomtest(dominant_n, directional_n, 0.5, alternative="two-sided").pvalue) if directional_n else np.nan
        rows.append(
            {
                "group": group,
                "variant": "native_usage",
                "method": "cnmf_usage_native",
                "source": "cnmf_usage_native",
                "context": context,
                "n_samples": int(sub["sample_label"].nunique()),
                "n_member_programs": int(member_counts.get(group, 0)),
                "final_label": str(sub["final_label"].dropna().iloc[0]) if sub["final_label"].notna().any() else "",
                "total_pairs": int(total_pairs),
                "weighted_mean_dz": weighted_mean,
                "meta_z": meta_z,
                "meta_p": meta_p,
                "n_HH_enriched": n_enriched,
                "n_HH_depleted": n_depleted,
                "direction_consistency_fraction": float(dominant_n / directional_n) if directional_n else np.nan,
                "dominant_direction": dominant_direction,
                "binomial_direction_p": binom_p,
                "median_abs_dz": float(np.median(np.abs(dz))),
            }
        )
    meta = pd.DataFrame(rows)
    meta["meta_fdr_bh"] = np.nan
    meta["direction_fdr_bh"] = np.nan
    for _, idx in meta.groupby("context").groups.items():
        meta.loc[idx, "meta_fdr_bh"] = bh_adjust_values(meta.loc[idx, "meta_p"]).values
        meta.loc[idx, "direction_fdr_bh"] = bh_adjust_values(meta.loc[idx, "binomial_direction_p"]).values
    order = ["MP1", "MP2", "MP3", "MP4", "MP5", "MP6", "C1", "MP7", "C2"]
    meta["group"] = pd.Categorical(meta["group"], order, ordered=True)
    meta["context"] = pd.Categorical(meta["context"], ["centre", "neighborhood"], ordered=True)
    meta = meta.sort_values(["context", "group"]).reset_index(drop=True)
    meta["group"] = meta["group"].astype(str)
    meta["context"] = meta["context"].astype(str)
    return meta


def export_native_usage_meta_only() -> tuple[Path, Path, Path, pd.DataFrame, dict[str, int]]:
    USAGE_OUT.mkdir(parents=True, exist_ok=True)
    assignment = pd.read_csv(ASSIGNMENT_FILE)
    member, coverage = load_native_usage_member_effects(assignment)
    per_sample = collapse_native_usage_per_sample(member)
    meta = meta_analyse_native_usage(per_sample, assignment)

    member_path = USAGE_OUT / "mp_usage_native_member_effects.csv"
    per_sample_path = USAGE_OUT / "mp_usage_native_per_sample.csv"
    meta_path = USAGE_OUT / "mp_usage_native_meta_analysis.csv"
    member.to_csv(member_path, index=False)
    per_sample.to_csv(per_sample_path, index=False)
    meta.to_csv(meta_path, index=False)
    update_report_native_usage_section()
    return meta_path, per_sample_path, member_path, meta, coverage


def update_report_native_usage_section() -> None:
    if not REPORT_FILE.exists():
        return
    text = REPORT_FILE.read_text(encoding="utf-8")
    section_title = "\n## Native cNMF Usage HH Meta-Analysis Method\n"
    if section_title in text:
        text = text.split(section_title)[0].rstrip() + "\n"
    section = f"""
## Native cNMF Usage HH Meta-Analysis Method

This step aggregates SNAI1-ac HH-vs-matched-nonHH effects to the metaprogram level using each member program's native cNMF usage. It does not build, score, or use any gene signature, and it does not use EnrichMap.

Inputs are the original 1:1 optimal malignant-fraction matched HH-vs-nonHH per-program usage contrast tables that produced the lab-meeting composite figures: hh_spot_level_programme_contrasts.csv for centre spots and hh_neighborhood_programme_contrasts.csv for 1-ring neighborhoods. The per-program effect size is Cohen's dz for paired HH minus matched-nonHH usage differences, with positive dz meaning HH-enriched.

Groups are MP1, MP2, MP3, MP4, MP5, MP6, C1, MP7, and C2 from program_cluster_assignment_v2. Nesting is retained, so MP5 members also contribute to C1 and MP7 members also contribute to C2. Junk programs are not group members.

To guard against within-sample pseudo-replication, if a sample contributes more than one member program to a group, member-program dz values are collapsed to one per-sample dz by averaging before meta-analysis. The independent unit is therefore the sample. n_samples is variable by group and reflects samples where at least one member program of that group was fit.

For each group and context, the meta table reports weighted_mean_dz using n_pairs as weights, total_pairs, meta_z, meta_p, n_HH_enriched, n_HH_depleted, direction_consistency_fraction, dominant_direction, two-sided binomial_direction_p, median_abs_dz, and BH FDR values across the nine groups within each context. Direction consistency is treated as a primary result.

Native-usage outputs:

- D:\\HGSOC_Spatial_Atlas\\05_analysis_ready\\S3_cNMF_Tumor_Programs\\jaccard_raw_matrices\\inspection_exports_average\\variantB_nonjunk_manual_cut_v2\\native_usage_hh_meta\\mp_usage_native_meta_analysis.csv
- D:\\HGSOC_Spatial_Atlas\\05_analysis_ready\\S3_cNMF_Tumor_Programs\\jaccard_raw_matrices\\inspection_exports_average\\variantB_nonjunk_manual_cut_v2\\native_usage_hh_meta\\mp_usage_native_per_sample.csv
- D:\\HGSOC_Spatial_Atlas\\05_analysis_ready\\S3_cNMF_Tumor_Programs\\jaccard_raw_matrices\\inspection_exports_average\\variantB_nonjunk_manual_cut_v2\\native_usage_hh_meta\\mp_usage_native_member_effects.csv
"""
    REPORT_FILE.write_text(text.rstrip() + "\n" + section, encoding="utf-8")


def export_locked_linkage_figures_only() -> tuple[Path, Path, Path]:
    MP_TRUE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    native_meta = pd.read_csv(USAGE_OUT / "mp_usage_native_meta_analysis.csv")
    native_sample = pd.read_csv(USAGE_OUT / "mp_usage_native_per_sample.csv")
    enrich_meta = pd.read_csv(MP_TRUE_META)

    forest_path = MP_TRUE_FIG_DIR / "mp_native_usage_forest_centre_neighborhood.png"
    dumbbell_path = MP_TRUE_FIG_DIR / "mp_enrichmap_vs_native_usage_centre_dumbbell.png"
    plot_native_usage_forest(native_sample, native_meta, forest_path)
    plot_reconciliation_dumbbell(native_meta, enrich_meta, dumbbell_path)
    update_mp_true_manifest([forest_path, dumbbell_path])
    update_report_locked_interpretation_section(forest_path, dumbbell_path)
    return forest_path, dumbbell_path, MP_TRUE_MANIFEST


def plot_native_usage_forest(native_sample: pd.DataFrame, native_meta: pd.DataFrame, path: Path) -> None:
    order = ["MP1", "MP2", "MP3", "MP4", "MP5", "MP6", "C1", "MP7", "C2"]
    y_lookup = {group: idx for idx, group in enumerate(order)}
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True, sharex=True)
    for ax, context in zip(axes, ["centre", "neighborhood"], strict=False):
        sub_sample = native_sample.loc[native_sample["context"].eq(context)].copy()
        sub_meta = native_meta.loc[native_meta["context"].eq(context)].copy()
        for group in order:
            vals = sub_sample.loc[sub_sample["group"].eq(group), "dz"].astype(float).to_numpy()
            y = y_lookup[group]
            if len(vals):
                ax.scatter(vals, np.full(len(vals), y), color="0.65", s=24, alpha=0.8, zorder=2)
            meta_row = sub_meta.loc[sub_meta["group"].eq(group)]
            if not meta_row.empty:
                x = float(meta_row["weighted_mean_dz"].iloc[0])
                ax.scatter([x], [y], marker="D", s=70, color="#2C7FB8", edgecolor="white", linewidth=0.8, zorder=3)
        ax.axvline(0, color="black", lw=1)
        ax.set_title(context)
        ax.set_xlabel("Cohen's dz (HH - matched non-HH)")
        ax.set_yticks(list(y_lookup.values()))
        ax.set_yticklabels(order)
        ax.invert_yaxis()
    all_x = pd.concat([native_sample["dz"], native_meta["weighted_mean_dz"]]).astype(float)
    lim = max(0.5, float(np.nanmax(np.abs(all_x))) * 1.15)
    for ax in axes:
        ax.set_xlim(-lim, lim)
        for group in order:
            row = native_meta.loc[(native_meta["context"].eq(ax.get_title())) & (native_meta["group"].eq(group))]
            if not row.empty:
                pct = float(row["direction_consistency_fraction"].iloc[0]) * 100
                ax.text(lim * 0.98, y_lookup[group], f"{pct:.0f}%", va="center", ha="right", fontsize=8)
    axes[0].set_ylabel("Jaccard metaprogramme")
    fig.suptitle("Native cNMF usage: SNAI1-ac HH vs matched non-HH", y=0.98)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_reconciliation_dumbbell(native_meta: pd.DataFrame, enrich_meta: pd.DataFrame, path: Path) -> None:
    order = ["MP1", "MP2", "MP3", "MP4", "MP5", "MP6", "C1", "MP7", "C2"]
    native = native_meta.loc[native_meta["context"].eq("centre"), ["group", "weighted_mean_dz"]].rename(
        columns={"weighted_mean_dz": "native_usage_centre_dz"}
    )
    enrich = enrich_meta.loc[
        enrich_meta["method"].eq("enrichmap") & enrich_meta["variant"].eq("strict"),
        ["group", "weighted_mean_dz"],
    ].rename(columns={"weighted_mean_dz": "enrichmap_strict_dz"})
    df = pd.DataFrame({"group": order}).merge(enrich, on="group", how="left").merge(native, on="group", how="left")
    y_lookup = {group: idx for idx, group in enumerate(order)}
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in df.iterrows():
        y = y_lookup[row["group"]]
        x1 = float(row["enrichmap_strict_dz"]) if pd.notna(row["enrichmap_strict_dz"]) else np.nan
        x2 = float(row["native_usage_centre_dz"]) if pd.notna(row["native_usage_centre_dz"]) else np.nan
        if np.isfinite(x1) and np.isfinite(x2):
            ax.plot([x1, x2], [y, y], color="0.65", lw=1.5, zorder=1)
        if np.isfinite(x1):
            ax.scatter([x1], [y], color="#D95F02", s=55, label="EnrichMap strict" if y == 0 else None, zorder=2)
        if np.isfinite(x2):
            ax.scatter([x2], [y], color="#2C7FB8", s=55, label="Native usage centre" if y == 0 else None, zorder=3)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(list(y_lookup.values()))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("Weighted mean Cohen's dz (HH - matched non-HH)")
    ax.set_ylabel("Jaccard metaprogramme")
    ax.set_title("EnrichMap strict vs native cNMF usage centre effect")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def update_mp_true_manifest(paths: list[Path]) -> None:
    manifest = json.loads(MP_TRUE_MANIFEST.read_text(encoding="utf-8")) if MP_TRUE_MANIFEST.exists() else {}
    figures = list(manifest.get("figures", []))
    for path in paths:
        path_str = str(path)
        if path_str not in figures:
            figures.append(path_str)
    manifest["figures"] = figures
    manifest["native_usage_linkage_figures"] = [str(path) for path in paths]
    MP_TRUE_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def update_report_locked_interpretation_section(forest_path: Path, dumbbell_path: Path) -> None:
    if not REPORT_FILE.exists():
        return
    text = REPORT_FILE.read_text(encoding="utf-8")
    section_title = "\n## Locked Interpretation\n"
    if section_title in text:
        text = text.split(section_title)[0].rstrip() + "\n"
    section = f"""
## Locked Interpretation

[TO CONFIRM: verbatim locked finding text was not present in the instruction block available to this script. Insert the exact locked finding here before treating this report section as final.]

Interpretation guardrail: a significant effect-size meta-p with approximately 50/50 direction consistency is sample-dependent, not a reproducible state.

The EnrichMap identity-signature result is retained only as a sensitivity demonstrating the tumour-identity/density confound, and is superseded by the native-usage result for any cross-sample claim.

Convergence with GASTON and SpottedPy findings: [TO CONFIRM: record the exact convergence wording approved for the thesis.]

Held-out independent cohort: [TO CONFIRM: record the designated held-out cohort name/path for future replication of any directional hint, e.g. hypoxia. No directional hint is claimed here.]

Locked-linkage figures:

- {forest_path}
- {dumbbell_path}
"""
    REPORT_FILE.write_text(text.rstrip() + "\n" + section, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jaccard metaprogram pipeline utilities.")
    parser.add_argument(
        "--final-labels-only",
        action="store_true",
        help="Only write final labels, labelled cluster summary, and report section; do not rerun analysis.",
    )
    parser.add_argument(
        "--native-usage-meta-only",
        action="store_true",
        help="Only aggregate native cNMF usage HH-vs-matched effects to metaprogram level; do not rerun analysis.",
    )
    parser.add_argument(
        "--locked-linkage-figures-only",
        action="store_true",
        help="Only regenerate locked SNAI1-ac linkage figures from existing tables and update manifest/report pointers.",
    )
    parser.add_argument(
        "--manual-cut-heatmap-only",
        action="store_true",
        help="Only recreate the Variant B non-junk manual assignment table and polished manual-cut heatmap.",
    )
    parser.add_argument(
        "--manual-cut-rho-ordered-heatmap-only",
        action="store_true",
        help="Only draw an inspection heatmap ordered by per-program SNAI1-ac Spearman rho.",
    )
    parser.add_argument(
        "--program-loading-elbow-only",
        action="store_true",
        help="Only export raw non-negative cNMF gene_spectra_tpm weight concentration/elbow diagnostics.",
    )
    parser.add_argument(
        "--pos-neg-variantb-only",
        action="store_true",
        help="Only build the Variant B top50-positive plus top50-negative signed-feature Jaccard branch.",
    )
    parser.add_argument(
        "--neg-variantb-only",
        action="store_true",
        help="Only build the Variant B top50-negative signed-feature Jaccard branch.",
    )
    parser.add_argument(
        "--coarse-recluster-only",
        action="store_true",
        help="Only recluster manual coarse clusters A/B/C independently for inspection.",
    )
    parser.add_argument(
        "--family-recluster-only",
        action="store_true",
        help="Only recluster selected old/reference families independently for inspection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.final_labels_only:
        paths = export_final_labels_only()
        print("Final labels recorded without rerunning analysis.")
        for path in paths:
            print(path)
        return
    if args.native_usage_meta_only:
        meta_path, per_sample_path, member_path, meta, coverage = export_native_usage_meta_only()
        print("Native cNMF usage HH meta-analysis exported without signature scoring.")
        print(meta_path)
        print(per_sample_path)
        print(member_path)
        print(f"Coverage: {coverage}")
        print(meta[["group", "context", "n_samples", "n_member_programs"]].to_string(index=False))
        print("No gene signature was used.")
        return
    if args.locked_linkage_figures_only:
        forest_path, dumbbell_path, manifest_path = export_locked_linkage_figures_only()
        print("Locked linkage figures regenerated from existing tables.")
        print(forest_path)
        print(dumbbell_path)
        print(manifest_path)
        print("No models, signatures, scoring, or relabelling were run.")
        return
    if args.manual_cut_heatmap_only:
        assignment_path, heatmap_path, coph = export_manual_cut_assignment_and_heatmap()
        print("Variant B non-junk manual cut assignment and heatmap recreated.")
        print(assignment_path)
        print(heatmap_path)
        print(f"Cophenetic correlation: {coph:.6f}")
        print("No matrix construction, clustering choice, signature scoring, ORA, or relabelling was run.")
        return
    if args.manual_cut_rho_ordered_heatmap_only:
        heatmap_path, order_path = export_manual_cut_rho_ordered_heatmap()
        print("Variant B non-junk rho-ordered inspection heatmap exported.")
        print(heatmap_path)
        print(order_path)
        print("Rows/columns are sorted by Spearman rho descending; no clustering choice or assignment was changed.")
        return
    if args.program_loading_elbow_only:
        paths = export_program_loading_elbow_diagnostics()
        summary = pd.read_csv(paths["summary"])
        print("Raw non-negative cNMF gene_spectra_tpm weight distribution diagnostics exported.")
        for path in paths.values():
            print(path)
        cols = [
            "elbow_rank_gene_weight",
            "n_genes_for_80pct_total_weight",
            "n_genes_for_90pct_total_weight",
            "top50_fraction_total_weight",
            "top100_fraction_total_weight",
        ]
        print(summary[cols].describe().loc[["min", "50%", "mean", "max"]].to_string())
        print("Raw cNMF gene-weight distributions and cumulative concentration were exported; no cutoff was chosen or applied.")
        return
    if args.pos_neg_variantb_only:
        paths = export_pos_neg_variant_b()
        print("Variant B pos+neg signed-feature Jaccard branch exported.")
        for path in paths.values():
            print(path)
        distribution = pd.read_csv(paths["distribution"])
        nonjunk_distribution = pd.read_csv(paths["nonjunk_distribution"])
        print("Full matrix distribution:")
        print(distribution.to_string(index=False))
        print("Non-junk matrix distribution:")
        print(nonjunk_distribution.to_string(index=False))
        print("Existing positive-only Variant B files were not modified.")
        return
    if args.neg_variantb_only:
        paths = export_neg_variant_b()
        print("Variant B negative-tail signed-feature Jaccard branch exported.")
        for path in paths.values():
            print(path)
        distribution = pd.read_csv(paths["distribution"])
        nonjunk_distribution = pd.read_csv(paths["nonjunk_distribution"])
        print("Full matrix distribution:")
        print(distribution.to_string(index=False))
        print("Non-junk matrix distribution:")
        print(nonjunk_distribution.to_string(index=False))
        print("Existing positive-only and pos+neg Variant B files were not modified.")
        return
    if args.coarse_recluster_only:
        summary = export_coarse_recluster_inspection()
        print("Variant B non-junk coarse A/B/C reclustering inspection exported.")
        print(COARSE_RECLUSTER_DIR)
        print(summary.to_string(index=False))
        print("No new cluster assignments, signatures, scoring, ORA, or relabelling were run.")
        return
    if args.family_recluster_only:
        summary = export_family_recluster_inspection()
        print("Variant B selected-family reclustering inspection exported.")
        print(FAMILY_RECLUSTER_DIR)
        print(summary.to_string(index=False))
        print("Families used:")
        for family_label in FAMILY_RECLUSTER_LABELS:
            print(f"- {family_label}")
        print("No new cluster assignments, signatures, scoring, ORA, or relabelling were run.")
        return

    np.random.seed(SEED)
    assignment = pd.read_csv(ASSIGNMENT_FILE)
    program_ids = assignment["program_id"].tolist()
    top_genes, top_scores = load_program_scores(program_ids, top_n=100)

    core_summary = export_core_reproduction(assignment, top_genes)
    recurrence_summary = run_recurrence_diagnostics(assignment, top_genes, top_scores)
    locked, signature_checks, cluster_summary = export_locked_signatures(assignment)
    update_report_signature_section(signature_checks)
    ora_result, universe_n = run_signature_ora(program_ids)
    update_report_ora_section(universe_n)

    print("Jaccard metaprogram pipeline update complete.")
    print(f"Seed: {SEED}")
    print(f"Core reproducibility outputs: {REPRO_OUT}")
    print(f"Recurrence/specificity outputs: {OUT}")
    print(core_summary.to_string(index=False))
    print(recurrence_summary.to_string(index=False))
    print(signature_checks.to_string(index=False))
    print(f"Locked signature rows: {len(locked)}")
    print(f"Cluster summary rows: {len(cluster_summary)}")
    print(f"ORA universe N: {universe_n}")
    print(f"ORA significant rows: {len(ora_result)}")
    print("No labels were assigned.")


if __name__ == "__main__":
    main()
