from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

import family_topgene_coherence_plots as base


HALLMARK_GENE_SETS = base.ROOT / "05_analysis_ready" / "Signature" / "hallmark_gene_sets.json"
FIG_OUT = base.OUT / "figures" / "gene_recurrence_functional_order"
TABLE_OUT = base.OUT / "tables" / "gene_recurrence_functional_order"

GROUP_RULES = [
    {
        "group": "Chemokine/cytokine",
        "terms": {
            "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
            "HALLMARK_INFLAMMATORY_RESPONSE",
            "HALLMARK_IL6_JAK_STAT3_SIGNALING",
            "HALLMARK_IL2_STAT5_SIGNALING",
        },
        "patterns": [r"^CXCL\d", r"^CCL\d", r"^IL\d", r"^TNF"],
    },
    {
        "group": "Antigen presentation",
        "terms": {"HALLMARK_ALLOGRAFT_REJECTION"},
        "patterns": [r"^HLA-", r"^B2M$", r"^TAP", r"^PSMB", r"^CD74$"],
    },
    {
        "group": "IFN/antiviral",
        "terms": {
            "HALLMARK_INTERFERON_ALPHA_RESPONSE",
            "HALLMARK_INTERFERON_GAMMA_RESPONSE",
        },
        "patterns": [
            r"^IFI",
            r"^IFIT",
            r"^IFITM",
            r"^OAS",
            r"^MX\d",
            r"^ISG",
            r"^GBP",
            r"^RSAD2$",
            r"^DDX58$",
            r"^STAT1$",
            r"^IRF7$",
            r"^WARS$",
            r"^WARS1$",
            r"^HERC",
            r"^HELZ2$",
            r"^SAMD9",
            r"^TRIM22$",
            r"^XAF1$",
            r"^BST2$",
            r"^PLSCR1$",
            r"^UBE2L6$",
            r"^SHFL$",
            r"^ZBP1$",
            r"^SP110$",
            r"^APOL",
        ],
    },
    {
        "group": "Complement/coagulation",
        "terms": {"HALLMARK_COMPLEMENT", "HALLMARK_COAGULATION"},
        "patterns": [
            r"^C1Q",
            r"^C1R$",
            r"^C1S$",
            r"^C[2345679]$",
            r"^C8[ABG]$",
            r"^CF[ABDHI]$",
            r"^SERPIN",
            r"^A2M$",
            r"^CLU$",
            r"^F[0-9]+$",
        ],
    },
    {
        "group": "Myeloid/phagolysosome",
        "terms": {"HALLMARK_ALLOGRAFT_REJECTION", "HALLMARK_COMPLEMENT"},
        "patterns": [
            r"^CD68$",
            r"^CD14$",
            r"^CD163$",
            r"^CSF1R$",
            r"^TYROBP$",
            r"^TREM2$",
            r"^FCGR",
            r"^LAPTM5$",
            r"^LYZ$",
            r"^AIF1$",
            r"^SPI1$",
            r"^CT[SHDLBZ]",
            r"^LGMN$",
            r"^LIPA$",
            r"^APOE$",
            r"^APOC",
            r"^GPNMB$",
        ],
    },
    {
        "group": "ECM/EMT/adhesion",
        "terms": {
            "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
            "HALLMARK_APICAL_JUNCTION",
            "HALLMARK_TGF_BETA_SIGNALING",
        },
        "patterns": [
            r"^COL",
            r"^MMP",
            r"^TIMP",
            r"^FN1$",
            r"^POSTN$",
            r"^DCN$",
            r"^LUM$",
            r"^SPARC$",
            r"^FAP$",
            r"^BGN$",
            r"^VCAN$",
            r"^THBS",
            r"^FBLN",
            r"^FBN",
            r"^TNC$",
            r"^TAGLN$",
            r"^ACTA2$",
            r"^LGALS1$",
        ],
    },
    {
        "group": "Vascular/pericyte",
        "terms": {"HALLMARK_ANGIOGENESIS"},
        "patterns": [
            r"^VWF$",
            r"^PECAM1$",
            r"^CDH5$",
            r"^KDR$",
            r"^FLT1$",
            r"^ENG$",
            r"^RGS5$",
            r"^MCAM$",
            r"^CLEC14A$",
            r"^ESAM$",
            r"^CD93$",
            r"^DLL4$",
            r"^PLVAP$",
            r"^ESM1$",
            r"^GJA4$",
            r"^CLDN5$",
            r"^ROBO4$",
            r"^TIE1$",
            r"^NOTCH3$",
        ],
    },
    {
        "group": "Cell cycle/DNA repair",
        "terms": {
            "HALLMARK_E2F_TARGETS",
            "HALLMARK_G2M_CHECKPOINT",
            "HALLMARK_MITOTIC_SPINDLE",
            "HALLMARK_DNA_REPAIR",
            "HALLMARK_MYC_TARGETS_V1",
            "HALLMARK_MYC_TARGETS_V2",
        },
        "patterns": [r"^MKI67$", r"^TOP2A$", r"^CDC", r"^CDK", r"^CENP", r"^MCM", r"^PCNA$"],
    },
    {
        "group": "Metabolism/OXPHOS",
        "terms": {
            "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
            "HALLMARK_GLYCOLYSIS",
            "HALLMARK_FATTY_ACID_METABOLISM",
            "HALLMARK_CHOLESTEROL_HOMEOSTASIS",
            "HALLMARK_ADIPOGENESIS",
            "HALLMARK_BILE_ACID_METABOLISM",
            "HALLMARK_PEROXISOME",
            "HALLMARK_MTORC1_SIGNALING",
            "HALLMARK_XENOBIOTIC_METABOLISM",
        },
        "patterns": [r"^NDUF", r"^COX", r"^ATP5", r"^UQCR", r"^SLC2A1$", r"^LDHA$", r"^PGK1$"],
    },
    {
        "group": "Ribosome/translation",
        "terms": {"HALLMARK_MYC_TARGETS_V1", "HALLMARK_MYC_TARGETS_V2", "HALLMARK_UNFOLDED_PROTEIN_RESPONSE"},
        "patterns": [r"^RPL", r"^RPS", r"^MRPL", r"^MRPS", r"^EEF", r"^EIF", r"^FAU$", r"^UBB$"],
    },
    {
        "group": "Hypoxia/stress",
        "terms": {
            "HALLMARK_HYPOXIA",
            "HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY",
            "HALLMARK_P53_PATHWAY",
            "HALLMARK_APOPTOSIS",
            "HALLMARK_UNFOLDED_PROTEIN_RESPONSE",
        },
        "patterns": [r"^DDIT4$", r"^NDRG1$", r"^CA9$", r"^VEGFA$", r"^HMOX1$", r"^ATF3$", r"^HSPA"],
    },
    {
        "group": "Secretory/epithelial",
        "terms": {
            "HALLMARK_PROTEIN_SECRETION",
            "HALLMARK_APICAL_SURFACE",
            "HALLMARK_APICAL_JUNCTION",
            "HALLMARK_ESTROGEN_RESPONSE_EARLY",
            "HALLMARK_ESTROGEN_RESPONSE_LATE",
        },
        "patterns": [r"^KRT", r"^EPCAM$", r"^MUC", r"^TFF", r"^WFDC2$", r"^SLPI$", r"^LCN2$", r"^SLC34A2$"],
    },
]

GROUP_COLORS = {
    "Chemokine/cytokine": "#4C78A8",
    "Antigen presentation": "#72B7B2",
    "IFN/antiviral": "#E45756",
    "Complement/coagulation": "#F58518",
    "Myeloid/phagolysosome": "#54A24B",
    "ECM/EMT/adhesion": "#B279A2",
    "Vascular/pericyte": "#9D755D",
    "Cell cycle/DNA repair": "#FF9DA6",
    "Metabolism/OXPHOS": "#EECA3B",
    "Ribosome/translation": "#8CD17D",
    "Hypoxia/stress": "#B6992D",
    "Secretory/epithelial": "#499894",
    "Other hallmark": "#BAB0AC",
    "No pathway annotation": "#D3D3D3",
}


def ensure_dirs() -> None:
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    TABLE_OUT.mkdir(parents=True, exist_ok=True)


def load_hallmark_gene_sets() -> dict[str, set[str]]:
    with HALLMARK_GENE_SETS.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {term: {str(gene).upper() for gene in genes} for term, genes in raw.items()}


def gene_to_hallmarks(hallmarks: dict[str, set[str]]) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = defaultdict(set)
    for term, genes in hallmarks.items():
        for gene in genes:
            mapping[gene].add(term)
    return dict(mapping)


def regex_features(gene: str) -> set[str]:
    gene_upper = gene.upper()
    features = set()
    for rule in GROUP_RULES:
        for pattern in rule["patterns"]:
            if re.match(pattern, gene_upper):
                features.add(f"PATTERN::{rule['group']}::{pattern}")
    return features


def assign_functional_group(gene: str, terms: set[str]) -> tuple[str, int]:
    gene_upper = gene.upper()
    best_group = "No pathway annotation"
    best_score = 0
    for rule in GROUP_RULES:
        term_hits = len(terms & rule["terms"])
        pattern_hits = sum(1 for pattern in rule["patterns"] if re.match(pattern, gene_upper))
        score = term_hits + 3 * pattern_hits
        if score > best_score:
            best_group = rule["group"]
            best_score = score
    if best_score == 0 and terms:
        return "Other hallmark", len(terms)
    return best_group, best_score


def build_gene_table(
    family: pd.DataFrame,
    score_df: pd.DataFrame,
    top50: dict[str, list[str]],
    ranks: dict[str, dict[str, int]],
    gene_to_terms: dict[str, set[str]],
    max_genes: int,
) -> pd.DataFrame:
    gene_counts: defaultdict[str, int] = defaultdict(int)
    for pid in family["program_id"].astype(str):
        for gene in top50[pid]:
            gene_counts[gene] += 1

    rows = []
    for gene, recurrence in gene_counts.items():
        scores = []
        best_rank = 10**9
        for pid in family["program_id"].astype(str):
            if gene in top50[pid]:
                scores.append(float(score_df.loc[pid, gene]) if gene in score_df.columns else np.nan)
                best_rank = min(best_rank, ranks[pid][gene])
        terms = gene_to_terms.get(gene.upper(), set())
        group, support_score = assign_functional_group(gene, terms)
        rows.append(
            {
                "gene": gene,
                "recurrence_n": recurrence,
                "mean_top50_score": float(np.nanmean(scores)) if scores else np.nan,
                "best_rank": best_rank,
                "functional_group": group,
                "functional_support_score": support_score,
                "n_hallmark_terms": len(terms),
                "hallmark_terms": ";".join(sorted(terms)),
            }
        )
    ranked = pd.DataFrame(rows).sort_values(
        ["recurrence_n", "mean_top50_score", "best_rank", "gene"],
        ascending=[False, False, True, True],
    )
    return ranked.head(max_genes).reset_index(drop=True)


def annotation_features(gene: str, gene_to_terms: dict[str, set[str]], selected_terms: list[str]) -> set[str]:
    terms = gene_to_terms.get(gene.upper(), set())
    features = {term for term in selected_terms if term in terms}
    features.update(regex_features(gene))
    return features


def cluster_within_group(group_table: pd.DataFrame, gene_to_terms: dict[str, set[str]], selected_terms: list[str]) -> list[str]:
    genes = group_table["gene"].astype(str).tolist()
    if len(genes) <= 2:
        return genes

    feature_sets = [annotation_features(gene, gene_to_terms, selected_terms) for gene in genes]
    feature_names = sorted(set().union(*feature_sets))
    if not feature_names:
        return genes

    matrix = np.array([[1 if feature in features else 0 for feature in feature_names] for features in feature_sets])
    if np.all(matrix.sum(axis=1) == 0):
        return genes

    distances = pdist(matrix, metric="jaccard")
    if not np.isfinite(distances).all() or np.allclose(distances, 0):
        return genes

    z = linkage(distances, method="average", optimal_ordering=True)
    clustered = [genes[idx] for idx in leaves_list(z)]
    rank = {gene: idx for idx, gene in enumerate(genes)}
    left_mean = np.mean([rank[gene] for gene in clustered[: max(1, len(clustered) // 2)]])
    right_mean = np.mean([rank[gene] for gene in clustered[max(1, len(clustered) // 2) :]])
    if left_mean > right_mean:
        clustered = list(reversed(clustered))
    return clustered


def functional_gene_order(gene_table: pd.DataFrame, gene_to_terms: dict[str, set[str]]) -> tuple[list[str], pd.DataFrame]:
    all_terms = sorted(
        term
        for term, count in pd.Series(
            [
                term
                for gene in gene_table["gene"].astype(str)
                for term in gene_to_terms.get(gene.upper(), set())
            ]
        ).value_counts().items()
        if count >= 2
    )

    ordered_genes = []
    order_rows = []
    for rule in GROUP_RULES:
        group = rule["group"]
        sub = gene_table.loc[gene_table["functional_group"] == group].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(
            ["recurrence_n", "mean_top50_score", "best_rank", "gene"],
            ascending=[False, False, True, True],
        )
        for gene in cluster_within_group(sub, gene_to_terms, all_terms):
            ordered_genes.append(gene)

    for group in ["Other hallmark", "No pathway annotation"]:
        sub = gene_table.loc[gene_table["functional_group"] == group].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(
            ["recurrence_n", "mean_top50_score", "best_rank", "gene"],
            ascending=[False, False, True, True],
        )
        ordered_genes.extend(sub["gene"].astype(str).tolist())

    order_index = {gene: idx for idx, gene in enumerate(ordered_genes)}
    ordered_table = gene_table.copy()
    ordered_table["functional_order"] = ordered_table["gene"].map(order_index)
    ordered_table = ordered_table.sort_values("functional_order").reset_index(drop=True)

    for idx, row in ordered_table.iterrows():
        order_rows.append({**row.to_dict(), "functional_order": idx + 1})
    return ordered_genes, pd.DataFrame(order_rows)


def block_ranges(ordered_table: pd.DataFrame) -> list[tuple[str, int, int]]:
    blocks: list[tuple[str, int, int]] = []
    current_group = None
    start = 0
    groups = ordered_table["functional_group"].tolist()
    for idx, group in enumerate(groups):
        if current_group is None:
            current_group = group
            start = idx
        elif group != current_group:
            blocks.append((str(current_group), start, idx - 1))
            current_group = group
            start = idx
    if current_group is not None:
        blocks.append((str(current_group), start, len(groups) - 1))
    return blocks


def plot_family_recurrence_dotplot_functional(
    family: pd.DataFrame,
    score_df: pd.DataFrame,
    top50: dict[str, list[str]],
    ranks: dict[str, dict[str, int]],
    program_order: list[str],
    output_path: Path,
    gene_to_terms: dict[str, set[str]],
    max_genes: int = 60,
) -> pd.DataFrame:
    gene_table = build_gene_table(family, score_df, top50, ranks, gene_to_terms, max_genes=max_genes)
    selected_genes, ordered_table = functional_gene_order(gene_table, gene_to_terms)

    gene_counts = dict(zip(gene_table["gene"], gene_table["recurrence_n"]))
    group_by_gene = dict(zip(ordered_table["gene"], ordered_table["functional_group"]))
    rows = []
    for x_idx, pid in enumerate(program_order):
        for y_idx, gene in enumerate(selected_genes):
            if gene not in top50[pid]:
                continue
            rank = ranks[pid][gene]
            rows.append(
                {
                    "program_id": pid,
                    "program_label": base.sample_program_label(pid),
                    "gene": gene,
                    "functional_group": group_by_gene[gene],
                    "x": x_idx,
                    "y": y_idx,
                    "rank": rank,
                    "rank_size": 18 + (51 - rank) * 2.0,
                    "score": float(score_df.loc[pid, gene]) if gene in score_df.columns else np.nan,
                    "recurrence_n": int(gene_counts[gene]),
                }
            )
    dot = pd.DataFrame(rows)

    width = max(8.5, 0.45 * len(program_order) + 5.0)
    height = max(6.5, 0.16 * len(selected_genes) + 2.4)
    fig, ax = plt.subplots(figsize=(width, height))

    for group, start, end in block_ranges(ordered_table):
        color = GROUP_COLORS.get(group, "#D3D3D3")
        ax.axhspan(start - 0.5, end + 0.5, color=color, alpha=0.08, linewidth=0, zorder=0)
        ax.axhline(end + 0.5, color="#c7c7c7", linewidth=0.8, zorder=1)
        ax.text(
            1.01,
            (start + end) / 2.0,
            group,
            color="#333333",
            fontsize=6.5,
            va="center",
            ha="left",
            transform=ax.get_yaxis_transform(),
        )

    if not dot.empty:
        ax.scatter(
            dot["x"],
            dot["y"],
            s=dot["rank_size"],
            color="#c95f4f",
            edgecolors="black",
            linewidths=0.15,
            alpha=0.82,
            zorder=2,
        )

    ax.set_xticks(range(len(program_order)))
    ax.set_xticklabels([base.sample_program_label(pid) for pid in program_order], rotation=90, fontsize=6)
    ax.set_yticks(range(len(selected_genes)))
    y_labels = [f"{gene} ({int(gene_counts[gene])})" for gene in selected_genes]
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Sample and program")
    ax.set_ylabel("Top recurrent top50 genes grouped by functional/pathway annotation")
    label = str(family["family_label"].iloc[0])
    ax.set_title(f"{label}: top-gene recurrence, functional y-axis order", fontsize=12, pad=10)
    ax.grid(color="#e8e8e8", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.text(
        0.01,
        0.01,
        "Dots mark top50 membership; larger dots are higher ranked within that program. Y-axis groups use local Hallmark pathway membership plus curated gene-family rules.",
        fontsize=7,
        ha="left",
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.02, 0.88, 1))
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)

    return ordered_table


def main() -> None:
    ensure_dirs()
    hallmarks = load_hallmark_gene_sets()
    gene_terms = gene_to_hallmarks(hallmarks)

    ann = base.load_primary_annotations()
    score_df, top_genes, ranks = base.read_program_scores(ann)
    program_ids = ann["program_id"].astype(str).tolist()
    _, frac, _ = base.overlap_matrices(program_ids, top_genes[50], n_top=50)

    recurrence_tables = []
    for (family_id, family_label), family in ann.groupby(["family_id", "family_label"], sort=True):
        ids = family["program_id"].astype(str).tolist()
        order = base.clustered_order(frac.loc[ids, ids])
        family_slug = base.safe_name(f"{family_id}_{family_label}")
        table = plot_family_recurrence_dotplot_functional(
            family,
            score_df,
            top_genes[50],
            ranks[50],
            order,
            FIG_OUT / f"{family_slug}_gene_recurrence_dotplot_functional_order.png",
            gene_terms,
            max_genes=60,
        )
        table.insert(0, "family_id", family_id)
        table.insert(1, "family_label", family_label)
        table.to_csv(TABLE_OUT / f"{family_slug}_gene_recurrence_functional_order.csv", index=False)
        recurrence_tables.append(table)

    pd.concat(recurrence_tables, ignore_index=True).to_csv(
        TABLE_OUT / "family_gene_recurrence_functional_order_table.csv",
        index=False,
    )
    print(f"Wrote functional-order gene recurrence figures to: {FIG_OUT}")
    print(f"Wrote functional-order gene recurrence tables to: {TABLE_OUT}")
    print(f"Families plotted: {ann['family_id'].nunique()}")


if __name__ == "__main__":
    main()
