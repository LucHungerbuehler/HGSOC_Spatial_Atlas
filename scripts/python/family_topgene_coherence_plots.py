from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


ROOT = Path(r"D:\HGSOC_Spatial_Atlas")
RUN = ROOT / "05_analysis_ready" / "20260424_definition3b_definition4_raw_geneNMF"
CNMF_RUNS = ROOT / "05_analysis_ready" / "S3_cNMF_Tumor_Programs" / "cnmf_runs"
FAMILY_TABLE = RUN / "08_snaI1_tme_family_definitions" / "program_to_snaI1_tme_family_v1.csv"
OUT = RUN / "12_family_topgene_coherence"

PRIMARY_ROLES = {
    "primary_tumor_intrinsic",
    "primary_tumor_spot_context",
    "context_primary_or_sensitivity",
}

PROGRAM_RE = re.compile(r"^(denisenko_2022|ju_2024|yamamoto_2025)__(.+)__K(\d+)__P(\d+)$")


def ensure_dirs() -> None:
    for subdir in [
        OUT / "tables",
        OUT / "figures" / "global_overlap",
        OUT / "figures" / "family_overlap",
        OUT / "figures" / "gene_recurrence",
    ]:
        subdir.mkdir(parents=True, exist_ok=True)


def boolish(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    return re.sub(r"_+", "_", value)


def short_program_id(program_id: str) -> str:
    match = re.search(r"(K\d+__P\d+)$", str(program_id))
    return match.group(1) if match else str(program_id)


def sample_program_label(program_id: str) -> str:
    match = PROGRAM_RE.match(str(program_id))
    if not match:
        return short_program_id(program_id).replace("__", "_")
    sample, k_value, p_value = match.group(2), match.group(3), match.group(4)
    return f"{sample}_K{k_value}_P{p_value}"


def load_primary_annotations() -> pd.DataFrame:
    ann = pd.read_csv(FAMILY_TABLE)
    keep = (
        boolish(ann["include_primary_snaI1_tme"])
        & ann["analysis_role"].isin(PRIMARY_ROLES)
        & ~boolish(ann["technical_flag"])
        & ~ann["family_id"].fillna("").str.startswith("F90")
    )
    ann = ann.loc[keep].copy()
    ann = ann.sort_values(["family_id", "dataset", "sample_id_on_disk", "program_id"]).reset_index(drop=True)
    return ann


def spectra_path(program_id: str) -> tuple[Path, int]:
    match = PROGRAM_RE.match(program_id)
    if not match:
        raise ValueError(f"Cannot parse program_id: {program_id}")
    dataset, sample, k_value, p_value = match.group(1), match.group(2), int(match.group(3)), int(match.group(4))
    sample_label = f"{dataset}__{sample}"
    path = CNMF_RUNS / sample_label / f"{sample_label}.gene_spectra_score.k_{k_value}.dt_0_5.txt"
    return path, p_value


def read_program_scores(ann: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, dict[str, list[str]]], dict[int, dict[str, dict[str, int]]]]:
    cache: dict[Path, pd.DataFrame] = {}
    score_rows: dict[str, pd.Series] = {}
    top_genes: dict[int, dict[str, list[str]]] = {50: {}, 100: {}}
    ranks: dict[int, dict[str, dict[str, int]]] = {50: {}, 100: {}}

    for program_id in ann["program_id"].astype(str):
        path, p_value = spectra_path(program_id)
        if path not in cache:
            df = pd.read_csv(path, sep="\t", index_col=0)
            df.index = [int(str(idx).replace("GEP", "").replace(".0", "")) for idx in df.index]
            cache[path] = df
        scores = cache[path].loc[p_value].astype(float)
        ordered = scores.sort_values(ascending=False)
        genes_100 = [str(gene) for gene in ordered.index[:100]]
        genes_50 = genes_100[:50]
        score_rows[program_id] = scores.rename(program_id)
        top_genes[50][program_id] = genes_50
        top_genes[100][program_id] = genes_100
        ranks[50][program_id] = {gene: rank for rank, gene in enumerate(genes_50, start=1)}
        ranks[100][program_id] = {gene: rank for rank, gene in enumerate(genes_100, start=1)}

    score_df = pd.DataFrame(score_rows).T
    return score_df, top_genes, ranks


def overlap_matrices(
    program_ids: list[str],
    top_genes: dict[str, list[str]],
    n_top: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sets = {pid: set(top_genes[pid]) for pid in program_ids}
    count = pd.DataFrame(0, index=program_ids, columns=program_ids, dtype=float)
    frac = count.copy()
    jaccard = count.copy()
    for pid_a in program_ids:
        set_a = sets[pid_a]
        for pid_b in program_ids:
            set_b = sets[pid_b]
            shared = len(set_a & set_b)
            union = len(set_a | set_b)
            count.loc[pid_a, pid_b] = shared
            frac.loc[pid_a, pid_b] = shared / float(n_top)
            jaccard.loc[pid_a, pid_b] = shared / union if union else np.nan
    return count, frac, jaccard


def clustered_order(matrix: pd.DataFrame) -> list[str]:
    if len(matrix) <= 2:
        return list(matrix.index)
    dist = 1.0 - matrix.to_numpy(dtype=float)
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, 1.0)
    condensed = squareform(dist, checks=False)
    z = linkage(condensed, method="average", optimal_ordering=True)
    return [matrix.index[i] for i in leaves_list(z)]


def global_family_order(ann: pd.DataFrame, frac: pd.DataFrame) -> list[str]:
    order: list[str] = []
    for _, family in ann.groupby(["family_id", "family_label"], sort=True):
        ids = family["program_id"].astype(str).tolist()
        if len(ids) > 2:
            ids = clustered_order(frac.loc[ids, ids])
        order.extend(ids)
    return order


def plot_overlap_heatmap(
    matrix: pd.DataFrame,
    ann: pd.DataFrame,
    order: list[str],
    output_path: Path,
    title: str,
    n_top: int = 50,
    label_size: int = 5,
    draw_boundaries: bool = True,
    cbar_label: str | None = None,
    vmax: float = 1.0,
) -> None:
    ordered = matrix.loc[order, order]
    labels = [short_program_id(pid) for pid in order]
    size = max(7.0, min(22.0, 0.19 * len(order) + 4.0))
    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(
        ordered,
        cmap="viridis",
        vmin=0,
        vmax=vmax,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": cbar_label or f"Top{n_top} overlap fraction (shared genes / {n_top})"},
        ax=ax,
    )
    ax.set_title(title, fontsize=12, pad=12)
    ax.tick_params(axis="x", labelrotation=90, labelsize=label_size)
    ax.tick_params(axis="y", labelsize=label_size)

    if draw_boundaries:
        meta = ann.set_index("program_id").loc[order]
        boundaries = []
        last = None
        for idx, family_id in enumerate(meta["family_id"].tolist()):
            if last is not None and family_id != last:
                boundaries.append(idx)
            last = family_id
        for b in boundaries:
            ax.axhline(b, color="white", linewidth=1.2)
            ax.axvline(b, color="white", linewidth=1.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_family_recurrence_dotplot(
    family: pd.DataFrame,
    score_df: pd.DataFrame,
    top50: dict[str, list[str]],
    ranks: dict[str, dict[str, int]],
    order: list[str],
    output_path: Path,
    max_genes: int = 60,
) -> pd.DataFrame:
    gene_counts: defaultdict[str, int] = defaultdict(int)
    for pid in family["program_id"].astype(str):
        for gene in top50[pid]:
            gene_counts[gene] += 1

    gene_scores = []
    for gene, recurrence in gene_counts.items():
        scores = []
        best_rank = 10**9
        for pid in family["program_id"].astype(str):
            if gene in top50[pid]:
                scores.append(float(score_df.loc[pid, gene]) if gene in score_df.columns else np.nan)
                best_rank = min(best_rank, ranks[pid][gene])
        gene_scores.append(
            {
                "gene": gene,
                "recurrence_n": recurrence,
                "mean_top50_score": float(np.nanmean(scores)) if scores else np.nan,
                "best_rank": best_rank,
            }
        )
    gene_table = pd.DataFrame(gene_scores).sort_values(
        ["recurrence_n", "mean_top50_score", "best_rank", "gene"],
        ascending=[False, False, True, True],
    )
    selected_genes = gene_table.head(max_genes)["gene"].tolist()

    rows = []
    for x_idx, pid in enumerate(order):
        for y_idx, gene in enumerate(selected_genes):
            if gene not in top50[pid]:
                continue
            rank = ranks[pid][gene]
            rows.append(
                {
                    "program_id": pid,
                    "program_label": sample_program_label(pid),
                    "gene": gene,
                    "x": x_idx,
                    "y": y_idx,
                    "rank": rank,
                    "rank_size": 18 + (51 - rank) * 2.0,
                    "score": float(score_df.loc[pid, gene]) if gene in score_df.columns else np.nan,
                    "recurrence_n": int(gene_counts[gene]),
                }
            )
    dot = pd.DataFrame(rows)

    width = max(7.5, 0.45 * len(order) + 3.5)
    height = max(6.0, 0.16 * len(selected_genes) + 2.2)
    fig, ax = plt.subplots(figsize=(width, height))
    if not dot.empty:
        vmax = float(np.nanpercentile(np.abs(dot["score"].to_numpy(dtype=float)), 98))
        vmax = max(vmax, 1e-6)
        scatter = ax.scatter(
            dot["x"],
            dot["y"],
            s=dot["rank_size"],
            c=dot["score"],
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            edgecolors="black",
            linewidths=0.15,
            alpha=0.9,
        )
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("gene_spectra_score loading")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([sample_program_label(pid) for pid in order], rotation=90, fontsize=6)
    ax.set_yticks(range(len(selected_genes)))
    y_labels = [f"{gene} ({int(gene_counts[gene])})" for gene in selected_genes]
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Sample and program")
    ax.set_ylabel("Top recurrent top50 genes; label shows recurrence count")
    label = str(family["family_label"].iloc[0])
    ax.set_title(f"{label}: top-gene recurrence across programs", fontsize=12, pad=10)
    ax.grid(color="#e8e8e8", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.text(
        0.01,
        0.01,
        "Dot shown only when gene is in that program's top50; larger dots are higher ranked within the program.",
        fontsize=7,
        ha="left",
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)

    return gene_table


def coherence_summary(ann: pd.DataFrame, frac: pd.DataFrame) -> pd.DataFrame:
    rows = []
    program_ids = ann["program_id"].astype(str).tolist()
    for (family_id, family_label), family in ann.groupby(["family_id", "family_label"], sort=True):
        ids = family["program_id"].astype(str).tolist()
        in_values = []
        if len(ids) > 1:
            sub = frac.loc[ids, ids].to_numpy(dtype=float)
            tri = sub[np.triu_indices_from(sub, k=1)]
            in_values = tri[np.isfinite(tri)].tolist()
        out_values = []
        outside = [pid for pid in program_ids if pid not in ids]
        if outside:
            vals = frac.loc[ids, outside].to_numpy(dtype=float).ravel()
            out_values = vals[np.isfinite(vals)].tolist()
        rows.append(
            {
                "family_id": family_id,
                "family_label": family_label,
                "n_programs": len(ids),
                "n_samples": family["sample_label"].nunique(),
                "median_within_top50_overlap_fraction": float(np.median(in_values)) if in_values else np.nan,
                "max_within_top50_overlap_fraction": float(np.max(in_values)) if in_values else np.nan,
                "median_outside_top50_overlap_fraction": float(np.median(out_values)) if out_values else np.nan,
                "within_minus_outside_median": (
                    float(np.median(in_values) - np.median(out_values)) if in_values and out_values else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("family_id")


def main() -> None:
    ensure_dirs()
    ann = load_primary_annotations()
    score_df, top_genes, ranks = read_program_scores(ann)
    program_ids = ann["program_id"].astype(str).tolist()
    count, frac, jaccard = overlap_matrices(program_ids, top_genes[50], n_top=50)
    count100, frac100, jaccard100 = overlap_matrices(program_ids, top_genes[100], n_top=100)

    count.to_csv(OUT / "tables" / "program_top50_shared_gene_count_matrix.csv")
    frac.to_csv(OUT / "tables" / "program_top50_overlap_fraction_matrix.csv")
    jaccard.to_csv(OUT / "tables" / "program_top50_jaccard_matrix.csv")
    count100.to_csv(OUT / "tables" / "program_top100_shared_gene_count_matrix.csv")
    frac100.to_csv(OUT / "tables" / "program_top100_overlap_fraction_matrix.csv")
    jaccard100.to_csv(OUT / "tables" / "program_top100_jaccard_matrix.csv")
    ann.to_csv(OUT / "tables" / "primary_program_family_order_metadata.csv", index=False)
    coherence_summary(ann, frac).to_csv(OUT / "tables" / "family_top50_coherence_summary.csv", index=False)

    global_order = global_family_order(ann, frac)
    pd.DataFrame({"program_id": global_order}).merge(ann, on="program_id", how="left").to_csv(
        OUT / "tables" / "global_overlap_heatmap_program_order.csv",
        index=False,
    )
    plot_overlap_heatmap(
        frac,
        ann,
        global_order,
        OUT / "figures" / "global_overlap" / "global_program_top50_overlap_fraction_grouped_by_family.png",
        "Global top50 overlap among primary non-technical programs",
        n_top=50,
        label_size=4,
        draw_boundaries=True,
    )
    plot_overlap_heatmap(
        jaccard,
        ann,
        global_order,
        OUT / "figures" / "global_overlap" / "global_program_top50_jaccard_grouped_by_family.png",
        "Global top50 Jaccard similarity among primary non-technical programs",
        n_top=50,
        label_size=4,
        draw_boundaries=True,
        cbar_label="Top50 Jaccard similarity",
    )
    plot_overlap_heatmap(
        frac100,
        ann,
        global_order,
        OUT / "figures" / "global_overlap" / "global_program_top100_overlap_fraction_grouped_by_family.png",
        "Global top100 overlap among primary non-technical programs",
        n_top=100,
        label_size=4,
        draw_boundaries=True,
    )
    plot_overlap_heatmap(
        jaccard100,
        ann,
        global_order,
        OUT / "figures" / "global_overlap" / "global_program_top100_jaccard_grouped_by_family.png",
        "Global top100 Jaccard similarity among primary non-technical programs",
        n_top=100,
        label_size=4,
        draw_boundaries=True,
        cbar_label="Top100 Jaccard similarity",
    )

    top50_jaccard_order = clustered_order(jaccard)
    pd.DataFrame({"program_id": top50_jaccard_order}).merge(ann, on="program_id", how="left").to_csv(
        OUT / "tables" / "global_top50_jaccard_clustered_program_order.csv",
        index=False,
    )
    plot_overlap_heatmap(
        jaccard,
        ann,
        top50_jaccard_order,
        OUT / "figures" / "global_overlap" / "global_program_top50_jaccard_clustered_by_jaccard.png",
        "Global top50 Jaccard similarity, hierarchically clustered",
        n_top=50,
        label_size=4,
        draw_boundaries=False,
        cbar_label="Top50 Jaccard similarity",
    )
    plot_overlap_heatmap(
        frac,
        ann,
        top50_jaccard_order,
        OUT / "figures" / "global_overlap" / "global_program_top50_overlap_fraction_clustered_by_jaccard.png",
        "Global top50 overlap fraction, ordered by top50 Jaccard clustering",
        n_top=50,
        label_size=4,
        draw_boundaries=False,
    )

    top100_jaccard_order = clustered_order(jaccard100)
    pd.DataFrame({"program_id": top100_jaccard_order}).merge(ann, on="program_id", how="left").to_csv(
        OUT / "tables" / "global_top100_jaccard_clustered_program_order.csv",
        index=False,
    )
    plot_overlap_heatmap(
        jaccard100,
        ann,
        top100_jaccard_order,
        OUT / "figures" / "global_overlap" / "global_program_top100_jaccard_clustered_by_jaccard.png",
        "Global top100 Jaccard similarity, hierarchically clustered",
        n_top=100,
        label_size=4,
        draw_boundaries=False,
        cbar_label="Top100 Jaccard similarity",
    )
    plot_overlap_heatmap(
        frac100,
        ann,
        top100_jaccard_order,
        OUT / "figures" / "global_overlap" / "global_program_top100_overlap_fraction_clustered_by_jaccard.png",
        "Global top100 overlap fraction, ordered by top100 Jaccard clustering",
        n_top=100,
        label_size=4,
        draw_boundaries=False,
    )

    recurrence_tables = []
    for (family_id, family_label), family in ann.groupby(["family_id", "family_label"], sort=True):
        ids = family["program_id"].astype(str).tolist()
        family_frac = frac.loc[ids, ids]
        order = clustered_order(family_frac)
        family_slug = safe_name(f"{family_id}_{family_label}")
        pd.DataFrame({"program_id": order}).merge(ann, on="program_id", how="left").to_csv(
            OUT / "tables" / f"{family_slug}_clustered_program_order.csv",
            index=False,
        )
        plot_overlap_heatmap(
            frac,
            ann,
            order,
            OUT / "figures" / "family_overlap" / f"{family_slug}_top50_overlap_fraction_clustered.png",
            f"{family_label}: top50 overlap, clustered",
            n_top=50,
            label_size=7,
            draw_boundaries=False,
        )
        gene_table = plot_family_recurrence_dotplot(
            family,
            score_df,
            top_genes[50],
            ranks[50],
            order,
            OUT / "figures" / "gene_recurrence" / f"{family_slug}_gene_recurrence_dotplot.png",
            max_genes=60,
        )
        gene_table.insert(0, "family_id", family_id)
        gene_table.insert(1, "family_label", family_label)
        recurrence_tables.append(gene_table)

    pd.concat(recurrence_tables, ignore_index=True).to_csv(
        OUT / "tables" / "family_gene_recurrence_ranked_table.csv",
        index=False,
    )

    print(f"Wrote top-gene coherence outputs to: {OUT}")
    print(f"Programs plotted: {len(program_ids)}")
    print(f"Families plotted: {ann['family_id'].nunique()}")


if __name__ == "__main__":
    main()
