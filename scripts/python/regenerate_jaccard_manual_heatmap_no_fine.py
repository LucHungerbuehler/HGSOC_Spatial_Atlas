from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

import jaccard_metaprogram_pipeline as pipe


OUTFILE = (
    pipe.MANUAL_DIR
    / "variantB_nonjunk_manual_cut_v2_heatmap_no_fine.png"
)


def draw_heatmap_without_fine(output_path: Path) -> None:
    matrix, z, _order, _coph = pipe.variant_b_nonjunk_matrix_and_order()
    assignment = pd.read_csv(pipe.ASSIGNMENT_FILE)
    ref_map = pipe.family_label_map_from_snapshot()
    corr = pd.read_csv(
        pipe.PROGRAMME_SNAI1AC_CORR_FILE,
        usecols=["program_id", "spearman_rho"],
    ).drop_duplicates("program_id")
    rho_map = corr.set_index("program_id")["spearman_rho"].astype(float).to_dict()

    assignment = assignment.set_index("program_id").reindex(matrix.index)
    assignment.index.name = "program_id"
    assignment = assignment.reset_index()

    coarse_palette = {
        "A": "#4C78A8",
        "B": "#54A24B",
        "C": "#E45756",
        "unassigned": "#B9B9B9",
    }
    family_labels = sorted({ref_map.get(pid, "missing") for pid in matrix.index})
    family_colors = pipe.family_palette_from_snapshot()
    rho_norm = mcolors.TwoSlopeNorm(vmin=-0.4, vcenter=0.0, vmax=0.4)
    rho_cmap = plt.get_cmap("vlag")

    def rho_to_color(program_id: str) -> str:
        rho = rho_map.get(program_id)
        if pd.isna(rho):
            return "#D9D9D9"
        clipped = float(max(-0.4, min(0.4, rho)))
        return mcolors.to_hex(rho_cmap(rho_norm(clipped)))

    row_colors = pd.DataFrame(
        {
            "coarse": assignment["cluster_coarse"]
            .map(coarse_palette)
            .fillna("#D9D9D9")
            .to_numpy(),
        },
        index=assignment["program_id"],
    )
    col_colors = pd.DataFrame(
        {
            "reference family": [
                family_colors.get(ref_map.get(pid, "missing"), "#D9D9D9")
                for pid in matrix.index
            ],
            "SNAI1-ac rho": [rho_to_color(pid) for pid in matrix.index],
        },
        index=matrix.index,
    )

    plot_matrix = matrix.copy()
    plot_matrix.values[range(plot_matrix.shape[0]), range(plot_matrix.shape[1])] = float("nan")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    sns.set_theme(style="white")

    plot_z = pipe.linkage_for_visible_dendrogram(z)
    grid = sns.clustermap(
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
        cbar_pos=None,
    )
    grid.ax_heatmap.set_xlabel("")
    grid.ax_heatmap.set_ylabel("")
    if grid.ax_row_colors is not None:
        grid.ax_row_colors.set_xlabel("")
        grid.ax_row_colors.set_ylabel("")
        grid.ax_row_colors.set_xticks([])
        grid.ax_row_colors.set_xticklabels([])

    family_handles = [
        Patch(facecolor=family_colors[label], label=label) for label in family_labels
    ]
    coarse_handles = [
        Patch(facecolor=color, label=label) for label, color in coarse_palette.items()
    ]
    rho_mappable = plt.cm.ScalarMappable(norm=rho_norm, cmap=rho_cmap)
    rho_mappable.set_array([])
    rho_cax = grid.fig.add_axes([0.020, 0.780, 0.026, 0.145])
    rho_cbar = grid.fig.colorbar(rho_mappable, cax=rho_cax)
    rho_cbar.set_label("Usage-SNAI1-ac rho", fontsize=8)
    rho_cbar.set_ticks([-0.4, 0.0, 0.4])
    rho_cbar.ax.tick_params(labelsize=8)
    rho_cbar.ax.yaxis.set_label_position("left")

    jac_cax = grid.fig.add_axes([0.110, 0.780, 0.026, 0.145])
    jac_cbar = grid.fig.colorbar(grid.ax_heatmap.collections[0], cax=jac_cax)
    jac_cbar.set_label("Jaccard similarity", fontsize=8)
    jac_cbar.set_ticks([0.0, 0.1, 0.2])
    jac_cbar.ax.tick_params(labelsize=8)

    family_legend = grid.fig.legend(
        handles=family_handles,
        title="reference family",
        loc="upper left",
        bbox_to_anchor=(0.88, 0.745),
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )
    coarse_legend = grid.fig.legend(
        handles=coarse_handles,
        title="coarse",
        loc="upper left",
        bbox_to_anchor=(0.88, 0.505),
        frameon=False,
    )
    grid.fig.add_artist(family_legend)
    grid.fig.add_artist(coarse_legend)
    grid.fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(grid.fig)


if __name__ == "__main__":
    draw_heatmap_without_fine(OUTFILE)
    print(OUTFILE)
