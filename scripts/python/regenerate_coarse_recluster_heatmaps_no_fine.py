from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

import jaccard_metaprogram_pipeline as pipe


FINAL_MP_MEMBERS_FILE = (
    pipe.MANUAL_DIR
    / "subcluster_signatures_scoring"
    / "signatures"
    / "manual_subcluster_program_members.csv"
)

FINAL_MP_ORDER = [
    "not_assigned_to_MP",
    "MP1",
    "MP2",
    "MP3",
    "MP4",
    "MP5",
    "MP6",
    "MP7",
    "MP8",
]

FINAL_MP_COLORS = {
    "not_assigned_to_MP": "#B9B9B9",
    "MP1": "#4C78A8",
    "MP2": "#F58518",
    "MP3": "#E45756",
    "MP4": "#72B7B2",
    "MP5": "#54A24B",
    "MP6": "#B279A2",
    "MP7": "#9D755D",
    "MP8": "#FF9DA6",
}


def draw_coarse_heatmap_without_fine(
    coarse: str,
    matrix: pd.DataFrame,
    z: np.ndarray,
    ref_map: dict[str, str],
    rho_map: dict[str, float],
    final_mp_map: dict[str, str],
    output_path: Path,
) -> None:
    family_labels = sorted({ref_map.get(pid, "missing") for pid in matrix.index})
    family_colors = pipe.family_palette_from_snapshot()
    final_mp_labels = [
        final_mp_map.get(pid, "not_assigned_to_MP") for pid in matrix.index
    ]
    rho_norm = mcolors.TwoSlopeNorm(vmin=-0.4, vcenter=0.0, vmax=0.4)
    rho_cmap = plt.get_cmap("vlag")

    def rho_to_color(program_id: str) -> str:
        rho = rho_map.get(program_id)
        if pd.isna(rho):
            return "#D9D9D9"
        clipped = float(np.clip(rho, -0.4, 0.4))
        return mcolors.to_hex(rho_cmap(rho_norm(clipped)))

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
    row_colors = pd.DataFrame(
        {
            "final MP": [
                FINAL_MP_COLORS.get(label, "#D9D9D9") for label in final_mp_labels
            ],
        },
        index=matrix.index,
    )

    plot_matrix = matrix.copy()
    np.fill_diagonal(plot_matrix.values, np.nan)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")

    sns.set_theme(style="white")
    size = 7.5 if len(matrix) <= 25 else 9.5
    plot_z = pipe.linkage_for_visible_dendrogram(z)
    grid = sns.clustermap(
        plot_matrix,
        row_linkage=plot_z,
        col_linkage=plot_z,
        col_colors=col_colors,
        row_colors=row_colors,
        cmap=cmap,
        vmin=0,
        vmax=0.2,
        xticklabels=False,
        yticklabels=False,
        linewidths=0,
        figsize=(size, size),
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

    rho_mappable = plt.cm.ScalarMappable(norm=rho_norm, cmap=rho_cmap)
    rho_mappable.set_array([])
    rho_cax = grid.fig.add_axes([0.006, 0.780, 0.024, 0.145])
    rho_cbar = grid.fig.colorbar(rho_mappable, cax=rho_cax)
    rho_cbar.set_label("Usage-SNAI1-ac rho", fontsize=8)
    rho_cbar.set_ticks([-0.4, 0.0, 0.4])
    rho_cbar.ax.tick_params(labelsize=8)
    rho_cbar.ax.yaxis.set_label_position("left")

    jac_cax = grid.fig.add_axes([0.074, 0.780, 0.024, 0.145])
    jac_cbar = grid.fig.colorbar(grid.ax_heatmap.collections[0], cax=jac_cax)
    jac_cbar.set_label("Jaccard similarity", fontsize=8)
    jac_cbar.set_ticks([0.0, 0.1, 0.2])
    jac_cbar.ax.tick_params(labelsize=8)

    family_handles = [
        Patch(facecolor=family_colors[label], label=label) for label in family_labels
    ]
    family_legend = grid.fig.legend(
        handles=family_handles,
        title="reference family",
        loc="upper left",
        bbox_to_anchor=(0.88, 0.745),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    grid.fig.add_artist(family_legend)

    present_final_mps = [
        label for label in FINAL_MP_ORDER if label in set(final_mp_labels)
    ]
    final_mp_handles = [
        Patch(facecolor=FINAL_MP_COLORS[label], label=label)
        for label in present_final_mps
    ]
    final_mp_anchor_y = max(0.34, 0.745 - 0.038 * (len(family_handles) + 1.4))
    final_mp_legend = grid.fig.legend(
        handles=final_mp_handles,
        title="final MP",
        loc="upper left",
        bbox_to_anchor=(0.88, final_mp_anchor_y),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    grid.fig.add_artist(final_mp_legend)
    grid.fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(grid.fig)


def main() -> None:
    pipe.COARSE_RECLUSTER_DIR.mkdir(parents=True, exist_ok=True)
    assignment = pd.read_csv(pipe.ASSIGNMENT_FILE)
    matrix_full = pd.read_csv(pipe.VARIANT_B_MATRIX, index_col=0)
    ref_map = pipe.family_label_map_from_snapshot()
    corr = pd.read_csv(
        pipe.PROGRAMME_SNAI1AC_CORR_FILE,
        usecols=["program_id", "spearman_rho"],
    ).drop_duplicates("program_id")
    rho_map = corr.set_index("program_id")["spearman_rho"].astype(float).to_dict()
    final_mp_members = pd.read_csv(
        FINAL_MP_MEMBERS_FILE,
        usecols=["program_id", "subcluster_id"],
    ).drop_duplicates("program_id")
    final_mp_map = (
        final_mp_members.set_index("program_id")["subcluster_id"].astype(str).to_dict()
    )

    written = []
    for coarse in ["A", "B", "C"]:
        program_ids = assignment.loc[
            assignment["cluster_coarse"].eq(coarse),
            "program_id",
        ].tolist()
        submatrix = matrix_full.loc[program_ids, program_ids]
        z, _order, _coph = pipe.average_linkage_from_jaccard(submatrix)
        out = pipe.COARSE_RECLUSTER_DIR / f"variantB_nonjunk_recluster_coarse_{coarse}.png"
        draw_coarse_heatmap_without_fine(
            coarse,
            submatrix,
            z,
            ref_map,
            rho_map,
            final_mp_map,
            out,
        )
        written.append(out)

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
