from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(r"D:\HGSOC_Spatial_Atlas")
UMAP_DIR = (
    ROOT
    / "05_analysis_ready"
    / "S3_cNMF_Tumor_Programs"
    / "jaccard_raw_matrices"
    / "inspection_exports_average"
    / "variantB_nonjunk_manual_cut_v2"
    / "robustness_alllevels_umap_final_mp1_8"
)
UMAP_COORDS = UMAP_DIR / "umap_coords.csv"
OUTFILE = UMAP_DIR / "umap_companion_panels.png"
FAMILY_SNAPSHOT = (
    ROOT
    / "05_analysis_ready"
    / "20260424_definition3b_definition4_raw_geneNMF"
    / "11_research_synthesis"
    / "tables"
    / "program_family_annotation_snapshot.csv"
)


def family_palette(labels: list[str]) -> dict[str, str]:
    colors = sns.color_palette("tab20", n_colors=len(labels)).as_hex()
    return dict(zip(labels, colors, strict=False))


def main() -> None:
    coords = pd.read_csv(UMAP_COORDS)
    family = pd.read_csv(FAMILY_SNAPSHOT, usecols=["program_id", "family_label"]).drop_duplicates("program_id")
    coords = coords.merge(family, on="program_id", how="left")
    coords["family_label"] = coords["family_label"].fillna("missing")

    family_order = sorted(coords["family_label"].unique())
    palette_family = family_palette(family_order)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))

    sns.scatterplot(data=coords, x="x", y="y", hue="group", s=30, linewidth=0, ax=axes[0])
    axes[0].set_title("Final MP assignment")

    sns.scatterplot(
        data=coords,
        x="x",
        y="y",
        hue="family_label",
        hue_order=family_order,
        palette=palette_family,
        s=30,
        linewidth=0,
        ax=axes[1],
    )
    axes[1].set_title("Reference family")

    sns.scatterplot(data=coords, x="x", y="y", hue="dataset", s=30, linewidth=0, ax=axes[2])
    axes[2].set_title("Cohort")

    x_pad = 0.45
    y_pad = 0.45
    xlim = (coords["x"].min() - x_pad, coords["x"].max() + x_pad)
    ylim = (coords["y"].min() - y_pad, coords["y"].max() + y_pad)
    for ax in axes:
        ax.set_xlabel("component 1")
        ax.set_ylabel("component 2")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    axes[0].legend(loc="upper right", fontsize=7, frameon=True)
    axes[1].legend(
        title="reference family",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=7,
        title_fontsize=8,
        frameon=True,
    )
    axes[2].legend(loc="upper right", fontsize=7, frameon=True)

    fig.tight_layout()
    fig.savefig(OUTFILE, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(OUTFILE)


if __name__ == "__main__":
    main()
