"""Create report figures for S2b ILR composition and joint predictability.

Inputs are the analysis-ready S2b outputs on D:. Figures are written into the
local workspace copy so they can be reviewed and moved into the report.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


INPUT_DIR = Path(
    r"D:\HGSOC_Spatial_Atlas\05_analysis_ready"
    r"\S2b_CellType_Composition_Correlation\ilr_first_pass"
)
OUTPUT_DIR = (
    Path(r"D:\HGSOC_Spatial_Atlas")
    / "05_analysis_ready"
    / "S2b_CellType_Composition_Correlation"
    / "figures"
)


BALANCE_LABELS = {
    "b1": "Malignant\nvs microenvironment",
    "b2": "Stromal\nvs immune",
    "b3": "CAF\nvs endothelial",
    "b4": "Myeloid\nvs lymphoid",
    "b9": "T/NK\nvs B/plasma",
}


TREE_NODES = {
    "root": (0.05, 0.50, "All SpaCET\nlineages"),
    "mal": (0.27, 0.72, "Malignant"),
    "tme": (0.27, 0.28, "Microenvironment"),
    "stromal": (0.50, 0.48, "Stromal\nCAF, Endothelial"),
    "immune": (0.50, 0.13, "Immune"),
    "caf": (0.72, 0.62, "CAF"),
    "endo": (0.72, 0.42, "Endothelial"),
    "myeloid": (0.72, 0.22, "Myeloid\nMac, DC, Mast,\nNeutrophil"),
    "lymphoid": (0.72, 0.04, "Lymphoid\nT, NK, B,\nPlasma"),
}

TREE_EDGES = [
    ("root", "mal", "b1 +"),
    ("root", "tme", "b1 -"),
    ("tme", "stromal", "b2 +"),
    ("tme", "immune", "b2 -"),
    ("stromal", "caf", "b3 +"),
    ("stromal", "endo", "b3 -"),
    ("immune", "myeloid", "b4 +"),
    ("immune", "lymphoid", "b4 -"),
]


def save_all(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(OUTPUT_DIR / f"{stem}{suffix}", bbox_inches="tight", dpi=300)


NODE_HALF_WIDTH = 0.083
NODE_HALF_HEIGHT = 0.045


def draw_node(ax: plt.Axes, xy: tuple[float, float], text: str, fc: str) -> None:
    x, y = xy
    box = FancyBboxPatch(
        (x - NODE_HALF_WIDTH, y - NODE_HALF_HEIGHT),
        NODE_HALF_WIDTH * 2,
        NODE_HALF_HEIGHT * 2,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.0,
        edgecolor="#4B5563",
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=7.6)


def draw_tree_edges(ax: plt.Axes) -> None:
    label_offsets = {
        ("root", "mal"): (-0.005, 0.032),
        ("root", "tme"): (-0.005, -0.028),
        ("tme", "stromal"): (-0.006, 0.034),
        ("tme", "immune"): (-0.006, -0.032),
        ("stromal", "caf"): (0.006, 0.030),
        ("stromal", "endo"): (0.010, -0.030),
        ("immune", "myeloid"): (0.006, 0.030),
        ("immune", "lymphoid"): (0.010, -0.030),
    }
    for parent, child, label in TREE_EDGES:
        x0, y0, _ = TREE_NODES[parent]
        x1, y1, _ = TREE_NODES[child]
        ax.annotate(
            "",
            xy=(x1 - NODE_HALF_WIDTH, y1),
            xytext=(x0 + NODE_HALF_WIDTH, y0),
            arrowprops=dict(arrowstyle="-", color="#6B7280", lw=1.15),
        )
        dx, dy = label_offsets[(parent, child)]
        ax.text(
            (x0 + x1) / 2 + dx,
            (y0 + y1) / 2 + dy,
            label,
            ha="center",
            va="center",
            fontsize=7.6,
            color="#374151",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.88),
        )


def plot_balance_tree() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    ax.set_axis_off()
    ax.set_xlim(-0.05, 0.86)
    ax.set_ylim(-0.05, 0.82)

    draw_tree_edges(ax)

    for node_id, (x, y, text) in TREE_NODES.items():
        if node_id == "mal":
            fc = "#FECACA"
        elif node_id in {"stromal", "caf", "endo"}:
            fc = "#DBEAFE"
        elif node_id in {"immune", "myeloid", "lymphoid"}:
            fc = "#DCFCE7"
        else:
            fc = "#F3F4F6"
        draw_node(ax, (x, y), text, fc)

    ax.text(
        0.0,
        0.80,
        "A",
        fontsize=16,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax.text(
        0.07,
        0.80,
        "Biology-defined ILR balance tree",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax.text(
        0.07,
        0.755,
        "Primary balances used for the compositionally aware SNAI1-ac association analysis.",
        fontsize=9,
        color="#4B5563",
        ha="left",
        va="top",
    )

    save_all(fig, "s2b_ilr_balance_tree")
    plt.close(fig)


def plot_ilr_balance_dotplot(ax: plt.Axes) -> None:
    summary = pd.read_csv(INPUT_DIR / "ilr_balance_summary.csv")
    primary = summary[summary["balance"].isin(BALANCE_LABELS)].copy()
    primary["label"] = primary["balance"].map(BALANCE_LABELS)
    primary["order"] = primary["balance"].map({b: i for i, b in enumerate(BALANCE_LABELS)})
    primary = primary.sort_values("order", ascending=False)

    y = np.arange(len(primary))
    med = primary["median_r"].to_numpy(float)
    low = primary["iqr_low"].to_numpy(float)
    high = primary["iqr_high"].to_numpy(float)
    consistency = primary["sign_consistency"].to_numpy(float) * 100
    colors = plt.cm.viridis((consistency - 50) / 50)

    ax.axvline(0, color="#9CA3AF", lw=1)
    ax.hlines(y, low, high, color="#6B7280", lw=1.6)
    ax.scatter(med, y, s=95, c=colors, edgecolor="white", linewidth=0.8, zorder=3)
    for yi, x, c in zip(y, med, consistency):
        ax.text(
            x + (0.015 if x >= 0 else -0.015),
            yi,
            f"{c:.0f}%",
            ha="left" if x >= 0 else "right",
            va="center",
            fontsize=8,
            color="#374151",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(primary["label"], fontsize=9)
    ax.set_xlabel("Median Spearman r with SNAI1-ac")
    ax.set_title("B  Primary ILR balance associations", loc="left", fontweight="bold")
    ax.set_xlim(-0.24, 0.30)
    ax.grid(axis="x", color="#E5E7EB", lw=0.8)


def plot_predictability_ladder() -> None:
    ladder = pd.read_csv(INPUT_DIR / "cross_sample_generalization_ladder_summary.csv")
    model_order = ["composition_only", "hallmarks_only", "combined"]
    rung_order = ["within_sample_outer_cv", "LOSO", "LODO"]
    model_labels = {
        "composition_only": "Composition",
        "hallmarks_only": "Hallmarks",
        "combined": "Combined",
    }
    rung_labels = {
        "within_sample_outer_cv": "Within-sample CV",
        "LOSO": "Leave-one-sample-out",
        "LODO": "Leave-one-dataset-out",
    }
    colors = {
        "composition_only": "#4E79A7",
        "hallmarks_only": "#F28E2B",
        "combined": "#59A14F",
    }

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.84, bottom=0.18)
    x = np.arange(len(rung_order))
    width = 0.22
    for i, model in enumerate(model_order):
        sub = ladder.set_index(["model", "rung"]).loc[model]
        vals = np.array([sub.loc[r, "median_r2"] for r in rung_order], dtype=float)
        lows = np.array([sub.loc[r, "iqr_low"] for r in rung_order], dtype=float)
        highs = np.array([sub.loc[r, "iqr_high"] for r in rung_order], dtype=float)
        xpos = x + (i - 1) * width
        ax.bar(xpos, vals, width=width, color=colors[model], label=model_labels[model])
        ax.errorbar(
            xpos,
            vals,
            yerr=np.vstack([vals - lows, highs - vals]),
            fmt="none",
            ecolor="#374151",
            elinewidth=1,
            capsize=2,
        )
        for xx, yy in zip(xpos, vals):
            ax.text(xx, yy + 0.012, f"{yy:.2f}", ha="center", va="bottom", fontsize=7.8)

    ax.set_xticks(x)
    ax.set_xticklabels([rung_labels[r] for r in rung_order], fontsize=9)
    ax.set_ylabel("Median held-out $R^2$", fontsize=10)
    fig.text(
        0.10,
        0.94,
        "Joint predictability of SNAI1-ac from broad annotations",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax.set_ylim(0, 0.52)
    ax.grid(axis="y", color="#E5E7EB", lw=0.8)
    ax.tick_params(axis="both", labelsize=9)
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=9)
    save_all(fig, "s2b_joint_predictability_ladder")
    plt.close(fig)


def plot_combined_tree_and_ilr() -> None:
    fig = plt.figure(figsize=(12.2, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.28, 1.0], wspace=0.36)
    ax_tree = fig.add_subplot(gs[0, 0])
    ax_dot = fig.add_subplot(gs[0, 1])

    ax_tree.set_axis_off()
    ax_tree.set_xlim(-0.05, 0.86)
    ax_tree.set_ylim(-0.05, 0.82)
    draw_tree_edges(ax_tree)
    for node_id, (x, y, text) in TREE_NODES.items():
        if node_id == "mal":
            fc = "#FECACA"
        elif node_id in {"stromal", "caf", "endo"}:
            fc = "#DBEAFE"
        elif node_id in {"immune", "myeloid", "lymphoid"}:
            fc = "#DCFCE7"
        else:
            fc = "#F3F4F6"
        draw_node(ax_tree, (x, y), text, fc)
    ax_tree.set_title("A  Biology-defined ILR balance tree", loc="left", fontweight="bold")

    plot_ilr_balance_dotplot(ax_dot)
    save_all(fig, "s2b_ilr_tree_and_balance_associations")
    plt.close(fig)


def main() -> None:
    with open(INPUT_DIR / "balance_tree.json", "r", encoding="utf-8") as handle:
        json.load(handle)
    plot_balance_tree()
    plot_combined_tree_and_ilr()
    plot_predictability_ladder()
    print(f"Wrote figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
