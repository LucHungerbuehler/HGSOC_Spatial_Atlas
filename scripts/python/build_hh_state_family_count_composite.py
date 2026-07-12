from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(r"D:\HGSOC_Spatial_Atlas\05_analysis_ready")
HH_ROOT = ROOT / "20260424_definition3b_definition4_raw_geneNMF" / "08_hh_programme_characterization"
OUT_ROOT = HH_ROOT / "report_assets"
OUT_TABLES = OUT_ROOT / "tables"
OUT_FIGURES = OUT_ROOT / "figures"
COUNT_TABLE = OUT_TABLES / "hh_state_family_direction_counts.csv"
OUT_PNG = OUT_FIGURES / "hh_state_significant_program_family_counts_composite.png"
OUT_SVG = OUT_FIGURES / "hh_state_significant_program_family_counts_composite.svg"


CONTRAST_ORDER = [
    ("hh_vs_ll_unmatched", "HH versus LL unmatched"),
    ("hh_vs_matched_ll", "HH versus matched LL"),
    ("hh_vs_matched_nonhh", "HH versus matched non-HH"),
]

CONTEXT_ORDER = [
    ("spot_level", "Center spots"),
    ("neighborhood", "Immediate neighborhood"),
]

FAMILY_ORDER = [
    "ECM-remodelling myCAF",
    "Inflammatory/hypoxic CAF-stress",
    "Angiogenic vascular/pericyte",
    "IFN/TLS chemokine immune",
    "APC/TAM myeloid",
    "Ciliated epithelial context",
    "Malignant hypoxia/stress",
    "Malignant OXPHOS/metabolic",
    "Malignant proliferation/biosynthesis",
    "Malignant EMT/interface/secretory",
    "Technical/low-quality",
]

INK = "#1F2430"
MUTED = "#5F6675"
GRID = "#E6E8F0"
AXIS = "#C8CDD8"
HH_COLOR = "#bf3f3f"
CONTROL_COLOR = "#3f6fb5"


def load_counts() -> pd.DataFrame:
    counts = pd.read_csv(COUNT_TABLE)
    counts["n_significant_program_contrasts"] = pd.to_numeric(
        counts["n_significant_program_contrasts"], errors="coerce"
    ).fillna(0).astype(int)
    return counts


def count_for(
    counts: pd.DataFrame,
    *,
    contrast_id: str,
    context_type: str,
    family_label: str,
    state: str,
) -> int:
    value = counts.loc[
        counts["contrast_id"].eq(contrast_id)
        & counts["context_type"].eq(context_type)
        & counts["family_label"].eq(family_label)
        & counts["enriched_state"].eq(state),
        "n_significant_program_contrasts",
    ].sum()
    return int(value)


def family_axis_order(counts: pd.DataFrame) -> list[str]:
    present = set(counts["family_label"].astype(str))
    ordered = [family for family in FAMILY_ORDER if family in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def symmetric_limit(counts: pd.DataFrame) -> int:
    max_count = int(counts["n_significant_program_contrasts"].max())
    return int(np.ceil((max_count + 1) / 5) * 5)


def tick_values(limit: int) -> np.ndarray:
    step = max(1, int(limit / 3))
    return np.arange(-limit, limit + 1, step)


def style_axis(ax: plt.Axes) -> None:
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="both", labelsize=8, colors=INK, length=0)


def draw_panel(
    ax: plt.Axes,
    counts: pd.DataFrame,
    *,
    contrast_id: str,
    context_type: str,
    families: list[str],
    x_limit: int,
    show_family_labels: bool,
) -> None:
    control_label = counts.loc[counts["contrast_id"].eq(contrast_id), "control_label"].dropna().iloc[0]
    y = np.arange(len(families))
    hh_counts = [
        count_for(
            counts,
            contrast_id=contrast_id,
            context_type=context_type,
            family_label=family,
            state="HH",
        )
        for family in families
    ]
    control_counts = [
        count_for(
            counts,
            contrast_id=contrast_id,
            context_type=context_type,
            family_label=family,
            state=control_label,
        )
        for family in families
    ]

    ax.barh(y, [-count for count in control_counts], color=CONTROL_COLOR, height=0.72)
    ax.barh(y, hh_counts, color=HH_COLOR, height=0.72)

    ax.set_xlim(-x_limit, x_limit)
    ticks = tick_values(x_limit)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(abs(int(tick))) for tick in ticks])
    ax.set_yticks(y)
    ax.set_yticklabels(families if show_family_labels else [""] * len(families), fontsize=8, color=INK)
    ax.tick_params(axis="y", labelleft=show_family_labels)
    ax.invert_yaxis()
    style_axis(ax)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    counts = load_counts()
    families = family_axis_order(counts)
    x_limit = symmetric_limit(counts)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(16.8, max(7.2, 0.45 * len(families) + 2.1)),
        sharex=True,
        sharey=False,
    )

    for col_idx, (contrast_id, contrast_label) in enumerate(CONTRAST_ORDER):
        axes[0, col_idx].set_title(contrast_label, fontsize=11, fontweight="bold", color=INK, pad=10)
        for row_idx, (context_type, _) in enumerate(CONTEXT_ORDER):
            draw_panel(
                axes[row_idx, col_idx],
                counts,
                contrast_id=contrast_id,
                context_type=context_type,
                families=families,
                x_limit=x_limit,
                show_family_labels=col_idx == 0,
            )
            if row_idx == 1:
                axes[row_idx, col_idx].set_xlabel("Significant local program contrasts", fontsize=9, color=INK)

    fig.text(0.974, 0.70, CONTEXT_ORDER[0][1], rotation=270, ha="center", va="center", fontsize=11, color=INK)
    fig.text(0.974, 0.30, CONTEXT_ORDER[1][1], rotation=270, ha="center", va="center", fontsize=11, color=INK)

    handles = [
        Patch(facecolor=CONTROL_COLOR, label="Comparison enriched"),
        Patch(facecolor=HH_COLOR, label="HH enriched"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0.055, 0.055, 0.955, 0.965), w_pad=1.8, h_pad=1.6)
    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_SVG)
    plt.close(fig)
    print(OUT_PNG)
    print(OUT_SVG)


if __name__ == "__main__":
    main()
