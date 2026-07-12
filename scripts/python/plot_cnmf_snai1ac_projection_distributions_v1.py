"""
Plot SNAI1-ac projection score distributions across harmonized MP1-MP8.

Inputs are the clean K*-only projection table created by
build_cnmf_snai1ac_signature_projection_v1.py. The plotted MP labels are
harmonized metaprograms derived from reclustering raw sample-specific cNMF
programmes; they are not raw cNMF programmes themselves.
"""

from __future__ import annotations

import argparse
import json
import shutil
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


DATA_ROOT = Path(r"D:\HGSOC_Spatial_Atlas")
OUTPUT_ROOT = (
    DATA_ROOT
    / "05_analysis_ready"
    / "S3_cNMF_Tumor_Programs"
    / "snai1ac_signature_projection_onto_cnmf_programs_v1"
)
INPUT_TABLE = OUTPUT_ROOT / "tables" / "kstar_snai1ac_signature_projection_clean.csv"
SCRIPT_PATH = Path(__file__).resolve()

SCORE_COLUMNS = {
    "across": "across_sample_projection_z_score",
    "within": "within_sample_projection_z_score",
}

MP_ORDER = [
    "MP1 angiogenic/vascular",
    "MP2 iCAF-stress",
    "MP3 complement-CAF",
    "MP4 activated-myCAF",
    "MP5 IFN/TLS immune",
    "MP6 APC/TAM myeloid",
    "MP7 malignant hypoxia",
    "MP8 malignant acute-phase/secretory",
]

MP_SHORT = {
    "MP1 angiogenic/vascular": "MP1",
    "MP2 iCAF-stress": "MP2",
    "MP3 complement-CAF": "MP3",
    "MP4 activated-myCAF": "MP4",
    "MP5 IFN/TLS immune": "MP5",
    "MP6 APC/TAM myeloid": "MP6",
    "MP7 malignant hypoxia": "MP7",
    "MP8 malignant acute-phase/secretory": "MP8",
}

MP_TICK_LABEL = {
    "MP1": "MP1\nangiogenic/\nvascular",
    "MP2": "MP2\niCAF-\nstress",
    "MP3": "MP3\ncomplement-\nCAF",
    "MP4": "MP4\nactivated-\nmyCAF",
    "MP5": "MP5\nIFN/TLS\nimmune",
    "MP6": "MP6\nAPC/TAM\nmyeloid",
    "MP7": "MP7\nmalignant\nhypoxia",
    "MP8": "MP8\nacute-phase/\nsecretory",
}

FONT_FAMILY = ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]
MONO_FONT_FAMILY = ["SF Mono", "Menlo", "Consolas", "DejaVu Sans Mono", "monospace"]

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLOR_FAMILIES = {
    "blue": {
        "open": TOKENS["panel"],
        "xlight": "#EAF1FE",
        "light": "#CEDFFE",
        "base": "#A3BEFA",
        "mid": "#5477C4",
        "dark": "#2E4780",
    },
    "pink": {
        "open": TOKENS["panel"],
        "xlight": "#FCDAD6",
        "light": "#F5BACC",
        "base": "#F390CA",
        "mid": "#BD569B",
        "dark": "#8A3A6F",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SNAI1-ac projection distributions by harmonized MP.")
    parser.add_argument("--input-table", type=Path, default=INPUT_TABLE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "figure.edgecolor": "none",
            "savefig.facecolor": TOKENS["surface"],
            "savefig.edgecolor": "none",
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.monospace": MONO_FONT_FAMILY,
            "patch.linewidth": 1.0,
        },
    )


def add_chart_header(fig: plt.Figure, ax: plt.Axes, title: str, subtitle: str) -> None:
    title = textwrap.fill(str(title).strip(), width=88, break_long_words=False)
    subtitle = textwrap.fill(str(subtitle).strip(), width=128, break_long_words=False)
    if not title or not subtitle:
        raise ValueError("Every shipped chart needs a title and subtitle.")

    title_lines = title.count("\n") + 1
    subtitle_lines = subtitle.count("\n") + 1
    ax.set_title("")
    fig.subplots_adjust(
        top=max(0.68, 0.86 - 0.04 * (title_lines - 1) - 0.028 * (subtitle_lines - 1)),
        bottom=0.21,
        left=0.08,
        right=0.98,
    )
    left = ax.get_position().x0
    fig.text(
        left,
        0.985,
        title,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
        color=TOKENS["ink"],
        linespacing=1.08,
    )
    fig.text(
        left,
        0.925 - 0.04 * (title_lines - 1),
        subtitle,
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
        linespacing=1.18,
    )
    sns.despine(ax=ax)


def load_plot_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"program_id", "mp1_8_name", *SCORE_COLUMNS.values()} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[df["mp1_8_name"].isin(MP_ORDER)].copy()
    if df.empty:
        raise ValueError("No MP1-MP8 rows found after excluding not_assigned_to_MP.")

    for column in SCORE_COLUMNS.values():
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["mp_short"] = df["mp1_8_name"].map(MP_SHORT)
    df["mp_short"] = pd.Categorical(df["mp_short"], categories=[MP_SHORT[label] for label in MP_ORDER], ordered=True)
    return df.sort_values(["mp_short", "program_id"]).reset_index(drop=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mp in MP_ORDER:
        subset = df[df["mp1_8_name"].eq(mp)]
        for score_name, column in SCORE_COLUMNS.items():
            values = subset[column].dropna()
            rows.append(
                {
                    "mp1_8_name": mp,
                    "score_type": score_name,
                    "n_programs": int(values.size),
                    "median": float(values.median()) if values.size else np.nan,
                    "q25": float(values.quantile(0.25)) if values.size else np.nan,
                    "q75": float(values.quantile(0.75)) if values.size else np.nan,
                    "min": float(values.min()) if values.size else np.nan,
                    "max": float(values.max()) if values.size else np.nan,
                }
            )
    return pd.DataFrame(rows)


def common_y_limits(df: pd.DataFrame) -> tuple[float, float]:
    values = pd.concat([df[column] for column in SCORE_COLUMNS.values()], ignore_index=True).dropna()
    lower = float(values.min())
    upper = float(values.max())
    pad = max(0.25, (upper - lower) * 0.08)
    return lower - pad, upper + pad


def tick_labels(df: pd.DataFrame) -> list[str]:
    counts = df.groupby("mp_short", observed=False).size().to_dict()
    labels = []
    for mp_short in [MP_SHORT[label] for label in MP_ORDER]:
        labels.append(f"{MP_TICK_LABEL[mp_short]}\nn={counts.get(mp_short, 0)}")
    return labels


def format_axis(ax: plt.Axes, df: pd.DataFrame, y_label: str, y_limits: tuple[float, float]) -> None:
    ax.axhline(0, color=TOKENS["ink"], linewidth=1.0, linestyle=":", zorder=0)
    ax.set_xlabel("")
    ax.set_ylabel(y_label, color=TOKENS["ink"], fontsize=10)
    ax.set_ylim(*y_limits)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=7))
    ax.grid(axis="y", color=TOKENS["grid"], linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])
    ax.tick_params(axis="y", colors=TOKENS["muted"], labelsize=8.5)
    ax.tick_params(axis="x", colors=TOKENS["ink"], labelsize=8.2, length=0, pad=7)
    ax.set_xticks(range(len(MP_ORDER)), tick_labels(df))


def draw_boxstrip(
    ax: plt.Axes,
    df: pd.DataFrame,
    score_col: str,
    family: dict[str, str],
    y_label: str,
    y_limits: tuple[float, float],
) -> None:
    order = [MP_SHORT[label] for label in MP_ORDER]
    sns.boxplot(
        data=df,
        x="mp_short",
        y=score_col,
        order=order,
        ax=ax,
        width=0.54,
        showfliers=False,
        color=family["xlight"],
        boxprops={"facecolor": family["xlight"], "edgecolor": family["dark"], "linewidth": 1.0},
        whiskerprops={"color": family["dark"], "linewidth": 1.0},
        capprops={"color": family["dark"], "linewidth": 1.0},
        medianprops={"color": TOKENS["ink"], "linewidth": 1.15},
    )
    sns.stripplot(
        data=df,
        x="mp_short",
        y=score_col,
        order=order,
        ax=ax,
        jitter=0.22,
        size=4.2,
        alpha=0.76,
        color=family["base"],
        edgecolor=family["dark"],
        linewidth=0.55,
        zorder=3,
    )
    format_axis(ax, df, y_label, y_limits)


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    for suffix in [".png", ".svg"]:
        fig.savefig(out_base.with_suffix(suffix), dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_single(
    df: pd.DataFrame,
    score_col: str,
    out_base: Path,
    title: str,
    subtitle: str,
    y_label: str,
    family: dict[str, str],
    y_limits: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    draw_boxstrip(ax, df, score_col, family, y_label, y_limits)
    add_chart_header(fig, ax, title, subtitle)
    save_figure(fig, out_base)


def plot_combined(df: pd.DataFrame, out_base: Path, y_limits: tuple[float, float]) -> None:
    long_df = df.melt(
        id_vars=["program_id", "mp1_8_name", "mp_short"],
        value_vars=[SCORE_COLUMNS["across"], SCORE_COLUMNS["within"]],
        var_name="score_column",
        value_name="projection_z_score",
    )
    score_labels = {
        SCORE_COLUMNS["across"]: "Across-sample z-score",
        SCORE_COLUMNS["within"]: "Within-sample z-score",
    }
    score_order = ["Across-sample z-score", "Within-sample z-score"]
    long_df["score_type"] = long_df["score_column"].map(score_labels)

    palette = {
        "Across-sample z-score": COLOR_FAMILIES["blue"]["xlight"],
        "Within-sample z-score": COLOR_FAMILIES["pink"]["xlight"],
    }
    point_palette = {
        "Across-sample z-score": COLOR_FAMILIES["blue"]["base"],
        "Within-sample z-score": COLOR_FAMILIES["pink"]["base"],
    }

    order = [MP_SHORT[label] for label in MP_ORDER]
    fig, ax = plt.subplots(figsize=(12.6, 6.4))
    fig.patch.set_facecolor(TOKENS["surface"])
    sns.boxplot(
        data=long_df,
        x="mp_short",
        y="projection_z_score",
        hue="score_type",
        order=order,
        hue_order=score_order,
        palette=palette,
        ax=ax,
        width=0.72,
        dodge=True,
        showfliers=False,
        linewidth=1.0,
        boxprops={"edgecolor": TOKENS["ink"], "linewidth": 1.0},
        whiskerprops={"color": TOKENS["ink"], "linewidth": 1.0},
        capprops={"color": TOKENS["ink"], "linewidth": 1.0},
        medianprops={"color": TOKENS["ink"], "linewidth": 1.15},
    )
    sns.stripplot(
        data=long_df,
        x="mp_short",
        y="projection_z_score",
        hue="score_type",
        order=order,
        hue_order=score_order,
        palette=point_palette,
        ax=ax,
        dodge=True,
        jitter=0.16,
        size=3.8,
        alpha=0.72,
        edgecolor=TOKENS["ink"],
        linewidth=0.45,
        zorder=3,
    )

    handles, labels = ax.get_legend_handles_labels()
    legend_map = {}
    for handle, label in zip(handles, labels):
        if label in score_order and label not in legend_map:
            legend_map[label] = handle
    ax.legend(
        [legend_map[label] for label in score_order],
        score_order,
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        frameon=False,
        ncol=2,
        borderaxespad=0,
        fontsize=9,
    )

    format_axis(ax, df, "Projection z-score", y_limits)
    title = "SNAI1-ac projection score distributions by harmonized metaprogram"
    subtitle = (
        "K*-selected programmes assigned to harmonized MP1-MP8; paired boxes show across- and "
        "within-sample z-scores for each MP, points show individual programmes, and the dotted line marks z=0."
    )
    add_chart_header(fig, ax, title, subtitle, )
    save_figure(fig, out_base)


def write_manifest(output_root: Path, df: pd.DataFrame, summary: pd.DataFrame, figure_names: list[str]) -> None:
    manifest = {
        "branch": "snai1ac_signature_projection_onto_cnmf_programs_v1",
        "input_table": str(INPUT_TABLE),
        "n_mp_assigned_programs_plotted": int(len(df)),
        "excluded_rows": "not_assigned_to_MP rows are intentionally excluded from MP1-MP8 distribution plots.",
        "score_columns": SCORE_COLUMNS,
        "figures": figure_names,
        "summary_table": "tables/kstar_snai1ac_signature_projection_mp_distribution_summary.csv",
        "script": str(SCRIPT_PATH),
        "wording_note": "MP labels are harmonized metaprograms derived from reclustering raw sample-specific programmes; they are not raw cNMF programmes.",
    }
    (output_root / "figures" / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary.to_csv(output_root / "tables" / "kstar_snai1ac_signature_projection_mp_distribution_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    use_chart_theme()

    output_root = args.output_root
    figure_dir = output_root / "figures"
    table_dir = output_root / "tables"
    script_dir = output_root / "scripts_used"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    script_dir.mkdir(parents=True, exist_ok=True)

    df = load_plot_data(args.input_table)
    summary = summarize(df)
    y_limits = common_y_limits(df)

    figure_specs = [
        (
            SCORE_COLUMNS["across"],
            figure_dir / "kstar_snai1ac_projection_across_sample_by_mp_boxstrip",
            "Across-sample SNAI1-ac projection by harmonized metaprogram",
            "K*-selected programmes assigned to harmonized MP1-MP8; boxes show median and IQR, points show individual programmes, and the dotted line marks z=0.",
            "Across-sample projection z-score",
            COLOR_FAMILIES["blue"],
        ),
        (
            SCORE_COLUMNS["within"],
            figure_dir / "kstar_snai1ac_projection_within_sample_by_mp_boxstrip",
            "Within-sample SNAI1-ac projection by harmonized metaprogram",
            "K*-selected programmes assigned to harmonized MP1-MP8; boxes show median and IQR, points show individual programmes, and the dotted line marks z=0.",
            "Within-sample projection z-score",
            COLOR_FAMILIES["pink"],
        ),
    ]
    for score_col, out_base, title, subtitle, y_label, family in figure_specs:
        plot_single(df, score_col, out_base, title, subtitle, y_label, family, y_limits)

    combined_base = figure_dir / "kstar_snai1ac_projection_across_vs_within_by_mp_boxstrip"
    plot_combined(df, combined_base, y_limits)

    figure_names = [
        "figures/kstar_snai1ac_projection_across_sample_by_mp_boxstrip.png",
        "figures/kstar_snai1ac_projection_within_sample_by_mp_boxstrip.png",
        "figures/kstar_snai1ac_projection_across_vs_within_by_mp_boxstrip.png",
    ]
    write_manifest(output_root, df, summary, figure_names)
    shutil.copy2(SCRIPT_PATH, script_dir / SCRIPT_PATH.name)

    print(f"Wrote {len(figure_names)} PNG figures and SVG companions to {figure_dir}")
    print(f"Wrote summary table with {len(summary)} rows")


if __name__ == "__main__":
    main()
