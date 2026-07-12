from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import build_hh_composite_figure as base  # noqa: E402
import hh_programme_characterization as hh  # noqa: E402


ROOT = Path(r"D:\HGSOC_Spatial_Atlas\05_analysis_ready")
RUN_DIR = ROOT / "20260424_definition3b_definition4_raw_geneNMF"
SAMPLE_ROOT = RUN_DIR / "02_definition3b_mixture_programme_niches"
OUT_DIR = RUN_DIR / "08_hh_programme_characterization" / "report_assets" / "figures"
MANIFEST_PATH = OUT_DIR / "hotspot_three_contrast_composite_manifest.csv"

CONTRASTS = [
    ("A", "HH versus LL unmatched", "ll_unmatched"),
    ("B", "HH versus matched LL", "ll_malignant_matched"),
    ("C", "HH versus matched non-HH", "malignant_matched_nonhh"),
]

HH_FILL = "#f4d7d1"
HH_EDGE = "#8f2f24"
CONTROL_FILL = "#d6e1f2"
CONTROL_EDGE = "#33527c"
INK = "#1F2430"
MUTED = "#5F6675"
GRID = "#E6E8F0"


def short_label(program_id: str) -> str:
    short = base.short_program_id(program_id)
    return short.replace("__", "_")


def load_family_annotations() -> dict[str, str]:
    path = RUN_DIR / "11_research_synthesis" / "tables" / "program_family_annotation_snapshot.csv"
    df = pd.read_csv(path)
    return dict(zip(df["program_id"].astype(str), df["alignment_category_draft"].astype(str)))


def sample_display_label(sample_label: str) -> str:
    return sample_label.split("__", 1)[1] if "__" in sample_label else sample_label


def sample_output_path(sample_label: str) -> Path:
    return OUT_DIR / f"{sample_label}_hotspot_three_contrast_composite_horizontal.png"


def discover_samples() -> list[str]:
    return [path.name for path in sorted(SAMPLE_ROOT.iterdir()) if path.is_dir()]


def contrast_effect_matrix(data: dict[str, object], effect_col: str, order: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    contrasts = pd.concat([data["spot_contrasts"], data["neighbor_contrasts"]], ignore_index=True)
    effect = (
        contrasts.pivot(index="context_type", columns="program_id", values=effect_col)
        .reindex(index=["spot_level", "neighborhood"], columns=order)
    )
    fdr = (
        contrasts.pivot(index="context_type", columns="program_id", values="fdr_bh")
        .reindex(index=["spot_level", "neighborhood"], columns=order)
    )
    return effect, fdr


def primary_program_order(data: dict[str, object], effect_col: str) -> list[str]:
    contrasts = pd.concat([data["spot_contrasts"], data["neighbor_contrasts"]], ignore_index=True)
    return (
        contrasts.groupby("program_id")[effect_col]
        .mean()
        .sort_values(ascending=False)
        .index.astype(str)
        .tolist()
    )


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C8CDD8")
    ax.spines["bottom"].set_color("#C8CDD8")
    ax.tick_params(axis="both", labelsize=8, colors=INK)


def draw_boxplots(
    ax: plt.Axes,
    hh_frame: pd.DataFrame,
    control_frame: pd.DataFrame,
    contrast_df: pd.DataFrame,
    order: list[str],
    effect_col: str,
    y_max: float,
    show_ylabel: bool,
) -> None:
    x = np.arange(len(order), dtype=float)
    hh_values = [hh_frame[p].to_numpy(dtype=float) for p in order]
    control_values = [control_frame[p].to_numpy(dtype=float) for p in order]

    hh_box = ax.boxplot(
        hh_values,
        positions=x - 0.18,
        widths=0.24,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
    )
    control_box = ax.boxplot(
        control_values,
        positions=x + 0.18,
        widths=0.24,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
    )
    hh.style_boxplot(hh_box, HH_FILL, HH_EDGE)
    hh.style_boxplot(control_box, CONTROL_FILL, CONTROL_EDGE)

    lookup = contrast_df.set_index("program_id")
    for idx, program_id in enumerate(order):
        vals = np.concatenate([hh_values[idx], control_values[idx]])
        vals = vals[np.isfinite(vals)]
        local_top = float(np.max(vals)) if len(vals) else 0.0
        bracket_y = min(y_max * 0.94, local_top + y_max * 0.045)
        bracket_low = bracket_y - y_max * 0.018
        ax.plot(
            [x[idx] - 0.18, x[idx] - 0.18, x[idx] + 0.18, x[idx] + 0.18],
            [bracket_low, bracket_y, bracket_y, bracket_low],
            color=INK,
            linewidth=0.8,
        )
        fdr = float(lookup.loc[program_id, "fdr_bh"])
        label = hh.significance_stars(fdr)
        ax.text(x[idx], bracket_y + y_max * 0.018, label, ha="center", va="bottom", fontsize=7.5, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels([short_label(p) for p in order], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(-0.04, y_max)
    ax.set_ylabel("Program usage score" if show_ylabel else "", fontsize=9, color=INK)
    style_axes(ax)


def draw_heatmap(
    ax: plt.Axes,
    effect: pd.DataFrame,
    fdr: pd.DataFrame,
    order: list[str],
    spot_n: int,
    neighbor_n: int,
    vmax: float,
    show_ylabel: bool,
):
    matrix = effect.to_numpy(dtype=float)
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = float(matrix[row_idx, col_idx])
            q = float(fdr.iat[row_idx, col_idx])
            label = hh.build_annotation(value, q)
            text_color = "white" if np.isfinite(value) and abs(value) >= 0.55 * vmax else INK
            ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=8, color=text_color)

    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([short_label(p) for p in order], rotation=45, ha="right", fontsize=8)
    ax.set_yticks([0, 1])
    labels = [f"Centre n={spot_n}", f"Neighbour n={neighbor_n}"] if show_ylabel else [f"n={spot_n}", f"n={neighbor_n}"]
    ax.set_yticklabels(labels, fontsize=8, color=INK)
    ax.set_xticks(np.arange(-0.5, len(order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def context_n(data: dict[str, object], config: dict[str, str]) -> tuple[int, int]:
    spot = data["spot_contrasts"]
    neigh = data["neighbor_contrasts"]
    return int(spot[config["n_col"]].iloc[0]), int(neigh[config["n_col"]].iloc[0])


def draw_footer_legend(fig: plt.Figure) -> None:
    legend = fig.legend(
        handles=[
            Patch(facecolor=HH_FILL, edgecolor=HH_EDGE, label="HH"),
            Patch(facecolor=CONTROL_FILL, edgecolor=CONTROL_EDGE, label="comparison"),
        ],
        title="State colours",
        loc="lower left",
        bbox_to_anchor=(0.055, 0.012),
        frameon=False,
        fontsize=10.0,
        title_fontsize=10.2,
        ncol=2,
        handlelength=1.7,
        columnspacing=1.4,
        borderaxespad=0,
    )
    legend.get_title().set_fontweight("bold")
    legend.get_title().set_color(INK)
    for text in legend.get_texts():
        text.set_color(INK)


def draw_statistics_footer(fig: plt.Figure) -> None:
    fig.text(
        0.985,
        0.024,
        "Asterisks indicate FDR < 0.05. A uses Mann-Whitney U. B and C use Wilcoxon signed-rank.",
        ha="right",
        va="bottom",
        fontsize=10.0,
        color=INK,
    )


def draw_unavailable_panel(fig: plt.Figure, gs: plt.GridSpec, col: int, item: dict[str, object], show_ylabel: bool) -> None:
    ax_top = fig.add_subplot(gs[0, col])
    ax_top.axis("off")
    ax_top.set_title(f"{item['panel']}  {item['title']}", fontsize=11, fontweight="bold", color=INK, pad=17)
    ax_top.text(
        0.5,
        1.015,
        "Centre spots",
        transform=ax_top.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="normal",
        color=INK,
    )

    ax_heat = fig.add_subplot(gs[1, col])
    ax_heat.axis("off")
    ax_heat.text(
        0.5,
        0.52,
        "Contrast unavailable",
        transform=ax_heat.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=INK,
    )
    reason = str(item.get("error", "")).strip()
    if reason:
        ax_heat.text(
            0.5,
            0.30,
            textwrap.fill(reason, width=36, break_long_words=False),
            transform=ax_heat.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color=MUTED,
            linespacing=1.15,
        )
    if show_ylabel:
        ax_heat.text(
            -0.08,
            0.50,
            "Centre and neighbour summaries",
            transform=ax_heat.transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=8,
            color=INK,
        )

    ax_bottom = fig.add_subplot(gs[2, col])
    ax_bottom.axis("off")
    ax_bottom.set_title("Immediate neighbours", fontsize=10, color=INK, pad=7)


def load_sample_contrasts(sample_label: str) -> list[dict[str, object]]:
    loaded: list[dict[str, object]] = []
    for panel, title, mode in CONTRASTS:
        config = base.mode_config(mode)
        try:
            data = base.load_sample_matching_data(RUN_DIR, sample_label, mode)
        except Exception as exc:  # noqa: BLE001
            loaded.append(
                {
                    "panel": panel,
                    "title": title,
                    "mode": mode,
                    "config": config,
                    "available": False,
                    "error": str(exc),
                }
            )
            continue
        loaded.append(
            {
                "panel": panel,
                "title": title,
                "mode": mode,
                "config": config,
                "data": data,
                "available": True,
                "error": "",
            }
        )
    return loaded


def render_sample(sample_label: str, annotations: dict[str, str]) -> dict[str, object]:
    loaded = load_sample_contrasts(sample_label)
    available = [item for item in loaded if item["available"]]
    if not available:
        return {
            "sample_label": sample_label,
            "output_path": "",
            "rendered": False,
            "available_contrasts": 0,
            "missing_contrasts": "; ".join(f"{item['mode']}: {item['error']}" for item in loaded),
        }

    primary_config = available[0]["config"]
    primary_data = available[0]["data"]
    order = primary_program_order(primary_data, primary_config["effect_col"])
    order_numeric = sorted(order, key=base.program_sort_key)

    effects = []
    y_values = []
    for item in available:
        data = item["data"]
        config = item["config"]
        effect, fdr = contrast_effect_matrix(data, config["effect_col"], order)
        item["effect"] = effect
        item["fdr"] = fdr
        effects.append(effect.to_numpy(dtype=float))
        for frame_name in ["hh_spot", "control_spot", "hh_neighbor", "control_neighbor"]:
            frame = data[frame_name]
            y_values.append(frame[order].to_numpy(dtype=float).ravel())

    vmax = float(np.nanmax(np.abs(np.concatenate([x.ravel() for x in effects]))))
    vmax = max(vmax, 0.05)
    vmax = float(np.ceil(vmax * 10) / 10)
    y_max = float(np.nanmax(np.concatenate(y_values)))
    y_max = max(1.0, np.ceil((y_max + 0.08) * 10) / 10)

    fig = plt.figure(figsize=(20.0, 11.5), dpi=300)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=3,
        width_ratios=[1.0, 1.0, 1.0],
        height_ratios=[1.35, 0.78, 1.35],
        left=0.055,
        right=0.965,
        top=0.940,
        bottom=0.105,
        wspace=0.08,
        hspace=0.24,
    )

    heat_axes = []
    heat_im = None
    for col, item in enumerate(loaded):
        show_ylabel = col == 0
        if not item["available"]:
            draw_unavailable_panel(fig, gs, col, item, show_ylabel)
            continue

        data = item["data"]
        config = item["config"]

        ax_top = fig.add_subplot(gs[0, col])
        draw_boxplots(
            ax_top,
            data["hh_spot"],
            data["control_spot"],
            data["spot_contrasts"],
            order,
            config["effect_col"],
            y_max,
            show_ylabel,
        )
        ax_top.set_title(f"{item['panel']}  {item['title']}", fontsize=11, fontweight="bold", color=INK, pad=17)
        ax_top.text(
            0.5,
            1.015,
            "Centre spots",
            transform=ax_top.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="normal",
            color=INK,
        )

        ax_heat = fig.add_subplot(gs[1, col])
        spot_n, neighbor_n = context_n(data, config)
        heat_im = draw_heatmap(
            ax_heat,
            item["effect"],
            item["fdr"],
            order,
            spot_n,
            neighbor_n,
            vmax,
            show_ylabel,
        )
        heat_axes.append(ax_heat)

        ax_bottom = fig.add_subplot(gs[2, col])
        draw_boxplots(
            ax_bottom,
            data["hh_neighbor"],
            data["control_neighbor"],
            data["neighbor_contrasts"],
            order,
            config["effect_col"],
            y_max,
            show_ylabel,
        )
        ax_bottom.set_title("Immediate neighbours", fontsize=10, color=INK, pad=7)

    if heat_im is not None and heat_axes:
        cax = heat_axes[-1].inset_axes([1.012, 0.0, 0.028, 1.0])
        cb = fig.colorbar(heat_im, cax=cax)
        cb.ax.tick_params(labelsize=8, colors=INK)

    draw_footer_legend(fig)
    draw_statistics_footer(fig)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = sample_output_path(sample_label)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return {
        "sample_label": sample_label,
        "output_path": str(out_path),
        "rendered": True,
        "available_contrasts": len(available),
        "missing_contrasts": "; ".join(f"{item['mode']}: {item['error']}" for item in loaded if not item["available"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build three-contrast SNAI1-ac hotspot composites for one or more samples.")
    parser.add_argument(
        "--sample",
        action="append",
        help="Sample label to render. Repeat to render multiple samples. Defaults to all sample folders.",
    )
    return parser.parse_args()


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

    args = parse_args()
    samples = args.sample if args.sample else discover_samples()
    annotations = load_family_annotations()
    manifest = []
    for sample_label in samples:
        row = render_sample(sample_label, annotations)
        manifest.append(row)
        print(f"{sample_label}\t{row['rendered']}\t{row['output_path']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest).to_csv(MANIFEST_PATH, index=False)
    print(MANIFEST_PATH)


if __name__ == "__main__":
    main()
