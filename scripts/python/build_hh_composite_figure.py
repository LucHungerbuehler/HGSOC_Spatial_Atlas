from __future__ import annotations

import argparse
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import hh_programme_characterization as hh


SHORT_PROGRAM_PATTERN = re.compile(r"(K\d+__P\d+)$")
KP_PATTERN = re.compile(r"K(\d+)__P(\d+)$")
NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a single composite HH figure from existing heatmap and violin/boxplot panels."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--annotation-xlsx", required=True)
    parser.add_argument("--sample-label", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--comparison-mode",
        default="malignant_matched_nonhh",
        choices=["malignant_matched_nonhh", "ll_unmatched", "ll_malignant_matched"],
    )
    parser.add_argument(
        "--omit-family-labels",
        action="store_true",
        help="Do not print program family labels in the right-hand annotation legend.",
    )
    return parser.parse_args()


def short_program_id(program_id: str) -> str:
    match = SHORT_PROGRAM_PATTERN.search(str(program_id))
    return match.group(1) if match else str(program_id)


def program_sort_key(program_id: str) -> tuple[int, int, str]:
    short_id = short_program_id(program_id)
    match = KP_PATTERN.search(short_id)
    if match:
        return (int(match.group(2)), int(match.group(1)), short_id)
    return (10**9, 10**9, short_id)


def sample_spot_table_path(run_dir: Path, sample_label: str) -> Path:
    return run_dir / "02_definition3b_mixture_programme_niches" / sample_label / "tables" / "spot_level_table.csv"


def rename_ll_paired_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(
        columns={
            "matched_nonhh_mean": "matched_ll_mean",
            "mean_difference_hh_minus_nonhh": "mean_difference_hh_minus_ll",
            "median_difference_hh_minus_nonhh": "median_difference_hh_minus_ll",
            "cohens_dz_hh_minus_nonhh": "cohens_dz_hh_minus_ll",
        }
    )


def mode_config(comparison_mode: str) -> dict[str, str]:
    if comparison_mode == "malignant_matched_nonhh":
        return {
            "branch_dir": "",
            "spot_table": "hh_spot_level_programme_contrasts.csv",
            "neighbor_table": "hh_neighborhood_programme_contrasts.csv",
            "effect_col": "mean_difference_hh_minus_nonhh",
            "n_col": "n_pairs",
            "control_label": "Matched non-HH",
            "difference_label": "matched non-HH",
            "stats_note": "FDR-adjusted Wilcoxon signed-rank results.",
        }
    if comparison_mode == "ll_unmatched":
        return {
            "branch_dir": "hh_vs_ll_unmatched",
            "spot_table": "hh_ll_spot_level_programme_contrasts.csv",
            "neighbor_table": "hh_ll_neighborhood_programme_contrasts.csv",
            "effect_col": "mean_difference_hh_minus_ll",
            "n_col": "hh_n",
            "control_label": "LL",
            "difference_label": "LL",
            "stats_note": "FDR-adjusted Mann-Whitney U results.",
        }
    if comparison_mode == "ll_malignant_matched":
        return {
            "branch_dir": "hh_vs_ll_malignant_matched",
            "spot_table": "hh_ll_matched_spot_level_programme_contrasts.csv",
            "neighbor_table": "hh_ll_matched_neighborhood_programme_contrasts.csv",
            "effect_col": "mean_difference_hh_minus_ll",
            "n_col": "n_pairs",
            "control_label": "Matched LL",
            "difference_label": "matched LL",
            "stats_note": "FDR-adjusted Wilcoxon signed-rank results.",
        }
    raise ValueError(f"Unsupported comparison mode: {comparison_mode}")


def load_sample_matching_data(run_dir: Path, sample_label: str, comparison_mode: str) -> dict[str, object]:
    sample_path = sample_spot_table_path(run_dir, sample_label)
    frame = pd.read_csv(sample_path)
    program_cols = hh.programme_columns(frame)
    spot_frame = hh.make_valid_frame(frame, program_cols)

    hh_spot = spot_frame.loc[spot_frame["LISA_category"].astype(str) == "High-High"].reset_index(drop=True)
    neighbor_frame = hh.compute_neighbor_vectors(spot_frame, program_cols)
    hh_neighbor = neighbor_frame.loc[neighbor_frame["LISA_category"].astype(str) == "High-High"].reset_index(drop=True)

    if comparison_mode == "malignant_matched_nonhh":
        control_spot = spot_frame.loc[spot_frame["LISA_category"].astype(str) != "High-High"].reset_index(drop=True)
        control_neighbor = neighbor_frame.loc[
            neighbor_frame["LISA_category"].astype(str) != "High-High"
        ].reset_index(drop=True)
        hh_spot, control_spot, _ = hh.optimal_malignant_matching(hh_spot, control_spot)
        hh_neighbor, control_neighbor, _ = hh.optimal_malignant_matching(hh_neighbor, control_neighbor)
        spot_contrasts = hh.paired_programme_contrasts(
            hh_spot,
            control_spot,
            program_cols=program_cols,
            context_type="spot_level",
        )
        neighbor_contrasts = hh.paired_programme_contrasts(
            hh_neighbor,
            control_neighbor,
            program_cols=program_cols,
            context_type="neighborhood",
        )
    elif comparison_mode == "ll_unmatched":
        control_spot = spot_frame.loc[spot_frame["LISA_category"].astype(str) == "Low-Low"].reset_index(drop=True)
        control_neighbor = neighbor_frame.loc[
            neighbor_frame["LISA_category"].astype(str) == "Low-Low"
        ].reset_index(drop=True)
        if hh_spot.empty or control_spot.empty or hh_neighbor.empty or control_neighbor.empty:
            raise ValueError(f"HH/LL pools unavailable for sample {sample_label}")
        spot_contrasts = hh.unpaired_programme_contrasts(
            hh_spot,
            control_spot,
            program_cols=program_cols,
            context_type="spot_level",
        )
        neighbor_contrasts = hh.unpaired_programme_contrasts(
            hh_neighbor,
            control_neighbor,
            program_cols=program_cols,
            context_type="neighborhood",
        )
    elif comparison_mode == "ll_malignant_matched":
        control_spot = spot_frame.loc[spot_frame["LISA_category"].astype(str) == "Low-Low"].reset_index(drop=True)
        control_neighbor = neighbor_frame.loc[
            neighbor_frame["LISA_category"].astype(str) == "Low-Low"
        ].reset_index(drop=True)
        if hh_spot.empty or control_spot.empty or hh_neighbor.empty or control_neighbor.empty:
            raise ValueError(f"HH/LL pools unavailable for sample {sample_label}")
        hh_spot, control_spot, _ = hh.optimal_malignant_matching(hh_spot, control_spot)
        hh_neighbor, control_neighbor, _ = hh.optimal_malignant_matching(hh_neighbor, control_neighbor)
        spot_contrasts = rename_ll_paired_contrasts(
            hh.paired_programme_contrasts(
                hh_spot,
                control_spot,
                program_cols=program_cols,
                context_type="spot_level",
            )
        )
        neighbor_contrasts = rename_ll_paired_contrasts(
            hh.paired_programme_contrasts(
                hh_neighbor,
                control_neighbor,
                program_cols=program_cols,
                context_type="neighborhood",
            )
        )
    else:
        raise ValueError(f"Unsupported comparison mode: {comparison_mode}")

    return {
        "sample_label": sample_label,
        "program_cols": program_cols,
        "hh_spot": hh_spot,
        "control_spot": control_spot,
        "spot_contrasts": spot_contrasts,
        "hh_neighbor": hh_neighbor,
        "control_neighbor": control_neighbor,
        "neighbor_contrasts": neighbor_contrasts,
    }


def load_annotation_map(annotation_xlsx: Path) -> dict[str, str]:
    with zipfile.ZipFile(annotation_xlsx) as zf:
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}

        target = None
        for sheet in workbook.find("a:sheets", NS):
            if sheet.attrib["name"] == "kstar_program_annotations_v0_2_":
                rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
                target = "xl/" + rel_map[rid]
                break
        if target is None:
            raise KeyError("Sheet 'kstar_program_annotations_v0_2_' not found in annotation workbook.")

        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            sst = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in sst.findall("a:si", NS):
                shared_strings.append("".join(t.text or "" for t in si.iterfind(".//a:t", NS)))

        def cell_value(cell: ET.Element) -> str:
            cell_type = cell.attrib.get("t")
            value = cell.find("a:v", NS)
            if value is None:
                inline = cell.find("a:is", NS)
                return "".join(t.text or "" for t in inline.iterfind(".//a:t", NS)) if inline is not None else ""
            raw = value.text or ""
            if cell_type == "s":
                return shared_strings[int(raw)] if raw.isdigit() and int(raw) < len(shared_strings) else raw
            return raw

        worksheet = ET.fromstring(zf.read(target))
        rows = []
        for row in worksheet.find("a:sheetData", NS).findall("a:row", NS):
            rows.append([cell_value(cell) for cell in row.findall("a:c", NS)])

    header = rows[0]
    records = [dict(zip(header, row)) for row in rows[1:] if row]
    return {
        record["program_id"]: str(record.get("alignment_category_draft", "")).strip()
        for record in records
        if record.get("program_id")
    }


def load_family_map(run_dir: Path) -> dict[str, str]:
    family_path = run_dir / "11_research_synthesis" / "tables" / "program_family_annotation_snapshot.csv"
    family_df = pd.read_csv(family_path)
    required = {"program_id", "family_id", "family_label"}
    missing = required.difference(family_df.columns)
    if missing:
        raise ValueError(f"Family annotation snapshot is missing columns: {sorted(missing)}")

    family_map: dict[str, str] = {}
    for row in family_df.to_dict("records"):
        program_id = str(row.get("program_id", "")).strip()
        if not program_id:
            continue
        family_id = str(row.get("family_id", "")).strip()
        family_label = str(row.get("family_label", "")).strip()
        if family_label and family_id:
            family_map[program_id] = f"{family_label} ({family_id})"
        elif family_label:
            family_map[program_id] = family_label
        elif family_id:
            family_map[program_id] = family_id
    return family_map


def get_program_order(run_dir: Path, sample_label: str, comparison_mode: str) -> list[str]:
    config = mode_config(comparison_mode)
    tables_dir = run_dir / "08_hh_programme_characterization" / config["branch_dir"] / "tables"
    spot = pd.read_csv(tables_dir / config["spot_table"])
    neighbor = pd.read_csv(tables_dir / config["neighbor_table"])
    combined = pd.concat([spot, neighbor], ignore_index=True)
    sample_df = combined.loc[combined["sample_label"] == sample_label].copy()
    if sample_df.empty:
        raise ValueError(f"No HH contrast rows found for sample {sample_label}")
    order = (
        sample_df.groupby("program_id")[config["effect_col"]]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    return order


def plot_distribution_axis(
    ax: plt.Axes,
    matched_hh: pd.DataFrame,
    matched_control: pd.DataFrame,
    contrast_df: pd.DataFrame,
    context_label: str,
    effect_col: str,
) -> None:
    program_order = (
        contrast_df.sort_values([effect_col, "fdr_bh"], ascending=[False, True])["program_id"]
        .tolist()
    )
    hh_data = [matched_hh[program_id].to_numpy(dtype=float) for program_id in program_order]
    control_data = [matched_control[program_id].to_numpy(dtype=float) for program_id in program_order]
    positions = np.arange(len(program_order), dtype=float)
    hh_positions = positions - 0.18
    control_positions = positions + 0.18

    all_values = np.concatenate(hh_data + control_data) if program_order else np.asarray([0.0])
    finite_values = all_values[np.isfinite(all_values)]
    data_min = float(np.min(finite_values)) if len(finite_values) else 0.0
    data_max = float(np.max(finite_values)) if len(finite_values) else 1.0
    data_range = max(data_max - data_min, 0.08)

    hh_box = ax.boxplot(
        hh_data,
        positions=hh_positions,
        widths=0.20,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
    )
    control_box = ax.boxplot(
        control_data,
        positions=control_positions,
        widths=0.20,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
    )
    hh.style_boxplot(hh_box, "#f4d7d1", "#8f2f24")
    hh.style_boxplot(control_box, "#d6e1f2", "#33527c")

    contrast_lookup = contrast_df.set_index("program_id")
    annotation_tops: list[float] = []
    for idx, program_id in enumerate(program_order):
        pair_top = float(max(np.max(hh_data[idx]), np.max(control_data[idx])))
        bracket_y = pair_top + 0.05 * data_range
        ax.plot(
            [hh_positions[idx], hh_positions[idx], control_positions[idx], control_positions[idx]],
            [bracket_y - 0.015 * data_range, bracket_y, bracket_y, bracket_y - 0.015 * data_range],
            color="black",
            linewidth=0.8,
        )
        ax.text(
            positions[idx],
            bracket_y + 0.01 * data_range,
            hh.significance_stars(float(contrast_lookup.loc[program_id, "fdr_bh"])),
            ha="center",
            va="bottom",
            fontsize=8,
        )
        annotation_tops.append(bracket_y + 0.07 * data_range)

    y_lower = min(0.0, data_min - 0.06 * data_range)
    y_upper = max(annotation_tops) if annotation_tops else data_max + 0.1 * data_range
    y_upper = max(y_upper, data_max + 0.12 * data_range)
    ax.set_ylim(y_lower, y_upper)

    ax.set_xticks(positions)
    ax.set_xticklabels([short_program_id(program_id) for program_id in program_order], rotation=45, ha="right")
    ax.set_ylabel("Programme usage score")
    ax.set_xlabel("")
    ax.set_title(f"{context_label}", fontsize=10, pad=8)
    ax.grid(axis="y", alpha=0.18, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_heatmap_axis(
    ax: plt.Axes,
    spot_contrasts: pd.DataFrame,
    neighbor_contrasts: pd.DataFrame,
    effect_col: str,
    n_col: str,
) -> None:
    sample_df = pd.concat([spot_contrasts, neighbor_contrasts], ignore_index=True)
    program_order = (
        sample_df.groupby("program_id")[effect_col]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    effect = (
        sample_df.pivot(index="context_type", columns="program_id", values=effect_col)
        .reindex(index=["spot_level", "neighborhood"], columns=program_order)
    )
    fdr = sample_df.pivot(index="context_type", columns="program_id", values="fdr_bh").reindex(
        index=["spot_level", "neighborhood"],
        columns=program_order,
    )
    pairs = sample_df.groupby("context_type")[n_col].first().to_dict()
    global_abs_max = float(np.nanmax(np.abs(sample_df[effect_col].to_numpy(dtype=float))))
    global_abs_max = max(global_abs_max, 0.05)

    im = ax.imshow(
        effect.to_numpy(dtype=float),
        cmap="RdBu_r",
        vmin=-global_abs_max,
        vmax=global_abs_max,
        aspect="auto",
        interpolation="nearest",
    )
    for row_idx in range(effect.shape[0]):
        for col_idx in range(effect.shape[1]):
            value = float(effect.iat[row_idx, col_idx])
            pval = float(fdr.iat[row_idx, col_idx])
            label = hh.build_annotation(value, pval)
            text_color = "white" if np.isfinite(value) and abs(value) >= 0.55 * global_abs_max else "black"
            ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=8, color=text_color)

    ax.set_xticks(range(len(program_order)))
    ax.set_xticklabels([short_program_id(program_id) for program_id in program_order], rotation=45, ha="right")
    ax.set_yticks([0, 1])
    ax.set_yticklabels([f"Spot-level (n={int(pairs.get('spot_level', 0))})", f"Neighborhood (n={int(pairs.get('neighborhood', 0))})"])
    ax.set_xticks(np.arange(-0.5, len(program_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im


def build_composite(
    run_dir: Path,
    annotation_xlsx: Path,
    sample_label: str,
    output_path: Path,
    comparison_mode: str,
    omit_family_labels: bool = False,
) -> None:
    config = mode_config(comparison_mode)
    sample_data = load_sample_matching_data(
        run_dir=run_dir,
        sample_label=sample_label,
        comparison_mode=comparison_mode,
    )
    spot_contrasts = sample_data["spot_contrasts"]
    neighbor_contrasts = sample_data["neighbor_contrasts"]
    program_order = get_program_order(run_dir=run_dir, sample_label=sample_label, comparison_mode=comparison_mode)
    annotation_map = load_annotation_map(annotation_xlsx)
    family_map = {} if omit_family_labels else load_family_map(run_dir)
    annotation_lines = [
        (
            short_program_id(program_id),
            annotation_map.get(program_id, "annotation missing"),
            "" if omit_family_labels else family_map.get(program_id, "family missing"),
        )
        for program_id in sorted(program_order, key=program_sort_key)
    ]

    fig = plt.figure(figsize=(14, 17))
    grid = fig.add_gridspec(
        nrows=3,
        ncols=2,
        width_ratios=[4.2, 1.9],
        height_ratios=[1.6, 1.0, 1.6],
        hspace=0.20,
        wspace=0.10,
    )

    ax_spot = fig.add_subplot(grid[0, 0])
    plot_distribution_axis(
        ax=ax_spot,
        matched_hh=sample_data["hh_spot"],
        matched_control=sample_data["control_spot"],
        contrast_df=spot_contrasts,
        context_label="Spot-level",
        effect_col=config["effect_col"],
    )

    ax_heat = fig.add_subplot(grid[1, 0])
    im = plot_heatmap_axis(
        ax=ax_heat,
        spot_contrasts=spot_contrasts,
        neighbor_contrasts=neighbor_contrasts,
        effect_col=config["effect_col"],
        n_col=config["n_col"],
    )
    fig.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.015)

    ax_neighborhood = fig.add_subplot(grid[2, 0])
    plot_distribution_axis(
        ax=ax_neighborhood,
        matched_hh=sample_data["hh_neighbor"],
        matched_control=sample_data["control_neighbor"],
        contrast_df=neighbor_contrasts,
        context_label="Neighborhood",
        effect_col=config["effect_col"],
    )

    right_ax = fig.add_subplot(grid[:, 1])
    right_ax.axis("off")
    right_ax.text(0.0, 0.985, sample_label, fontsize=17, fontweight="bold", va="top")
    right_ax.text(0.0, 0.94, "Panel key", fontsize=12, fontweight="bold", va="top")
    legend = right_ax.legend(
        handles=[
            Patch(facecolor="#c44536", edgecolor="#8f2f24", alpha=0.55, label="HH"),
            Patch(facecolor="#4a6fa5", edgecolor="#33527c", alpha=0.55, label=config["control_label"]),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, 0.905),
        frameon=False,
        fontsize=11,
    )
    right_ax.add_artist(legend)
    right_ax.text(
        0.0,
        0.84,
        "Rows:\n1. Spot-level distribution\n2. HH contrast heatmap\n3. Neighborhood distribution",
        fontsize=11,
        va="top",
    )
    right_ax.text(0.0, 0.73, "Program annotation legend", fontsize=12, fontweight="bold", va="top")

    y = 0.70
    step = 0.074 if omit_family_labels else (0.092 if len(annotation_lines) <= 6 else 0.074)
    for short_id, label, family_label in annotation_lines:
        right_ax.text(
            0.0,
            y,
            f"{short_id}",
            fontsize=11,
            fontweight="bold",
            va="top",
        )
        right_ax.text(
            0.0,
            y - 0.023,
            label,
            fontsize=10,
            color="#333333",
            va="top",
            wrap=True,
        )
        if not omit_family_labels:
            right_ax.text(
                0.0,
                y - 0.045,
                f"Family: {family_label}",
                fontsize=11,
                fontweight="bold",
                color="#555555",
                va="top",
                wrap=True,
            )
        y -= step

    right_ax.text(
        0.0,
        0.04,
        "Heatmap cells show raw mean difference:\n"
        f"mean(program usage in HH) - mean(program usage in {config['difference_label']})\n\n"
        "Asterisks in the heatmap indicate FDR < 0.05.\n"
        f"Significance labels in boxplot panels are {config['stats_note']}",
        fontsize=10,
        color="#333333",
        va="bottom",
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    build_composite(
        run_dir=Path(args.run_dir),
        annotation_xlsx=Path(args.annotation_xlsx),
        sample_label=args.sample_label,
        output_path=Path(args.output_path),
        comparison_mode=str(args.comparison_mode),
        omit_family_labels=bool(args.omit_family_labels),
    )


if __name__ == "__main__":
    main()
