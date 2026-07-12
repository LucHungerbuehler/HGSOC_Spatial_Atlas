"""Compact comparison of C-SIDE Model 1 and Model 2 outputs.

Model 1: SNAI1-ac only, saved as cside_all_results.csv.
Model 2: SNAI1-ac + SpaCET malignant fraction, saved as cside_2cov_all_results.csv.

This script intentionally does not rebuild the full S2e evidence ladder. It
checks whether Model 1 is internally well formed, compares Model 1 and Model 2
where both exist, and asks whether the Model-2 robust core is supported by the
unadjusted Model-1 fit.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(r"D:\HGSOC_Spatial_Atlas")
RCTD_ROOT = ROOT / "scRNA_reference" / "rctd_outputs"
S2E_ROOT = ROOT / "05_analysis_ready" / "S2e_CSIDE_CellTypeSpecific_DE_Audit"
OUT_DIR = S2E_ROOT / "10_model1_vs_model2_compact_audit"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
SCRIPT_PATH = Path(__file__).resolve()

DATASETS = ("denisenko_2022", "ju_2024", "yamamoto_2025")
ROBUST_CORE_PATH = (
    S2E_ROOT
    / "07_report_ready_packaging"
    / "tables"
    / "cside_robust_core_73_gene_celltype_associations.csv"
)


def read_csv_auto(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return []
    first = text.splitlines()[0]
    delimiter = ";" if first.count(";") > first.count(",") else ","
    return list(csv.DictReader(text.splitlines(), delimiter=delimiter))


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: str | float | int | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def is_true(value: str | bool) -> bool:
    return str(value).strip().lower() == "true"


def signed_z(row: dict[str, str]) -> float:
    z = fnum(row.get("Z_score"))
    log_fc = fnum(row.get("log_fc"))
    if not math.isfinite(z) or not math.isfinite(log_fc):
        return float("nan")
    return z if log_fc >= 0 else -z


def sign_value(value: float) -> int:
    if not math.isfinite(value) or value == 0:
        return 0
    return 1 if value > 0 else -1


def safe_div(num: float, den: float) -> float:
    return num / den if den else float("nan")


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return float("nan")
    xvals = [x for x, _ in pairs]
    yvals = [y for _, y in pairs]
    mx = sum(xvals) / len(xvals)
    my = sum(yvals) / len(yvals)
    vx = sum((x - mx) ** 2 for x in xvals)
    vy = sum((y - my) ** 2 for y in yvals)
    if vx <= 0 or vy <= 0:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def stouffer(values: list[float]) -> float:
    finite = [x for x in values if math.isfinite(x)]
    if not finite:
        return float("nan")
    return sum(finite) / math.sqrt(len(finite))


def q05(row: dict[str, str]) -> bool:
    try:
        return fnum(row.get("p_val")) < 0.05
    except Exception:
        return False


def sample_dirs() -> list[tuple[str, str, Path]]:
    dirs: list[tuple[str, str, Path]] = []
    for dataset in DATASETS:
        dataset_dir = RCTD_ROOT / dataset
        if not dataset_dir.exists():
            continue
        for sample_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            dirs.append((dataset, sample_dir.name, sample_dir))
    return dirs


def load_model_results(prefix: str) -> tuple[dict[tuple[str, str, str, str], dict], dict]:
    """Return row map keyed by dataset/sample/cell_type/gene and validation stats."""
    row_map: dict[tuple[str, str, str, str], dict] = {}
    stats = {
        "files": 0,
        "rows": 0,
        "nonconverged_rows": 0,
        "max_p_delta": 0.0,
        "rows_p_delta_gt_1e_8": 0,
        "max_logfc_delta": 0.0,
        "rows_logfc_delta_gt_1e_8": 0,
    }
    filename = f"{prefix}_all_results.csv"
    for dataset, sample, sample_dir in sample_dirs():
        path = sample_dir / filename
        if not path.exists():
            continue
        stats["files"] += 1
        for row in read_csv_auto(path):
            row = dict(row)
            row["dataset"] = dataset
            row["sample"] = sample
            row["sample_label"] = f"{dataset}__{sample}"
            row["signed_z"] = signed_z(row)
            key = (dataset, sample, row["cell_type"], row["gene"])
            row_map[key] = row
            stats["rows"] += 1
            if not is_true(row.get("conv", "")):
                stats["nonconverged_rows"] += 1
            z = fnum(row.get("Z_score"))
            p_val = fnum(row.get("p_val"))
            if math.isfinite(z) and math.isfinite(p_val):
                p_from_z = math.erfc(abs(z) / math.sqrt(2.0))
                p_delta = abs(p_val - p_from_z)
                stats["max_p_delta"] = max(stats["max_p_delta"], p_delta)
                if p_delta > 1e-8:
                    stats["rows_p_delta_gt_1e_8"] += 1
            log_fc = fnum(row.get("log_fc"))
            mean_0 = fnum(row.get("mean_0"))
            mean_1 = fnum(row.get("mean_1"))
            if all(math.isfinite(x) for x in (log_fc, mean_0, mean_1)):
                logfc_delta = abs(log_fc - (mean_1 - mean_0))
                stats["max_logfc_delta"] = max(stats["max_logfc_delta"], logfc_delta)
                if logfc_delta > 1e-8:
                    stats["rows_logfc_delta_gt_1e_8"] += 1
    return row_map, stats


def load_significant(prefix: str) -> dict[tuple[str, str, str, str], dict]:
    sig_map: dict[tuple[str, str, str, str], dict] = {}
    filename = f"{prefix}_significant.csv"
    for dataset, sample, sample_dir in sample_dirs():
        path = sample_dir / filename
        if not path.exists():
            continue
        for row in read_csv_auto(path):
            row = dict(row)
            row["dataset"] = dataset
            row["sample"] = sample
            row["sample_label"] = f"{dataset}__{sample}"
            row["signed_z"] = signed_z(row)
            sig_map[(dataset, sample, row["cell_type"], row["gene"])] = row
    return sig_map


def summarize_overlap(model1: dict, model2: dict, sig1: dict, sig2: dict) -> tuple[list[dict], list[dict]]:
    by_sample: dict[tuple[str, str], dict[str, list | int]] = defaultdict(
        lambda: {"z1": [], "z2": [], "lf1": [], "lf2": [], "same": 0, "n": 0}
    )
    by_ct: dict[str, dict[str, list | int]] = defaultdict(
        lambda: {"z1": [], "z2": [], "lf1": [], "lf2": [], "same": 0, "n": 0}
    )
    common_keys = sorted(set(model1) & set(model2))
    for key in common_keys:
        dataset, sample, cell_type, _gene = key
        row1 = model1[key]
        row2 = model2[key]
        z1 = row1["signed_z"]
        z2 = row2["signed_z"]
        lf1 = fnum(row1.get("log_fc"))
        lf2 = fnum(row2.get("log_fc"))
        same = sign_value(z1) == sign_value(z2) and sign_value(z1) != 0
        for group_key, bucket in (((dataset, sample), by_sample[(dataset, sample)]), (cell_type, by_ct[cell_type])):
            bucket["z1"].append(z1)
            bucket["z2"].append(z2)
            bucket["lf1"].append(lf1)
            bucket["lf2"].append(lf2)
            bucket["same"] += int(same)
            bucket["n"] += 1

    sample_rows = []
    for (dataset, sample, sample_dir) in sample_dirs():
        m1_rows = sum(1 for key in model1 if key[0] == dataset and key[1] == sample)
        m2_rows = sum(1 for key in model2 if key[0] == dataset and key[1] == sample)
        s1_rows = sum(1 for key in sig1 if key[0] == dataset and key[1] == sample)
        s2_rows = sum(1 for key in sig2 if key[0] == dataset and key[1] == sample)
        bucket = by_sample.get((dataset, sample), {"z1": [], "z2": [], "lf1": [], "lf2": [], "same": 0, "n": 0})
        sample_rows.append(
            {
                "dataset": dataset,
                "sample": sample,
                "has_model1": bool(m1_rows),
                "has_model2": bool(m2_rows),
                "model1_rows": m1_rows,
                "model2_rows": m2_rows,
                "model1_significant_rows": s1_rows,
                "model2_significant_rows": s2_rows,
                "overlap_rows": bucket["n"],
                "signed_z_pearson_model1_vs_model2": pearson(bucket["z1"], bucket["z2"]),
                "logfc_pearson_model1_vs_model2": pearson(bucket["lf1"], bucket["lf2"]),
                "direction_agreement_fraction": safe_div(bucket["same"], bucket["n"]),
            }
        )

    ct_rows = []
    all_cell_types = sorted({key[2] for key in set(model1) | set(model2) | set(sig1) | set(sig2)})
    for cell_type in all_cell_types:
        bucket = by_ct.get(cell_type, {"z1": [], "z2": [], "lf1": [], "lf2": [], "same": 0, "n": 0})
        m1_sig = {key for key in sig1 if key[2] == cell_type}
        m2_sig = {key for key in sig2 if key[2] == cell_type}
        ct_rows.append(
            {
                "cell_type": cell_type,
                "model1_rows": sum(1 for key in model1 if key[2] == cell_type),
                "model2_rows": sum(1 for key in model2 if key[2] == cell_type),
                "overlap_rows": bucket["n"],
                "signed_z_pearson_model1_vs_model2": pearson(bucket["z1"], bucket["z2"]),
                "logfc_pearson_model1_vs_model2": pearson(bucket["lf1"], bucket["lf2"]),
                "direction_agreement_fraction": safe_div(bucket["same"], bucket["n"]),
                "model1_significant_rows": len(m1_sig),
                "model2_significant_rows": len(m2_sig),
                "shared_significant_rows": len(m1_sig & m2_sig),
                "model1_only_significant_rows": len(m1_sig - m2_sig),
                "model2_only_significant_rows": len(m2_sig - m1_sig),
            }
        )
    return sample_rows, ct_rows


def recurrent_sig_rows(keys: set[tuple[str, str, str, str]]) -> list[dict]:
    grouped: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for dataset, sample, cell_type, gene in keys:
        grouped[(gene, cell_type)].add((dataset, sample))
    rows = []
    for (gene, cell_type), samples in grouped.items():
        datasets = sorted({d for d, _ in samples})
        rows.append(
            {
                "gene": gene,
                "cell_type": cell_type,
                "n_significant_samples": len(samples),
                "n_datasets": len(datasets),
                "datasets": ";".join(datasets),
                "example_samples": ";".join(f"{d}/{s}" for d, s in sorted(samples)[:8]),
            }
        )
    return sorted(rows, key=lambda r: (-r["n_significant_samples"], -r["n_datasets"], r["cell_type"], r["gene"]))


def robust_core_support(model1: dict, model2: dict) -> list[dict]:
    robust = read_csv_auto(ROBUST_CORE_PATH)
    rows = []
    for row in robust:
        gene = row["gene"]
        cell_type = row["cell_type"]
        model2_direction = 1 if row["direction"] == "positive" else -1
        common = []
        model1_values = []
        model2_values = []
        model1_lf = []
        for key, row2 in model2.items():
            dataset, sample, ct, g = key
            if ct != cell_type or g != gene:
                continue
            row1 = model1.get(key)
            if row1 is None:
                continue
            z1 = row1["signed_z"]
            z2 = row2["signed_z"]
            if sign_value(z1) == 0 or sign_value(z2) == 0:
                continue
            common.append(key)
            model1_values.append(z1)
            model2_values.append(z2)
            model1_lf.append(fnum(row1.get("log_fc")))
        m1_same_as_model2_sample = sum(sign_value(a) == sign_value(b) for a, b in zip(model1_values, model2_values))
        m1_same_as_robust = sum(sign_value(a) == model2_direction for a in model1_values)
        m1_positive = sum(sign_value(a) > 0 for a in model1_values)
        m1_negative = sum(sign_value(a) < 0 for a in model1_values)
        rows.append(
            {
                "cell_type": cell_type,
                "gene": gene,
                "model2_robust_direction": row["direction"],
                "model2_robust_n_samples": row["n_samples"],
                "model2_robust_sign_consistency": row["sign_consistency"],
                "model2_random_q": row["random_q"],
                "model2_is_snai1ac_signature_gene": row["is_snai1ac_signature_gene"],
                "model1_common_samples": len(common),
                "model1_same_sample_direction_as_model2_n": m1_same_as_model2_sample,
                "model1_same_sample_direction_as_model2_fraction": safe_div(m1_same_as_model2_sample, len(common)),
                "model1_supports_model2_robust_direction_n": m1_same_as_robust,
                "model1_supports_model2_robust_direction_fraction": safe_div(m1_same_as_robust, len(common)),
                "model1_positive_samples": m1_positive,
                "model1_negative_samples": m1_negative,
                "model1_mean_signed_z": sum(model1_values) / len(model1_values) if model1_values else float("nan"),
                "model1_stouffer_z": stouffer(model1_values),
                "model1_mean_logfc": sum(model1_lf) / len(model1_lf) if model1_lf else float("nan"),
            }
        )
    return rows


def make_figure(ct_rows: list[dict], robust_rows: list[dict], path_png: Path, path_svg: Path) -> str:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except Exception as exc:
        return f"plot_skipped: {exc!r}"

    tokens = {
        "surface": "#FCFCFD",
        "panel": "#FFFFFF",
        "ink": "#1F2430",
        "muted": "#6F768A",
        "grid": "#E6E8F0",
        "axis": "#D7DBE7",
        "blue": "#A3BEFA",
        "blue_dark": "#2E4780",
        "orange": "#F0986E",
        "orange_dark": "#804126",
        "gold": "#FFE15B",
        "neutral": "#C5CAD3",
    }
    order = ["Epithelial", "Fibroblast", "Macrophage", "CAF", "Endothelial", "B_cell", "T_cell_prolif"]
    ct_lookup = {row["cell_type"]: row for row in ct_rows}
    plot_ct = [ct for ct in order if ct in ct_lookup]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor=tokens["surface"])
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.09, top=0.86, hspace=0.42, wspace=0.28)
    fig.text(0.10, 0.95, "C-SIDE Model 1 vs Model 2 compact audit", fontsize=16, weight="bold", color=tokens["ink"])
    fig.text(
        0.10,
        0.915,
        "Model 1 = SNAI1-ac only; Model 2 = SNAI1-ac + SpaCET malignant fraction. Values use rows present in both models.",
        fontsize=10,
        color=tokens["muted"],
    )

    def style_axis(ax):
        ax.set_facecolor(tokens["panel"])
        ax.grid(axis="x", color=tokens["grid"], linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(tokens["axis"])
        ax.spines["bottom"].set_color(tokens["axis"])
        ax.tick_params(colors=tokens["muted"], labelsize=9)

    # A. Signed-Z agreement
    ax = axes[0, 0]
    vals = [ct_lookup[ct]["signed_z_pearson_model1_vs_model2"] for ct in plot_ct]
    ax.barh(plot_ct, vals, color=tokens["blue"], edgecolor=tokens["blue_dark"], linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_title("A. Signed-Z agreement", loc="left", fontsize=11, weight="bold", color=tokens["ink"])
    ax.set_xlabel("Pearson r", color=tokens["muted"])
    for y, value in enumerate(vals):
        ax.text(min(value + 0.02, 0.98), y, f"{value:.2f}", va="center", fontsize=9, color=tokens["ink"])
    style_axis(ax)

    # B. Direction agreement
    ax = axes[0, 1]
    vals = [ct_lookup[ct]["direction_agreement_fraction"] for ct in plot_ct]
    ax.barh(plot_ct, vals, color=tokens["gold"], edgecolor="#736422", linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_title("B. Direction agreement", loc="left", fontsize=11, weight="bold", color=tokens["ink"])
    ax.set_xlabel("Fraction same sign", color=tokens["muted"])
    for y, value in enumerate(vals):
        ax.text(min(value + 0.02, 0.98), y, f"{value:.2f}", va="center", fontsize=9, color=tokens["ink"])
    style_axis(ax)

    # C. Significant-row overlap
    ax = axes[1, 0]
    shared = [ct_lookup[ct]["shared_significant_rows"] for ct in plot_ct]
    m1_only = [ct_lookup[ct]["model1_only_significant_rows"] for ct in plot_ct]
    m2_only = [ct_lookup[ct]["model2_only_significant_rows"] for ct in plot_ct]
    y_positions = list(range(len(plot_ct)))
    ax.barh(y_positions, shared, color=tokens["blue"], edgecolor=tokens["blue_dark"], label="Shared", linewidth=0.7)
    left = shared[:]
    ax.barh(y_positions, m1_only, left=left, color=tokens["neutral"], edgecolor="#7A828F", label="Model 1 only", linewidth=0.7)
    left = [a + b for a, b in zip(left, m1_only)]
    ax.barh(y_positions, m2_only, left=left, color=tokens["orange"], edgecolor=tokens["orange_dark"], label="Model 2 only", linewidth=0.7)
    ax.set_yticks(y_positions, plot_ct)
    ax.invert_yaxis()
    ax.set_title("C. Nominal significant rows", loc="left", fontsize=11, weight="bold", color=tokens["ink"])
    ax.set_xlabel("Rows with p < 0.05 in C-SIDE significant files", color=tokens["muted"])
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style_axis(ax)

    # D. Model-2 robust core support in Model 1
    ax = axes[1, 1]
    bins = {
        ">=0.8 same\ndirection": 0,
        "0.6-0.8 same\ndirection": 0,
        "<0.6 same\ndirection": 0,
        "no M1\ncommon rows": 0,
    }
    for row in robust_rows:
        n = int(row["model1_common_samples"])
        value = row["model1_supports_model2_robust_direction_fraction"]
        if n == 0 or not math.isfinite(value):
            bins["no M1\ncommon rows"] += 1
        elif value >= 0.8:
            bins[">=0.8 same\ndirection"] += 1
        elif value >= 0.6:
            bins["0.6-0.8 same\ndirection"] += 1
        else:
            bins["<0.6 same\ndirection"] += 1
    labels = list(bins)
    values = [bins[k] for k in labels]
    colors = [tokens["blue"], "#CEDFFE", tokens["orange"], tokens["neutral"]]
    ax.bar(labels, values, color=colors, edgecolor=tokens["blue_dark"], linewidth=0.7)
    ax.set_title("D. Model-2 robust core in Model 1", loc="left", fontsize=11, weight="bold", color=tokens["ink"])
    ax.set_ylabel("Gene-cell-type associations", color=tokens["muted"])
    for x, value in enumerate(values):
        ax.text(x, value + 0.5, str(value), ha="center", fontsize=9, color=tokens["ink"])
    style_axis(ax)
    ax.grid(axis="y", color=tokens["grid"], linewidth=0.8)
    ax.grid(axis="x", visible=False)

    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    fig.savefig(path_svg, bbox_inches="tight")
    plt.close(fig)
    return "ok"


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    model1, model1_stats = load_model_results("cside")
    model2, model2_stats = load_model_results("cside_2cov")
    sig1 = load_significant("cside")
    sig2 = load_significant("cside_2cov")

    sample_rows, ct_rows = summarize_overlap(model1, model2, sig1, sig2)
    write_csv(TABLE_DIR / "model1_model2_sample_inventory.csv", sample_rows)
    write_csv(TABLE_DIR / "model1_model2_celltype_comparison.csv", ct_rows)

    sig1_keys = set(sig1)
    sig2_keys = set(sig2)
    write_csv(
        TABLE_DIR / "model1_only_recurrent_significant_gene_celltypes_top50.csv",
        recurrent_sig_rows(sig1_keys - sig2_keys)[:50],
    )
    write_csv(
        TABLE_DIR / "model2_only_recurrent_significant_gene_celltypes_top50.csv",
        recurrent_sig_rows(sig2_keys - sig1_keys)[:50],
    )
    write_csv(
        TABLE_DIR / "shared_recurrent_significant_gene_celltypes_top50.csv",
        recurrent_sig_rows(sig1_keys & sig2_keys)[:50],
    )

    robust_rows = robust_core_support(model1, model2)
    write_csv(TABLE_DIR / "model2_robust_core_model1_support.csv", robust_rows)

    common_keys = set(model1) & set(model2)
    all_z1 = [model1[key]["signed_z"] for key in common_keys]
    all_z2 = [model2[key]["signed_z"] for key in common_keys]
    all_lf1 = [fnum(model1[key].get("log_fc")) for key in common_keys]
    all_lf2 = [fnum(model2[key].get("log_fc")) for key in common_keys]
    all_same = sum(sign_value(model1[key]["signed_z"]) == sign_value(model2[key]["signed_z"]) for key in common_keys)
    robust_support_values = [
        row["model1_supports_model2_robust_direction_fraction"]
        for row in robust_rows
        if math.isfinite(row["model1_supports_model2_robust_direction_fraction"])
    ]
    robust_ge_08 = sum(v >= 0.8 for v in robust_support_values)
    robust_ge_06 = sum(v >= 0.6 for v in robust_support_values)
    missing_m1 = [
        {"dataset": row["dataset"], "sample": row["sample"]}
        for row in sample_rows
        if not row["has_model1"]
    ]

    figure_status = make_figure(
        ct_rows,
        robust_rows,
        FIG_DIR / "model1_vs_model2_compact_audit.png",
        FIG_DIR / "model1_vs_model2_compact_audit.svg",
    )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(SCRIPT_PATH),
        "input_root": str(RCTD_ROOT),
        "output_dir": str(OUT_DIR),
        "model1": {
            "definition": "SNAI1-ac only; cside_all_results.csv",
            **model1_stats,
            "significant_rows": len(sig1),
            "missing_samples": missing_m1,
        },
        "model2": {
            "definition": "SNAI1-ac + SpaCET malignant fraction; cside_2cov_all_results.csv",
            **model2_stats,
            "significant_rows": len(sig2),
        },
        "overlap": {
            "common_gene_celltype_sample_rows": len(common_keys),
            "signed_z_pearson": pearson(all_z1, all_z2),
            "logfc_pearson": pearson(all_lf1, all_lf2),
            "direction_agreement_fraction": safe_div(all_same, len(common_keys)),
            "shared_significant_rows": len(sig1_keys & sig2_keys),
            "model1_only_significant_rows": len(sig1_keys - sig2_keys),
            "model2_only_significant_rows": len(sig2_keys - sig1_keys),
        },
        "model2_robust_core_support_in_model1": {
            "n_robust_core_gene_celltype_associations": len(robust_rows),
            "n_with_model1_common_samples": sum(int(row["model1_common_samples"]) > 0 for row in robust_rows),
            "n_with_model1_support_fraction_ge_0_8": robust_ge_08,
            "n_with_model1_support_fraction_ge_0_6": robust_ge_06,
            "median_model1_support_fraction": sorted(robust_support_values)[len(robust_support_values) // 2]
            if robust_support_values
            else float("nan"),
        },
        "figure_status": figure_status,
        "outputs": {
            "sample_inventory": str(TABLE_DIR / "model1_model2_sample_inventory.csv"),
            "celltype_comparison": str(TABLE_DIR / "model1_model2_celltype_comparison.csv"),
            "model2_robust_core_model1_support": str(TABLE_DIR / "model2_robust_core_model1_support.csv"),
            "model1_only_top50": str(TABLE_DIR / "model1_only_recurrent_significant_gene_celltypes_top50.csv"),
            "model2_only_top50": str(TABLE_DIR / "model2_only_recurrent_significant_gene_celltypes_top50.csv"),
            "shared_top50": str(TABLE_DIR / "shared_recurrent_significant_gene_celltypes_top50.csv"),
            "figure_png": str(FIG_DIR / "model1_vs_model2_compact_audit.png"),
            "figure_svg": str(FIG_DIR / "model1_vs_model2_compact_audit.svg"),
        },
    }

    (OUT_DIR / "run_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = [
        "# C-SIDE Model 1 vs Model 2 Compact Audit",
        "",
        "Purpose: compare the saved SNAI1-ac-only C-SIDE model against the SNAI1-ac + SpaCET malignant-fraction model without rebuilding the full S2e evidence ladder.",
        "",
        "## Main Findings",
        "",
        f"- Model 1 files found: {model1_stats['files']} samples; missing Model 1 samples: {missing_m1}.",
        f"- Model 1 internal canaries: {model1_stats['nonconverged_rows']} non-converged rows; max p-value delta from Z = {model1_stats['max_p_delta']:.3g}; max logFC delta from mean_1 - mean_0 = {model1_stats['max_logfc_delta']:.3g}.",
        f"- Common Model 1/Model 2 rows: {len(common_keys):,}; signed-Z Pearson r = {summary['overlap']['signed_z_pearson']:.3f}; direction agreement = {summary['overlap']['direction_agreement_fraction']:.3f}.",
        f"- Nominal significant rows: Model 1 = {len(sig1):,}; Model 2 = {len(sig2):,}; shared = {len(sig1_keys & sig2_keys):,}; Model 1 only = {len(sig1_keys - sig2_keys):,}; Model 2 only = {len(sig2_keys - sig1_keys):,}.",
        f"- Model-2 robust core support in Model 1: {robust_ge_08}/{len(robust_rows)} associations have >=0.8 same-direction support in Model 1 among common samples; {robust_ge_06}/{len(robust_rows)} have >=0.6 support.",
        "",
        "## Interpretation",
        "",
        "Model 1 is internally well formed as a saved unadjusted C-SIDE fit, but it is incomplete by one sample and was not the model carried through the report-facing S2e ladder.",
        "Model 2 is more conservative: many Model-1 nominal hits do not survive malignant-fraction adjustment, while most Model-2 robust-core associations are already directionally visible in Model 1.",
        "",
        "## Outputs",
        "",
    ]
    for name, path in summary["outputs"].items():
        md.append(f"- {name}: `{path}`")
    (OUT_DIR / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
