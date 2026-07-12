"""Deep audit of saved C-SIDE Model 1 and Model 2 outputs.

Model 1: expression ~ SNAI1-ac, saved as cside_all_results.csv.
Model 2: expression ~ SNAI1-ac + SpaCET malignant fraction, saved as
cside_2cov_all_results.csv.

This script does not rerun C-SIDE. It mines saved result rows to answer:

1. Which gene-cell-type signals are retained, attenuated, or emergent after
   malignant-fraction adjustment?
2. Which broad gene categories dominate those signal classes?
3. Does the external SpaCET malignant covariate broadly align with the RCTD
   epithelial weights used by C-SIDE?
4. How much Model-1 support is present for the 73 report-facing Model-2 robust
   core associations?
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Iterable


ROOT = Path(r"D:\HGSOC_Spatial_Atlas")
RCTD_ROOT = ROOT / "scRNA_reference" / "rctd_outputs"
INPUT_ROOT = ROOT / "scRNA_reference" / "rctd_inputs"
S2E_ROOT = ROOT / "05_analysis_ready" / "S2e_CSIDE_CellTypeSpecific_DE_Audit"
OUT_DIR = S2E_ROOT / "11_model1_vs_model2_deep_gene_audit"
TABLE_DIR = OUT_DIR / "tables"
SCRIPT_PATH = Path(__file__).resolve()

DATASETS = ("denisenko_2022", "ju_2024", "yamamoto_2025")
MAIN_CELL_TYPES = ("Epithelial", "Fibroblast", "Macrophage", "CAF", "Endothelial")
ROBUST_CORE_PATH = (
    S2E_ROOT
    / "07_report_ready_packaging"
    / "tables"
    / "cside_robust_core_73_gene_celltype_associations.csv"
)
SIGNATURE_PATH = ROOT / "05_analysis_ready" / "Signature" / "snai1_acetylation_signature_full.csv"
MODEL_SETUP_PATH = (
    S2E_ROOT
    / "09_section25_report_assets"
    / "tables"
    / "section25_cside_model_setup_by_sample.csv"
)


EPITHELIAL_HGSOC_MARKERS = {
    "MUC16",
    "MUC1",
    "MSLN",
    "WFDC2",
    "FOLR1",
    "SLPI",
    "CLDN3",
    "CLDN4",
    "EPCAM",
    "TACSTD2",
    "KRT7",
    "KRT8",
    "KRT18",
    "KRT19",
    "BCAM",
    "CD24",
    "PAX8",
    "SOX17",
    "FXYD3",
    "S100A6",
    "S100A11",
    "LCN2",
    "CRABP2",
}

ECM_STROMAL_GENES = {
    "AEBP1",
    "AHNAK",
    "ACTA2",
    "CALD1",
    "COL1A1",
    "COL1A2",
    "COL3A1",
    "COL5A1",
    "COL6A1",
    "COL6A2",
    "COL6A3",
    "DCN",
    "FBN1",
    "FN1",
    "IGFBP5",
    "LUM",
    "MGP",
    "POSTN",
    "SPARC",
    "TAGLN",
    "VIM",
    "LGALS1",
}

IMMUNE_IFN_ANTIGEN_GENES = {
    "B2M",
    "C3",
    "IFI6",
    "IFI27",
    "IFI44",
    "IFI44L",
    "IFIT1",
    "IFIT2",
    "IFIT3",
    "IFITM1",
    "IFITM2",
    "IFITM3",
    "ISG15",
    "MX1",
    "OAS1",
    "OAS2",
    "OAS3",
    "PSMB8",
    "STAT1",
    "TAP1",
}

STRESS_HEATSHOCK_GENES = {
    "ATF3",
    "ATF4",
    "DDIT3",
    "DNAJA1",
    "DNAJB1",
    "FOS",
    "HSPA1A",
    "HSPA1B",
    "HSPA5",
    "HSPA8",
    "HSP90AA1",
    "HSP90AB1",
    "JUN",
}

CELL_CYCLE_GENES = {
    "BIRC5",
    "CCNB1",
    "CCND1",
    "CDK1",
    "HMGB2",
    "MKI67",
    "NUSAP1",
    "PCNA",
    "PTTG1",
    "TOP2A",
    "UBE2C",
}


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


def fnum(value: object) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def finite(values: Iterable[float]) -> list[float]:
    return [x for x in values if math.isfinite(x)]


def signed_z(row: dict[str, object]) -> float:
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


def pearson(xs: Iterable[float], ys: Iterable[float]) -> float:
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


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            out[order[k]] = rank
        i = j
    return out


def spearman(xs: Iterable[float], ys: Iterable[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return float("nan")
    return pearson(ranks([x for x, _ in pairs]), ranks([y for _, y in pairs]))


def stouffer(values: Iterable[float]) -> float:
    vals = finite(values)
    if not vals:
        return float("nan")
    return sum(vals) / math.sqrt(len(vals))


def is_sig(row: dict[str, object]) -> bool:
    return fnum(row.get("p_val")) < 0.05


def fmt_float(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def load_signature_genes() -> set[str]:
    if not SIGNATURE_PATH.exists():
        return set()
    rows = read_csv_auto(SIGNATURE_PATH)
    return {str(row.get("Gene", "")).strip() for row in rows if str(row.get("Gene", "")).strip()}


def gene_tags(gene: str, signature_genes: set[str]) -> list[str]:
    gene = gene.strip()
    tags: list[str] = []
    if gene in signature_genes:
        tags.append("SNAI1ac_signature_gene")
    if gene.startswith("MT-"):
        tags.append("mitochondrial")
    if gene.startswith(("RPL", "RPS", "MRPL", "MRPS")) or gene in {"UBA52", "FAU"}:
        tags.append("ribosomal_translation")
    if gene.startswith("HLA-") or gene in IMMUNE_IFN_ANTIGEN_GENES:
        tags.append("interferon_antigen_presentation")
    if gene.startswith("COL") or gene in ECM_STROMAL_GENES:
        tags.append("ECM_stromal")
    if gene in EPITHELIAL_HGSOC_MARKERS:
        tags.append("epithelial_HGSOC_marker_like")
    if gene in STRESS_HEATSHOCK_GENES or gene.startswith("HSP"):
        tags.append("stress_heatshock")
    if gene in CELL_CYCLE_GENES:
        tags.append("cell_cycle")
    return tags or ["other"]


def primary_tag(tags: list[str]) -> str:
    priority = [
        "SNAI1ac_signature_gene",
        "mitochondrial",
        "ribosomal_translation",
        "ECM_stromal",
        "epithelial_HGSOC_marker_like",
        "interferon_antigen_presentation",
        "stress_heatshock",
        "cell_cycle",
        "other",
    ]
    for tag in priority:
        if tag in tags:
            return tag
    return "other"


def sample_dirs() -> list[tuple[str, str, Path]]:
    dirs: list[tuple[str, str, Path]] = []
    for dataset in DATASETS:
        dataset_dir = RCTD_ROOT / dataset
        if not dataset_dir.exists():
            continue
        for sample_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            dirs.append((dataset, sample_dir.name, sample_dir))
    return dirs


def load_model_results(filename: str) -> tuple[dict[tuple[str, str, str, str], dict], dict]:
    row_map: dict[tuple[str, str, str, str], dict] = {}
    stats = {
        "files": 0,
        "rows": 0,
        "nonconverged_rows": 0,
        "max_p_delta": 0.0,
        "max_logfc_delta": 0.0,
    }
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
            row["log_fc_num"] = fnum(row.get("log_fc"))
            row["p_val_num"] = fnum(row.get("p_val"))
            key = (dataset, sample, row["cell_type"], row["gene"])
            row_map[key] = row
            stats["rows"] += 1
            if str(row.get("conv", "")).strip().lower() != "true":
                stats["nonconverged_rows"] += 1
            z = fnum(row.get("Z_score"))
            p_val = fnum(row.get("p_val"))
            if math.isfinite(z) and math.isfinite(p_val):
                stats["max_p_delta"] = max(stats["max_p_delta"], abs(p_val - math.erfc(abs(z) / math.sqrt(2.0))))
            log_fc = fnum(row.get("log_fc"))
            mean_0 = fnum(row.get("mean_0"))
            mean_1 = fnum(row.get("mean_1"))
            if all(math.isfinite(x) for x in (log_fc, mean_0, mean_1)):
                stats["max_logfc_delta"] = max(stats["max_logfc_delta"], abs(log_fc - (mean_1 - mean_0)))
    return row_map, stats


def load_significant(filename: str) -> dict[tuple[str, str, str, str], dict]:
    sig_map: dict[tuple[str, str, str, str], dict] = {}
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
            row["log_fc_num"] = fnum(row.get("log_fc"))
            row["p_val_num"] = fnum(row.get("p_val"))
            sig_map[(dataset, sample, row["cell_type"], row["gene"])] = row
    return sig_map


def summarize_signal_classes(
    sig1: dict[tuple[str, str, str, str], dict],
    sig2: dict[tuple[str, str, str, str], dict],
    signature_genes: set[str],
) -> tuple[list[dict], list[dict]]:
    buckets: dict[tuple[str, str], dict] = defaultdict(
        lambda: {
            "event_count": 0,
            "gene_celltypes": set(),
            "genes": set(),
            "samples": set(),
            "datasets": set(),
            "model1_signed_z": [],
            "model2_signed_z": [],
            "model1_log_fc": [],
            "model2_log_fc": [],
            "tags": Counter(),
        }
    )
    tag_buckets: dict[tuple[str, str, str], dict] = defaultdict(
        lambda: {
            "event_count": 0,
            "gene_celltypes": set(),
            "genes": set(),
            "samples": set(),
            "datasets": set(),
        }
    )

    for key in sorted(set(sig1) | set(sig2)):
        dataset, sample, cell_type, gene = key
        if key in sig1 and key in sig2:
            signal_class = "shared_significant"
        elif key in sig1:
            signal_class = "model1_only_significant"
        else:
            signal_class = "model2_only_significant"

        b = buckets[(signal_class, cell_type)]
        b["event_count"] += 1
        b["gene_celltypes"].add((cell_type, gene))
        b["genes"].add(gene)
        b["samples"].add(f"{dataset}__{sample}")
        b["datasets"].add(dataset)
        if key in sig1:
            b["model1_signed_z"].append(sig1[key]["signed_z"])
            b["model1_log_fc"].append(sig1[key]["log_fc_num"])
        if key in sig2:
            b["model2_signed_z"].append(sig2[key]["signed_z"])
            b["model2_log_fc"].append(sig2[key]["log_fc_num"])

        tags = gene_tags(gene, signature_genes)
        for tag in tags:
            b["tags"][tag] += 1
            tb = tag_buckets[(signal_class, cell_type, tag)]
            tb["event_count"] += 1
            tb["gene_celltypes"].add((cell_type, gene))
            tb["genes"].add(gene)
            tb["samples"].add(f"{dataset}__{sample}")
            tb["datasets"].add(dataset)

    class_rows = []
    for (signal_class, cell_type), b in sorted(buckets.items()):
        top_tags = ";".join(f"{tag}:{count}" for tag, count in b["tags"].most_common(6))
        class_rows.append(
            {
                "signal_class": signal_class,
                "cell_type": cell_type,
                "sample_level_significant_events": b["event_count"],
                "unique_gene_celltypes": len(b["gene_celltypes"]),
                "unique_genes": len(b["genes"]),
                "samples_touched": len(b["samples"]),
                "datasets_touched": len(b["datasets"]),
                "model1_mean_signed_z_among_events": mean_or_nan(b["model1_signed_z"]),
                "model2_mean_signed_z_among_events": mean_or_nan(b["model2_signed_z"]),
                "model1_mean_log_fc_among_events": mean_or_nan(b["model1_log_fc"]),
                "model2_mean_log_fc_among_events": mean_or_nan(b["model2_log_fc"]),
                "top_gene_tags_by_event_count": top_tags,
            }
        )

    tag_rows = []
    for (signal_class, cell_type, tag), b in sorted(tag_buckets.items()):
        tag_rows.append(
            {
                "signal_class": signal_class,
                "cell_type": cell_type,
                "gene_tag": tag,
                "sample_level_significant_events": b["event_count"],
                "unique_gene_celltypes": len(b["gene_celltypes"]),
                "unique_genes": len(b["genes"]),
                "samples_touched": len(b["samples"]),
                "datasets_touched": len(b["datasets"]),
            }
        )

    class_rows.sort(key=lambda r: (r["cell_type"], r["signal_class"]))
    tag_rows.sort(key=lambda r: (r["signal_class"], r["cell_type"], -r["sample_level_significant_events"], r["gene_tag"]))
    return class_rows, tag_rows


def mean_or_nan(values: Iterable[float]) -> float:
    vals = finite(values)
    return sum(vals) / len(vals) if vals else float("nan")


def aggregate_gene_celltypes(
    model1: dict[tuple[str, str, str, str], dict],
    model2: dict[tuple[str, str, str, str], dict],
    sig1: dict[tuple[str, str, str, str], dict],
    sig2: dict[tuple[str, str, str, str], dict],
    signature_genes: set[str],
) -> list[dict]:
    groups: dict[tuple[str, str], dict] = defaultdict(
        lambda: {
            "datasets": set(),
            "samples": set(),
            "z1": [],
            "z2": [],
            "lf1": [],
            "lf2": [],
            "same_direction": 0,
            "n_common": 0,
            "model1_sig_samples": set(),
            "model2_sig_samples": set(),
            "shared_sig_samples": set(),
            "model1_only_sig_samples": set(),
            "model2_only_sig_samples": set(),
            "model1_sig_datasets": set(),
            "model2_sig_datasets": set(),
            "shared_sig_datasets": set(),
            "model1_only_sig_datasets": set(),
            "model2_only_sig_datasets": set(),
        }
    )

    for key in sorted(set(model1) & set(model2)):
        dataset, sample, cell_type, gene = key
        g = groups[(cell_type, gene)]
        label = f"{dataset}__{sample}"
        row1 = model1[key]
        row2 = model2[key]
        z1 = row1["signed_z"]
        z2 = row2["signed_z"]
        g["datasets"].add(dataset)
        g["samples"].add(label)
        g["z1"].append(z1)
        g["z2"].append(z2)
        g["lf1"].append(row1["log_fc_num"])
        g["lf2"].append(row2["log_fc_num"])
        g["same_direction"] += int(sign_value(z1) == sign_value(z2) and sign_value(z1) != 0)
        g["n_common"] += 1

        s1 = key in sig1
        s2 = key in sig2
        if s1:
            g["model1_sig_samples"].add(label)
            g["model1_sig_datasets"].add(dataset)
        if s2:
            g["model2_sig_samples"].add(label)
            g["model2_sig_datasets"].add(dataset)
        if s1 and s2:
            g["shared_sig_samples"].add(label)
            g["shared_sig_datasets"].add(dataset)
        elif s1:
            g["model1_only_sig_samples"].add(label)
            g["model1_only_sig_datasets"].add(dataset)
        elif s2:
            g["model2_only_sig_samples"].add(label)
            g["model2_only_sig_datasets"].add(dataset)

    rows = []
    for (cell_type, gene), g in groups.items():
        tags = gene_tags(gene, signature_genes)
        model1_stouf = stouffer(g["z1"])
        model2_stouf = stouffer(g["z2"])
        m1_sig_n = len(g["model1_sig_samples"])
        m2_sig_n = len(g["model2_sig_samples"])
        shared_n = len(g["shared_sig_samples"])
        m1_only_n = len(g["model1_only_sig_samples"])
        m2_only_n = len(g["model2_only_sig_samples"])
        if shared_n >= 3 and len(g["shared_sig_datasets"]) >= 2:
            recurrent_class = "retained_shared_recurrent"
        elif m1_only_n >= 3 and len(g["model1_only_sig_datasets"]) >= 2:
            recurrent_class = "model1_attenuated_recurrent"
        elif m2_only_n >= 3 and len(g["model2_only_sig_datasets"]) >= 2:
            recurrent_class = "model2_emergent_recurrent"
        elif m1_sig_n or m2_sig_n:
            recurrent_class = "nonrecurrent_or_mixed_sample_signal"
        else:
            recurrent_class = "not_sample_significant"
        rows.append(
            {
                "cell_type": cell_type,
                "gene": gene,
                "tags": ";".join(tags),
                "primary_tag": primary_tag(tags),
                "is_snai1ac_signature_gene": gene in signature_genes,
                "n_common_samples": g["n_common"],
                "n_common_datasets": len(g["datasets"]),
                "common_datasets": ";".join(sorted(g["datasets"])),
                "common_samples": ";".join(sorted(g["samples"])),
                "model1_mean_signed_z": mean_or_nan(g["z1"]),
                "model2_mean_signed_z": mean_or_nan(g["z2"]),
                "model1_stouffer_z": model1_stouf,
                "model2_stouffer_z": model2_stouf,
                "delta_stouffer_model2_minus_model1": model2_stouf - model1_stouf,
                "delta_abs_stouffer_model2_minus_model1": abs(model2_stouf) - abs(model1_stouf),
                "model1_mean_log_fc": mean_or_nan(g["lf1"]),
                "model2_mean_log_fc": mean_or_nan(g["lf2"]),
                "delta_mean_log_fc_model2_minus_model1": mean_or_nan(g["lf2"]) - mean_or_nan(g["lf1"]),
                "same_direction_fraction_model1_vs_model2": safe_div(g["same_direction"], g["n_common"]),
                "model1_sig_samples": m1_sig_n,
                "model2_sig_samples": m2_sig_n,
                "shared_sig_samples": shared_n,
                "model1_only_sig_samples": m1_only_n,
                "model2_only_sig_samples": m2_only_n,
                "model1_sig_datasets": len(g["model1_sig_datasets"]),
                "model2_sig_datasets": len(g["model2_sig_datasets"]),
                "shared_sig_datasets": len(g["shared_sig_datasets"]),
                "model1_only_sig_datasets": len(g["model1_only_sig_datasets"]),
                "model2_only_sig_datasets": len(g["model2_only_sig_datasets"]),
                "recurrent_signal_class": recurrent_class,
            }
        )

    rows.sort(
        key=lambda r: (
            r["recurrent_signal_class"],
            r["cell_type"],
            -max(r["model1_sig_samples"], r["model2_sig_samples"]),
            r["gene"],
        )
    )
    return rows


def top_gene_celltype_tables(meta_rows: list[dict]) -> dict[str, list[dict]]:
    attenuated = [
        row
        for row in meta_rows
        if row["model1_only_sig_samples"] > 0
        and row["model1_sig_samples"] >= 3
        and row["model1_sig_samples"] >= row["model2_sig_samples"]
    ]
    emergent = [
        row
        for row in meta_rows
        if row["model2_only_sig_samples"] > 0
        and row["model2_sig_samples"] >= 3
        and row["model2_sig_samples"] >= row["model1_sig_samples"]
    ]
    retained = [
        row
        for row in meta_rows
        if row["shared_sig_samples"] > 0
        and row["model1_sig_samples"] >= 3
        and row["model2_sig_samples"] >= 3
    ]
    attenuated.sort(
        key=lambda r: (
            -r["model1_only_sig_samples"],
            -r["model1_only_sig_datasets"],
            r["delta_abs_stouffer_model2_minus_model1"],
            r["cell_type"],
            r["gene"],
        )
    )
    emergent.sort(
        key=lambda r: (
            -r["model2_only_sig_samples"],
            -r["model2_only_sig_datasets"],
            -r["delta_abs_stouffer_model2_minus_model1"],
            r["cell_type"],
            r["gene"],
        )
    )
    retained.sort(
        key=lambda r: (
            -r["shared_sig_samples"],
            -r["shared_sig_datasets"],
            -abs(r["model2_stouffer_z"]),
            r["cell_type"],
            r["gene"],
        )
    )
    return {
        "top_model1_attenuated_after_malignant_adjustment.csv": attenuated[:250],
        "top_model2_emergent_after_malignant_adjustment.csv": emergent[:250],
        "top_retained_shared_gene_celltypes.csv": retained[:250],
    }


def summarize_recurrent_classes(meta_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    by_ct: dict[tuple[str, str], dict] = defaultdict(
        lambda: {
            "gene_celltypes": 0,
            "genes": set(),
            "m1_sig_samples": 0,
            "m2_sig_samples": 0,
            "shared_sig_samples": 0,
            "m1_only_sig_samples": 0,
            "m2_only_sig_samples": 0,
            "tags": Counter(),
        }
    )
    by_tag: dict[tuple[str, str], dict] = defaultdict(
        lambda: {
            "gene_celltypes": 0,
            "genes": set(),
            "cell_types": set(),
            "m1_sig_samples": 0,
            "m2_sig_samples": 0,
            "shared_sig_samples": 0,
            "m1_only_sig_samples": 0,
            "m2_only_sig_samples": 0,
        }
    )
    for row in meta_rows:
        cls = row["recurrent_signal_class"]
        if cls in {"not_sample_significant", "nonrecurrent_or_mixed_sample_signal"}:
            continue
        key = (cls, row["cell_type"])
        b = by_ct[key]
        b["gene_celltypes"] += 1
        b["genes"].add(row["gene"])
        for tag in str(row["tags"]).split(";"):
            b["tags"][tag] += 1
            tb = by_tag[(cls, tag)]
            tb["gene_celltypes"] += 1
            tb["genes"].add(row["gene"])
            tb["cell_types"].add(row["cell_type"])
            tb["m1_sig_samples"] += row["model1_sig_samples"]
            tb["m2_sig_samples"] += row["model2_sig_samples"]
            tb["shared_sig_samples"] += row["shared_sig_samples"]
            tb["m1_only_sig_samples"] += row["model1_only_sig_samples"]
            tb["m2_only_sig_samples"] += row["model2_only_sig_samples"]
        b["m1_sig_samples"] += row["model1_sig_samples"]
        b["m2_sig_samples"] += row["model2_sig_samples"]
        b["shared_sig_samples"] += row["shared_sig_samples"]
        b["m1_only_sig_samples"] += row["model1_only_sig_samples"]
        b["m2_only_sig_samples"] += row["model2_only_sig_samples"]

    ct_rows = []
    for (cls, ct), b in sorted(by_ct.items()):
        ct_rows.append(
            {
                "recurrent_signal_class": cls,
                "cell_type": ct,
                "gene_celltypes": b["gene_celltypes"],
                "unique_genes": len(b["genes"]),
                "model1_sig_samples_sum": b["m1_sig_samples"],
                "model2_sig_samples_sum": b["m2_sig_samples"],
                "shared_sig_samples_sum": b["shared_sig_samples"],
                "model1_only_sig_samples_sum": b["m1_only_sig_samples"],
                "model2_only_sig_samples_sum": b["m2_only_sig_samples"],
                "top_tags_by_gene_celltype_count": ";".join(f"{tag}:{count}" for tag, count in b["tags"].most_common(8)),
            }
        )
    tag_rows = []
    for (cls, tag), b in sorted(by_tag.items()):
        tag_rows.append(
            {
                "recurrent_signal_class": cls,
                "gene_tag": tag,
                "gene_celltypes": b["gene_celltypes"],
                "unique_genes": len(b["genes"]),
                "cell_types_touched": len(b["cell_types"]),
                "model1_sig_samples_sum": b["m1_sig_samples"],
                "model2_sig_samples_sum": b["m2_sig_samples"],
                "shared_sig_samples_sum": b["shared_sig_samples"],
                "model1_only_sig_samples_sum": b["m1_only_sig_samples"],
                "model2_only_sig_samples_sum": b["m2_only_sig_samples"],
            }
        )
    ct_rows.sort(key=lambda r: (r["recurrent_signal_class"], r["cell_type"]))
    tag_rows.sort(key=lambda r: (r["recurrent_signal_class"], -r["gene_celltypes"], r["gene_tag"]))
    return ct_rows, tag_rows


def covariate_alignment_by_sample() -> list[dict]:
    rows = []
    for dataset, sample, sample_dir in sample_dirs():
        meta_path = INPUT_ROOT / dataset / sample / "metadata.csv"
        weights_path = sample_dir / "rctd_weights.csv"
        if not meta_path.exists() or not weights_path.exists():
            rows.append(
                {
                    "dataset": dataset,
                    "sample": sample,
                    "status": "missing_metadata_or_weights",
                    "metadata_path": str(meta_path),
                    "rctd_weights_path": str(weights_path),
                }
            )
            continue
        meta_rows = read_csv_auto(meta_path)
        weight_rows = read_csv_auto(weights_path)
        meta_by_barcode = {}
        for row in meta_rows:
            barcode = row.get("") or row.get("barcode") or row.get("spot") or next(iter(row.values()))
            meta_by_barcode[str(barcode)] = row
        values = {
            "snai": [],
            "mal": [],
            "rctd_epi": [],
            "rctd_nonepi": [],
            "rctd_fibro": [],
            "rctd_caf": [],
            "rctd_macro": [],
        }
        for row in weight_rows:
            barcode = row.get("") or row.get("barcode") or row.get("spot") or next(iter(row.values()))
            meta = meta_by_barcode.get(str(barcode))
            if not meta:
                continue
            snai = fnum(meta.get("SNAI1-ac_score"))
            mal = fnum(meta.get("Malignant"))
            epi = fnum(row.get("Epithelial"))
            if not all(math.isfinite(x) for x in (snai, mal, epi)):
                continue
            numeric_weights = []
            for key, val in row.items():
                if key == "" or key == "Epithelial":
                    continue
                num = fnum(val)
                if math.isfinite(num):
                    numeric_weights.append(num)
            values["snai"].append(snai)
            values["mal"].append(mal)
            values["rctd_epi"].append(epi)
            values["rctd_nonepi"].append(sum(numeric_weights))
            values["rctd_fibro"].append(fnum(row.get("Fibroblast")))
            values["rctd_caf"].append(fnum(row.get("CAF")))
            values["rctd_macro"].append(fnum(row.get("Macrophage")))
        rows.append(
            {
                "dataset": dataset,
                "sample": sample,
                "status": "ok",
                "n_common_barcodes": len(values["mal"]),
                "pearson_spacet_malignant_vs_rctd_epithelial": pearson(values["mal"], values["rctd_epi"]),
                "spearman_spacet_malignant_vs_rctd_epithelial": spearman(values["mal"], values["rctd_epi"]),
                "pearson_spacet_malignant_vs_rctd_non_epithelial_sum": pearson(values["mal"], values["rctd_nonepi"]),
                "pearson_snai1ac_vs_spacet_malignant": pearson(values["snai"], values["mal"]),
                "pearson_snai1ac_vs_rctd_epithelial": pearson(values["snai"], values["rctd_epi"]),
                "pearson_spacet_malignant_vs_rctd_fibroblast": pearson(values["mal"], values["rctd_fibro"]),
                "pearson_spacet_malignant_vs_rctd_caf": pearson(values["mal"], values["rctd_caf"]),
                "pearson_spacet_malignant_vs_rctd_macrophage": pearson(values["mal"], values["rctd_macro"]),
                "spacet_malignant_median": median(finite(values["mal"])) if finite(values["mal"]) else float("nan"),
                "rctd_epithelial_median": median(finite(values["rctd_epi"])) if finite(values["rctd_epi"]) else float("nan"),
                "metadata_path": str(meta_path),
                "rctd_weights_path": str(weights_path),
            }
        )
    return rows


def summarize_covariate_alignment(rows: list[dict]) -> list[dict]:
    metrics = [
        "pearson_spacet_malignant_vs_rctd_epithelial",
        "spearman_spacet_malignant_vs_rctd_epithelial",
        "pearson_spacet_malignant_vs_rctd_non_epithelial_sum",
        "pearson_snai1ac_vs_spacet_malignant",
        "pearson_snai1ac_vs_rctd_epithelial",
        "pearson_spacet_malignant_vs_rctd_fibroblast",
        "pearson_spacet_malignant_vs_rctd_caf",
        "pearson_spacet_malignant_vs_rctd_macrophage",
    ]
    out = []
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    for metric in metrics:
        vals = finite(fnum(row.get(metric)) for row in ok_rows)
        out.append(
            {
                "metric": metric,
                "n_samples": len(vals),
                "min": min(vals) if vals else float("nan"),
                "median": median(vals) if vals else float("nan"),
                "max": max(vals) if vals else float("nan"),
                "n_positive": sum(1 for x in vals if x > 0),
                "n_negative": sum(1 for x in vals if x < 0),
            }
        )
    return out


def robust_core_deep_support(
    meta_rows: list[dict],
    model1: dict[tuple[str, str, str, str], dict],
    model2: dict[tuple[str, str, str, str], dict],
    signature_genes: set[str],
) -> tuple[list[dict], list[dict]]:
    lookup = {(row["cell_type"], row["gene"]): row for row in meta_rows}
    rows = []
    for core in read_csv_auto(ROBUST_CORE_PATH):
        ct = core["cell_type"]
        gene = core["gene"]
        robust_direction = core["direction"]
        robust_sign = 1 if robust_direction == "positive" else -1
        agg = lookup.get((ct, gene), {})
        common_keys = [
            key
            for key in sorted(set(model1) & set(model2))
            if key[2] == ct and key[3] == gene
        ]
        m1_same = 0
        m2_same = 0
        for key in common_keys:
            m1_same += int(sign_value(model1[key]["signed_z"]) == robust_sign)
            m2_same += int(sign_value(model2[key]["signed_z"]) == robust_sign)
        m1_support = safe_div(m1_same, len(common_keys))
        m2_support = safe_div(m2_same, len(common_keys))
        if not math.isfinite(m1_support):
            status = "no_model1_common_samples"
        elif m1_support >= 0.8:
            status = "strong_model1_directional_support"
        elif m1_support >= 0.6:
            status = "partial_model1_directional_support"
        else:
            status = "weak_model1_directional_support"
        tags = gene_tags(gene, signature_genes)
        row = dict(core)
        row.update(
            {
                "tags": ";".join(tags),
                "primary_tag": primary_tag(tags),
                "model1_support_status": status,
                "model1_common_samples": len(common_keys),
                "model1_same_direction_fraction_vs_model2_core_direction": m1_support,
                "model2_same_direction_fraction_on_common_samples": m2_support,
                "model1_stouffer_z_common": agg.get("model1_stouffer_z", float("nan")),
                "model2_stouffer_z_common": agg.get("model2_stouffer_z", float("nan")),
                "delta_abs_stouffer_model2_minus_model1": agg.get("delta_abs_stouffer_model2_minus_model1", float("nan")),
                "model1_sig_samples": agg.get("model1_sig_samples", 0),
                "model2_sig_samples": agg.get("model2_sig_samples", 0),
                "shared_sig_samples": agg.get("shared_sig_samples", 0),
                "model1_only_sig_samples": agg.get("model1_only_sig_samples", 0),
                "model2_only_sig_samples": agg.get("model2_only_sig_samples", 0),
            }
        )
        rows.append(row)

    summary: dict[tuple[str, str], dict] = defaultdict(
        lambda: {
            "associations": 0,
            "genes": set(),
            "m1_support_values": [],
            "tags": Counter(),
        }
    )
    for row in rows:
        key = (row["model1_support_status"], row["cell_type"])
        b = summary[key]
        b["associations"] += 1
        b["genes"].add(row["gene"])
        if math.isfinite(fnum(row["model1_same_direction_fraction_vs_model2_core_direction"])):
            b["m1_support_values"].append(fnum(row["model1_same_direction_fraction_vs_model2_core_direction"]))
        for tag in str(row["tags"]).split(";"):
            b["tags"][tag] += 1
    summary_rows = []
    for (status, ct), b in sorted(summary.items()):
        vals = finite(b["m1_support_values"])
        summary_rows.append(
            {
                "model1_support_status": status,
                "cell_type": ct,
                "associations": b["associations"],
                "unique_genes": len(b["genes"]),
                "median_model1_same_direction_fraction": median(vals) if vals else float("nan"),
                "top_tags": ";".join(f"{tag}:{count}" for tag, count in b["tags"].most_common(8)),
            }
        )
    return rows, summary_rows


def build_readme(
    model1_stats: dict,
    model2_stats: dict,
    sig1: dict,
    sig2: dict,
    meta_rows: list[dict],
    signal_class_rows: list[dict],
    cov_summary: list[dict],
    robust_rows: list[dict],
) -> None:
    def metric_row(name: str) -> dict:
        return next(row for row in cov_summary if row["metric"] == name)

    common_pairs = [row for row in meta_rows if row["n_common_samples"] > 0]
    all_z1 = []
    all_z2 = []
    for row in common_pairs:
        all_z1.append(row["model1_stouffer_z"])
        all_z2.append(row["model2_stouffer_z"])
    recurrent_counts = Counter(row["recurrent_signal_class"] for row in meta_rows)
    robust_status_counts = Counter(row["model1_support_status"] for row in robust_rows)

    class_lookup = {(row["signal_class"], row["cell_type"]): row for row in signal_class_rows}
    main_ct_lines = []
    for ct in MAIN_CELL_TYPES:
        m1_only = class_lookup.get(("model1_only_significant", ct), {}).get("sample_level_significant_events", 0)
        m2_only = class_lookup.get(("model2_only_significant", ct), {}).get("sample_level_significant_events", 0)
        shared = class_lookup.get(("shared_significant", ct), {}).get("sample_level_significant_events", 0)
        main_ct_lines.append(f"- {ct}: shared={shared}, Model1-only={m1_only}, Model2-only={m2_only}")

    top_attenuated = top_gene_celltype_tables(meta_rows)["top_model1_attenuated_after_malignant_adjustment.csv"][:12]
    top_emergent = top_gene_celltype_tables(meta_rows)["top_model2_emergent_after_malignant_adjustment.csv"][:12]
    top_retained = top_gene_celltype_tables(meta_rows)["top_retained_shared_gene_celltypes.csv"][:12]

    def compact_top(rows: list[dict]) -> str:
        lines = []
        for row in rows:
            lines.append(
                f"- {row['cell_type']}:{row['gene']} "
                f"(M1_sig={row['model1_sig_samples']}, M2_sig={row['model2_sig_samples']}, "
                f"M1_only={row['model1_only_sig_samples']}, M2_only={row['model2_only_sig_samples']}, "
                f"tag={row['primary_tag']}, M1_Stouffer={fmt_float(row['model1_stouffer_z'])}, "
                f"M2_Stouffer={fmt_float(row['model2_stouffer_z'])})"
            )
        return "\n".join(lines)

    cov_mal_epi = metric_row("pearson_spacet_malignant_vs_rctd_epithelial")
    cov_snai_mal = metric_row("pearson_snai1ac_vs_spacet_malignant")
    cov_snai_epi = metric_row("pearson_snai1ac_vs_rctd_epithelial")
    readme = f"""# C-SIDE Model 1 vs Model 2 Deep Gene Audit

Generated: {datetime.now().isoformat(timespec="seconds")}

Script: `{SCRIPT_PATH}`

## What Was Audited

- Model 1: saved `cside_all_results.csv` / `cside_significant.csv` files.
- Model 2: saved `cside_2cov_all_results.csv` / `cside_2cov_significant.csv` files.
- SpaCET malignant covariate in `rctd_inputs/*/*/metadata.csv`.
- RCTD epithelial weights in `rctd_outputs/*/*/rctd_weights.csv`.
- Report-facing 73-association robust core from `{ROBUST_CORE_PATH}`.

## Core Sanity Checks

- Model 1 result files: {model1_stats['files']}; rows: {model1_stats['rows']}; non-converged rows: {model1_stats['nonconverged_rows']}; max p-from-Z delta: {model1_stats['max_p_delta']:.3g}; max logFC reconstruction delta: {model1_stats['max_logfc_delta']:.3g}.
- Model 2 result files: {model2_stats['files']}; rows: {model2_stats['rows']}; non-converged rows: {model2_stats['nonconverged_rows']}; max p-from-Z delta: {model2_stats['max_p_delta']:.3g}; max logFC reconstruction delta: {model2_stats['max_logfc_delta']:.3g}.
- Sample-level significant events: Model 1={len(sig1)}, Model 2={len(sig2)}, shared={len(set(sig1) & set(sig2))}, Model1-only={len(set(sig1) - set(sig2))}, Model2-only={len(set(sig2) - set(sig1))}.
- Gene-cell-type recurrent classes: {dict(recurrent_counts)}.

## Cross-Source Covariate Check

- SpaCET Malignant vs RCTD Epithelial Pearson r: median={fmt_float(cov_mal_epi['median'])}, range={fmt_float(cov_mal_epi['min'])} to {fmt_float(cov_mal_epi['max'])}, positive samples={cov_mal_epi['n_positive']}/{cov_mal_epi['n_samples']}.
- SNAI1-ac vs SpaCET Malignant Pearson r: median={fmt_float(cov_snai_mal['median'])}, range={fmt_float(cov_snai_mal['min'])} to {fmt_float(cov_snai_mal['max'])}, positive samples={cov_snai_mal['n_positive']}/{cov_snai_mal['n_samples']}.
- SNAI1-ac vs RCTD Epithelial Pearson r: median={fmt_float(cov_snai_epi['median'])}, range={fmt_float(cov_snai_epi['min'])} to {fmt_float(cov_snai_epi['max'])}, positive samples={cov_snai_epi['n_positive']}/{cov_snai_epi['n_samples']}.

Interpretation: the SpaCET malignant covariate is not the same object as the RCTD epithelial weight, but it is a spot-level tumor-content covariate that is positively aligned with the RCTD epithelial axis in the saved data. The two-covariate model is therefore not internally incoherent just because the covariate came from SpaCET. The key caveat is that the coefficient is a conditional SNAI1-ac association at fixed SpaCET malignant fraction, not an RCTD-only or causal estimate.

## Sample-Level Significant Event Split By Main Cell Type

{chr(10).join(main_ct_lines)}

## Top Model-1 Signals Attenuated After Malignant Adjustment

{compact_top(top_attenuated)}

## Top Model-2 Signals Emerging After Malignant Adjustment

{compact_top(top_emergent)}

## Top Retained Shared Signals

{compact_top(top_retained)}

## Model-2 Robust Core Support In Model 1

- Support-status counts: {dict(robust_status_counts)}
- Strong support means the Model-1 signed effect matches the Model-2 robust-core direction in at least 80% of common samples.
- Partial support means 60-80%; weak support means below 60%.

## Main Read

Model 1 is not garbage. It is internally coherent and useful as the unadjusted SNAI1-ac signal layer. But the row data show why Model 2 became report-facing: adjustment removes a large recurrent layer enriched for stromal/ECM and mitochondrial/ribosomal/stress-like signal, while preserving most of the robust Model-2 core directionally in Model 1. That means Model 2 is not simply inventing a new result; it is filtering a tumor-content-coupled version of the same analysis toward a stricter conditional question.

## Output Tables

- `tables/covariate_cross_source_alignment_by_sample.csv`
- `tables/covariate_cross_source_alignment_summary.csv`
- `tables/sample_level_signal_class_summary_by_celltype.csv`
- `tables/sample_level_signal_class_summary_by_tag.csv`
- `tables/gene_celltype_model1_model2_meta_comparison.csv`
- `tables/recurrent_signal_class_summary_by_celltype.csv`
- `tables/recurrent_signal_class_summary_by_tag.csv`
- `tables/top_model1_attenuated_after_malignant_adjustment.csv`
- `tables/top_model2_emergent_after_malignant_adjustment.csv`
- `tables/top_retained_shared_gene_celltypes.csv`
- `tables/model2_robust_core_deep_support.csv`
- `tables/model2_robust_core_deep_support_summary.csv`
"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    signature_genes = load_signature_genes()

    model1, model1_stats = load_model_results("cside_all_results.csv")
    model2, model2_stats = load_model_results("cside_2cov_all_results.csv")
    sig1 = load_significant("cside_significant.csv")
    sig2 = load_significant("cside_2cov_significant.csv")

    signal_class_rows, signal_tag_rows = summarize_signal_classes(sig1, sig2, signature_genes)
    write_csv(TABLE_DIR / "sample_level_signal_class_summary_by_celltype.csv", signal_class_rows)
    write_csv(TABLE_DIR / "sample_level_signal_class_summary_by_tag.csv", signal_tag_rows)

    meta_rows = aggregate_gene_celltypes(model1, model2, sig1, sig2, signature_genes)
    write_csv(TABLE_DIR / "gene_celltype_model1_model2_meta_comparison.csv", meta_rows)

    recurrent_ct_rows, recurrent_tag_rows = summarize_recurrent_classes(meta_rows)
    write_csv(TABLE_DIR / "recurrent_signal_class_summary_by_celltype.csv", recurrent_ct_rows)
    write_csv(TABLE_DIR / "recurrent_signal_class_summary_by_tag.csv", recurrent_tag_rows)

    for filename, rows in top_gene_celltype_tables(meta_rows).items():
        write_csv(TABLE_DIR / filename, rows)

    cov_rows = covariate_alignment_by_sample()
    cov_summary = summarize_covariate_alignment(cov_rows)
    write_csv(TABLE_DIR / "covariate_cross_source_alignment_by_sample.csv", cov_rows)
    write_csv(TABLE_DIR / "covariate_cross_source_alignment_summary.csv", cov_summary)

    robust_rows, robust_summary_rows = robust_core_deep_support(meta_rows, model1, model2, signature_genes)
    write_csv(TABLE_DIR / "model2_robust_core_deep_support.csv", robust_rows)
    write_csv(TABLE_DIR / "model2_robust_core_deep_support_summary.csv", robust_summary_rows)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(SCRIPT_PATH),
        "out_dir": str(OUT_DIR),
        "model1_stats": model1_stats,
        "model2_stats": model2_stats,
        "model1_significant_events": len(sig1),
        "model2_significant_events": len(sig2),
        "shared_significant_events": len(set(sig1) & set(sig2)),
        "model1_only_significant_events": len(set(sig1) - set(sig2)),
        "model2_only_significant_events": len(set(sig2) - set(sig1)),
        "signature_path": str(SIGNATURE_PATH),
        "n_signature_genes_loaded": len(signature_genes),
    }
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    build_readme(model1_stats, model2_stats, sig1, sig2, meta_rows, signal_class_rows, cov_summary, robust_rows)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
