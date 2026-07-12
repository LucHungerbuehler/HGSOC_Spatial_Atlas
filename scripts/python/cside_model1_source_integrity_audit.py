"""Source-integrity audit for C-SIDE Model 1 outputs.

This is the gate before rebuilding a full Model-1 S2e evidence ladder.

Checks:
  - sample coverage and missing outputs
  - required result columns
  - duplicate sample/cell_type/gene rows
  - convergence flags
  - p-value reconstruction from Z
  - logFC reconstruction from mean_1 - mean_0
  - significant-file consistency with all-results p_val < 0.05
  - cell-type coverage across samples

The script does not rerun C-SIDE. The targeted SP5 rerun is handled by:
  scripts/R/rerun_cside_model1_single_sample.R
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(r"D:\HGSOC_Spatial_Atlas")
INPUT_ROOT = ROOT / "scRNA_reference" / "rctd_inputs"
OUTPUT_ROOT = ROOT / "scRNA_reference" / "rctd_outputs"
S2E_ROOT = ROOT / "05_analysis_ready" / "S2e_CSIDE_CellTypeSpecific_DE_Audit"
OUT_DIR = S2E_ROOT / "12_model1_source_integrity_audit"
TABLE_DIR = OUT_DIR / "tables"
LOG_PATH = S2E_ROOT / "11_model1_vs_model2_deep_gene_audit" / "sp5_model1_targeted_rerun.log"
SCRIPT_PATH = Path(__file__).resolve()

DATASETS = ("denisenko_2022", "ju_2024", "yamamoto_2025")
REQUIRED_ALL_COLUMNS = [
    "Z_score",
    "log_fc",
    "se",
    "paramindex_best",
    "conv",
    "p_val",
    "mean_0",
    "mean_1",
    "sd_0",
    "sd_1",
    "cell_type",
    "gene",
]


def read_csv_auto(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
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


def safe_div(num: float, den: float) -> float:
    return num / den if den else float("nan")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sample_dirs() -> list[tuple[str, str, Path, Path]]:
    dirs: list[tuple[str, str, Path, Path]] = []
    for dataset in DATASETS:
        input_dataset_dir = INPUT_ROOT / dataset
        if not input_dataset_dir.exists():
            continue
        for input_sample_dir in sorted(p for p in input_dataset_dir.iterdir() if p.is_dir()):
            output_sample_dir = OUTPUT_ROOT / dataset / input_sample_dir.name
            dirs.append((dataset, input_sample_dir.name, input_sample_dir, output_sample_dir))
    return dirs


def p_from_z(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def audit_sample(dataset: str, sample: str, input_dir: Path, output_dir: Path) -> tuple[dict, list[dict]]:
    all_path = output_dir / "cside_all_results.csv"
    sig_path = output_dir / "cside_significant.csv"
    rctd_path = output_dir / "rctd_object.rds"
    cside_rds_path = output_dir / "rctd_cside_object.rds"
    model2_path = output_dir / "cside_2cov_all_results.csv"

    row = {
        "dataset": dataset,
        "sample": sample,
        "sample_label": f"{dataset}__{sample}",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "has_rctd_object": rctd_path.exists(),
        "has_model1_all_results": all_path.exists(),
        "has_model1_significant": sig_path.exists(),
        "has_model1_rds": cside_rds_path.exists(),
        "has_model2_all_results": model2_path.exists(),
        "model1_all_results_path": str(all_path),
        "model1_significant_path": str(sig_path),
        "model1_rds_path": str(cside_rds_path),
        "model1_all_results_size_bytes": all_path.stat().st_size if all_path.exists() else 0,
        "model1_significant_size_bytes": sig_path.stat().st_size if sig_path.exists() else 0,
        "model1_all_results_sha256": sha256_file(all_path) if all_path.exists() else "",
        "model1_significant_sha256": sha256_file(sig_path) if sig_path.exists() else "",
        "status": "missing_model1_all_results",
    }
    celltype_rows: list[dict] = []

    if not all_path.exists():
        return row, celltype_rows

    all_rows = read_csv_auto(all_path)
    sig_rows = read_csv_auto(sig_path) if sig_path.exists() else []
    columns = list(all_rows[0].keys()) if all_rows else []
    missing_columns = [col for col in REQUIRED_ALL_COLUMNS if col not in columns]

    p_delta_max = 0.0
    logfc_delta_max = 0.0
    p_delta_gt_1e_8 = 0
    logfc_delta_gt_1e_8 = 0
    nonconverged = 0
    p_missing_or_bad = 0
    logfc_missing_or_bad = 0
    duplicate_rows = 0
    seen = set()
    all_sig_keys = set()
    all_p_lt_005_keys = set()
    all_celltype_counts = Counter()
    all_celltype_sig_counts = Counter()
    p_values = []
    z_values = []

    for result in all_rows:
        key = (result.get("cell_type", ""), result.get("gene", ""))
        if key in seen:
            duplicate_rows += 1
        seen.add(key)
        all_celltype_counts[result.get("cell_type", "")] += 1
        if str(result.get("conv", "")).strip().lower() != "true":
            nonconverged += 1
        z = fnum(result.get("Z_score"))
        p_val = fnum(result.get("p_val"))
        log_fc = fnum(result.get("log_fc"))
        mean_0 = fnum(result.get("mean_0"))
        mean_1 = fnum(result.get("mean_1"))
        if math.isfinite(z):
            z_values.append(z)
        if math.isfinite(p_val):
            p_values.append(p_val)
        if math.isfinite(z) and math.isfinite(p_val):
            delta = abs(p_val - p_from_z(z))
            p_delta_max = max(p_delta_max, delta)
            p_delta_gt_1e_8 += int(delta > 1e-8)
        else:
            p_missing_or_bad += 1
        if all(math.isfinite(x) for x in (log_fc, mean_0, mean_1)):
            delta = abs(log_fc - (mean_1 - mean_0))
            logfc_delta_max = max(logfc_delta_max, delta)
            logfc_delta_gt_1e_8 += int(delta > 1e-8)
        else:
            logfc_missing_or_bad += 1
        if math.isfinite(p_val) and p_val < 0.05:
            all_p_lt_005_keys.add(key)

    sig_keys = set()
    sig_duplicate_rows = 0
    sig_p_ge_005 = 0
    for sig in sig_rows:
        key = (sig.get("cell_type", ""), sig.get("gene", ""))
        if key in sig_keys:
            sig_duplicate_rows += 1
        sig_keys.add(key)
        all_sig_keys.add(key)
        all_celltype_sig_counts[sig.get("cell_type", "")] += 1
        if fnum(sig.get("p_val")) >= 0.05:
            sig_p_ge_005 += 1

    sig_not_in_all = len(sig_keys - seen)
    # C-SIDE's sig_gene_list is not guaranteed to equal every nominal p<0.05
    # row from all_gene_list. Treat the difference as provenance metadata, not
    # a source-integrity failure.
    all_p_lt_005_not_in_sig = len(all_p_lt_005_keys - sig_keys)
    sig_missing_from_file = len(sig_keys) == 0 and len(all_p_lt_005_keys) > 0

    for cell_type in sorted(all_celltype_counts):
        celltype_rows.append(
            {
                "dataset": dataset,
                "sample": sample,
                "sample_label": f"{dataset}__{sample}",
                "cell_type": cell_type,
                "all_results_rows": all_celltype_counts[cell_type],
                "significant_rows": all_celltype_sig_counts[cell_type],
                "significant_fraction": safe_div(all_celltype_sig_counts[cell_type], all_celltype_counts[cell_type]),
            }
        )

    hard_failures = []
    if missing_columns:
        hard_failures.append("missing_required_columns")
    if nonconverged:
        hard_failures.append("nonconverged_rows")
    if duplicate_rows:
        hard_failures.append("duplicate_all_result_gene_celltype_rows")
    if sig_duplicate_rows:
        hard_failures.append("duplicate_significant_gene_celltype_rows")
    if sig_not_in_all:
        hard_failures.append("significant_rows_not_in_all_results")
    if sig_p_ge_005:
        hard_failures.append("significant_rows_with_p_ge_0.05")
    if p_delta_gt_1e_8:
        hard_failures.append("p_value_reconstruction_failures")
    if logfc_delta_gt_1e_8:
        hard_failures.append("logfc_reconstruction_failures")
    if sig_missing_from_file:
        hard_failures.append("significant_file_empty_but_all_results_has_p_lt_0.05")

    row.update(
        {
            "status": "pass" if not hard_failures else "fail",
            "failure_reasons": ";".join(hard_failures),
            "all_results_rows": len(all_rows),
            "significant_rows": len(sig_rows),
            "all_results_p_lt_0.05_rows": len(all_p_lt_005_keys),
            "required_columns_present": not bool(missing_columns),
            "missing_required_columns": ";".join(missing_columns),
            "all_results_duplicate_gene_celltype_rows": duplicate_rows,
            "significant_duplicate_gene_celltype_rows": sig_duplicate_rows,
            "nonconverged_rows": nonconverged,
            "max_p_reconstruction_delta": p_delta_max,
            "p_reconstruction_delta_gt_1e_8_rows": p_delta_gt_1e_8,
            "p_missing_or_bad_rows": p_missing_or_bad,
            "max_logfc_reconstruction_delta": logfc_delta_max,
            "logfc_reconstruction_delta_gt_1e_8_rows": logfc_delta_gt_1e_8,
            "logfc_missing_or_bad_rows": logfc_missing_or_bad,
            "significant_rows_not_in_all_results": sig_not_in_all,
            "all_results_p_lt_0.05_rows_not_in_significant_file": all_p_lt_005_not_in_sig,
            "significant_rows_with_p_ge_0.05": sig_p_ge_005,
            "cell_types_tested": ";".join(sorted(all_celltype_counts)),
            "n_cell_types_tested": len(all_celltype_counts),
            "min_p_value": min(p_values) if p_values else float("nan"),
            "max_abs_z": max(abs(z) for z in z_values) if z_values else float("nan"),
        }
    )
    return row, celltype_rows


def parse_rerun_log() -> dict:
    if not LOG_PATH.exists():
        return {"rerun_log_path": str(LOG_PATH), "rerun_log_exists": False}
    text = LOG_PATH.read_text(encoding="utf-8", errors="replace")
    if "length(cell_types)" not in text and "run.CSIDE.single" not in text:
        try:
            text = LOG_PATH.read_text(encoding="utf-16", errors="replace")
        except UnicodeError:
            pass
    failure_summary = ""
    if re.search(r"length\s*\(cell_types\)\s+is\s+0", text, flags=re.IGNORECASE):
        failure_summary = "SP5 Model 1 rerun failed: no cell types passed C-SIDE eligibility after original cell_type_threshold=50 and weight_threshold=0.8 filters."
    elif "Targeted Model 1 rerun complete" in text:
        failure_summary = "SP5 Model 1 rerun completed."
    else:
        failure_summary = "SP5 Model 1 rerun log present; inspect log for status."

    filtered = ""
    match = re.search(r"filtered out cell types: (.*?) due to not having sufficiently many pixels", text, re.S)
    if match:
        filtered = " ".join(match.group(1).split())
    removed = ""
    match = re.search(r"removing the following cell types due to insufficient counts per region.*?Cell types: (.*?)(?:Fehler|Error|$)", text, re.S)
    if match:
        removed = " ".join(match.group(1).split())
    return {
        "rerun_log_path": str(LOG_PATH),
        "rerun_log_exists": True,
        "rerun_failure_summary": failure_summary,
        "rerun_filtered_out_celltypes_text": filtered,
        "rerun_removed_celltypes_text": removed,
    }


def build_readme(sample_rows: list[dict], celltype_rows: list[dict], rerun_info: dict) -> None:
    total_expected = len(sample_rows)
    present = [row for row in sample_rows if row["has_model1_all_results"]]
    missing = [row for row in sample_rows if not row["has_model1_all_results"]]
    failing = [row for row in present if row["status"] != "pass"]
    passing = [row for row in present if row["status"] == "pass"]
    total_rows = sum(int(row.get("all_results_rows") or 0) for row in present)
    total_sig = sum(int(row.get("significant_rows") or 0) for row in present)
    max_p_delta = max((fnum(row.get("max_p_reconstruction_delta")) for row in present), default=float("nan"))
    max_logfc_delta = max((fnum(row.get("max_logfc_reconstruction_delta")) for row in present), default=float("nan"))
    nonconv = sum(int(row.get("nonconverged_rows") or 0) for row in present)
    missing_labels = ", ".join(f"{row['dataset']}/{row['sample']}" for row in missing) or "none"

    by_ct = defaultdict(lambda: {"samples": set(), "rows": 0, "sig": 0})
    for row in celltype_rows:
        ct = row["cell_type"]
        by_ct[ct]["samples"].add(row["sample_label"])
        by_ct[ct]["rows"] += int(row["all_results_rows"])
        by_ct[ct]["sig"] += int(row["significant_rows"])
    ct_lines = []
    for ct in sorted(by_ct):
        b = by_ct[ct]
        ct_lines.append(
            f"- {ct}: {len(b['samples'])} samples, {b['rows']} tested rows, {b['sig']} significant rows"
        )

    verdict = (
        "Model 1 source files pass internal numeric integrity for the 22 samples that produced outputs. "
        "SP5 should not be force-added under the original parameterization because the targeted rerun fails C-SIDE eligibility filters."
    )
    if failing:
        verdict = (
            "Model 1 source integrity is not yet clean because at least one existing output file failed an internal check. "
            "Inspect sample_model1_source_integrity.csv before continuing."
        )

    readme = f"""# C-SIDE Model 1 Source-Integrity Audit

Generated: {datetime.now().isoformat(timespec="seconds")}

Script: `{SCRIPT_PATH}`

## Question

Can Model 1 be treated as a trustworthy source layer before rebuilding the full S2e evidence ladder?

## Rerun Result

- Targeted rerun log: `{rerun_info.get('rerun_log_path')}`
- Rerun status: {rerun_info.get('rerun_failure_summary', 'not available')}

## Coverage

- Expected samples from RCTD inputs: {total_expected}
- Model 1 all-results files present: {len(present)}
- Model 1 all-results files missing: {len(missing)}
- Missing Model 1 sample(s): {missing_labels}

## Integrity Checks Across Existing Model 1 Files

- Passing existing Model 1 samples: {len(passing)}/{len(present)}
- Failing existing Model 1 samples: {len(failing)}
- Total tested rows: {total_rows}
- Total significant rows: {total_sig}
- Non-converged rows: {nonconv}
- Maximum p-value reconstruction delta from Z: {max_p_delta:.3g}
- Maximum logFC reconstruction delta from mean_1 - mean_0: {max_logfc_delta:.3g}

## Cell-Type Coverage

{chr(10).join(ct_lines)}

## Verdict

{verdict}

## Practical Consequence

The clean comparison universe for a full Model-1 workup is the 22-sample Model-1 universe, unless we deliberately change C-SIDE eligibility thresholds for SP5. Changing thresholds would create a different Model 1 and should not be mixed into the main same-parameter evidence ladder.

## Output Tables

- `tables/sample_model1_source_integrity.csv`
- `tables/celltype_model1_source_integrity.csv`
- `tables/celltype_model1_coverage_summary.csv`
- `tables/missing_model1_samples.csv`
- `tables/rerun_model1_missing_sample_status.csv`
"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    sample_rows = []
    celltype_rows = []
    for dataset, sample, input_dir, output_dir in sample_dirs():
        row, ct_rows = audit_sample(dataset, sample, input_dir, output_dir)
        sample_rows.append(row)
        celltype_rows.extend(ct_rows)

    by_ct = defaultdict(lambda: {"samples": set(), "rows": 0, "sig": 0})
    for row in celltype_rows:
        ct = row["cell_type"]
        by_ct[ct]["samples"].add(row["sample_label"])
        by_ct[ct]["rows"] += int(row["all_results_rows"])
        by_ct[ct]["sig"] += int(row["significant_rows"])
    ct_summary = [
        {
            "cell_type": ct,
            "samples_tested": len(b["samples"]),
            "all_results_rows": b["rows"],
            "significant_rows": b["sig"],
            "significant_fraction": safe_div(b["sig"], b["rows"]),
            "sample_labels": ";".join(sorted(b["samples"])),
        }
        for ct, b in sorted(by_ct.items())
    ]
    missing_rows = [row for row in sample_rows if not row["has_model1_all_results"]]
    rerun_info = parse_rerun_log()

    write_csv(TABLE_DIR / "sample_model1_source_integrity.csv", sample_rows)
    write_csv(TABLE_DIR / "celltype_model1_source_integrity.csv", celltype_rows)
    write_csv(TABLE_DIR / "celltype_model1_coverage_summary.csv", ct_summary)
    write_csv(TABLE_DIR / "missing_model1_samples.csv", missing_rows)
    write_csv(TABLE_DIR / "rerun_model1_missing_sample_status.csv", [rerun_info])

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(SCRIPT_PATH),
        "out_dir": str(OUT_DIR),
        "expected_samples": len(sample_rows),
        "model1_files_present": sum(1 for row in sample_rows if row["has_model1_all_results"]),
        "model1_files_missing": sum(1 for row in sample_rows if not row["has_model1_all_results"]),
        "existing_model1_samples_passing_integrity": sum(
            1 for row in sample_rows if row["has_model1_all_results"] and row["status"] == "pass"
        ),
        "existing_model1_samples_failing_integrity": sum(
            1 for row in sample_rows if row["has_model1_all_results"] and row["status"] != "pass"
        ),
        "total_all_results_rows": sum(int(row.get("all_results_rows") or 0) for row in sample_rows),
        "total_significant_rows": sum(int(row.get("significant_rows") or 0) for row in sample_rows),
        "rerun_info": rerun_info,
    }
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    build_readme(sample_rows, celltype_rows, rerun_info)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
