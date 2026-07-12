"""
Calculate SpottedPy v2 Hallmark and SpaCET hotspots/coldspots.

This completes the revised hotspot layer before distance statistics. It reuses
the 39 Hallmark score_genes tables from the neighborhood module, adds SpaCET
fraction variables from the full-slide h5ad/core hotspot objects, and calls
SpottedPy hotspots with the same primary k=10, p=0.05, 999-permutation,
relative-to-sample settings used for the revised SNAI1-family, MP, and K*
hotspots.

This script intentionally does not plot figures and does not compute distances
or GEE. Run from PowerShell using spottedpy_env Python.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spottedpy_v2_hallmark_scoregenes_neighborhood_preflight import (  # noqa: E402
    HALLMARK_GROUPS,
    SPACET_COLS,
    compact_hallmark_name,
    hallmark_col,
)
from spottedpy_v2_hotspot_preflight import (  # noqa: E402
    HotspotSpec,
    add_spatial_obs,
    import_spottedpy,
    minmax,
    register_anndata_null_reader,
    require,
    run_specs,
    safe_name,
    stop,
    valid_component_mask,
)


DATA_ROOT = Path(r"D:\HGSOC_Spatial_Atlas")
ANALYSIS_ROOT = DATA_ROOT / "05_analysis_ready"
CODE_ROOT = Path(r"C:\Users\luchu\Documents\MSc\Master Thesis\Code\HGSOC_Spatial_Atlas")

RUN_ROOT = ANALYSIS_ROOT / "SpottedPy_v2_paper_aligned"
INPUT_MANIFEST = RUN_ROOT / "01_inputs_qc" / "tables" / "spottedpy_v2_live_input_manifest.csv"
HOTSPOT_ROOT = RUN_ROOT / "04_hotspots_preflight_revised_scoring_policy"
HALLMARK_SCORE_DIR = RUN_ROOT / "02_neighborhood_enrichment" / "tables" / "hallmark_scoregenes_by_sample"

TABLE_DIR = HOTSPOT_ROOT / "tables"
H5AD_DIR = HOTSPOT_ROOT / "h5ad" / "hallmark_spacet_full"
SAMPLE_TABLE_DIR = TABLE_DIR / "hallmark_spacet_hotspots_by_sample"
SCRIPT_DIR = HOTSPOT_ROOT / "scripts_used"
RUN_LOG = HOTSPOT_ROOT / "spottedpy_v2_hallmark_spacet_hotspots.log"

NEIGHBOURS_PRIMARY = 10
P_VALUE = 0.05
PERMUTATIONS = 999
SEED = 100


def log(message: str) -> None:
    text = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(text, flush=True)
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def prepare_dirs() -> None:
    for directory in [TABLE_DIR, H5AD_DIR, SAMPLE_TABLE_DIR, SCRIPT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), SCRIPT_DIR / Path(__file__).name)
    for helper in [
        "spottedpy_v2_hotspot_preflight.py",
        "spottedpy_v2_hallmark_scoregenes_neighborhood_preflight.py",
    ]:
        src = CODE_ROOT / "scripts" / helper
        if src.exists():
            shutil.copy2(src, SCRIPT_DIR / helper)


def load_manifest() -> pd.DataFrame:
    require(INPUT_MANIFEST)
    manifest = pd.read_csv(INPUT_MANIFEST)
    if len(manifest) != 23:
        stop(f"Expected 23 samples in input manifest; found {len(manifest)}")
    sample_limit = os.environ.get("SPOTTEDPY_V2_SAMPLE_LIMIT", "").strip()
    if sample_limit:
        manifest = manifest.sort_values(["dataset", "sample"]).head(int(sample_limit)).copy()
    return manifest


def core_h5ad_path(sample_label: str) -> Path:
    return HOTSPOT_ROOT / "h5ad" / "core_full" / f"{safe_name(sample_label)}__core_full_and_tumor_hotspots.h5ad"


def hallmark_scores_path(sample_label: str) -> Path:
    return HALLMARK_SCORE_DIR / f"{safe_name(sample_label)}__hallmark_scoregenes.csv.gz"


def sample_output_path(sample_label: str) -> Path:
    return H5AD_DIR / f"{safe_name(sample_label)}__hallmark_spacet_full_hotspots.h5ad"


def sample_summary_path(sample_label: str) -> Path:
    return SAMPLE_TABLE_DIR / f"{safe_name(sample_label)}__summary.csv"


def sample_scaling_path(sample_label: str) -> Path:
    return SAMPLE_TABLE_DIR / f"{safe_name(sample_label)}__scaling.csv"


def sample_component_path(sample_label: str) -> Path:
    return SAMPLE_TABLE_DIR / f"{safe_name(sample_label)}__component_numbering.csv"


def spot_ids(adata) -> pd.Series:
    if "spot" in adata.obs:
        return adata.obs["spot"].astype(str)
    return pd.Series(adata.obs.index.astype(str), index=adata.obs.index)


def make_lightweight_adata(base):
    import anndata as ad

    obs = base.obs.copy()
    var = pd.DataFrame(index=pd.Index([], name=base.var_names.name))
    lite = ad.AnnData(X=np.zeros((base.n_obs, 0), dtype=np.float32), obs=obs, var=var)
    if "spatial" not in base.obsm:
        stop("Base h5ad lacks .obsm['spatial']")
    lite.obsm["spatial"] = np.asarray(base.obsm["spatial"]).copy()
    return lite


def ordered_hallmarks() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for group, names in HALLMARK_GROUPS.items():
        for name in names:
            rows.append((group, name))
    return rows


def add_hallmark_specs(adata, scores: pd.DataFrame) -> tuple[list[HotspotSpec], list[dict[str, Any]]]:
    specs: list[HotspotSpec] = []
    scaling_records: list[dict[str, Any]] = []
    if "spot_id" not in scores.columns:
        stop("Hallmark score table lacks spot_id")
    score_by_spot = scores.set_index("spot_id")
    spot_index = spot_ids(adata)
    for group, hallmark in ordered_hallmarks():
        raw_col = hallmark_col(hallmark)
        if raw_col not in score_by_spot.columns:
            stop(f"Missing Hallmark score column: {raw_col}")
        compact = compact_hallmark_name(hallmark)
        scaled_col = f"spv2_full_hallmark_{safe_name(compact)}_scoregenes_minmax"
        adata.obs[raw_col] = spot_index.map(score_by_spot[raw_col])
        adata.obs[scaled_col], audit = minmax(adata.obs[raw_col])
        spec = HotspotSpec(
            family="hallmark",
            variable_id=hallmark,
            raw_col=raw_col,
            scaled_col=scaled_col,
            title=compact,
            domain="full",
            scale_method="full_score_genes_hallmark_then_minmax",
        )
        specs.append(spec)
        scaling_records.append(
            {
                "family": "hallmark",
                "domain": "full",
                "variable_id": hallmark,
                "hallmark_group": group,
                "raw_col": raw_col,
                "scaled_col": scaled_col,
                "scale_method": spec.scale_method,
                **audit,
            }
        )
    return specs, scaling_records


def add_spacet_specs(adata) -> tuple[list[HotspotSpec], list[dict[str, Any]], list[dict[str, Any]]]:
    specs: list[HotspotSpec] = []
    scaling_records: list[dict[str, Any]] = []
    missing_records: list[dict[str, Any]] = []
    for col in SPACET_COLS:
        variable_id = f"SpaCET_{safe_name(col)}"
        if col not in adata.obs.columns:
            missing_records.append(
                {
                    "family": "spacet",
                    "domain": "full",
                    "variable_id": variable_id,
                    "title": f"SpaCET {col}",
                    "raw_col": col,
                    "scaled_col": "",
                    "scale_method": "full_spacet_fraction_minmax",
                    "status": "skipped_missing_raw_column",
                    "error": f"Missing SpaCET column {col}",
                    "n_roi_spots": 0,
                    "n_hot_spots": 0,
                    "n_cold_spots": 0,
                    "n_hot_components": 0,
                    "n_cold_components": 0,
                    "seconds": 0,
                }
            )
            continue
        scaled_col = f"spv2_full_spacet_{safe_name(col)}_minmax"
        adata.obs[scaled_col], audit = minmax(adata.obs[col])
        spec = HotspotSpec(
            family="spacet",
            variable_id=variable_id,
            raw_col=col,
            scaled_col=scaled_col,
            title=f"SpaCET {col}",
            domain="full",
            scale_method="full_spacet_fraction_minmax",
        )
        specs.append(spec)
        scaling_records.append(
            {
                "family": "spacet",
                "domain": "full",
                "variable_id": variable_id,
                "raw_col": col,
                "scaled_col": scaled_col,
                "scale_method": spec.scale_method,
                **audit,
            }
        )
    return specs, scaling_records, missing_records


def parse_component_numbers(labels: pd.Series) -> list[int]:
    values = labels.astype(str)
    valid = values[valid_component_mask(values)]
    numbers = sorted({int(str(value).split("_", 1)[0]) for value in valid})
    return numbers


def parse_component_number(label: str) -> int | None:
    match = re.match(r"^(\d+)(?:_|$)", str(label))
    if not match:
        return None
    return int(match.group(1))


def compact_component_label(old_label: str, new_number: int) -> str:
    text = str(old_label)
    match = re.match(r"^\d+(_.*)?$", text)
    suffix = match.group(1) if match and match.group(1) else ""
    return f"{new_number}{suffix}"


def repair_component_numbering(adata, specs: list[HotspotSpec]) -> None:
    for spec in specs:
        for state in ["hot", "cold"]:
            number_col = f"{spec.scaled_col}_{state}_number"
            if number_col not in adata.obs.columns:
                continue
            values = adata.obs[number_col].astype(str)
            valid = values[valid_component_mask(values)]
            unique_labels = sorted(
                valid.unique(),
                key=lambda value: parse_component_number(value)
                if parse_component_number(value) is not None
                else 10**9,
            )
            old_numbers = sorted(
                {
                    parse_component_number(label)
                    for label in unique_labels
                    if parse_component_number(label) is not None
                }
            )
            number_map = {old: new for new, old in enumerate(old_numbers)}
            label_map = {}
            for label in unique_labels:
                old_number = parse_component_number(label)
                if old_number is not None:
                    label_map[label] = compact_component_label(label, number_map[old_number])
            adata.obs[number_col] = values.replace(label_map)


def component_audit(adata, specs: list[HotspotSpec], sample_label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        for state in ["hot", "cold"]:
            number_col = f"{spec.scaled_col}_{state}_number"
            if number_col not in adata.obs.columns:
                numbers: list[int] = []
            else:
                numbers = parse_component_numbers(adata.obs[number_col])
            expected = list(range(len(numbers))) if numbers else []
            missing = sorted(set(expected).difference(numbers))
            rows.append(
                {
                    "sample_label": sample_label,
                    "family": spec.family,
                    "domain": spec.domain,
                    "variable_id": spec.variable_id,
                    "state": state,
                    "number_col": number_col,
                    "n_components": len(numbers),
                    "numbers": ",".join(map(str, numbers)),
                    "expected_numbers": ",".join(map(str, expected)),
                    "missing_between_min_max": ",".join(map(str, missing)),
                    "needs_relabel": numbers != expected,
                }
            )
    return rows


def completed_sample(sample_label: str) -> bool:
    paths = [sample_output_path(sample_label), sample_summary_path(sample_label), sample_scaling_path(sample_label), sample_component_path(sample_label)]
    return all(path.exists() for path in paths)


def run_specs_with_progress(sp, adata, specs: list[HotspotSpec], sample_label: str, progress: dict[str, int]) -> tuple[Any, list[dict[str, Any]]]:
    all_records: list[dict[str, Any]] = []
    for spec in specs:
        progress["done"] += 1
        log(
            f"[{progress['done']}/{progress['total']}] "
            f"{sample_label} {spec.family} {spec.variable_id}"
        )
        adata, records = run_specs(sp, adata, [spec], sample_label)
        all_records.extend(records)
    return adata, all_records


def process_sample(sp, row: dict[str, Any], progress: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import anndata as ad

    dataset = str(row["dataset"])
    sample = str(row["sample"])
    sample_label = str(row["sample_label"])
    if completed_sample(sample_label) and os.environ.get("SPOTTEDPY_V2_FORCE", "").strip() != "1":
        progress["done"] += len(ordered_hallmarks()) + len(SPACET_COLS)
        log(f"Reusing completed Hallmark/SpaCET hotspot sample {sample_label}")
        log(f"[{progress['done']}/{progress['total']}] completed through {sample_label}")
        return (
            pd.read_csv(sample_summary_path(sample_label)),
            pd.read_csv(sample_scaling_path(sample_label)),
            pd.read_csv(sample_component_path(sample_label)),
        )

    log(f"Hallmark/SpaCET hotspot calculation {sample_label}")
    base_path = core_h5ad_path(sample_label)
    score_path = hallmark_scores_path(sample_label)
    require(base_path)
    require(score_path)

    base = ad.read_h5ad(base_path)
    adata = make_lightweight_adata(base)
    add_spatial_obs(adata, dataset, sample)
    scores = pd.read_csv(score_path)

    hallmark_specs, hallmark_scaling = add_hallmark_specs(adata, scores)
    spacet_specs, spacet_scaling, missing_spacet = add_spacet_specs(adata)
    specs = hallmark_specs + spacet_specs

    t0 = time.time()
    adata, summary_records = run_specs_with_progress(sp, adata, specs, sample_label, progress)
    repair_component_numbering(adata, specs)
    summary_records.extend(missing_spacet)
    summary = pd.DataFrame({"dataset": dataset, "sample": sample, **record} for record in summary_records)
    scaling = pd.DataFrame({"dataset": dataset, "sample": sample, "sample_label": sample_label, **record} for record in hallmark_scaling + spacet_scaling)
    components = pd.DataFrame({"dataset": dataset, "sample": sample, **record} for record in component_audit(adata, specs, sample_label))

    adata.write_h5ad(sample_output_path(sample_label))
    summary.to_csv(sample_summary_path(sample_label), index=False)
    scaling.to_csv(sample_scaling_path(sample_label), index=False)
    components.to_csv(sample_component_path(sample_label), index=False)
    log(f"Finished {sample_label} in {round(time.time() - t0, 1)} seconds")
    return summary, scaling, components


def write_branch_readme(summary: pd.DataFrame, component_audit_df: pd.DataFrame) -> None:
    ok = summary[summary["status"].eq("ok")]
    lines = [
        "# SpottedPy v2 Hallmark And SpaCET Hotspots",
        "",
        "This calculation-only layer completes the revised hotspot inputs before distance statistics.",
        "It does not create Hallmark/SpaCET hotspot figures, distance tables, GEE models, or sensitivity runs.",
        "",
        "## Parameters",
        "",
        f"- `neighbours_parameters`: {NEIGHBOURS_PRIMARY}",
        f"- `p_value`: {P_VALUE}",
        f"- `permutations`: {PERMUTATIONS}",
        f"- `seed`: {SEED}",
        "- `relative_to_batch`: True",
        "- batch grain: one sample/patient",
        "",
        "## Inputs",
        "",
        "- Hallmarks: 39 score_genes tables from `02_neighborhood_enrichment/tables/hallmark_scoregenes_by_sample`.",
        "- SpaCET: full-slide fraction columns from the revised core h5ad objects.",
        "- Base h5ad: lightweight copies of `04_hotspots_preflight_revised_scoring_policy/h5ad/core_full` preserving core SNAI1-family hotspot columns.",
        "",
        "## Outputs",
        "",
        "- `h5ad/hallmark_spacet_full/*__hallmark_spacet_full_hotspots.h5ad`",
        "- `tables/spottedpy_v2_hallmark_spacet_hotspot_summary.csv`",
        "- `tables/spottedpy_v2_hallmark_spacet_scaling_audit.csv`",
        "- `tables/spottedpy_v2_hallmark_spacet_component_numbering_audit.csv`",
        "",
        "## Counts",
        "",
        f"- Hotspot tests represented: {len(summary)}",
        f"- Successful hotspot tests: {len(ok)}",
        f"- Tests with any hot spots: {int((summary['n_hot_spots'] > 0).sum())}",
        f"- Tests with any cold spots: {int((summary['n_cold_spots'] > 0).sum())}",
        f"- Numbering rows needing relabel: {int(component_audit_df['needs_relabel'].astype(str).str.lower().eq('true').sum()) if len(component_audit_df) else 0}",
        "",
    ]
    (HOTSPOT_ROOT / "README_hallmark_spacet_hotspots.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    prepare_dirs()
    register_anndata_null_reader()
    sp = import_spottedpy()
    manifest = load_manifest().sort_values(["dataset", "sample"]).copy()
    n_specs_per_sample = len(ordered_hallmarks()) + len(SPACET_COLS)
    progress = {
        "done": 0,
        "total": int(len(manifest) * n_specs_per_sample),
    }
    log(
        f"Starting Hallmark/SpaCET hotspot layer: {len(manifest)} samples, "
        f"{n_specs_per_sample} variables/sample, {progress['total']} total hotspot calls"
    )

    summaries: list[pd.DataFrame] = []
    scalings: list[pd.DataFrame] = []
    components: list[pd.DataFrame] = []
    for row in manifest.to_dict("records"):
        summary, scaling, component = process_sample(sp, row, progress)
        summaries.append(summary)
        scalings.append(scaling)
        components.append(component)

    summary_df = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    scaling_df = pd.concat(scalings, ignore_index=True) if scalings else pd.DataFrame()
    component_df = pd.concat(components, ignore_index=True) if components else pd.DataFrame()
    summary_df.to_csv(TABLE_DIR / "spottedpy_v2_hallmark_spacet_hotspot_summary.csv", index=False)
    scaling_df.to_csv(TABLE_DIR / "spottedpy_v2_hallmark_spacet_scaling_audit.csv", index=False)
    component_df.to_csv(TABLE_DIR / "spottedpy_v2_hallmark_spacet_component_numbering_audit.csv", index=False)

    run_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__)),
        "output_root": str(HOTSPOT_ROOT),
        "input_manifest": str(INPUT_MANIFEST),
        "hallmark_score_dir": str(HALLMARK_SCORE_DIR),
        "h5ad_output_dir": str(H5AD_DIR),
        "neighbours_parameters": NEIGHBOURS_PRIMARY,
        "p_value": P_VALUE,
        "permutations": PERMUTATIONS,
        "seed": SEED,
        "relative_to_batch": True,
        "batch_grain": "sample/patient",
        "n_samples": int(manifest["sample_label"].nunique()) if len(manifest) else 0,
        "n_hallmarks": len(ordered_hallmarks()),
        "n_spacet_variables": len(SPACET_COLS),
        "n_hotspot_rows": int(len(summary_df)),
        "n_h5ad_files": int(len(list(H5AD_DIR.glob("*__hallmark_spacet_full_hotspots.h5ad")))),
        "does_not_run": [
            "distance statistics",
            "GEE",
            "neighborhood enrichment",
            "k=8 sensitivity",
            "Hallmark/SpaCET hotspot plotting",
        ],
    }
    (HOTSPOT_ROOT / "run_manifest_hallmark_spacet_hotspots.json").write_text(
        json.dumps(run_manifest, indent=2),
        encoding="utf-8",
    )
    write_branch_readme(summary_df, component_df)
    log(json.dumps(run_manifest, indent=2))


if __name__ == "__main__":
    main()
