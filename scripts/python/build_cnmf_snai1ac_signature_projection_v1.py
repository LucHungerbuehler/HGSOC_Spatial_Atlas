"""
Build one clean K*-only table projecting SNAI1-ac signature weights onto cNMF spectra.

The table is intentionally narrow:
  program_id
  alignment_category_draft
  within_sample_projection_z_score
  across_sample_projection_z_score
  family_label
  mp1_8_name

Projection is the absolute-weight-normalized dot product between the weighted
SNAI1-ac signature and each K* programme spectrum, then z-scored within sample
across that sample's K* programmes and across all K* programmes.

The MP labels are harmonized metaprograms derived from reclustering raw
sample-specific programmes; they are not the raw cNMF programmes themselves.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


DATA_ROOT = Path(r"D:\HGSOC_Spatial_Atlas")
CNMF_ROOT = DATA_ROOT / "05_analysis_ready" / "S3_cNMF_Tumor_Programs"
SIGNATURE_WEIGHTS = DATA_ROOT / "05_analysis_ready" / "Signature" / "snai1_ac_weights.json"
OUTPUT_ROOT = CNMF_ROOT / "snai1ac_signature_projection_onto_cnmf_programs_v1"
SCRIPT_PATH = Path(__file__).resolve()

ALIGNMENT_TABLE = CNMF_ROOT / "jaccard_commonspace_substrate" / "extracted_program_top50_commonspace.csv"
MANUAL_MP_ROOT = (
    CNMF_ROOT
    / "jaccard_raw_matrices"
    / "inspection_exports_average"
    / "variantB_nonjunk_manual_cut_v2"
)
FAMILY_TABLE = MANUAL_MP_ROOT / "variantB_nonjunk_manual_cut_v2_ordered_by_snai1ac_rho.csv"
MP_MEMBERS_TABLE = MANUAL_MP_ROOT / "subcluster_signatures_scoring" / "signatures" / "manual_subcluster_program_members.csv"
MP_NAME_MAP = {
    "MP1": "MP1 angiogenic/vascular",
    "MP2": "MP2 iCAF-stress",
    "MP3": "MP3 complement-CAF",
    "MP4": "MP4 activated-myCAF",
    "MP5": "MP5 IFN/TLS immune",
    "MP6": "MP6 APC/TAM myeloid",
    "MP7": "MP7 malignant hypoxia",
    "MP8": "MP8 malignant acute-phase/secretory",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean K* SNAI1-ac projection table.")
    parser.add_argument("--cnmf-root", type=Path, default=CNMF_ROOT)
    parser.add_argument("--signature-weights", type=Path, default=SIGNATURE_WEIGHTS)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def load_signature_weights(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        return {str(gene): float(weight) for gene, weight in json.load(handle).items()}


def project_one_spectrum(row: pd.Series, gene_columns: list[str], signature_weights: dict[str, float]) -> float:
    pairs = []
    for column in gene_columns:
        gene = column.replace("__gene__", "", 1)
        if gene in signature_weights:
            pairs.append((column, signature_weights[gene]))
    abs_sum = sum(abs(weight) for _, weight in pairs)
    if abs_sum <= 0:
        return math.nan

    projection = 0.0
    for column, weight in pairs:
        value = pd.to_numeric(row[column], errors="coerce")
        if pd.notna(value):
            projection += (weight / abs_sum) * float(value)
    return float(projection)


def zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    sd = numeric.std(ddof=0)
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series([math.nan] * len(numeric), index=numeric.index)
    return (numeric - numeric.mean()) / sd


def load_kstar_projection(cnmf_root: Path, signature_weights: dict[str, float]) -> pd.DataFrame:
    manifest = pd.read_csv(cnmf_root / "sample_manifest.csv")
    manifest = manifest[manifest["eligible_for_cnmf"].astype(str).str.lower().eq("true")].copy()

    rows = []
    for _, sample in manifest.sort_values(["dataset", "sample_id_on_disk"]).iterrows():
        sample_id = str(sample["sample_id_on_disk"])
        sample_label = str(sample["sample_label"])
        spectra_path = cnmf_root / "per_sample" / sample_id / "extracted_program_spectra.csv"
        spectra = pd.read_csv(spectra_path)
        spectra = spectra[spectra["is_k_star"].astype(str).str.lower().eq("true")].copy()
        gene_columns = [column for column in spectra.columns if str(column).startswith("__gene__")]

        spectra["signature_projection_absnorm_dot"] = spectra.apply(
            lambda row: project_one_spectrum(row, gene_columns, signature_weights),
            axis=1,
        )
        spectra["within_sample_projection_z_score"] = zscore(spectra["signature_projection_absnorm_dot"])
        for _, row in spectra.iterrows():
            rows.append(
                {
                    "sample_label": sample_label,
                    "program_id": str(row["program_id"]),
                    "signature_projection_absnorm_dot": row["signature_projection_absnorm_dot"],
                    "within_sample_projection_z_score": row["within_sample_projection_z_score"],
                }
            )
    projection = pd.DataFrame(rows)
    projection["across_sample_projection_z_score"] = zscore(projection["signature_projection_absnorm_dot"])
    return projection


def build_clean_table(cnmf_root: Path, signature_weights_path: Path) -> pd.DataFrame:
    signature_weights = load_signature_weights(signature_weights_path)
    projection = load_kstar_projection(cnmf_root, signature_weights)

    alignment = pd.read_csv(ALIGNMENT_TABLE)[["program_id", "alignment_category_draft"]].drop_duplicates("program_id")
    family = pd.read_csv(FAMILY_TABLE)[["program_id", "family_label"]].drop_duplicates("program_id")
    mp_labels = pd.read_csv(MP_MEMBERS_TABLE)[["program_id", "subcluster_id"]].drop_duplicates("program_id")
    mp_labels["mp1_8_name"] = mp_labels["subcluster_id"].map(MP_NAME_MAP)

    clean = projection.merge(alignment, on="program_id", how="left")
    clean = clean.merge(family, on="program_id", how="left")
    clean = clean.merge(mp_labels[["program_id", "mp1_8_name"]], on="program_id", how="left")

    clean["family_label"] = clean["family_label"].fillna("not_assigned_to_MP")
    clean["mp1_8_name"] = clean["mp1_8_name"].fillna("not_assigned_to_MP")

    clean = clean[
        [
            "program_id",
            "alignment_category_draft",
            "within_sample_projection_z_score",
            "across_sample_projection_z_score",
            "family_label",
            "mp1_8_name",
        ]
    ].sort_values(["program_id"])
    return clean


def write_outputs(clean: pd.DataFrame, output_root: Path) -> Path:
    table_dir = output_root / "tables"
    script_dir = output_root / "scripts_used"
    table_dir.mkdir(parents=True, exist_ok=True)
    script_dir.mkdir(parents=True, exist_ok=True)

    for old_csv in table_dir.glob("*.csv"):
        old_csv.unlink()

    output_path = table_dir / "kstar_snai1ac_signature_projection_clean.csv"
    clean.to_csv(output_path, index=False)
    shutil.copy2(SCRIPT_PATH, script_dir / SCRIPT_PATH.name)

    n_assigned = int((clean["mp1_8_name"] != "not_assigned_to_MP").sum())
    readme = "\n".join(
        [
            "# K* SNAI1-ac Signature Projection",
            "",
            "Clean K*-only table for inspecting how each representative cNMF programme spectrum aligns with the weighted SNAI1-ac signature.",
            "",
            "Projection is computed as an absolute-weight-normalized dot product between signature weights and programme spectrum weights, then z-scored within each sample across that sample's K* programmes.",
            "",
            "The across-sample z-score uses the same programme-level projection values but standardizes them across all 144 K* programmes.",
            "",
            "MP labels are harmonized metaprograms derived from reclustering raw sample-specific programmes so SNAI1-ac associations can be compared across samples. They are not raw cNMF programmes.",
            "",
            "Primary table:",
            "",
            "- `tables/kstar_snai1ac_signature_projection_clean.csv`",
            "",
            f"Rows: {len(clean)} K* programmes.",
            f"MP-assigned rows: {n_assigned}.",
            f"Rows not assigned to MP by design: {len(clean) - n_assigned}.",
            "",
        ]
    )
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    manifest = {
        "branch": "snai1ac_signature_projection_onto_cnmf_programs_v1",
        "primary_table": "tables/kstar_snai1ac_signature_projection_clean.csv",
        "n_kstar_programs": int(len(clean)),
        "n_mp_assigned": n_assigned,
        "n_not_assigned_to_mp": int(len(clean) - n_assigned),
        "cnmf_root": str(CNMF_ROOT),
        "signature_weights": str(SIGNATURE_WEIGHTS),
        "alignment_source": str(ALIGNMENT_TABLE),
        "family_source": str(FAMILY_TABLE),
        "mp_membership_source": str(MP_MEMBERS_TABLE),
        "mp_label_source": "user-confirmed MP1-MP8 harmonized metaprogram names",
        "wording_note": "MP labels are harmonized metaprograms derived from reclustering raw sample-specific programmes; they are not raw cNMF programmes.",
        "script": str(SCRIPT_PATH),
    }
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    clean = build_clean_table(args.cnmf_root, args.signature_weights)
    if len(clean) != 144:
        raise RuntimeError(f"Expected 144 K* programmes, found {len(clean)}")
    output_path = write_outputs(clean, args.output_root)
    n_assigned = int((clean["mp1_8_name"] != "not_assigned_to_MP").sum())
    print(f"Wrote {len(clean)} K* rows to {output_path}", flush=True)
    print(f"MP-assigned rows: {n_assigned}; not assigned by design: {len(clean) - n_assigned}", flush=True)


if __name__ == "__main__":
    main()
