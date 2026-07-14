"""Build report-facing GASTON supplementary tables from the promoted analysis branch."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
GASTON_ROOT = Path(
    os.environ.get(
        "HGSOC_GASTON_ROOT",
        r"D:\HGSOC_Spatial_Atlas\05_analysis_ready\GASTON_method_v1",
    )
)
OUT_DIR = REPO_ROOT / "supplementary_data" / "gaston"
NATIVE_ROOT = GASTON_ROOT / "07_gradient_review" / "02_gaston_native_gradient_identity"
TABLE_DIR = NATIVE_ROOT / "tables"
ROOT_MANIFEST = REPO_ROOT / "supplementary_data_manifest.csv"
CHECKSUM_FILE = REPO_ROOT / "checksums_sha256.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def observed_fitted_metrics() -> pd.DataFrame:
    manifest = pd.read_csv(
        GASTON_ROOT / "04_isodepth_score_alignment" / "sample_alignment_manifest.csv"
    )
    records: list[dict[str, object]] = []

    for row in manifest.itertuples(index=False):
        stem = f"{row.dataset}__{row.sample}__whole__{row.feature_method}"
        alignment = pd.read_csv(row.alignment_table)
        mapping = pd.read_csv(TABLE_DIR / f"{stem}__domain_order_mapping.csv")
        fits = pd.read_csv(TABLE_DIR / f"{stem}__snai1ac_score_piecewise_fits.csv")
        domain_to_segment = dict(
            zip(
                mapping["original_domain_label"].astype(int),
                mapping["gradient_segment"].astype(int),
            )
        )
        alignment["gradient_segment"] = (
            alignment["gaston_domain_selected"].astype(int).map(domain_to_segment)
        )

        for score_column, score_fits in fits.groupby("score_column", sort=False):
            parameters = score_fits.set_index("gradient_segment")[["intercept", "slope"]]
            predicted = np.asarray(
                [
                    parameters.loc[int(segment), "intercept"]
                    + parameters.loc[int(segment), "slope"] * isodepth
                    for segment, isodepth in zip(
                        alignment["gradient_segment"],
                        alignment["gaston_isodepth_malignant_oriented"],
                    )
                ],
                dtype=float,
            )
            observed = pd.to_numeric(alignment[score_column], errors="coerce").to_numpy()
            finite = np.isfinite(observed) & np.isfinite(predicted)
            observed = observed[finite]
            predicted = predicted[finite]
            pearson_r = float(np.corrcoef(observed, predicted)[0, 1])
            tss = float(np.sum((observed - observed.mean()) ** 2))
            rss = float(np.sum((observed - predicted) ** 2))
            slopes = pd.to_numeric(score_fits["slope"], errors="coerce").to_numpy()

            records.append(
                {
                    "dataset": row.dataset,
                    "sample": row.sample,
                    "feature_method": row.feature_method,
                    "analysis_tier": row.analysis_tier,
                    "selected_k": int(row.selected_k),
                    "score_column": score_column,
                    "score_label": score_fits["score_label"].iloc[0],
                    "n_spots": int(finite.sum()),
                    "observed_vs_fitted_pearson_r": pearson_r,
                    "in_sample_r2": float(1 - rss / tss),
                    "n_positive_domain_slopes": int((slopes > 0).sum()),
                    "n_negative_domain_slopes": int((slopes < 0).sum()),
                    "n_zero_domain_slopes": int((slopes == 0).sum()),
                    "both_positive_and_negative_slopes": bool(
                        (slopes > 0).any() and (slopes < 0).any()
                    ),
                }
            )

    return pd.DataFrame.from_records(records).sort_values(
        ["dataset", "sample", "score_column"]
    )


def gene_gradient_sample_summary() -> pd.DataFrame:
    manifest = pd.read_csv(NATIVE_ROOT / "gene_gradient_manifest.csv")
    signature_counts: list[int] = []
    classified_counts: list[int] = []

    for row in manifest.itertuples(index=False):
        stem = f"{row.dataset}__{row.sample}__{row.layer}__{row.feature_method}"
        patterns = pd.read_csv(TABLE_DIR / f"{stem}__gene_gradient_patterns.csv.gz")
        is_signature = patterns["is_snai1ac_signature_gene"].astype(bool)
        is_classified = (patterns["n_continuous_segments"] > 0) | (
            patterns["n_discontinuous_boundaries"] > 0
        )
        signature_counts.append(int(is_signature.sum()))
        classified_counts.append(int((is_signature & is_classified).sum()))

    output = manifest.copy()
    output["n_snai1ac_signature_genes_passing_filter"] = signature_counts
    output["n_snai1ac_signature_genes_classified"] = classified_counts
    return output.sort_values(["dataset", "sample"])


def signature_gradient_class_summary() -> pd.DataFrame:
    paths = sorted(TABLE_DIR.glob("*__gradient_identity_class_summary.csv"))
    output = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    return output.sort_values(["dataset", "sample", "class_type", "class_id"])


def update_root_manifest() -> None:
    manifest = pd.read_csv(ROOT_MANIFEST, dtype=str).fillna("")
    manifest = manifest[
        ~manifest["proposed_repo_path"].str.startswith("supplementary_data/gaston/")
    ].copy()
    source_lookup = {
        "gradient_identity_evidence.csv": (
            GASTON_ROOT
            / "07_gradient_review"
            / "01_gradient_identity_review"
            / "gradient_identity_evidence.csv"
        ),
        "cohort_relationship_summary.csv": (
            GASTON_ROOT / "05_relationship_review" / "cohort_relationship_summary.csv"
        ),
    }
    context_lookup = {
        "gradient_identity_evidence.csv": ("1552", "supp:table17_gradient_identity"),
        "cohort_relationship_summary.csv": ("1558", "subsec:supp_gaston_score_diagnostics"),
        "domainwise_score_fit_summary.csv": ("1560", "subsec:supp_gaston_score_diagnostics"),
        "gene_gradient_sample_summary.csv": ("1572", "subsec:supp_gaston_gene_gradient"),
        "signature_gradient_class_summary.csv": ("1574", "subsec:supp_gaston_gene_gradient"),
    }
    rows = []
    for name, (line, context) in context_lookup.items():
        release_path = OUT_DIR / name
        source_path = source_lookup.get(name, release_path)
        repo_path = f"supplementary_data/gaston/{name}"
        rows.append(
            {
                "line": line,
                "kind": "path",
                "label_context": context,
                "raw_reference": repo_path,
                "normalized_reference": repo_path,
                "resolved_path": str(source_path),
                "exists": "True",
                "size_bytes": str(release_path.stat().st_size),
                "sha256": sha256(release_path),
                "extension": release_path.suffix,
                "release_decision": "github_small_data",
                "proposed_repo_path": repo_path,
                "notes": (
                    "Report-facing GASTON supplementary output rebuilt from "
                    "GASTON_method_v1 by scripts/python/build_gaston_supplementary_data.py."
                ),
            }
        )
    updated = pd.concat([manifest, pd.DataFrame(rows)], ignore_index=True)
    updated.to_csv(ROOT_MANIFEST, index=False)


def update_checksums() -> None:
    records = {}
    for path in REPO_ROOT.rglob("*"):
        if (
            not path.is_file()
            or ".git" in path.parts
            or "__pycache__" in path.parts
            or path.suffix == ".pyc"
            or path == CHECKSUM_FILE
        ):
            continue
        records[path.relative_to(REPO_ROOT).as_posix()] = sha256(path)

    existing_order = []
    if CHECKSUM_FILE.exists():
        for line in CHECKSUM_FILE.read_text(encoding="ascii").splitlines():
            parts = line.split("  ", maxsplit=1)
            if len(parts) == 2 and parts[1] in records:
                existing_order.append(parts[1])
    new_paths = sorted(set(records) - set(existing_order), key=str.lower)
    ordered_paths = existing_order + new_paths
    CHECKSUM_FILE.write_text(
        "".join(f"{records[path]}  {path}\n" for path in ordered_paths),
        encoding="ascii",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_files = {
        "gradient_identity_evidence.csv": (
            GASTON_ROOT
            / "07_gradient_review"
            / "01_gradient_identity_review"
            / "gradient_identity_evidence.csv"
        ),
        "cohort_relationship_summary.csv": (
            GASTON_ROOT / "05_relationship_review" / "cohort_relationship_summary.csv"
        ),
    }
    for name, source in source_files.items():
        shutil.copy2(source, OUT_DIR / name)

    tables = {
        "domainwise_score_fit_summary.csv": observed_fitted_metrics(),
        "gene_gradient_sample_summary.csv": gene_gradient_sample_summary(),
        "signature_gradient_class_summary.csv": signature_gradient_class_summary(),
    }
    for name, table in tables.items():
        table.to_csv(OUT_DIR / name, index=False)

    outputs = []
    for path in sorted(OUT_DIR.glob("*.csv")):
        outputs.append(
            {
                "file": path.name,
                "rows": int(len(pd.read_csv(path))),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest_path = OUT_DIR / "gaston_supplementary_data_manifest.json"
    manifest_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    update_root_manifest()
    update_checksums()
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
