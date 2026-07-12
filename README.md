# HGSOC Spatial Atlas Thesis Code Archive

This repository contains the analysis code and lightweight supplementary data
tables for a Master thesis on SNAI1-ac-associated spatial transcriptomic
patterns in high-grade serous ovarian cancer.

The repository is intended as a transparent code-availability archive. It is
not a polished one-command reproduction pipeline and is not expected to run
unchanged on another machine. Many scripts preserve the local absolute paths
used during the thesis analyses; these paths are provenance for where the
reported outputs were generated, not an accidental omission.

## What Is Included

- `scripts/python/` and `scripts/R/`: curated thesis-facing analysis scripts.
- `supplementary_data/`: small derived tables and workbooks referenced by the
  thesis supplement.
- `manifests/methods_to_scripts_manifest.csv`: maps Methods/Results areas to
  the included scripts.
- `supplementary_data_manifest.csv`: maps supplement table references to copied
  GitHub-relative files and original local source paths.
- `docs/methods_to_scripts_run_order.md`: high-level map of scripts by analysis
  section.
- `docs/supplementary_data_index.md`: reader-facing index for Supplementary
  Tables S1-S17.
- `docs/environment_notes.md`: notes on the multiple Python/R environments used
  during analysis.

## What Is Not Included

Raw Visium data, large `.h5ad`/`.rds` objects, rendered figure archives, and
large intermediate analysis folders are not included in this GitHub archive.
The repository therefore documents the analysis code and small derived
supplementary tables, while larger data dependencies remain external to this
code archive.

## Main Caveats

- The old `scripts/python/main.py` development orchestrator is not the official
  final thesis pipeline. The final analyses are represented by the curated
  script list in `manifests/methods_to_scripts_manifest.csv`.
- Absolute paths inside scripts are intentionally retained as provenance.
- Multiple local software environments were used; the included `environment.yml`
  is a partial baseline, not a complete lockfile for every script.
- Scripts were developed over the course of the thesis and some are support,
  sensitivity, audit, or figure-building scripts rather than independent
  primary analyses.
- The small supplementary data files in `supplementary_data/` are derived
  outputs, not raw patient-level sequencing data.

For the most compact entry points, start with:

1. `docs/supplementary_data_index.md`
2. `docs/methods_to_scripts_run_order.md`
3. `docs/environment_notes.md`
4. `manifests/methods_to_scripts_manifest.csv`
5. `docs/script_release_audit_summary.md`
