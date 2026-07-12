# Script Release Audit Summary

Updated: 2026-07-12

The GitHub-first package now stages 89 scripts. This set was rebuilt after a stricter audit against both the Results/Supplement manuscript graphics and the local branch/run-manifest evidence.

## Audit Results

- `methods_to_scripts_manifest.csv`: 89 scripts, 0 missing source files.
- `results_branch_script_coverage.csv`: all true branch-level script references are covered; remaining branch hits are resource/regex false positives.
- `results_figure_to_script_links.csv`: 156 manuscript graphics traced to local output paths and candidate scripts.
- `results_figure_unresolved_script_candidates.csv`: 0 unresolved missing script candidates.
- `excluded_script_decisions.csv`: documented exclusions for exploratory, superseded, presentation-only, or non-authoritative review scripts.

## Main Additions Beyond The First Pass

- S2b ILR and malignant-threshold figure builders.
- cNMF ridge/model-fit, univariate, HH-state, family-top-gene, final MP, and supplementary figure builders.
- K* annotation supplement and GO-audited MP enrichment scripts.
- SpottedPy distance design, SpottedPy-native GEE plotting, and all-in-one heatmap refresh scripts.
- Final excluded-script pass additions: tumor-only Hallmark analysis, the upstream Definition 3b / raw Definition 4 cNMF pipeline, and its shared `analysis_utils.py` helper.

## Release Scope Caveats

The package intentionally still contains local absolute paths. `hardcoded_path_audit.csv` currently reports 147 path hits, retained as provenance for the local thesis analysis environment. This GitHub package is a code-availability archive with small derived supplementary data tables, not a fully portable one-command rerun pipeline.

The historical `scripts/python/main.py` development orchestrator is not the controlling entry point for the final thesis analyses. Use `manifests/methods_to_scripts_manifest.csv` and `docs/methods_to_scripts_run_order.md` as the authoritative script map.

Multiple local Python/R environments were used across the thesis analyses. See `docs/environment_notes.md`; the included `environment.yml` is a partial baseline and not a complete lockfile for every script.
