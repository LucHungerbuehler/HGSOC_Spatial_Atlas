# Release Readiness Summary

Updated: 2026-07-12

This GitHub package is ready as a thesis code-availability archive. It is not
presented as a fully portable one-command reproduction pipeline.

## Included

- 89 curated analysis scripts.
- 20 small derived supplementary data files.
- Methods-to-scripts manifest, excluded-script decisions, hardcoded-path audit,
  checksums, and supplementary data manifest.
- Reader-facing supplementary data index.

## Safety Checks

- No staged source script is missing.
- No `.h5ad`, `.rds`, `.rda`, `.loom`, `.h5`, `.hdf5`, archive, or compressed
  large-object file is present in the GitHub package.
- No file in the GitHub package is larger than 25 MiB.
- `scripts/python/main.py` is not staged as the final thesis entry point.
- The supplement file references now use GitHub-relative `supplementary_data/...`
  paths rather than machine-specific analysis paths.

## Caveats Addressed

- Repository structure is documented in `README.md`.
- Important scripts are mapped in `manifests/methods_to_scripts_manifest.csv`
  and `docs/methods_to_scripts_run_order.md`.
- Absolute paths inside scripts are documented as analysis provenance.
- The historical `main.py` development orchestrator is explicitly not treated
  as the final thesis pipeline.
- Large raw and intermediate data objects are intentionally excluded.

## Remaining Manual Step

Before public upload, add a final repository license after supervisor/project
approval and replace any placeholder GitHub URL in the thesis text if needed.
