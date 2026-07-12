# Public Release Checklist

- Review `manifests/methods_to_scripts_manifest.csv`.
- Review `manifests/hardcoded_path_audit.csv` and confirm local paths are documented as provenance.
- Confirm the README states that this is a thesis code archive, not a one-command portable pipeline.
- Confirm `scripts/python/main.py` is not described as the final thesis orchestrator.
- Confirm that only lightweight supplementary data tables are committed under `supplementary_data/`.
- Keep raw Visium, scRNA reference objects, `.h5ad`, `.rds`, and large image archives outside GitHub.
- Add a final repository license after supervisor/project approval.
- Update manuscript/supplement file references to GitHub-relative `supplementary_data/...` paths.
