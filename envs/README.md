# Environment Exports

These files document the local software environments used across the thesis
analysis scripts in this repository. They are included for code availability and
provenance, not as a promise that the full analysis can be rerun on another
machine without adapting paths and data locations.

## Conda Environments

| File | Local environment | Main use in this repository |
| --- | --- | --- |
| `IMLp1.environment.yml` | `IMLp1` | General Python analysis and plotting baseline. |
| `enrichmap_env.environment.yml` | `enrichmap_env` | EnrichMap scoring, h5ad-facing utilities, spatial transcriptomics plotting, and several thesis figure scripts. |
| `cnmf_env.environment.yml` | `cnmf_env` | cNMF programme extraction and sample-level cNMF execution. |
| `gaston_env.environment.yml` | `gaston_env` | GASTON method-aligned feature preparation, model training, and gradient follow-up scripts. |
| `spottedpy_env.environment.yml` | `spottedpy_env` | SpottedPy hotspot, neighborhood, and distance-analysis scripts. |

The root `environment.yml` is retained as a lightweight baseline for historical
compatibility. The fuller current export of that same baseline environment is
`envs/IMLp1.environment.yml`.

## R Environment

The R scripts were run with local R 4.4.3 rather than a conda environment.

- `R-4.4.3-version.txt`: detected local R version.
- `R-4.4.3-package-versions.csv`: package versions for R packages imported by
  the released R scripts and R-calling Python helpers.

## Scope Note

Other local conda environments existed on the workstation, including
`scvi_env` and `napari-env`, but no scripts in this GitHub release reference
those environments by name. They were therefore not exported into this archive.
