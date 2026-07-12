# Environment Notes

This repository was prepared as a thesis code-availability archive. The scripts
were not all run from one universal software environment, so the release now
includes exported specifications for the local environments that were referenced
by the curated script set.

The root `environment.yml` is retained as a lightweight historical baseline. The
fuller current export of that same local environment is
`envs/IMLp1.environment.yml`.

## Included Environment Files

| Analysis area | Environment file |
| --- | --- |
| General Python analysis and plotting baseline | `envs/IMLp1.environment.yml` |
| EnrichMap scoring, h5ad-facing utilities, and many spatial figure scripts | `envs/enrichmap_env.environment.yml` |
| cNMF programme extraction and sample-level cNMF execution | `envs/cnmf_env.environment.yml` |
| GASTON method-aligned feature preparation, model training, and gradient follow-up | `envs/gaston_env.environment.yml` |
| SpottedPy hotspot, neighborhood, and distance analyses | `envs/spottedpy_env.environment.yml` |
| RCTD/C-SIDE, ORA, and R meta-analysis scripts | `envs/R-4.4.3-version.txt` and `envs/R-4.4.3-package-versions.csv` |

## How These Were Selected

The environment set was chosen by scanning the released scripts and the
hardcoded-path audit for explicit interpreter/environment references. The
released scripts refer to `enrichmap_env`, `cnmf_env`, `gaston_env`, and
`spottedpy_env`; the baseline `IMLp1` environment was already present in the
archive. The R scripts and R-calling Python helpers point to local R 4.4.3 and
import the packages listed in `envs/R-4.4.3-package-versions.csv`.

Other conda environments existed on the local workstation, including `scvi_env`
and `napari-env`, but no script in this curated GitHub release referenced those
environment names. They are therefore not exported here.

## Reader Guidance

The purpose of this repository is to make the thesis code and small derived
supplementary data available. It is not presented as an executable
one-command reproduction pipeline. A reader trying to re-run a specific branch
should start from the script itself, the relevant manifest entry, the
environment file listed above, and the local path/provenance notes preserved in
`manifests/hardcoded_path_audit.csv`.

The conda exports were generated from the local environments at release
preparation time with build strings omitted. They are more informative than a
single baseline file, but they still do not remove the need to adapt data paths,
external data objects, and platform-specific packages.
