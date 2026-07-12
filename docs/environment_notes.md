# Environment Notes

This repository was prepared as a thesis code-availability archive. The scripts
were not all run from one universal software environment.

The included `environment.yml` records a lightweight baseline Python environment
from the local project checkout. It should not be read as a complete lockfile for
all analyses in this archive.

## Known Environment Split

| Analysis area | Environment notes |
| --- | --- |
| General Python analysis and plotting | Most pandas/numpy/scipy/sklearn/matplotlib/seaborn workflows can be interpreted from the included baseline `environment.yml`, but exact local package versions were not frozen for every script. |
| EnrichMap scoring and many h5ad-facing figure scripts | Several thesis runs were performed in the local `enrichmap_env` conda environment. This environment was used because it had the spatial transcriptomics and plotting stack needed for EnrichMap-derived score work. |
| cNMF programme extraction | cNMF execution was routed through a separate local `cnmf_env`; this is visible in `manifests/hardcoded_path_audit.csv` for `s3_tumor_cnmf_pipeline.py`. |
| GASTON model training and method-aligned runs | GASTON scripts were run in a separate local `gaston_env`; this is visible in `manifests/hardcoded_path_audit.csv` for `gaston_method_aligned_run.py`. |
| RCTD/C-SIDE and R meta-analysis scripts | R scripts require an R/Bioconductor setup rather than the Python `environment.yml`. Relevant scripts are under `scripts/R/`. |
| SpottedPy analyses | SpottedPy-related scripts depend on a Python environment with the SpottedPy stack and its spatial-statistics dependencies installed. |

## Reader Guidance

The purpose of this repository is to make the thesis code and small derived
supplementary data available. It is not presented as an executable environment
specification. A reader trying to re-run a specific branch should start from the
script itself, the relevant manifest entry, and the local path/provenance notes
preserved in `manifests/hardcoded_path_audit.csv`.

The most important point is that no single `python` or `conda env create`
command should be expected to reproduce every branch in this archive.
