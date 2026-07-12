# Supplementary Data Release Plan

Policy used for this staging pass:

- Small CSV/TSV/XLSX/JSON/TXT/MD files under 25 MiB are copied into `supplementary_data/`.
- Rendered figures are not copied by default; they are already in the supplement PDF and can be archived on Zenodo later.
- Large analysis objects such as `.h5ad`, `.rds`, archives, or files above normal GitHub limits are reserved for Zenodo or external source citation.
- Unresolved files require manual review before manuscript filepath replacement.

Small data files copied: 20
Zenodo-later/optional items: 110
Unresolved items: 0

Current decision: the thesis hand-in will use GitHub only for the curated code
archive and the small derived supplementary data files. Zenodo may still be used
later for larger optional archives or DOI minting, but it is not required for
the current GitHub package.
