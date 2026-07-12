"""
Build expanded Step 4 threshold diagnostics for 10-100% malignant fractions.

This wrapper keeps the original 50-90% diagnostic outputs untouched and writes
the expanded source tables/figures into a separate subfolder.
"""

from __future__ import annotations

import build_s2b_step4_threshold_diagnostics as base


OUTPUT_SUBDIR = "threshold_10_to_100_positions"


def load_full_h5ad_sample(dataset: str, sample: str, lineages: list[str]):
    h5 = base.READY / "visium" / dataset / sample / f"{sample}.h5ad"
    if not h5.exists():
        raise FileNotFoundError(h5)

    a = base.ad.read_h5ad(h5, backed="r")
    obs = a.obs.copy()
    a.file.close()
    obs = obs.reset_index(names="spot_id")

    hallmark_cols = [c for c in obs.columns if c.startswith("HALLMARK_") and c.endswith("_score")]
    required = list(dict.fromkeys(["spot_id", "SNAI1-ac_score", "Malignant", *lineages]))
    missing = [c for c in required if c not in obs.columns]
    if missing:
        raise RuntimeError(f"Missing expected h5ad obs columns for {dataset}/{sample}: {missing}")

    obs_cols = required + hallmark_cols + [c for c in base.QC_COLS if c in obs.columns and c not in required]
    df = obs[obs_cols].copy()
    df["dataset"] = dataset
    df["sample"] = sample
    df["sample_label"] = f"{dataset}__{sample}"
    return df


def main() -> None:
    base.THRESHOLDS = [i / 10 for i in range(1, 11)]
    base.OUT = base.OUT / OUTPUT_SUBDIR
    base.load_sample = load_full_h5ad_sample
    base.main()


if __name__ == "__main__":
    main()
