#!/usr/bin/env python
"""
Per-gene directional sign-test audit for C-SIDE outputs.

This analysis tests whether individual gene-cell-type associations show more
same-direction sample signs than expected from the cell-type-wide sign
background. It is intentionally separate from the set-level permutation test:
the unit here is one gene-cell-type association, so multiple-testing correction
is severe and the result is used as a per-gene calibration check.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


DEFAULT_ROOT = Path(r"D:\HGSOC_Spatial_Atlas\05_analysis_ready\S2e_CSIDE_CellTypeSpecific_DE_Audit")
DEFAULT_INPUT = DEFAULT_ROOT / "01_signed_ranking_audit" / "cside_2cov_signed_gene_level_all_samples.csv"
DEFAULT_OUTPUT = DEFAULT_ROOT / "08_per_gene_direction_sign_test"
MAIN_CELL_TYPES = ["Epithelial", "Fibroblast", "Macrophage", "CAF", "Endothelial"]


def bh_adjust(values: pd.Series) -> pd.Series:
    p = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return pd.Series(out, index=values.index)
    idx = np.where(valid)[0]
    order = idx[np.argsort(p[valid])]
    running = 1.0
    for rank in range(len(order), 0, -1):
        original_idx = order[rank - 1]
        running = min(running, p[original_idx] * len(order) / rank)
        out[original_idx] = min(running, 1.0)
    return pd.Series(out, index=values.index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--main-cell-types", nargs="+", default=MAIN_CELL_TYPES)
    return parser.parse_args()


def prepare_input(path: Path, main_cell_types: list[str]) -> pd.DataFrame:
    usecols = [
        "cell_type",
        "gene",
        "dataset",
        "sample_id_on_disk",
        "sample_label",
        "conv_bool",
        "signed_z",
    ]
    df = pd.read_csv(path, usecols=usecols)
    df = df.loc[df["cell_type"].isin(main_cell_types)].copy()
    df["signed_z"] = pd.to_numeric(df["signed_z"], errors="coerce")
    df = df.loc[df["conv_bool"].astype(str).str.lower().isin(["true", "1"])]
    df = df.loc[np.isfinite(df["signed_z"]) & (df["signed_z"] != 0)].copy()
    df["sign"] = np.sign(df["signed_z"]).astype(int)
    df["sample_key"] = df["dataset"].astype(str) + "__" + df["sample_id_on_disk"].astype(str)
    df["gene"] = df["gene"].astype(str).str.upper()
    return df


def exact_two_sided_binom(k: int, n: int, p: float) -> float:
    if n <= 0 or not np.isfinite(p) or p <= 0 or p >= 1:
        return np.nan
    return float(binomtest(k, n, p, alternative="two-sided").pvalue)


def build_gene_table(df: pd.DataFrame, min_samples: int) -> pd.DataFrame:
    bg = (
        df.groupby("cell_type")["sign"]
        .agg(total_signs="size", positive_signs=lambda s: int((s > 0).sum()))
        .reset_index()
    )
    bg["background_positive_fraction"] = bg["positive_signs"] / bg["total_signs"]
    bg_lookup = dict(zip(bg["cell_type"], bg["background_positive_fraction"]))

    rows = []
    for (cell_type, gene), subset in df.groupby(["cell_type", "gene"], sort=False):
        # One sign per sample-key. Duplicates are not expected; if present, keep
        # the first finite sign to keep the test at sample granularity.
        per_sample = subset.drop_duplicates("sample_key")
        n = int(len(per_sample))
        if n < min_samples:
            continue
        n_pos = int((per_sample["sign"] > 0).sum())
        n_neg = int((per_sample["sign"] < 0).sum())
        dominant = "positive" if n_pos >= n_neg else "negative"
        sign_consistency = max(n_pos, n_neg) / n
        p_bg = float(bg_lookup[cell_type])
        rows.append(
            {
                "cell_type": cell_type,
                "gene": gene,
                "n_samples": n,
                "n_datasets": int(per_sample["dataset"].nunique()),
                "n_positive": n_pos,
                "n_negative": n_neg,
                "dominant_direction": dominant,
                "sign_consistency": sign_consistency,
                "background_positive_fraction": p_bg,
                "p_empirical_background": exact_two_sided_binom(n_pos, n, p_bg),
                "p_binom_0_5": exact_two_sided_binom(n_pos, n, 0.5),
                "datasets": ";".join(sorted(per_sample["dataset"].unique())),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result["q_empirical_background_by_celltype"] = (
        result.groupby("cell_type", group_keys=False)["p_empirical_background"].apply(bh_adjust)
    )
    result["q_binom_0_5_by_celltype"] = (
        result.groupby("cell_type", group_keys=False)["p_binom_0_5"].apply(bh_adjust)
    )
    result["q_empirical_background_global"] = bh_adjust(result["p_empirical_background"])
    result["q_binom_0_5_global"] = bh_adjust(result["p_binom_0_5"])
    return result.sort_values(["q_empirical_background_global", "cell_type", "gene"]).reset_index(drop=True)


def summarize(
    result: pd.DataFrame,
    df: pd.DataFrame,
    min_samples: int,
    input_path: Path,
    main_cell_types: list[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    for cell_type, subset in result.groupby("cell_type", sort=False):
        n_ge_10 = subset.loc[subset["n_samples"] >= 10]
        top = subset.sort_values("q_empirical_background_global").head(1)
        rows.append(
            {
                "cell_type": cell_type,
                "n_gene_celltype_tests": int(len(subset)),
                "n_q_empirical_global_lt_0_05": int((subset["q_empirical_background_global"] < 0.05).sum()),
                "n_q_empirical_by_celltype_lt_0_05": int((subset["q_empirical_background_by_celltype"] < 0.05).sum()),
                "n_q_binom_0_5_global_lt_0_05": int((subset["q_binom_0_5_global"] < 0.05).sum()),
                "n_q_binom_0_5_by_celltype_lt_0_05": int((subset["q_binom_0_5_by_celltype"] < 0.05).sum()),
                "min_q_empirical_global": float(subset["q_empirical_background_global"].min()),
                "min_q_empirical_by_celltype": float(subset["q_empirical_background_by_celltype"].min()),
                "median_sign_consistency_all_tested": float(subset["sign_consistency"].median()),
                "median_sign_consistency_n_ge_10": float(n_ge_10["sign_consistency"].median()) if len(n_ge_10) else np.nan,
                "n_genes_n_ge_10": int(len(n_ge_10)),
                "top_gene_by_empirical_global_q": top.iloc[0]["gene"] if len(top) else "",
                "top_gene_empirical_global_q": float(top.iloc[0]["q_empirical_background_global"]) if len(top) else np.nan,
            }
        )

    summary = pd.DataFrame(rows).sort_values("cell_type")
    overall = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(input_path),
        "min_samples": min_samples,
        "main_cell_types": main_cell_types,
        "n_input_rows_after_filter": int(len(df)),
        "n_gene_celltype_tests": int(len(result)),
        "n_q_empirical_global_lt_0_05": int((result["q_empirical_background_global"] < 0.05).sum()),
        "n_q_empirical_by_celltype_lt_0_05": int((result["q_empirical_background_by_celltype"] < 0.05).sum()),
        "n_q_binom_0_5_global_lt_0_05": int((result["q_binom_0_5_global"] < 0.05).sum()),
        "n_q_binom_0_5_by_celltype_lt_0_05": int((result["q_binom_0_5_by_celltype"] < 0.05).sum()),
    }
    return summary, overall


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), args.output_dir / Path(__file__).name)

    df = prepare_input(args.input, args.main_cell_types)
    result = build_gene_table(df, args.min_samples)
    summary, overall = summarize(result, df, args.min_samples, args.input, args.main_cell_types)

    result.to_csv(args.output_dir / "per_gene_direction_sign_test_results.csv", index=False)
    summary.to_csv(args.output_dir / "per_gene_direction_sign_test_summary_by_celltype.csv", index=False)
    result.head(50).to_csv(args.output_dir / "per_gene_direction_sign_test_top50.csv", index=False)

    manifest = {
        **overall,
        "script": str(Path(__file__).resolve()),
        "output_dir": str(args.output_dir),
        "test_definition": {
            "unit": "gene-cell-type association",
            "signs": "finite non-zero signed_z signs from converged C-SIDE fits",
            "primary_null": "cell-type-wide empirical positive-sign fraction",
            "sensitivity_null": "p=0.5 binomial sign null",
            "multiplicity": "BH globally across main cell-type gene tests and separately within cell type",
        },
        "outputs": {
            "results": str(args.output_dir / "per_gene_direction_sign_test_results.csv"),
            "summary_by_celltype": str(args.output_dir / "per_gene_direction_sign_test_summary_by_celltype.csv"),
            "top50": str(args.output_dir / "per_gene_direction_sign_test_top50.csv"),
        },
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(overall, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
