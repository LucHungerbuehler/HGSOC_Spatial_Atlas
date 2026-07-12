#!/usr/bin/env python
"""
Sample-level sign-permutation test for C-SIDE directional reproducibility.

This tests whether more gene-cell-type associations show reproducible
directionality than expected under sample-level sign exchangeability. It does
not test effect magnitude.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


MAIN_CELL_TYPES = ["Epithelial", "Fibroblast", "Macrophage", "CAF", "Endothelial"]
THRESHOLDS = [0.70, 0.80, 0.90]


DEFAULT_ROOT = Path(r"D:\HGSOC_Spatial_Atlas\05_analysis_ready\S2e_CSIDE_CellTypeSpecific_DE_Audit")
DEFAULT_INPUT = DEFAULT_ROOT / "01_signed_ranking_audit" / "cside_2cov_signed_gene_level_all_samples.csv"
DEFAULT_META = DEFAULT_ROOT / "01_signed_ranking_audit" / "meta_gene_effects_signed_stouffer_iv_random_effects.csv"
DEFAULT_OUTPUT = DEFAULT_ROOT / "06_setlevel_permutation"


@dataclass
class PreparedMatrix:
    sign_matrix: np.ndarray
    row_table: pd.DataFrame
    sample_table: pd.DataFrame
    n_samples: np.ndarray
    n_datasets: np.ndarray
    dataset_membership: np.ndarray
    coverage_mask: np.ndarray
    cell_type_codes: np.ndarray
    cell_type_names: list[str]
    zero_count: int
    nonfinite_count: int
    duplicate_pairs: int


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    q_values = np.empty(n, dtype=float)
    running = 1.0
    for rank_idx in range(n - 1, -1, -1):
        idx = order[rank_idx]
        running = min(running, p_values[idx] * n / (rank_idx + 1))
        q_values[idx] = min(running, 1.0)
    return q_values


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-perm", type=int, default=10_000)
    parser.add_argument("--probe-perm", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--max-probe-seconds", type=float, default=10.0)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--skip-probe", action="store_true")
    return parser.parse_args()


def load_and_prepare(input_path: Path) -> PreparedMatrix:
    usecols = ["cell_type", "gene", "dataset", "sample_id_on_disk", "sample_label", "signed_z"]
    df = pd.read_csv(input_path, usecols=usecols)
    df = df[df["cell_type"].isin(MAIN_CELL_TYPES)].copy()
    df["signed_z"] = pd.to_numeric(df["signed_z"], errors="coerce")

    nonfinite = ~np.isfinite(df["signed_z"].to_numpy(dtype=float))
    zero = np.isfinite(df["signed_z"].to_numpy(dtype=float)) & (df["signed_z"].to_numpy(dtype=float) == 0)
    nonfinite_count = int(nonfinite.sum())
    zero_count = int(zero.sum())

    valid = ~(nonfinite | zero)
    valid_df = df.loc[valid].copy()
    valid_df["sample_key"] = valid_df["dataset"].astype(str) + "__" + valid_df["sample_id_on_disk"].astype(str)
    valid_df["sign"] = np.where(valid_df["signed_z"] > 0, 1, -1).astype(np.int8)

    duplicate_pairs = int(
        valid_df.duplicated(subset=["cell_type", "gene", "sample_key"], keep=False).sum()
    )
    if duplicate_pairs:
        # Collapse duplicated rows deterministically by summing signs. A zero sum
        # means the duplicate entries disagree and the pair becomes absent.
        valid_df = (
            valid_df.groupby(["cell_type", "gene", "dataset", "sample_id_on_disk", "sample_label", "sample_key"], as_index=False)
            .agg(sign_sum=("sign", "sum"))
        )
        valid_df["sign"] = np.sign(valid_df["sign_sum"]).astype(np.int8)
        valid_df = valid_df[valid_df["sign"] != 0].copy()

    row_keys = valid_df["cell_type"].astype(str) + "\t" + valid_df["gene"].astype(str)
    row_codes, row_uniques = pd.factorize(row_keys, sort=True)
    sample_codes, sample_uniques = pd.factorize(valid_df["sample_key"], sort=True)

    row_split = pd.Series(row_uniques).str.split("\t", n=1, expand=True)
    row_table = pd.DataFrame({"cell_type": row_split[0], "gene": row_split[1]})

    sample_info = valid_df[["sample_key", "dataset", "sample_id_on_disk", "sample_label"]].drop_duplicates("sample_key")
    sample_info = sample_info.set_index("sample_key").loc[list(sample_uniques)].reset_index()

    sign_matrix = np.zeros((len(row_uniques), len(sample_uniques)), dtype=np.int8)
    sign_matrix[row_codes, sample_codes] = valid_df["sign"].to_numpy(dtype=np.int8)

    n_samples = (sign_matrix != 0).sum(axis=1).astype(np.int16)

    datasets = sample_info["dataset"].astype(str).to_numpy()
    dataset_names = sorted(sample_info["dataset"].astype(str).unique())
    dataset_membership = np.zeros((sign_matrix.shape[0], len(dataset_names)), dtype=bool)
    for i, dataset in enumerate(dataset_names):
        cols = datasets == dataset
        dataset_membership[:, i] = (sign_matrix[:, cols] != 0).any(axis=1)
    n_datasets = dataset_membership.sum(axis=1).astype(np.int8)

    coverage_mask = (n_samples >= 5) & (n_datasets == 3)

    cell_type_names = MAIN_CELL_TYPES + ["pooled"]
    cell_type_to_code = {ct: i for i, ct in enumerate(MAIN_CELL_TYPES)}
    cell_type_codes = row_table["cell_type"].map(cell_type_to_code).to_numpy(dtype=np.int8)

    row_table["n_samples"] = n_samples
    row_table["n_datasets"] = n_datasets
    row_table["coverage_pass"] = coverage_mask
    for i, dataset in enumerate(dataset_names):
        row_table[f"present_in_{dataset}"] = dataset_membership[:, i]

    return PreparedMatrix(
        sign_matrix=sign_matrix,
        row_table=row_table,
        sample_table=sample_info,
        n_samples=n_samples,
        n_datasets=n_datasets,
        dataset_membership=dataset_membership,
        coverage_mask=coverage_mask,
        cell_type_codes=cell_type_codes,
        cell_type_names=cell_type_names,
        zero_count=zero_count,
        nonfinite_count=nonfinite_count,
        duplicate_pairs=duplicate_pairs,
    )


def compute_observed_counts(prep: PreparedMatrix) -> tuple[dict[tuple[str, float], int], np.ndarray]:
    sign_matrix = prep.sign_matrix
    n_pos = (sign_matrix > 0).sum(axis=1)
    n_neg = (sign_matrix < 0).sum(axis=1)
    consistency = np.divide(
        np.maximum(n_pos, n_neg),
        prep.n_samples,
        out=np.zeros_like(prep.n_samples, dtype=float),
        where=prep.n_samples > 0,
    )
    prep.row_table["n_positive"] = n_pos
    prep.row_table["n_negative"] = n_neg
    prep.row_table["consistency"] = consistency

    counts: dict[tuple[str, float], int] = {}
    for threshold in THRESHOLDS:
        pass_mask = prep.coverage_mask & (consistency >= threshold)
        for ct in MAIN_CELL_TYPES:
            ct_mask = prep.row_table["cell_type"].to_numpy() == ct
            counts[(ct, threshold)] = int((pass_mask & ct_mask).sum())
        counts[("pooled", threshold)] = int(pass_mask.sum())
    return counts, consistency


def permutation_counts(
    prep: PreparedMatrix,
    n_perm: int,
    seed: int,
    batch_size: int,
) -> tuple[dict[tuple[str, float], np.ndarray], dict[str, object]]:
    rng = np.random.default_rng(seed)
    n_cols = prep.sign_matrix.shape[1]
    n_rows = prep.sign_matrix.shape[0]

    pos = (prep.sign_matrix > 0).astype(np.int16)
    neg = (prep.sign_matrix < 0).astype(np.int16)
    n_samples_col = prep.n_samples.astype(np.int16)[:, None]
    coverage_col = prep.coverage_mask[:, None]

    null_counts: dict[tuple[str, float], np.ndarray] = {
        (ct, threshold): np.zeros(n_perm, dtype=np.int32)
        for threshold in THRESHOLDS
        for ct in prep.cell_type_names
    }

    ct_masks = {ct: (prep.row_table["cell_type"].to_numpy() == ct) for ct in MAIN_CELL_TYPES}

    example_coverage_identical = None
    example_dataset_identical = None
    flips_permutation_min = n_cols
    flips_permutation_max = n_cols

    write_at = 0
    for start in range(0, n_perm, batch_size):
        batch_n = min(batch_size, n_perm - start)
        flips = rng.choice(np.array([-1, 1], dtype=np.int8), size=(batch_n, n_cols))
        flips_permutation_min = min(flips_permutation_min, int((np.abs(flips) == 1).sum(axis=1).min()))
        flips_permutation_max = max(flips_permutation_max, int((np.abs(flips) == 1).sum(axis=1).max()))

        flip_pos = (flips == 1).astype(np.int16).T
        flip_neg = (flips == -1).astype(np.int16).T
        n_pos_perm = (pos @ flip_pos) + (neg @ flip_neg)
        n_neg_perm = n_samples_col - n_pos_perm

        consistency = np.divide(
            np.maximum(n_pos_perm, n_neg_perm),
            n_samples_col,
            out=np.zeros((n_rows, batch_n), dtype=float),
            where=n_samples_col > 0,
        )

        if example_coverage_identical is None:
            example_coverage_identical = bool(np.array_equal(prep.n_samples, prep.n_samples))
            example_dataset_identical = bool(np.array_equal(prep.dataset_membership, prep.dataset_membership))

        end = write_at + batch_n
        for threshold in THRESHOLDS:
            pass_mask = coverage_col & (consistency >= threshold)
            null_counts[("pooled", threshold)][write_at:end] = pass_mask.sum(axis=0)
            for ct, ct_mask in ct_masks.items():
                null_counts[(ct, threshold)][write_at:end] = pass_mask[ct_mask, :].sum(axis=0)
        write_at = end

    sanity = {
        "n_permutations": n_perm,
        "n_unique_sample_keys": n_cols,
        "flips_permutation_min": flips_permutation_min,
        "flips_permutation_max": flips_permutation_max,
        "coverage_identical_example_permutation": example_coverage_identical,
        "dataset_membership_identical_example_permutation": example_dataset_identical,
    }
    return null_counts, sanity


def summarize_results(
    observed_counts: dict[tuple[str, float], int],
    null_counts: dict[tuple[str, float], np.ndarray],
    n_perm: int,
) -> pd.DataFrame:
    rows = []
    for threshold in THRESHOLDS:
        for ct in MAIN_CELL_TYPES + ["pooled"]:
            key = (ct, threshold)
            null = null_counts[key]
            obs = observed_counts[key]
            raw_p = (1 + int((null >= obs).sum())) / (n_perm + 1)
            rows.append(
                {
                    "cell_type": ct,
                    "threshold_T": threshold,
                    "observed_count": obs,
                    "null_mean": float(null.mean()),
                    "null_p05": percentile(null, 5),
                    "null_p95": percentile(null, 95),
                    "null_max": int(null.max()),
                    "raw_p": raw_p,
                }
            )
    result = pd.DataFrame(rows)
    result["bh_q"] = bh_adjust(result["raw_p"].to_numpy())
    return result


def load_manual_core(meta_path: Path) -> pd.DataFrame:
    meta = pd.read_csv(meta_path)
    core = meta[
        meta["cell_type"].isin(MAIN_CELL_TYPES)
        & (meta["random_q"] < 0.05)
        & (meta["n_samples"] >= 5)
        & (meta["n_datasets"] == 3)
        & (meta["sign_consistency"] >= 0.8)
    ].copy()
    return core


def print_probe(probe_elapsed: float, probe_perm: int, target_perm: int, sanity: dict[str, object]) -> None:
    extrapolated = probe_elapsed * target_perm / probe_perm
    print("TIMING PROBE")
    print(f"  probe_permutations: {probe_perm}")
    print(f"  elapsed_seconds: {probe_elapsed:.3f}")
    print(f"  extrapolated_{target_perm}_seconds: {extrapolated:.1f}")
    print(f"  extrapolated_{target_perm}_minutes: {extrapolated / 60:.2f}")
    print(f"  flips_permutation_min: {sanity['flips_permutation_min']}")
    print(f"  flips_permutation_max: {sanity['flips_permutation_max']}")


def save_outputs(
    output_dir: Path,
    prep: PreparedMatrix,
    observed_counts: dict[tuple[str, float], int],
    observed_consistency: np.ndarray,
    null_counts: dict[tuple[str, float], np.ndarray],
    result: pd.DataFrame,
    manual_core: pd.DataFrame,
    sanity: dict[str, object],
    args: argparse.Namespace,
    elapsed_full: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    result.to_csv(output_dir / "setlevel_reproducibility_permutation.csv", index=False)

    wide = pd.DataFrame({"permutation": np.arange(1, args.n_perm + 1)})
    for threshold in THRESHOLDS:
        for ct in MAIN_CELL_TYPES + ["pooled"]:
            col = f"{ct}__T{threshold:.2f}"
            wide[col] = null_counts[(ct, threshold)]
    wide.to_csv(output_dir / "setlevel_reproducibility_null_counts_wide.csv", index=False)

    long_parts = []
    for threshold in THRESHOLDS:
        for ct in MAIN_CELL_TYPES + ["pooled"]:
            long_parts.append(
                pd.DataFrame(
                    {
                        "permutation": np.arange(1, args.n_perm + 1),
                        "cell_type": ct,
                        "threshold_T": threshold,
                        "permuted_count": null_counts[(ct, threshold)],
                    }
                )
            )
    pd.concat(long_parts, ignore_index=True).to_csv(
        output_dir / "setlevel_reproducibility_null_counts_long.csv", index=False
    )

    observed_rows = prep.row_table.copy()
    observed_rows["observed_consistency"] = observed_consistency
    for threshold in THRESHOLDS:
        observed_rows[f"passes_T{threshold:.2f}"] = (
            observed_rows["coverage_pass"] & (observed_rows["observed_consistency"] >= threshold)
        )
    observed_rows.to_csv(output_dir / "setlevel_reproducibility_observed_gene_celltype_table.csv", index=False)

    manual_summary = manual_core.groupby("cell_type").size().reindex(MAIN_CELL_TYPES, fill_value=0).reset_index()
    manual_summary.columns = ["cell_type", "manual_randomq_core_count"]
    manual_summary.loc[len(manual_summary)] = ["pooled", int(len(manual_core))]
    manual_summary.to_csv(output_dir / "manual_randomq_core_T080_reference_counts.csv", index=False)

    shutil.copy2(Path(__file__), output_dir / "cside_setlevel_reproducibility_permutation.py")

    manifest = {
        "script": str(Path(__file__)),
        "input": str(args.input),
        "meta": str(args.meta),
        "output_dir": str(output_dir),
        "n_perm": args.n_perm,
        "probe_perm": args.probe_perm,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "main_cell_types": MAIN_CELL_TYPES,
        "thresholds": THRESHOLDS,
        "counted_unit": "gene-cell-type association",
        "pooled_definition": "sum over the five main cell-type-specific association counts",
        "zero_signed_z_excluded": prep.zero_count,
        "nonfinite_signed_z_excluded": prep.nonfinite_count,
        "duplicate_gene_celltype_sample_rows": prep.duplicate_pairs,
        "n_rows_gene_celltype": int(prep.sign_matrix.shape[0]),
        "n_unique_sample_keys": int(prep.sign_matrix.shape[1]),
        "elapsed_full_seconds": elapsed_full,
        "sanity": sanity,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    print(f"Input: {args.input}")
    print(f"Meta reference: {args.meta}")
    print(f"Output dir: {args.output_dir}")
    print(f"Seed: {args.seed}")
    print(f"Main cell types: {', '.join(MAIN_CELL_TYPES)}")
    print("Counted unit: gene-cell-type association")

    prep = load_and_prepare(args.input)
    print("DATA SUMMARY")
    print(f"  gene-cell-type rows: {prep.sign_matrix.shape[0]}")
    print(f"  unique sample keys: {prep.sign_matrix.shape[1]}")
    print(f"  zero signed_z excluded: {prep.zero_count}")
    print(f"  nonfinite signed_z excluded: {prep.nonfinite_count}")
    print(f"  duplicate gene-cell-type-sample rows detected: {prep.duplicate_pairs}")
    print(f"  coverage-pass rows (n_samples >= 5 and n_datasets == 3): {int(prep.coverage_mask.sum())}")

    observed_counts, observed_consistency = compute_observed_counts(prep)
    manual_core = load_manual_core(args.meta)

    print("OBSERVED COUNTS AT T=0.80 VS MANUAL RANDOM_Q CORE")
    manual_split = manual_core.groupby("cell_type").size().to_dict()
    for ct in MAIN_CELL_TYPES:
        print(
            f"  {ct}: sign-only observed={observed_counts[(ct, 0.80)]}, "
            f"manual_randomq_core={int(manual_split.get(ct, 0))}"
        )
    print(
        f"  pooled: sign-only observed={observed_counts[('pooled', 0.80)]}, "
        f"manual_randomq_core={len(manual_core)}"
    )

    if not args.skip_probe:
        probe_start = time.perf_counter()
        _, probe_sanity = permutation_counts(prep, args.probe_perm, args.seed, args.batch_size)
        probe_elapsed = time.perf_counter() - probe_start
        print_probe(probe_elapsed, args.probe_perm, args.n_perm, probe_sanity)
        if args.probe_only:
            return 0
        if probe_elapsed > args.max_probe_seconds:
            print(
                f"STOP: {args.probe_perm} permutations took {probe_elapsed:.3f}s, "
                f"above the {args.max_probe_seconds:.3f}s probe threshold.",
                file=sys.stderr,
            )
            return 2

    full_start = time.perf_counter()
    null_counts, sanity = permutation_counts(prep, args.n_perm, args.seed, args.batch_size)
    elapsed_full = time.perf_counter() - full_start
    result = summarize_results(observed_counts, null_counts, args.n_perm)

    print("FULL RUN")
    print(f"  permutations: {args.n_perm}")
    print(f"  elapsed_seconds: {elapsed_full:.3f}")
    print(f"  elapsed_minutes: {elapsed_full / 60:.2f}")
    print("SANITY CHECKS")
    for key, value in sanity.items():
        print(f"  {key}: {value}")

    print("RESULTS")
    display = result.copy()
    display["obs_over_null_mean"] = display["observed_count"] / display["null_mean"]
    print(
        display[
            [
                "cell_type",
                "threshold_T",
                "observed_count",
                "null_mean",
                "obs_over_null_mean",
                "null_p05",
                "null_p95",
                "null_max",
                "raw_p",
                "bh_q",
            ]
        ].to_string(index=False)
    )

    save_outputs(
        args.output_dir,
        prep,
        observed_counts,
        observed_consistency,
        null_counts,
        result,
        manual_core,
        sanity,
        args,
        elapsed_full,
    )
    print(f"Saved outputs to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
