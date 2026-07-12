"""
Phase 1 and Phase 2 runner for tumor-only Visium cNMF.

Current workflow:
1. Build a real cohort manifest from processed and analysis-ready D: objects.
2. Export stripped tumor-only counts-only AnnData inputs from the rich Visium h5ad files.
3. Run per-sample cNMF first pass for K-selection review.
4. After manual K decisions are recorded, run consensus export and cross-sample analysis.

The script is intentionally log-heavy and stdout-light so long cNMF subprocesses do not
flood the terminal. Rich Visium h5ad files are read in the current Python environment,
while cNMF is executed through the dedicated cnmf_env interpreter.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import textwrap
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity

try:
    import libpysal
    from esda.moran import Moran
    SPATIAL_STATS_AVAILABLE = True
except Exception:
    libpysal = None
    Moran = None
    SPATIAL_STATS_AVAILABLE = False

try:
    from statsmodels.stats.multitest import multipletests
except Exception:
    multipletests = None


BASE_DIR = Path(r"D:\HGSOC_Spatial_Atlas")
PROCESSED_DIR = BASE_DIR / "02_processed_data" / "visium"
VISIUM_DIR = BASE_DIR / "05_analysis_ready" / "visium"
METADATA_DIR = BASE_DIR / "03_metadata" / "visium"
ROBUSTNESS_CACHE_DIR = BASE_DIR / "05_analysis_ready" / "Signature" / "robustness" / "h5ad_cache" / "visium"
OUTPUT_DIR = BASE_DIR / "05_analysis_ready" / "S3_cNMF_Tumor_Programs"

SAMPLE_MANIFEST_DIR = OUTPUT_DIR / "sample_manifests"
INPUT_DIR = OUTPUT_DIR / "inputs"
K_SELECTION_DIR = OUTPUT_DIR / "k_selection"
PER_SAMPLE_DIR = OUTPUT_DIR / "per_sample"
META_DIR = OUTPUT_DIR / "meta"
WORKSPACE_DIR = OUTPUT_DIR / "cnmf_runs"
LOG_DIR = OUTPUT_DIR / "logs"
SIMILARITY_DIR = META_DIR / "similarity"
CLUSTERING_DIR = META_DIR / "clustering"
RETENTION_DIR = META_DIR / "retention"
CONSENSUS_DIR = META_DIR / "consensus"
SIGNATURE_OVERLAP_DIR = META_DIR / "signature_overlap"
USAGE_DIR = META_DIR / "usage"
SPATIAL_DIR = META_DIR / "spatial"

ROOT_SAMPLE_MANIFEST_PATH = OUTPUT_DIR / "sample_manifest.csv"
MANUAL_K_CSV_PATH = K_SELECTION_DIR / "manual_k_decisions.csv"
MANUAL_K_XLSX_PATH = K_SELECTION_DIR / "manual_k_decisions.xlsx"
PARSED_MANUAL_K_PATH = K_SELECTION_DIR / "parsed_manual_k_decisions.csv"
MANUAL_K_VALIDATION_PATH = K_SELECTION_DIR / "manual_k_validation_report.csv"
README_POSTK_PATH = OUTPUT_DIR / "README_postK_summary.md"
DOC_PATH = Path(r"C:\Users\luchu\Documents\MSc\Master Thesis\Code\HGSOC_Spatial_Atlas\00_documentation\cNMF_postK_strategy_and_rationale.md")
SIGNATURE_DIR = BASE_DIR / "05_analysis_ready" / "Signature"
ORIGINAL_SIGNATURE_WEIGHTS_PATH = SIGNATURE_DIR / "snai1_ac_weights.json"
ROBUSTNESS_WEIGHTS_DIR = SIGNATURE_DIR / "robustness" / "weights"

ANALYSIS_DATASETS = ["denisenko_2022", "ju_2024", "yamamoto_2025"]

CNMF_PYTHON = Path(r"C:\Users\luchu\anaconda3\envs\cnmf_env\python.exe")
CNMF_SCRIPT = Path(r"C:\Users\luchu\anaconda3\envs\cnmf_env\Lib\site-packages\cnmf\cnmf.py")

INTERFACE_COL = "interface"
TUMOR_LABEL = "Tumor"
HIGH_PURITY_THRESHOLD = 0.75
MIN_GENE_SPOTS = 3
NUM_HIGHVAR_GENES = 2000
RANDOM_STATE = 42
K_VALUES = list(range(4, 13))
N_ITER = 100
LOCAL_DENSITY_THRESHOLD = 0.5
LOCAL_NEIGHBORHOOD_SIZE = 0.30
COSINE_CLUSTER_THRESHOLD = 0.35
JACCARD_TOP30_THRESHOLD = 0.20
JACCARD_TOP50_THRESHOLD = 0.15
JACCARD_TOP100_THRESHOLD = 0.10
REPRESENTATIVE_SPATIAL_K = 6
CONSENSUS_TOP_GENE_COUNT = 50
SIGNATURE_VARIANTS = [
    ("original", ORIGINAL_SIGNATURE_WEIGHTS_PATH),
    ("pc_thresholded", ROBUSTNESS_WEIGHTS_DIR / "snai1_ac_pc_thresholded.json"),
    ("pc_up", ROBUSTNESS_WEIGHTS_DIR / "snai1_ac_pc_up.json"),
]
SIGNATURE_SCORE_COLUMNS = {
    "original": "SNAI1-ac_score",
    "pc_thresholded": "SNAI1_ac_pc_thresholded_score",
    "pc_up": "SNAI1_ac_pc_up_score",
}

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tumor-only cNMF pipeline for Visium cohorts")
    parser.add_argument(
        "--stage",
        choices=["manifest", "pre_k", "k_review", "post_k"],
        default="pre_k",
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--samples",
        nargs="*",
        default=None,
        help="Optional sample filters using either sample_id or dataset__sample_id",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of eligible samples to process",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild outputs even when target files already exist",
    )
    parser.add_argument(
        "--k-decisions",
        type=Path,
        default=None,
        help="CSV file with manual K decisions for the post_k stage",
    )
    return parser.parse_args()


def ensure_dirs() -> None:
    for directory in [
        OUTPUT_DIR,
        SAMPLE_MANIFEST_DIR,
        INPUT_DIR,
        K_SELECTION_DIR,
        PER_SAMPLE_DIR,
        META_DIR,
        SIMILARITY_DIR,
        CLUSTERING_DIR,
        RETENTION_DIR,
        CONSENSUS_DIR,
        SIGNATURE_OVERLAP_DIR,
        USAGE_DIR,
        SPATIAL_DIR,
        WORKSPACE_DIR,
        LOG_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def parse_expected_samples(config_path: Path) -> list[str]:
    if not config_path.exists():
        return []
    samples: list[str] = []
    in_expected_block = False
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("expected_samples:"):
            in_expected_block = True
            continue
        if not in_expected_block:
            continue
        if not line or line.startswith("#"):
            continue
        if line.startswith("- "):
            samples.append(line[2:].strip().strip('"').strip("'"))
            continue
        break
    return samples


def config_name_for_sample(dataset: str, processed_sample: str, expected_samples: list[str]) -> tuple[str, str]:
    if processed_sample in expected_samples:
        return processed_sample, "exact_match"
    if dataset == "ju_2024":
        return "", "unresolved_config_mismatch"
    return "", "not_listed_in_config"


def processed_h5ad_path(dataset: str, sample_id: str) -> Path:
    return PROCESSED_DIR / dataset / f"{sample_id}.h5ad"


def analysis_ready_h5ad_path(dataset: str, sample_id: str) -> Path:
    return VISIUM_DIR / dataset / sample_id / f"{sample_id}.h5ad"


def robustness_h5ad_path(dataset: str, sample_id: str) -> Path:
    return ROBUSTNESS_CACHE_DIR / dataset / f"{sample_id}.h5ad"


def discover_sample_ids(dataset: str) -> list[str]:
    processed_dir = PROCESSED_DIR / dataset
    analysis_ready_dir = VISIUM_DIR / dataset

    processed_ids = []
    if processed_dir.exists():
        processed_ids = [path.stem for path in processed_dir.glob("*.h5ad")]

    analysis_ready_ids = []
    if analysis_ready_dir.exists():
        for sample_dir in sorted(p for p in analysis_ready_dir.iterdir() if p.is_dir()):
            sample_h5ad = sample_dir / f"{sample_dir.name}.h5ad"
            if sample_h5ad.exists():
                analysis_ready_ids.append(sample_dir.name)

    return sorted(set(processed_ids) | set(analysis_ready_ids))


def exact_tumor_masks(obs: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if INTERFACE_COL not in obs.columns:
        empty = pd.Series(False, index=obs.index)
        return empty, empty

    primary_mask = obs[INTERFACE_COL].astype(str) == TUMOR_LABEL
    if "Malignant" not in obs.columns:
        return primary_mask, pd.Series(False, index=obs.index)

    malignant = pd.to_numeric(obs["Malignant"], errors="coerce")
    high_purity_mask = primary_mask & (malignant >= HIGH_PURITY_THRESHOLD)
    return primary_mask, high_purity_mask


def finite_fraction(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    finite = np.isfinite(values)
    if finite.sum() == 0:
        return float("nan")
    return float(np.mean(values[finite]))


def build_cohort_manifest() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for dataset in ANALYSIS_DATASETS:
        expected_samples = parse_expected_samples(METADATA_DIR / dataset / "config.yaml")
        for sample_id in discover_sample_ids(dataset):
            processed_path = processed_h5ad_path(dataset, sample_id)
            analysis_ready_path = analysis_ready_h5ad_path(dataset, sample_id)
            robust_path = robustness_h5ad_path(dataset, sample_id)
            sample_label = f"{dataset}__{sample_id}"
            config_sample_name, config_status = config_name_for_sample(dataset, sample_id, expected_samples)

            processed_exists = processed_path.exists()
            analysis_ready_exists = analysis_ready_path.exists()
            robust_exists = robust_path.exists()
            raw_counts_layer_available = False
            spacet_interface_available = False
            malignant_available = False
            tumor_spots_primary = 0
            tumor_spots_high_purity = 0

            if analysis_ready_exists:
                adata = ad.read_h5ad(analysis_ready_path)
                raw_counts_layer_available = "counts" in adata.layers
                spacet_interface_available = INTERFACE_COL in adata.obs.columns
                malignant_available = "Malignant" in adata.obs.columns
                primary_mask, high_purity_mask = exact_tumor_masks(adata.obs)
                tumor_spots_primary = int(primary_mask.sum())
                tumor_spots_high_purity = int(high_purity_mask.sum())

            exclusion_reasons: list[str] = []
            if not analysis_ready_exists:
                exclusion_reasons.append("missing analysis-ready h5ad")
            if analysis_ready_exists and not raw_counts_layer_available:
                exclusion_reasons.append("missing raw counts layer")
            if analysis_ready_exists and not spacet_interface_available:
                exclusion_reasons.append("missing SpaCET interface labels")
            if analysis_ready_exists and spacet_interface_available and tumor_spots_primary == 0:
                exclusion_reasons.append('zero spots with interface == "Tumor"')

            eligible_for_cnmf = len(exclusion_reasons) == 0
            incomplete_downstream = processed_exists and not analysis_ready_exists

            rows.append(
                {
                    "dataset": dataset,
                    "sample_id_on_disk": sample_id,
                    "sample_label": sample_label,
                    "metadata_config_sample_name": config_sample_name,
                    "config_match_status": config_status,
                    "processed_h5ad_exists": processed_exists,
                    "analysis_ready_h5ad_exists": analysis_ready_exists,
                    "robustness_h5ad_exists": robust_exists,
                    "raw_counts_layer_available": raw_counts_layer_available,
                    "spacet_interface_available": spacet_interface_available,
                    "malignant_available": malignant_available,
                    "tumor_spots_primary": tumor_spots_primary,
                    "tumor_spots_high_purity": tumor_spots_high_purity,
                    "eligible_for_cnmf": eligible_for_cnmf,
                    "incomplete_downstream": incomplete_downstream,
                    "reason_for_exclusion": "; ".join(exclusion_reasons),
                    "processed_h5ad_path": str(processed_path),
                    "analysis_ready_h5ad_path": str(analysis_ready_path),
                    "robustness_h5ad_path": str(robust_path),
                }
            )

    manifest = pd.DataFrame(rows).sort_values(["dataset", "sample_id_on_disk"]).reset_index(drop=True)
    manifest.to_csv(SAMPLE_MANIFEST_DIR / "cohort_manifest.csv", index=False)
    manifest.to_csv(ROOT_SAMPLE_MANIFEST_PATH, index=False)
    return manifest


def select_samples(manifest: pd.DataFrame, requested: list[str] | None, max_samples: int | None) -> pd.DataFrame:
    selected = manifest.copy()
    if requested:
        requested_set = set(requested)
        keep_mask = selected["sample_id_on_disk"].isin(requested_set) | selected["sample_label"].isin(requested_set)
        selected = selected.loc[keep_mask].copy()
    selected = selected.loc[selected["eligible_for_cnmf"]].copy()
    if max_samples is not None:
        selected = selected.head(max_samples).copy()
    return selected.reset_index(drop=True)


def matrix_detected_in_spots(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray((matrix > 0).sum(axis=0)).ravel()
    return np.asarray((matrix > 0).sum(axis=0)).ravel()


def export_tumor_input(sample_row: pd.Series, force: bool) -> dict[str, object]:
    dataset = str(sample_row["dataset"])
    sample_id = str(sample_row["sample_id_on_disk"])
    sample_label = str(sample_row["sample_label"])
    source_path = Path(sample_row["analysis_ready_h5ad_path"])

    input_path = INPUT_DIR / f"{sample_label}__tumor_counts_minimal.h5ad"
    tumor_manifest_path = SAMPLE_MANIFEST_DIR / f"{sample_label}__tumor_spots.csv"
    gene_manifest_path = SAMPLE_MANIFEST_DIR / f"{sample_label}__gene_filter_summary.csv"

    if input_path.exists() and tumor_manifest_path.exists() and gene_manifest_path.exists() and not force:
        gene_summary = pd.read_csv(gene_manifest_path)
        return {
            "dataset": dataset,
            "sample_id_on_disk": sample_id,
            "sample_label": sample_label,
            "input_h5ad": str(input_path),
            "tumor_manifest": str(tumor_manifest_path),
            "n_tumor_spots": int(gene_summary.loc[0, "n_tumor_spots"]),
            "n_genes_pre_filter": int(gene_summary.loc[0, "n_genes_pre_filter"]),
            "n_genes_post_filter": int(gene_summary.loc[0, "n_genes_post_filter"]),
        }

    adata = ad.read_h5ad(source_path)
    primary_mask, high_purity_mask = exact_tumor_masks(adata.obs)
    primary_mask_np = primary_mask.to_numpy()

    counts = adata.layers["counts"] if "counts" in adata.layers else adata.X
    tumor_counts = counts[primary_mask_np, :]
    detected_in_spots = matrix_detected_in_spots(tumor_counts)
    nonzero_total = np.asarray(tumor_counts.sum(axis=0)).ravel() > 0
    gene_mask = nonzero_total & (detected_in_spots >= MIN_GENE_SPOTS)

    tumor_obs = adata.obs.loc[primary_mask].copy()
    tumor_obs = tumor_obs.assign(
        dataset=dataset,
        sample_id_on_disk=sample_id,
        sample_label=sample_label,
        primary_tumor=True,
        high_purity_tumor=high_purity_mask.loc[primary_mask].to_numpy(),
    )
    tumor_obs.index.name = "spot_id"

    keep_obs_cols = [
        "dataset",
        "sample_id_on_disk",
        "sample_label",
        "primary_tumor",
        "high_purity_tumor",
        INTERFACE_COL,
        "Malignant",
        "total_counts",
        "n_genes_by_counts",
        "SNAI1-ac_score",
    ]
    keep_obs_cols = [col for col in keep_obs_cols if col in tumor_obs.columns]
    tumor_obs.loc[:, keep_obs_cols].to_csv(tumor_manifest_path)

    tumor_var = adata.var.loc[gene_mask].copy()
    slim_adata = ad.AnnData(
        X=tumor_counts[:, gene_mask].copy(),
        obs=tumor_obs.loc[:, keep_obs_cols].copy(),
        var=tumor_var.copy(),
    )
    slim_adata.write_h5ad(input_path)

    gene_summary = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "sample_id_on_disk": sample_id,
                "sample_label": sample_label,
                "input_h5ad": str(input_path),
                "n_tumor_spots": int(primary_mask.sum()),
                "n_high_purity_tumor_spots": int(high_purity_mask.sum()),
                "n_genes_pre_filter": int(adata.n_vars),
                "n_genes_post_filter": int(gene_mask.sum()),
                "min_gene_spots": MIN_GENE_SPOTS,
            }
        ]
    )
    gene_summary.to_csv(gene_manifest_path, index=False)

    return {
        "dataset": dataset,
        "sample_id_on_disk": sample_id,
        "sample_label": sample_label,
        "input_h5ad": str(input_path),
        "tumor_manifest": str(tumor_manifest_path),
        "n_tumor_spots": int(primary_mask.sum()),
        "n_genes_pre_filter": int(adata.n_vars),
        "n_genes_post_filter": int(gene_mask.sum()),
    }


def cnmf_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(THREAD_ENV)
    return env


def run_logged_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write("\n=== COMMAND ===\n")
        log_handle.write(" ".join(command) + "\n")
        log_handle.write("=== OUTPUT ===\n")
        log_handle.flush()
        subprocess.run(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=True,
            env=cnmf_env(),
        )


def workspace_sample_dir(sample_label: str) -> Path:
    return WORKSPACE_DIR / sample_label


def k_selection_npz_path(sample_label: str) -> Path:
    return workspace_sample_dir(sample_label) / f"{sample_label}.k_selection_stats.df.npz"


def k_selection_png_path(sample_label: str) -> Path:
    return workspace_sample_dir(sample_label) / f"{sample_label}.k_selection.png"


def load_df_from_npz(npz_path: Path) -> pd.DataFrame:
    with np.load(npz_path, allow_pickle=True) as data:
        return pd.DataFrame(data=data["data"], index=data["index"], columns=data["columns"])


def run_cnmf_first_pass(sample_label: str, input_h5ad: Path, force: bool) -> None:
    metrics_csv = K_SELECTION_DIR / f"{sample_label}_k_selection_metrics.csv"
    metrics_png = K_SELECTION_DIR / f"{sample_label}_k_selection_plot.png"
    template_md = K_SELECTION_DIR / f"{sample_label}_manual_decision_template.md"
    if metrics_csv.exists() and metrics_png.exists() and template_md.exists() and not force:
        return

    base_args = [
        str(CNMF_PYTHON),
        str(CNMF_SCRIPT),
    ]

    common_args = [
        "--output-dir",
        str(WORKSPACE_DIR),
        "--name",
        sample_label,
    ]

    prepare_cmd = base_args + [
        "prepare",
        *common_args,
        "-c",
        str(input_h5ad),
        "-k",
        *[str(k) for k in K_VALUES],
        "-n",
        str(N_ITER),
        "--numgenes",
        str(NUM_HIGHVAR_GENES),
        "--seed",
        str(RANDOM_STATE),
    ]
    factorize_cmd = base_args + ["factorize", *common_args]
    combine_cmd = base_args + ["combine", *common_args]
    kplot_cmd = base_args + ["k_selection_plot", *common_args]

    run_logged_command(prepare_cmd, LOG_DIR / f"{sample_label}__prepare.log")
    run_logged_command(factorize_cmd, LOG_DIR / f"{sample_label}__factorize.log")
    run_logged_command(combine_cmd, LOG_DIR / f"{sample_label}__combine.log")
    run_logged_command(kplot_cmd, LOG_DIR / f"{sample_label}__k_selection.log")


def write_manual_decision_template(sample_row: pd.Series, metrics_df: pd.DataFrame, template_path: Path) -> None:
    metric_lines = []
    for _, row in metrics_df.iterrows():
        metric_lines.append(
            f"| {int(row['k'])} | {row['silhouette']:.4f} | {row['prediction_error']:.4f} |"
        )

    body = textwrap.dedent(
        f"""
        # Manual K Review: {sample_row['sample_label']}

        Dataset: `{sample_row['dataset']}`
        Sample ID: `{sample_row['sample_id_on_disk']}`
        Tumor spots: `{int(sample_row['tumor_spots_primary'])}`
        High-purity tumor spots: `{int(sample_row['tumor_spots_high_purity'])}`
        Local density threshold used for K-selection stats: `2.0` (cNMF internal skip-density mode)
        Consensus density threshold planned for final runs: `{LOCAL_DENSITY_THRESHOLD}`

        ## Review Checklist

        - Inspect the K-selection plot for a stable silhouette plateau versus reconstruction error decline.
        - Prefer K values that avoid obvious over-fragmentation into tiny or redundant programs.
        - Later confirm biological interpretability after consensus export before treating a program as robust.
        - Record the final chosen `K` in a CSV for the `post_k` stage.

        ## K-selection Metrics

        | K | silhouette | prediction_error |
        |---|------------|------------------|
        {os.linesep.join(metric_lines)}

        ## Decision

        - Final chosen K:
        - Rationale:
        - Notes on borderline alternatives:
        """
    ).strip() + "\n"
    template_path.write_text(body, encoding="utf-8")


def materialize_k_selection_outputs(sample_row: pd.Series, force: bool) -> None:
    sample_label = str(sample_row["sample_label"])
    metrics_npz = k_selection_npz_path(sample_label)
    plot_png = k_selection_png_path(sample_label)
    metrics_csv = K_SELECTION_DIR / f"{sample_label}_k_selection_metrics.csv"
    copied_plot = K_SELECTION_DIR / f"{sample_label}_k_selection_plot.png"
    template_md = K_SELECTION_DIR / f"{sample_label}_manual_decision_template.md"

    if not metrics_npz.exists():
        raise FileNotFoundError(f"Missing cNMF k-selection stats: {metrics_npz}")
    if not plot_png.exists():
        raise FileNotFoundError(f"Missing cNMF k-selection plot: {plot_png}")

    if force or not metrics_csv.exists():
        metrics_df = load_df_from_npz(metrics_npz)
        metrics_df = metrics_df.assign(
            dataset=str(sample_row["dataset"]),
            sample_id_on_disk=str(sample_row["sample_id_on_disk"]),
            sample_label=sample_label,
        )
        metrics_df.to_csv(metrics_csv, index=False)
    else:
        metrics_df = pd.read_csv(metrics_csv)

    if force or not copied_plot.exists():
        shutil.copy2(plot_png, copied_plot)

    if force or not template_md.exists():
        write_manual_decision_template(sample_row, metrics_df, template_md)


def run_pre_k(manifest: pd.DataFrame, force: bool) -> None:
    selected = manifest.copy()
    if selected.empty:
        print("No eligible samples selected for cNMF.")
        return

    exported_rows = []
    for _, sample_row in selected.iterrows():
        export_summary = export_tumor_input(sample_row, force=force)
        exported_rows.append(export_summary)
        print(
            f"Prepared tumor input for {export_summary['sample_label']} "
            f"({export_summary['n_tumor_spots']} tumor spots, "
            f"{export_summary['n_genes_post_filter']} genes after filter)"
        )
        run_cnmf_first_pass(
            sample_label=str(sample_row["sample_label"]),
            input_h5ad=Path(str(export_summary["input_h5ad"])),
            force=force,
        )
        materialize_k_selection_outputs(sample_row, force=force)
        print(f"Completed first-pass cNMF K-selection artifacts for {sample_row['sample_label']}")

    pd.DataFrame(exported_rows).to_csv(SAMPLE_MANIFEST_DIR / "tumor_input_summary.csv", index=False)


def k_metrics_csv_path(sample_label: str) -> Path:
    return K_SELECTION_DIR / f"{sample_label}_k_selection_metrics.csv"


def k_plot_output_path(sample_label: str) -> Path:
    return K_SELECTION_DIR / f"{sample_label}_k_selection_plot.png"


def k_template_output_path(sample_label: str) -> Path:
    return K_SELECTION_DIR / f"{sample_label}_manual_decision_template.md"


def compute_elbow_k(metrics_df: pd.DataFrame) -> int:
    ks = metrics_df["k"].astype(int).to_numpy()
    errors = metrics_df["prediction_error"].astype(float).to_numpy()
    if len(ks) < 3:
        return int(ks[np.argmin(errors)])

    log_errors = np.log(errors)
    second_diff = log_errors[:-2] - (2.0 * log_errors[1:-1]) + log_errors[2:]
    elbow_idx = int(np.argmax(second_diff)) + 1
    return int(ks[elbow_idx])


def relative_error_gains(metrics_df: pd.DataFrame) -> pd.Series:
    errors = metrics_df["prediction_error"].astype(float)
    prev = errors.shift(1)
    gains = (prev - errors) / prev * 100.0
    return gains


def build_k_review_outputs(selected_manifest: pd.DataFrame) -> None:
    review_rows: list[dict[str, object]] = []
    wide_rows: list[dict[str, object]] = []
    silhouette_heatmap_rows: list[np.ndarray] = []
    error_heatmap_rows: list[np.ndarray] = []
    ordered_sample_labels: list[str] = []
    missing_samples: list[str] = []

    selected_manifest = selected_manifest.sort_values(
        ["dataset", "tumor_spots_primary", "sample_id_on_disk"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    for _, sample_row in selected_manifest.iterrows():
        sample_label = str(sample_row["sample_label"])
        metrics_path = k_metrics_csv_path(sample_label)
        plot_path = k_plot_output_path(sample_label)
        template_path = k_template_output_path(sample_label)

        if not metrics_path.exists():
            missing_samples.append(sample_label)
            continue

        metrics_df = pd.read_csv(metrics_path).sort_values("k").reset_index(drop=True)
        gains = relative_error_gains(metrics_df)
        ks = metrics_df["k"].astype(int)
        sil = metrics_df["silhouette"].astype(float)
        err = metrics_df["prediction_error"].astype(float)

        max_silhouette = float(sil.max())
        best_silhouette_k = int(metrics_df.loc[sil.idxmax(), "k"])
        silhouette_95_smallest_k = int(ks.loc[sil >= 0.95 * max_silhouette].iloc[0])
        silhouette_98_smallest_k = int(ks.loc[sil >= 0.98 * max_silhouette].iloc[0])
        elbow_k = compute_elbow_k(metrics_df)
        min_error_k = int(metrics_df.loc[err.idxmin(), "k"])
        shortlist = sorted({best_silhouette_k, silhouette_98_smallest_k, elbow_k})

        review_rows.append(
            {
                "dataset": sample_row["dataset"],
                "sample_id_on_disk": sample_row["sample_id_on_disk"],
                "sample_label": sample_label,
                "tumor_spots_primary": int(sample_row["tumor_spots_primary"]),
                "tumor_spots_high_purity": int(sample_row["tumor_spots_high_purity"]),
                "k_best_silhouette": best_silhouette_k,
                "max_silhouette": max_silhouette,
                "k_smallest_within_95pct_max_silhouette": silhouette_95_smallest_k,
                "k_smallest_within_98pct_max_silhouette": silhouette_98_smallest_k,
                "k_error_elbow_log_curve": elbow_k,
                "k_min_prediction_error": min_error_k,
                "prediction_error_drop_pct_k4_to_k12": float((err.iloc[0] - err.iloc[-1]) / err.iloc[0] * 100.0),
                "review_shortlist": ";".join(str(k) for k in shortlist),
                "plot_filename": plot_path.name,
                "template_filename": template_path.name,
            }
        )

        wide_row: dict[str, object] = {
            "dataset": sample_row["dataset"],
            "sample_id_on_disk": sample_row["sample_id_on_disk"],
            "sample_label": sample_label,
            "tumor_spots_primary": int(sample_row["tumor_spots_primary"]),
        }
        for idx, k in enumerate(ks):
            wide_row[f"silhouette_k{k}"] = float(sil.iloc[idx])
            wide_row[f"prediction_error_k{k}"] = float(err.iloc[idx])
            if idx > 0:
                wide_row[f"relative_error_gain_pct_k{k}"] = float(gains.iloc[idx])
        wide_rows.append(wide_row)

        ordered_sample_labels.append(sample_label)
        silhouette_heatmap_rows.append(sil.to_numpy(dtype=float))
        if float(err.max()) > float(err.min()):
            error_scaled = ((err - err.min()) / (err.max() - err.min())).to_numpy(dtype=float)
        else:
            error_scaled = np.zeros(len(err), dtype=float)
        error_heatmap_rows.append(error_scaled)

    if missing_samples:
        raise FileNotFoundError(
            "Missing K-selection metrics for: " + ", ".join(missing_samples)
        )

    review_df = pd.DataFrame(review_rows).sort_values(
        ["dataset", "tumor_spots_primary", "sample_id_on_disk"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    wide_df = pd.DataFrame(wide_rows).sort_values(
        ["dataset", "tumor_spots_primary", "sample_id_on_disk"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    review_df.to_csv(K_SELECTION_DIR / "k_selection_review_summary.csv", index=False)
    wide_df.to_csv(K_SELECTION_DIR / "k_selection_review_wide.csv", index=False)

    heuristic_counts = []
    for column in [
        "k_best_silhouette",
        "k_smallest_within_95pct_max_silhouette",
        "k_smallest_within_98pct_max_silhouette",
        "k_error_elbow_log_curve",
    ]:
        counts = review_df[column].value_counts().sort_index()
        for k_value, count in counts.items():
            heuristic_counts.append(
                {
                    "heuristic": column,
                    "k": int(k_value),
                    "count_samples": int(count),
                }
            )
    heuristic_counts_df = pd.DataFrame(heuristic_counts).sort_values(["heuristic", "k"])
    heuristic_counts_df.to_csv(K_SELECTION_DIR / "k_selection_review_heuristic_counts.csv", index=False)

    ks = [int(k) for k in K_VALUES]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, max(8, 0.45 * len(ordered_sample_labels))),
        constrained_layout=True,
    )

    silhouette_matrix = np.vstack(silhouette_heatmap_rows)
    error_matrix = np.vstack(error_heatmap_rows)

    im1 = axes[0].imshow(silhouette_matrix, aspect="auto", cmap="viridis")
    axes[0].set_title("cNMF K-selection silhouette by sample")
    axes[0].set_yticks(np.arange(len(ordered_sample_labels)))
    axes[0].set_yticklabels(ordered_sample_labels, fontsize=8)
    axes[0].set_xticks(np.arange(len(ks)))
    axes[0].set_xticklabels(ks)
    axes[0].set_xlabel("K")
    fig.colorbar(im1, ax=axes[0], fraction=0.025, pad=0.02)

    im2 = axes[1].imshow(error_matrix, aspect="auto", cmap="magma_r")
    axes[1].set_title("Within-sample scaled prediction error by sample (0 = best)")
    axes[1].set_yticks(np.arange(len(ordered_sample_labels)))
    axes[1].set_yticklabels(ordered_sample_labels, fontsize=8)
    axes[1].set_xticks(np.arange(len(ks)))
    axes[1].set_xticklabels(ks)
    axes[1].set_xlabel("K")
    fig.colorbar(im2, ax=axes[1], fraction=0.025, pad=0.02)
    fig.savefig(K_SELECTION_DIR / "k_selection_review_heatmaps.png", dpi=250)
    plt.close(fig)

    heuristic_table_lines = []
    for heuristic, sub_df in heuristic_counts_df.groupby("heuristic"):
        heuristic_table_lines.append(f"### `{heuristic}`")
        heuristic_table_lines.append("")
        heuristic_table_lines.append("| K | Count |")
        heuristic_table_lines.append("|---|-------|")
        for _, row in sub_df.iterrows():
            heuristic_table_lines.append(f"| {int(row['k'])} | {int(row['count_samples'])} |")
        heuristic_table_lines.append("")

    per_sample_lines = [
        "| Dataset | Sample | Tumor spots | Best silhouette K | 98% silhouette K | Error elbow K | Shortlist | Plot | Template |",
        "|---------|--------|-------------|-------------------|------------------|---------------|-----------|------|----------|",
    ]
    for _, row in review_df.iterrows():
        per_sample_lines.append(
            f"| {row['dataset']} | {row['sample_id_on_disk']} | {int(row['tumor_spots_primary'])} | "
            f"{int(row['k_best_silhouette'])} | {int(row['k_smallest_within_98pct_max_silhouette'])} | "
            f"{int(row['k_error_elbow_log_curve'])} | {row['review_shortlist']} | "
            f"{row['plot_filename']} | {row['template_filename']} |"
        )

    note = textwrap.dedent(
        f"""
        # Cross-sample K Review Sheet

        Eligible samples reviewed: `{len(review_df)}`
        Datasets included: `{", ".join(sorted(review_df['dataset'].unique()))}`

        This sheet is a review aid, not a final K caller. The shortlist per sample is the union of:
        - `k_best_silhouette`
        - `k_smallest_within_98pct_max_silhouette`
        - `k_error_elbow_log_curve`

        Files generated alongside this note:
        - `k_selection_review_summary.csv`
        - `k_selection_review_wide.csv`
        - `k_selection_review_heuristic_counts.csv`
        - `k_selection_review_heatmaps.png`

        ## Heuristic Count Tables

        {os.linesep.join(heuristic_table_lines).rstrip()}

        ## Per-sample Review Table

        {os.linesep.join(per_sample_lines)}
        """
    ).strip() + "\n"
    (K_SELECTION_DIR / "k_selection_review_summary.md").write_text(note, encoding="utf-8")


def manual_k_input_path(k_decisions_path: Path | None) -> Path:
    candidates = [k_decisions_path, MANUAL_K_CSV_PATH, MANUAL_K_XLSX_PATH]
    for candidate in candidates:
        if candidate is not None and Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError(
        "Could not find manual K decisions file. Expected either "
        f"{MANUAL_K_CSV_PATH} or {MANUAL_K_XLSX_PATH}."
    )


def normalize_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def parse_k_window_values(raw_value: object) -> list[int]:
    if pd.isna(raw_value):
        return []
    tokens = [token.strip() for token in str(raw_value).replace(",", ";").split(";")]
    values = sorted({int(token) for token in tokens if token})
    return values


def read_manual_k_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported manual K decisions format: {path}")


def cnmf_text_paths(sample_label: str, k_value: int) -> dict[str, Path]:
    density_tag = str(LOCAL_DENSITY_THRESHOLD).replace(".", "_")
    sample_dir = workspace_sample_dir(sample_label)
    return {
        "usage": sample_dir / f"{sample_label}.usages.k_{k_value}.dt_{density_tag}.consensus.txt",
        "spectra_score": sample_dir / f"{sample_label}.gene_spectra_score.k_{k_value}.dt_{density_tag}.txt",
        "spectra_tpm": sample_dir / f"{sample_label}.gene_spectra_tpm.k_{k_value}.dt_{density_tag}.txt",
        "consensus_spectra": sample_dir / f"{sample_label}.spectra.k_{k_value}.dt_{density_tag}.consensus.txt",
    }


def merged_spectra_npz_path(sample_label: str, k_value: int) -> Path:
    return workspace_sample_dir(sample_label) / "cnmf_tmp" / f"{sample_label}.spectra.k_{k_value}.merged.df.npz"


def consensus_outputs_exist(sample_label: str, k_value: int) -> bool:
    expected = cnmf_text_paths(sample_label, k_value)
    return all(path.exists() for path in expected.values())


def read_tab_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", index_col=0)
    return df


def load_signature_gene_sets() -> dict[str, set[str]]:
    gene_sets: dict[str, set[str]] = {}
    for variant_name, weight_path in SIGNATURE_VARIANTS:
        with open(weight_path, "r", encoding="utf-8") as handle:
            weights = json.load(handle)
        positive_genes = {str(gene) for gene, weight in weights.items() if float(weight) > 0}
        gene_sets[variant_name] = positive_genes
    return gene_sets


def parse_and_validate_manual_k(
    manifest: pd.DataFrame,
    k_decisions_path: Path | None,
) -> pd.DataFrame:
    manual_path = manual_k_input_path(k_decisions_path)
    manual_df = read_manual_k_table(manual_path).copy()
    manual_df.columns = [str(col).strip() for col in manual_df.columns]

    required_columns = [
        "dataset",
        "sample_id",
        "include_sample",
        "k_star",
        "k_window_values",
        "rationale_short",
        "notes",
    ]
    missing_columns = [col for col in required_columns if col not in manual_df.columns]
    if missing_columns:
        raise ValueError(f"Manual K decisions file is missing columns: {missing_columns}")

    manual_df = manual_df.assign(
        dataset=manual_df["dataset"].astype(str).str.strip(),
        sample_id=manual_df["sample_id"].astype(str).str.strip(),
        include_sample=manual_df["include_sample"].apply(normalize_bool),
        k_star=pd.to_numeric(manual_df["k_star"], errors="coerce"),
        k_window_values_raw=manual_df["k_window_values"].astype(str).str.strip(),
        rationale_short=manual_df["rationale_short"].fillna("").astype(str),
        notes=manual_df["notes"].fillna("").astype(str),
    )
    manual_df["sample_label"] = manual_df["dataset"] + "__" + manual_df["sample_id"]
    manual_df["k_window_list"] = manual_df["k_window_values"].apply(parse_k_window_values)
    manual_df["k_window_values"] = manual_df["k_window_list"].apply(lambda vals: ";".join(str(v) for v in vals))
    manual_df["k_star"] = manual_df["k_star"].astype("Int64")

    manifest_lookup = manifest.rename(columns={"sample_id_on_disk": "sample_id"})
    merged = manual_df.merge(
        manifest_lookup,
        on=["dataset", "sample_id", "sample_label"],
        how="left",
        suffixes=("", "_manifest"),
    )

    validation_rows: list[dict[str, object]] = []
    has_error = False
    included_rows = []

    for _, row in merged.iterrows():
        exists_in_manifest = not pd.isna(row.get("eligible_for_cnmf"))
        k_window_list = row["k_window_list"]
        k_star = int(row["k_star"]) if not pd.isna(row["k_star"]) else None
        include_sample = bool(row["include_sample"])

        missing_k_outputs: list[int] = []
        if include_sample and exists_in_manifest:
            for k_value in k_window_list:
                merged_path = merged_spectra_npz_path(str(row["sample_label"]), k_value)
                if not merged_path.exists():
                    missing_k_outputs.append(k_value)

        errors: list[str] = []
        if not exists_in_manifest:
            errors.append("sample not found in real cohort manifest")
        if include_sample and exists_in_manifest and not bool(row["eligible_for_cnmf"]):
            errors.append("sample is not eligible for cNMF in real cohort manifest")
        if include_sample and (k_star is None):
            errors.append("missing k_star")
        if include_sample and not k_window_list:
            errors.append("empty k_window_values")
        if include_sample and k_star is not None and k_star not in k_window_list:
            errors.append("k_star is not a member of k_window_values")
        if include_sample and missing_k_outputs:
            errors.append("missing pre-K cNMF outputs for some K values")

        validation_rows.append(
            {
                "dataset": row["dataset"],
                "sample_id": row["sample_id"],
                "sample_label": row["sample_label"],
                "include_sample": include_sample,
                "k_star": k_star,
                "k_window_values": ";".join(str(v) for v in k_window_list),
                "exists_in_manifest": exists_in_manifest,
                "eligible_for_cnmf": bool(row["eligible_for_cnmf"]) if exists_in_manifest else False,
                "analysis_ready_h5ad_exists": bool(row["analysis_ready_h5ad_exists"]) if exists_in_manifest else False,
                "valid_k_star_in_window": (k_star in k_window_list) if k_star is not None else False,
                "missing_k_outputs": ";".join(str(v) for v in missing_k_outputs),
                "status": "valid" if not errors else "error",
                "message": "; ".join(errors),
            }
        )

        if errors:
            has_error = True
        if include_sample and not errors:
            included_rows.append(row)

    parsed_manual = merged.copy()
    parsed_manual = parsed_manual.rename(columns={"sample_id": "sample_id_on_disk"})
    parsed_manual["k_star"] = parsed_manual["k_star"].astype("Int64")
    parsed_manual["k_window_n"] = parsed_manual["k_window_list"].apply(len)
    parsed_manual["manual_k_source_path"] = str(manual_path)
    parsed_manual["manual_k_source_format"] = manual_path.suffix.lower()
    parsed_manual["k_window_values"] = parsed_manual["k_window_list"].apply(lambda vals: ";".join(str(v) for v in vals))
    parsed_manual.to_csv(PARSED_MANUAL_K_PATH, index=False)
    pd.DataFrame(validation_rows).to_csv(MANUAL_K_VALIDATION_PATH, index=False)
    parsed_manual.to_csv(ROOT_SAMPLE_MANIFEST_PATH, index=False)

    if has_error:
        raise RuntimeError(
            "Manual K validation failed. See "
            f"{MANUAL_K_VALIDATION_PATH} for details."
        )

    included_df = pd.DataFrame(included_rows).copy()
    if included_df.empty:
        raise RuntimeError("Manual K decisions contain no valid included samples.")

    included_df = included_df.rename(columns={"sample_id": "sample_id_on_disk"})
    included_df["k_star"] = included_df["k_star"].astype(int)
    included_df["k_window_values"] = included_df["k_window_list"].apply(lambda vals: ";".join(str(v) for v in vals))
    return included_df.reset_index(drop=True)


def sample_output_dir(sample_id: str) -> Path:
    path = PER_SAMPLE_DIR / sample_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_program_id(sample_label: str, k_value: int, local_program_index: int) -> str:
    return f"{sample_label}__K{k_value}__P{local_program_index}"


def run_cnmf_consensus(sample_label: str, k_value: int, force: bool) -> None:
    if consensus_outputs_exist(sample_label, k_value) and not force:
        return
    command = [
        str(CNMF_PYTHON),
        str(CNMF_SCRIPT),
        "consensus",
        "--output-dir",
        str(WORKSPACE_DIR),
        "--name",
        sample_label,
        "-k",
        str(k_value),
        "--local-density-threshold",
        str(LOCAL_DENSITY_THRESHOLD),
        "--local-neighborhood-size",
        str(LOCAL_NEIGHBORHOOD_SIZE),
    ]
    run_logged_command(command, LOG_DIR / f"{sample_label}__consensus_k{k_value}.log")


def safe_int_label(value) -> int:
    try:
        return int(value)
    except Exception:
        return int(str(value).replace("GEP", "").replace(".0", ""))


def top_gene_list(scores: pd.Series, n_top: int) -> list[str]:
    ordered = scores.sort_values(ascending=False)
    return [str(gene) for gene in ordered.index[:n_top]]


def top_gene_row(
    metadata: dict[str, object],
    gene_list: list[str],
    n_top: int,
) -> dict[str, object]:
    row = dict(metadata)
    for rank in range(n_top):
        row[f"gene_{rank + 1}"] = gene_list[rank] if rank < len(gene_list) else ""
    return row


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def extract_programs_for_sample(
    sample_row: pd.Series,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset = str(sample_row["dataset"])
    sample_id = str(sample_row["sample_id_on_disk"])
    sample_label = str(sample_row["sample_label"])
    k_star = int(sample_row["k_star"])
    k_window_list = list(sample_row["k_window_list"])
    output_dir = sample_output_dir(sample_id)

    spectra_output = output_dir / "extracted_program_spectra.csv"
    top30_output = output_dir / "extracted_program_top30.csv"
    top50_output = output_dir / "extracted_program_top50.csv"
    top100_output = output_dir / "extracted_program_top100.csv"
    usage_output = output_dir / "representative_usage_kstar.csv"
    summary_output = output_dir / "extraction_summary.json"

    if all(path.exists() for path in [spectra_output, top30_output, top50_output, top100_output, usage_output, summary_output]) and not force:
        spectra_df = pd.read_csv(spectra_output)
        metadata_df = spectra_df.loc[:, [col for col in spectra_df.columns if not col.startswith("__gene__")]].copy()
        metadata_df = metadata_df[
            [
                "program_id",
                "dataset",
                "sample_id_on_disk",
                "sample_label",
                "source_k",
                "local_program_index",
                "is_k_star",
            ]
        ].copy()
        score_df = spectra_df.set_index("program_id").loc[:, [col for col in spectra_df.columns if col.startswith("__gene__")]].copy()
        score_df.columns = [col.replace("__gene__", "") for col in score_df.columns]
        usage_df = pd.read_csv(usage_output).set_index("spot_id")
        usage_df = usage_df.loc[:, [col for col in usage_df.columns if col not in {"dataset", "sample_id_on_disk", "sample_label"}]]
        return metadata_df, score_df, usage_df

    metadata_rows: list[dict[str, object]] = []
    top30_rows: list[dict[str, object]] = []
    top50_rows: list[dict[str, object]] = []
    top100_rows: list[dict[str, object]] = []
    score_blocks: list[pd.DataFrame] = []
    representative_usage_df: pd.DataFrame | None = None

    for k_value in k_window_list:
        run_cnmf_consensus(sample_label, k_value, force=force)
        paths = cnmf_text_paths(sample_label, k_value)
        score_df = read_tab_matrix(paths["spectra_score"])
        usage_df = read_tab_matrix(paths["usage"])
        score_df.index = [safe_int_label(idx) for idx in score_df.index]
        usage_df.columns = [safe_int_label(col) for col in usage_df.columns]

        renamed_program_ids = {
            local_program_index: make_program_id(sample_label, k_value, local_program_index)
            for local_program_index in score_df.index
        }
        score_df = score_df.rename(index=renamed_program_ids)
        usage_df = usage_df.rename(columns=renamed_program_ids)
        score_blocks.append(score_df)

        for local_program_index, program_id in renamed_program_ids.items():
            program_scores = score_df.loc[program_id]
            metadata = {
                "program_id": program_id,
                "dataset": dataset,
                "sample_id_on_disk": sample_id,
                "sample_label": sample_label,
                "source_k": int(k_value),
                "local_program_index": int(local_program_index),
                "is_k_star": bool(k_value == k_star),
            }
            metadata_rows.append(metadata)
            top30_rows.append(top_gene_row(metadata, top_gene_list(program_scores, 30), 30))
            top50_rows.append(top_gene_row(metadata, top_gene_list(program_scores, 50), 50))
            top100_rows.append(top_gene_row(metadata, top_gene_list(program_scores, 100), 100))

        if k_value == k_star:
            representative_usage_df = usage_df.copy()

    if representative_usage_df is None:
        raise RuntimeError(f"No representative usage matrix was found for {sample_label} at k_star={k_star}")

    metadata_df = pd.DataFrame(metadata_rows).sort_values(["source_k", "local_program_index"]).reset_index(drop=True)
    score_df = pd.concat(score_blocks, axis=0, sort=True)
    score_df = score_df.reindex(metadata_df["program_id"])
    score_df = score_df.fillna(0.0)

    spectra_export = metadata_df.merge(
        score_df.reset_index().rename(columns={"index": "program_id"}),
        on="program_id",
        how="left",
    )
    gene_cols = [col for col in spectra_export.columns if col not in metadata_df.columns]
    spectra_export = spectra_export.rename(columns={col: f"__gene__{col}" for col in gene_cols})
    spectra_export.to_csv(spectra_output, index=False)
    pd.DataFrame(top30_rows).to_csv(top30_output, index=False)
    pd.DataFrame(top50_rows).to_csv(top50_output, index=False)
    pd.DataFrame(top100_rows).to_csv(top100_output, index=False)

    usage_export = representative_usage_df.copy()
    usage_export.index.name = "spot_id"
    usage_export.insert(0, "sample_label", sample_label)
    usage_export.insert(0, "sample_id_on_disk", sample_id)
    usage_export.insert(0, "dataset", dataset)
    usage_export.reset_index().to_csv(usage_output, index=False)

    write_json(
        summary_output,
        {
            "dataset": dataset,
            "sample_id_on_disk": sample_id,
            "sample_label": sample_label,
            "k_star": k_star,
            "k_window_values": k_window_list,
            "n_programs_extracted": int(metadata_df.shape[0]),
            "n_genes_union": int(score_df.shape[1]),
            "representative_usage_n_spots": int(representative_usage_df.shape[0]),
            "representative_usage_n_programs": int(representative_usage_df.shape[1]),
        },
    )

    usage_export = representative_usage_df.copy()
    usage_export.index.name = "spot_id"
    return metadata_df, score_df, usage_export


def standardize_program_spectra(score_df: pd.DataFrame) -> pd.DataFrame:
    values = score_df.to_numpy(dtype=float)
    means = np.mean(values, axis=1, keepdims=True)
    sds = np.std(values, axis=1, ddof=0, keepdims=True)
    sds[sds == 0] = 1.0
    standardized = (values - means) / sds
    return pd.DataFrame(standardized, index=score_df.index, columns=score_df.columns)


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def square_jaccard_matrix(top_gene_sets: dict[str, set[str]]) -> pd.DataFrame:
    program_ids = list(top_gene_sets.keys())
    matrix = np.eye(len(program_ids), dtype=float)
    for i in range(len(program_ids)):
        for j in range(i + 1, len(program_ids)):
            sim = jaccard_similarity(top_gene_sets[program_ids[i]], top_gene_sets[program_ids[j]])
            matrix[i, j] = sim
            matrix[j, i] = sim
    return pd.DataFrame(matrix, index=program_ids, columns=program_ids)


def threshold_cluster_labels(similarity_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    program_ids = list(similarity_df.index)
    adjacency = (similarity_df.to_numpy(dtype=float) >= threshold).astype(int)
    np.fill_diagonal(adjacency, 1)
    n_components, labels = connected_components(
        sparse.csr_matrix(adjacency),
        directed=False,
        return_labels=True,
    )
    cluster_sizes = Counter(labels)
    records = []
    for program_id, label in zip(program_ids, labels):
        records.append(
            {
                "program_id": program_id,
                "cluster_id": f"C{int(label) + 1:03d}",
                "cluster_size": int(cluster_sizes[label]),
                "threshold": threshold,
            }
        )
    return pd.DataFrame(records)


def pairwise_clustering_agreement(
    clustering_tables: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    records = []
    metric_names = list(clustering_tables.keys())
    for i, metric_a in enumerate(metric_names):
        for metric_b in metric_names[i + 1:]:
            df_a = clustering_tables[metric_a].sort_values("program_id")
            df_b = clustering_tables[metric_b].sort_values("program_id")
            if list(df_a["program_id"]) != list(df_b["program_id"]):
                raise ValueError("Program ordering mismatch between clustering tables")
            labels_a = df_a["cluster_id"].astype(str).to_numpy()
            labels_b = df_b["cluster_id"].astype(str).to_numpy()
            records.append(
                {
                    "metric_a": metric_a,
                    "metric_b": metric_b,
                    "adjusted_rand_index": float(adjusted_rand_score(labels_a, labels_b)),
                    "normalized_mutual_info": float(normalized_mutual_info_score(labels_a, labels_b)),
                    "n_programs": int(len(labels_a)),
                }
            )
    return pd.DataFrame(records)


def adjust_pvalues_bh(p_values: list[float] | np.ndarray) -> np.ndarray:
    pvals = np.asarray(p_values, dtype=float)
    adjusted = np.full(pvals.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(pvals)
    if finite_mask.sum() == 0:
        return adjusted

    finite_vals = pvals[finite_mask]
    if multipletests is not None:
        adjusted[finite_mask] = multipletests(finite_vals, method="fdr_bh")[1]
        return adjusted

    order = np.argsort(finite_vals)
    ranked = finite_vals[order]
    n_tests = len(ranked)
    bh = np.empty(n_tests, dtype=float)
    running = 1.0
    for idx in range(n_tests - 1, -1, -1):
        rank = idx + 1
        value = min(running, ranked[idx] * n_tests / rank)
        bh[idx] = value
        running = value
    restored = np.empty(n_tests, dtype=float)
    restored[order] = bh
    adjusted[finite_mask] = restored
    return adjusted


def build_top_gene_sets(score_df: pd.DataFrame, n_top: int) -> dict[str, set[str]]:
    return {
        str(program_id): set(top_gene_list(score_df.loc[program_id], n_top))
        for program_id in score_df.index
    }


def clustering_lookup(clustering_df: pd.DataFrame) -> dict[str, dict[str, object]]:
    return clustering_df.set_index("program_id")[["cluster_id", "cluster_size"]].to_dict("index")


def same_nontrivial_cluster(lookup: dict[str, dict[str, object]], program_a: str, program_b: str) -> bool:
    if program_a not in lookup or program_b not in lookup:
        return False
    record_a = lookup[program_a]
    record_b = lookup[program_b]
    return (
        str(record_a["cluster_id"]) == str(record_b["cluster_id"])
        and int(record_a["cluster_size"]) > 1
        and int(record_b["cluster_size"]) > 1
    )


def sample_coverage_tier(n_samples: int) -> str:
    if n_samples <= 1:
        return "sample_specific"
    if n_samples <= 3:
        return "subset_specific"
    return "cohort_recurrent"


def build_metaprogram_catalogue(
    metadata_df: pd.DataFrame,
    clustering_tables: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    program_ids = metadata_df["program_id"].astype(str).tolist()
    cosine_lookup = clustering_lookup(clustering_tables["cosine"])
    jaccard_lookups = {
        metric_name: clustering_lookup(table)
        for metric_name, table in clustering_tables.items()
        if metric_name != "cosine"
    }

    support_rows: list[dict[str, object]] = []
    adjacency = np.eye(len(program_ids), dtype=int)

    for index_a, index_b in combinations(range(len(program_ids)), 2):
        program_a = program_ids[index_a]
        program_b = program_ids[index_b]
        cosine_supported = same_nontrivial_cluster(cosine_lookup, program_a, program_b)
        jaccard_hits = {
            metric_name: same_nontrivial_cluster(metric_lookup, program_a, program_b)
            for metric_name, metric_lookup in jaccard_lookups.items()
        }
        robust_edge = cosine_supported and sum(bool(hit) for hit in jaccard_hits.values()) >= 2
        if robust_edge:
            adjacency[index_a, index_b] = 1
            adjacency[index_b, index_a] = 1
        support_rows.append(
            {
                "program_id_a": program_a,
                "program_id_b": program_b,
                "cosine_supported": cosine_supported,
                "jaccard_top30_supported": bool(jaccard_hits.get("jaccard_top30", False)),
                "jaccard_top50_supported": bool(jaccard_hits.get("jaccard_top50", False)),
                "jaccard_top100_supported": bool(jaccard_hits.get("jaccard_top100", False)),
                "robust_edge": robust_edge,
            }
        )

    _, labels = connected_components(
        sparse.csr_matrix(adjacency),
        directed=False,
        return_labels=True,
    )
    membership_df = metadata_df.copy()
    membership_df["candidate_metaprogram_id"] = [f"candidate_{int(label) + 1:03d}" for label in labels]

    support_df = pd.DataFrame(support_rows)
    candidate_rows: list[dict[str, object]] = []
    dropped_rows: list[dict[str, object]] = []

    for candidate_id, component in membership_df.groupby("candidate_metaprogram_id", sort=False):
        component = component.copy()
        component_program_ids = set(component["program_id"].astype(str))
        n_programs = int(component.shape[0])
        sample_count = int(component["sample_label"].nunique())
        unique_k_count = int(component["source_k"].nunique())
        within_sample_recurrence = bool((component.groupby("sample_label")["source_k"].nunique() >= 2).any())
        cross_sample_recurrence = sample_count >= 2
        has_kstar = bool(component["is_k_star"].astype(bool).any())
        redundant_within_sample = sample_count == 1 and unique_k_count == 1
        nontrivial = n_programs > 1
        retained = nontrivial and not redundant_within_sample and (within_sample_recurrence or cross_sample_recurrence)

        if n_programs > 1 and not support_df.empty:
            pair_support = support_df[
                support_df["program_id_a"].isin(component_program_ids)
                & support_df["program_id_b"].isin(component_program_ids)
            ].copy()
        else:
            pair_support = pd.DataFrame()

        if pair_support.empty:
            support_summary = "no_pairwise_support"
            cosine_pair_support = 0.0
            top30_pair_support = 0.0
            top50_pair_support = 0.0
            top100_pair_support = 0.0
            robust_pair_support = 0.0
        else:
            cosine_pair_support = float(pair_support["cosine_supported"].mean())
            top30_pair_support = float(pair_support["jaccard_top30_supported"].mean())
            top50_pair_support = float(pair_support["jaccard_top50_supported"].mean())
            top100_pair_support = float(pair_support["jaccard_top100_supported"].mean())
            robust_pair_support = float(pair_support["robust_edge"].mean())
            support_summary = (
                f"cosine={cosine_pair_support:.2f};"
                f"top30={top30_pair_support:.2f};"
                f"top50={top50_pair_support:.2f};"
                f"top100={top100_pair_support:.2f};"
                f"robust={robust_pair_support:.2f}"
            )

        if retained and cross_sample_recurrence and within_sample_recurrence:
            retained_reason = "retained_due_to_cross_sample_and_within_sample_recurrence"
        elif retained and cross_sample_recurrence:
            retained_reason = "retained_due_to_cross_sample_recurrence"
        elif retained and within_sample_recurrence:
            retained_reason = "retained_due_to_within_sample_recurrence"
        elif not nontrivial:
            retained_reason = "dropped_singleton_no_robust_support"
        elif redundant_within_sample:
            retained_reason = "dropped_single_sample_single_k_cluster"
        else:
            retained_reason = "dropped_failed_recurrence_rule"

        candidate_rows.append(
            {
                "candidate_metaprogram_id": candidate_id,
                "n_programs": n_programs,
                "within_sample_recurrence_flag": within_sample_recurrence,
                "cross_sample_recurrence_flag": cross_sample_recurrence,
                "sample_coverage_n": sample_count,
                "sample_coverage_tier": sample_coverage_tier(sample_count),
                "has_kstar_representative": has_kstar,
                "redundant_within_sample_flag": redundant_within_sample,
                "metric_support_summary": support_summary,
                "cosine_pair_support": cosine_pair_support,
                "jaccard_top30_pair_support": top30_pair_support,
                "jaccard_top50_pair_support": top50_pair_support,
                "jaccard_top100_pair_support": top100_pair_support,
                "robust_pair_support": robust_pair_support,
                "retained_reason": retained_reason,
                "retained": retained,
            }
        )

        if not retained:
            for _, program_row in component.iterrows():
                dropped_rows.append(
                    {
                        "candidate_metaprogram_id": candidate_id,
                        "program_id": str(program_row["program_id"]),
                        "dataset": str(program_row["dataset"]),
                        "sample_id_on_disk": str(program_row["sample_id_on_disk"]),
                        "sample_label": str(program_row["sample_label"]),
                        "source_k": int(program_row["source_k"]),
                        "local_program_index": int(program_row["local_program_index"]),
                        "drop_reason": retained_reason,
                    }
                )

    candidate_df = pd.DataFrame(candidate_rows)
    retained_df = candidate_df[candidate_df["retained"]].copy()
    retained_df = retained_df.sort_values(
        ["sample_coverage_n", "n_programs", "has_kstar_representative", "candidate_metaprogram_id"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    retained_df["metaprogram_id"] = [f"MP{index + 1:03d}" for index in range(len(retained_df))]

    membership_df = membership_df.merge(
        retained_df[["candidate_metaprogram_id", "metaprogram_id"]],
        on="candidate_metaprogram_id",
        how="left",
    )
    return membership_df, retained_df, pd.DataFrame(dropped_rows), support_df


def build_consensus_gene_tables(
    retained_df: pd.DataFrame,
    membership_df: pd.DataFrame,
    score_df: pd.DataFrame,
    standardized_score_df: pd.DataFrame,
    top50_sets: dict[str, set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    consensus_rows: list[dict[str, object]] = []
    composition_rows: list[dict[str, object]] = []
    contribution_rows: list[dict[str, object]] = []
    consensus_gene_map: dict[str, list[str]] = {}

    retained_members = membership_df[membership_df["metaprogram_id"].notna()].copy()
    candidate_lookup = retained_df.set_index("metaprogram_id").to_dict("index")

    for metaprogram_id, members in retained_members.groupby("metaprogram_id", sort=False):
        members = members.copy()
        member_program_ids = members["program_id"].astype(str).tolist()
        gene_counter: Counter[str] = Counter()
        gene_sample_support: defaultdict[str, set[str]] = defaultdict(set)

        for program_id in member_program_ids:
            sample_label = str(members.loc[members["program_id"] == program_id, "sample_label"].iloc[0])
            for gene in sorted(top50_sets.get(program_id, set())):
                gene_counter[gene] += 1
                gene_sample_support[gene].add(sample_label)

        ranking_rows = []
        for gene, count in gene_counter.items():
            ranking_rows.append(
                {
                    "gene": gene,
                    "member_occurrence_n": int(count),
                    "member_occurrence_fraction": float(count / len(member_program_ids)),
                    "sample_support_n": int(len(gene_sample_support[gene])),
                    "mean_standardized_loading": float(standardized_score_df.loc[member_program_ids, gene].mean()),
                    "mean_raw_loading": float(score_df.loc[member_program_ids, gene].mean()),
                }
            )
        ranking_df = pd.DataFrame(ranking_rows)
        if not ranking_df.empty:
            ranking_df = ranking_df.sort_values(
                ["member_occurrence_n", "sample_support_n", "mean_standardized_loading", "mean_raw_loading", "gene"],
                ascending=[False, False, False, False, True],
            ).reset_index(drop=True)
        consensus_genes = ranking_df["gene"].head(CONSENSUS_TOP_GENE_COUNT).tolist() if not ranking_df.empty else []
        consensus_gene_map[str(metaprogram_id)] = consensus_genes

        for rank_index, row in ranking_df.iterrows():
            consensus_rows.append(
                {
                    "metaprogram_id": str(metaprogram_id),
                    "candidate_metaprogram_id": str(candidate_lookup[str(metaprogram_id)]["candidate_metaprogram_id"]),
                    "sample_coverage_n": int(candidate_lookup[str(metaprogram_id)]["sample_coverage_n"]),
                    "sample_coverage_tier": str(candidate_lookup[str(metaprogram_id)]["sample_coverage_tier"]),
                    "gene": str(row["gene"]),
                    "consensus_rank": int(rank_index + 1),
                    "member_occurrence_n": int(row["member_occurrence_n"]),
                    "member_occurrence_fraction": float(row["member_occurrence_fraction"]),
                    "sample_support_n": int(row["sample_support_n"]),
                    "mean_standardized_loading": float(row["mean_standardized_loading"]),
                    "mean_raw_loading": float(row["mean_raw_loading"]),
                    "included_in_consensus_top_genes": bool(rank_index < CONSENSUS_TOP_GENE_COUNT),
                }
            )

        for _, member_row in members.iterrows():
            composition_rows.append(
                {
                    "metaprogram_id": str(metaprogram_id),
                    "candidate_metaprogram_id": str(member_row["candidate_metaprogram_id"]),
                    "program_id": str(member_row["program_id"]),
                    "dataset": str(member_row["dataset"]),
                    "sample_id_on_disk": str(member_row["sample_id_on_disk"]),
                    "sample_label": str(member_row["sample_label"]),
                    "source_k": int(member_row["source_k"]),
                    "local_program_index": int(member_row["local_program_index"]),
                    "is_k_star": bool(member_row["is_k_star"]),
                }
            )

        for (dataset, sample_id, sample_label), sample_rows in members.groupby(
            ["dataset", "sample_id_on_disk", "sample_label"],
            sort=False,
        ):
            contribution_rows.append(
                {
                    "metaprogram_id": str(metaprogram_id),
                    "dataset": str(dataset),
                    "sample_id_on_disk": str(sample_id),
                    "sample_label": str(sample_label),
                    "n_programs_total": int(sample_rows.shape[0]),
                    "n_programs_k_star": int(sample_rows["is_k_star"].astype(bool).sum()),
                    "source_ks": ";".join(str(int(k)) for k in sorted(sample_rows["source_k"].astype(int).unique())),
                    "sample_coverage_tier": str(candidate_lookup[str(metaprogram_id)]["sample_coverage_tier"]),
                }
            )

    return (
        pd.DataFrame(consensus_rows),
        pd.DataFrame(composition_rows),
        pd.DataFrame(contribution_rows),
        consensus_gene_map,
    )


def fisher_overlap_rows(
    retained_df: pd.DataFrame,
    consensus_gene_map: dict[str, list[str]],
    signature_gene_sets: dict[str, set[str]],
    universe_genes: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    retained_lookup = retained_df.set_index("metaprogram_id").to_dict("index")

    for metaprogram_id, consensus_genes in consensus_gene_map.items():
        metaprogram_genes = set(consensus_genes) & universe_genes
        for variant_name, signature_genes in signature_gene_sets.items():
            sig_genes = set(signature_genes) & universe_genes
            overlap = metaprogram_genes & sig_genes
            a = len(overlap)
            b = len(metaprogram_genes - sig_genes)
            c = len(sig_genes - metaprogram_genes)
            d = len(universe_genes) - a - b - c
            _, p_value = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
            rows.append(
                {
                    "metaprogram_id": metaprogram_id,
                    "candidate_metaprogram_id": str(retained_lookup[metaprogram_id]["candidate_metaprogram_id"]),
                    "signature_variant": variant_name,
                    "metaprogram_gene_n": int(len(metaprogram_genes)),
                    "signature_gene_n": int(len(sig_genes)),
                    "overlap_size": int(a),
                    "jaccard": float(a / len(metaprogram_genes | sig_genes)) if (metaprogram_genes | sig_genes) else np.nan,
                    "overlap_coefficient": float(a / min(len(metaprogram_genes), len(sig_genes))) if min(len(metaprogram_genes), len(sig_genes)) > 0 else np.nan,
                    "fisher_p_one_sided": float(p_value) if np.isfinite(p_value) else np.nan,
                    "sample_coverage_n": int(retained_lookup[metaprogram_id]["sample_coverage_n"]),
                    "sample_coverage_tier": str(retained_lookup[metaprogram_id]["sample_coverage_tier"]),
                }
            )

    overlap_df = pd.DataFrame(rows)
    overlap_df["fdr_bh"] = adjust_pvalues_bh(overlap_df["fisher_p_one_sided"].to_numpy(dtype=float))

    summary_rows: list[dict[str, object]] = []
    for metaprogram_id, subset in overlap_df.groupby("metaprogram_id", sort=False):
        best_row = subset.sort_values(["fdr_bh", "overlap_coefficient", "overlap_size"], ascending=[True, False, False]).iloc[0]
        significant_variants = subset.loc[subset["fdr_bh"] < 0.10, "signature_variant"].astype(str).tolist()
        concordant = (
            ("pc_up" in significant_variants)
            and len(significant_variants) >= 2
        )
        summary_rows.append(
            {
                "metaprogram_id": metaprogram_id,
                "best_signature_variant": str(best_row["signature_variant"]),
                "best_overlap_size": int(best_row["overlap_size"]),
                "best_overlap_coefficient": float(best_row["overlap_coefficient"]),
                "best_fdr_bh": float(best_row["fdr_bh"]),
                "significant_variants_fdr_lt_0_10": ";".join(significant_variants),
                "concordance_across_signature_variants": bool(concordant),
                "pc_up_overlap_size": int(subset.loc[subset["signature_variant"] == "pc_up", "overlap_size"].max()),
                "pc_up_fdr_bh": float(subset.loc[subset["signature_variant"] == "pc_up", "fdr_bh"].min()),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    overlap_df = overlap_df.merge(
        summary_df[["metaprogram_id", "concordance_across_signature_variants"]],
        on="metaprogram_id",
        how="left",
    )
    return overlap_df, summary_df


def load_tumor_sample_context(sample_row: pd.Series) -> tuple[ad.AnnData, pd.DataFrame]:
    analysis_path = Path(str(sample_row["analysis_ready_h5ad_path"]))
    adata = ad.read_h5ad(analysis_path)
    tumor_mask, _ = exact_tumor_masks(adata.obs)
    adata_tumor = adata[tumor_mask].copy()

    signature_df = pd.DataFrame(index=adata_tumor.obs_names)
    robustness_path = Path(str(sample_row["robustness_h5ad_path"]))
    if robustness_path.exists():
        robust_adata = ad.read_h5ad(robustness_path)
        robust_obs = robust_adata.obs.reindex(adata_tumor.obs_names)
        for variant_name, score_col in SIGNATURE_SCORE_COLUMNS.items():
            signature_df[variant_name] = pd.to_numeric(robust_obs.get(score_col), errors="coerce")
    else:
        for variant_name in SIGNATURE_SCORE_COLUMNS:
            signature_df[variant_name] = np.nan
    return adata_tumor, signature_df


def log1p_normalized_counts(adata: ad.AnnData):
    matrix = adata.layers["counts"] if "counts" in adata.layers else adata.X
    if sparse.issparse(matrix):
        matrix = matrix.tocsr().astype(float)
        totals = np.asarray(matrix.sum(axis=1)).ravel()
        scale = np.divide(1e4, totals, out=np.zeros_like(totals, dtype=float), where=totals > 0)
        normalized = matrix.multiply(scale[:, None]).tocsr()
        normalized.data = np.log1p(normalized.data)
        return normalized

    dense = np.asarray(matrix, dtype=float)
    totals = dense.sum(axis=1, keepdims=True)
    normalized = np.divide(dense, totals, out=np.zeros_like(dense), where=totals > 0) * 1e4
    return np.log1p(normalized)


def score_gene_set_from_matrix(
    log_matrix,
    var_names: pd.Index,
    genes: list[str],
) -> tuple[np.ndarray, int]:
    present_genes = [str(gene) for gene in genes if str(gene) in var_names]
    if len(present_genes) < 3:
        return np.full(log_matrix.shape[0], np.nan), len(present_genes)

    gene_indexer = var_names.get_indexer(present_genes)
    subset = log_matrix[:, gene_indexer]
    values = subset.toarray() if sparse.issparse(subset) else np.asarray(subset, dtype=float)
    gene_means = np.mean(values, axis=0, keepdims=True)
    gene_sds = np.std(values, axis=0, ddof=0, keepdims=True)
    gene_sds[gene_sds == 0] = 1.0
    z_values = (values - gene_means) / gene_sds
    return np.mean(z_values, axis=1), len(present_genes)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]
    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, int(len(x))
    rho, p_value = stats.spearmanr(x, y)
    return float(rho), float(p_value), int(len(x))


def compute_morans_i(coords: np.ndarray, scores: np.ndarray) -> tuple[float, float, int]:
    if not SPATIAL_STATS_AVAILABLE:
        return np.nan, np.nan, int(len(scores))
    values = np.asarray(scores, dtype=float)
    coord_array = np.asarray(coords, dtype=float)
    finite_mask = np.isfinite(values) & np.all(np.isfinite(coord_array), axis=1)
    values = values[finite_mask]
    coord_array = coord_array[finite_mask]
    if len(values) <= REPRESENTATIVE_SPATIAL_K or np.std(values) == 0:
        return np.nan, np.nan, int(len(values))
    k_value = min(REPRESENTATIVE_SPATIAL_K, len(values) - 1)
    weights = libpysal.weights.KNN.from_array(coord_array, k=k_value)
    weights.transform = "r"
    moran = Moran(values, weights)
    return float(moran.I), float(moran.p_sim), int(len(values))


def summarize_association_rows(assoc_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if assoc_df.empty:
        return pd.DataFrame(rows)
    for (metaprogram_id, score_type, signature_variant), subset in assoc_df.groupby(
        ["metaprogram_id", "score_type", "signature_variant"],
        sort=False,
    ):
        rho = subset["spearman_rho"].astype(float).to_numpy()
        finite_rho = rho[np.isfinite(rho)]
        fisher_mean = np.nan
        if len(finite_rho) > 0:
            clipped = np.clip(finite_rho, -0.999999, 0.999999)
            fisher_mean = float(np.tanh(np.mean(np.arctanh(clipped))))
        rows.append(
            {
                "metaprogram_id": str(metaprogram_id),
                "score_type": str(score_type),
                "signature_variant": str(signature_variant),
                "n_samples_tested": int(subset["sample_label"].nunique()),
                "mean_spearman_rho": float(np.nanmean(rho)) if np.isfinite(rho).any() else np.nan,
                "median_spearman_rho": float(np.nanmedian(rho)) if np.isfinite(rho).any() else np.nan,
                "fisher_mean_rho": fisher_mean,
                "pct_positive": float(np.mean(finite_rho > 0)) if len(finite_rho) else np.nan,
                "pct_p_lt_0_05": float(np.mean(subset["spearman_p"].astype(float).to_numpy() < 0.05)) if len(subset) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_spatial_rows(spatial_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if spatial_df.empty:
        return pd.DataFrame(rows)
    for (metaprogram_id, score_type), subset in spatial_df.groupby(["metaprogram_id", "score_type"], sort=False):
        values = subset["morans_I"].astype(float).to_numpy()
        pvals = subset["morans_p"].astype(float).to_numpy()
        finite_values = values[np.isfinite(values)]
        rows.append(
            {
                "metaprogram_id": str(metaprogram_id),
                "score_type": str(score_type),
                "n_samples_tested": int(subset["sample_label"].nunique()),
                "mean_morans_I": float(np.nanmean(values)) if np.isfinite(values).any() else np.nan,
                "median_morans_I": float(np.nanmedian(values)) if np.isfinite(values).any() else np.nan,
                "pct_positive": float(np.mean(finite_values > 0)) if len(finite_values) else np.nan,
                "pct_p_lt_0_05": float(np.mean(pvals < 0.05)) if np.isfinite(pvals).any() else np.nan,
                "spatially_coherent_flag": bool(np.isfinite(values).any() and ((np.nanmedian(values) > 0) or np.any((values > 0) & (pvals < 0.05)))),
            }
        )
    return pd.DataFrame(rows)


def choose_representative_programs(
    retained_membership_df: pd.DataFrame,
    cosine_similarity_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (metaprogram_id, dataset, sample_id, sample_label), subset in retained_membership_df.groupby(
        ["metaprogram_id", "dataset", "sample_id_on_disk", "sample_label"],
        sort=False,
    ):
        kstar_subset = subset[subset["is_k_star"].astype(bool)].copy()
        if kstar_subset.empty:
            rows.append(
                {
                    "metaprogram_id": str(metaprogram_id),
                    "dataset": str(dataset),
                    "sample_id_on_disk": str(sample_id),
                    "sample_label": str(sample_label),
                    "representative_available": False,
                    "representative_program_id": "",
                    "selection_reason": "no_k_star_member_in_metaprogram",
                }
            )
            continue

        if kstar_subset.shape[0] == 1:
            chosen_program_id = str(kstar_subset["program_id"].iloc[0])
            selection_reason = "single_k_star_member"
        else:
            member_programs = subset["program_id"].astype(str).tolist()
            centrality_scores = {}
            for program_id in kstar_subset["program_id"].astype(str).tolist():
                comparison_ids = [other_id for other_id in member_programs if other_id != program_id]
                if comparison_ids:
                    centrality_scores[program_id] = float(cosine_similarity_df.loc[program_id, comparison_ids].mean())
                else:
                    centrality_scores[program_id] = 1.0
            chosen_program_id = max(centrality_scores, key=centrality_scores.get)
            selection_reason = "highest_within_metaprogram_cosine_centrality"

        rows.append(
            {
                "metaprogram_id": str(metaprogram_id),
                "dataset": str(dataset),
                "sample_id_on_disk": str(sample_id),
                "sample_label": str(sample_label),
                "representative_available": True,
                "representative_program_id": chosen_program_id,
                "selection_reason": selection_reason,
            }
        )
    return pd.DataFrame(rows)


def build_usage_and_association_outputs(
    included_df: pd.DataFrame,
    representative_choices_df: pd.DataFrame,
    representative_usage_by_sample: dict[str, pd.DataFrame],
    consensus_gene_map: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    representative_rows: list[dict[str, object]] = []
    consensus_rows: list[dict[str, object]] = []
    association_rows: list[dict[str, object]] = []
    spatial_rows: list[dict[str, object]] = []

    for _, sample_row in included_df.iterrows():
        sample_label = str(sample_row["sample_label"])
        dataset = str(sample_row["dataset"])
        sample_id = str(sample_row["sample_id_on_disk"])
        adata_tumor, signature_scores = load_tumor_sample_context(sample_row)
        coords = np.asarray(adata_tumor.obsm["spatial"], dtype=float)
        obs_names = adata_tumor.obs_names.astype(str)
        signature_scores = signature_scores.reindex(obs_names)
        usage_df = representative_usage_by_sample.get(sample_label)
        if usage_df is None:
            usage_df = pd.DataFrame(index=obs_names)
        else:
            usage_df = usage_df.copy()
            usage_df.index = usage_df.index.astype(str)
            usage_df = usage_df.reindex(obs_names)

        log_matrix = log1p_normalized_counts(adata_tumor)
        var_names = pd.Index(map(str, adata_tumor.var_names))

        sample_choices = representative_choices_df[representative_choices_df["sample_label"] == sample_label].copy()
        for _, choice_row in sample_choices.iterrows():
            metaprogram_id = str(choice_row["metaprogram_id"])
            selected_program_id = str(choice_row["representative_program_id"])
            if bool(choice_row["representative_available"]) and selected_program_id in usage_df.columns:
                values = pd.to_numeric(usage_df[selected_program_id], errors="coerce").to_numpy(dtype=float)
                for spot_id, value in zip(obs_names, values):
                    representative_rows.append(
                        {
                            "dataset": dataset,
                            "sample_id_on_disk": sample_id,
                            "sample_label": sample_label,
                            "spot_id": str(spot_id),
                            "metaprogram_id": metaprogram_id,
                            "representative_program_id": selected_program_id,
                            "usage": float(value) if np.isfinite(value) else np.nan,
                        }
                    )
                moran_i, moran_p, n_used = compute_morans_i(coords, values)
                spatial_rows.append(
                    {
                        "dataset": dataset,
                        "sample_id_on_disk": sample_id,
                        "sample_label": sample_label,
                        "metaprogram_id": metaprogram_id,
                        "score_type": "representative_usage",
                        "morans_I": moran_i,
                        "morans_p": moran_p,
                        "n_spots_used": n_used,
                    }
                )
                for variant_name in SIGNATURE_SCORE_COLUMNS:
                    rho, p_value, n_used_assoc = safe_spearman(values, signature_scores[variant_name].to_numpy(dtype=float))
                    association_rows.append(
                        {
                            "dataset": dataset,
                            "sample_id_on_disk": sample_id,
                            "sample_label": sample_label,
                            "metaprogram_id": metaprogram_id,
                            "score_type": "representative_usage",
                            "signature_variant": variant_name,
                            "spearman_rho": rho,
                            "spearman_p": p_value,
                            "n_spots_used": n_used_assoc,
                        }
                    )

        for metaprogram_id, consensus_genes in consensus_gene_map.items():
            score_values, n_genes_used = score_gene_set_from_matrix(log_matrix, var_names, consensus_genes)
            for spot_id, value in zip(obs_names, score_values):
                consensus_rows.append(
                    {
                        "dataset": dataset,
                        "sample_id_on_disk": sample_id,
                        "sample_label": sample_label,
                        "spot_id": str(spot_id),
                        "metaprogram_id": str(metaprogram_id),
                        "consensus_score": float(value) if np.isfinite(value) else np.nan,
                        "n_genes_used": int(n_genes_used),
                        "n_genes_consensus": int(len(consensus_genes)),
                    }
                )
            moran_i, moran_p, n_used = compute_morans_i(coords, score_values)
            spatial_rows.append(
                {
                    "dataset": dataset,
                    "sample_id_on_disk": sample_id,
                    "sample_label": sample_label,
                    "metaprogram_id": str(metaprogram_id),
                    "score_type": "consensus_score",
                    "morans_I": moran_i,
                    "morans_p": moran_p,
                    "n_spots_used": n_used,
                }
            )
            for variant_name in SIGNATURE_SCORE_COLUMNS:
                rho, p_value, n_used_assoc = safe_spearman(score_values, signature_scores[variant_name].to_numpy(dtype=float))
                association_rows.append(
                    {
                        "dataset": dataset,
                        "sample_id_on_disk": sample_id,
                        "sample_label": sample_label,
                        "metaprogram_id": str(metaprogram_id),
                        "score_type": "consensus_score",
                        "signature_variant": variant_name,
                        "spearman_rho": rho,
                        "spearman_p": p_value,
                        "n_spots_used": n_used_assoc,
                    }
                )

    return (
        pd.DataFrame(representative_rows),
        pd.DataFrame(consensus_rows),
        pd.DataFrame(association_rows),
        pd.DataFrame(spatial_rows),
    )


def finalize_core_metaprograms(
    retained_df: pd.DataFrame,
    overlap_summary_df: pd.DataFrame,
    association_summary_df: pd.DataFrame,
    spatial_summary_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    retained_out = retained_df.copy()

    overlap_lookup = overlap_summary_df.set_index("metaprogram_id").to_dict("index") if not overlap_summary_df.empty else {}
    assoc_lookup = {}
    if not association_summary_df.empty:
        assoc_focus = association_summary_df[
            (association_summary_df["score_type"] == "representative_usage")
            & (association_summary_df["signature_variant"] == "pc_up")
        ].copy()
        assoc_lookup = assoc_focus.set_index("metaprogram_id").to_dict("index")
    spatial_lookup = {}
    if not spatial_summary_df.empty:
        spatial_focus = spatial_summary_df[spatial_summary_df["score_type"] == "representative_usage"].copy()
        spatial_lookup = spatial_focus.set_index("metaprogram_id").to_dict("index")

    recommendations = []
    core_flags = []
    for _, row in retained_out.iterrows():
        metaprogram_id = str(row["metaprogram_id"])
        spatial_record = spatial_lookup.get(metaprogram_id, {})
        overlap_record = overlap_lookup.get(metaprogram_id, {})
        assoc_record = assoc_lookup.get(metaprogram_id, {})

        spatially_coherent = bool(spatial_record.get("spatially_coherent_flag", False))
        pc_up_positive = bool(
            np.isfinite(float(assoc_record.get("median_spearman_rho", np.nan)))
            and float(assoc_record.get("median_spearman_rho", np.nan)) > 0
        )
        overlap_supported = bool(overlap_record.get("concordance_across_signature_variants", False)) or (
            np.isfinite(float(overlap_record.get("best_overlap_coefficient", np.nan)))
            and float(overlap_record.get("best_overlap_coefficient", np.nan)) > 0
        )

        if not bool(row["has_kstar_representative"]):
            recommendation = "not_recommended"
        elif row["sample_coverage_tier"] == "cohort_recurrent" and spatially_coherent:
            recommendation = "carry_into_cross_sample_variance_partition"
        elif row["sample_coverage_tier"] == "subset_specific" and spatially_coherent:
            recommendation = "carry_into_subset_or_sample_level_variance_partition"
        elif row["sample_coverage_tier"] == "sample_specific" and spatially_coherent and bool(row["within_sample_recurrence_flag"]):
            recommendation = "sample_specific_only"
        else:
            recommendation = "not_recommended"

        core_flag = bool(bool(row["has_kstar_representative"]) and spatially_coherent and (pc_up_positive or overlap_supported))
        core_flags.append(core_flag)
        recommendations.append(recommendation)

    retained_out["core_metaprogram"] = core_flags
    retained_out["variance_partition_recommendation"] = recommendations
    retained_out["safe_for_variance_partition"] = retained_out["variance_partition_recommendation"] != "not_recommended"
    core_df = retained_out[retained_out["core_metaprogram"]].copy()
    return retained_out, core_df


def write_postk_documentation(
    included_df: pd.DataFrame,
    retained_df: pd.DataFrame,
    core_df: pd.DataFrame,
    overlap_summary_df: pd.DataFrame,
) -> None:
    overlap_lookup = overlap_summary_df.set_index("metaprogram_id").to_dict("index") if not overlap_summary_df.empty else {}

    readme_lines = [
        "# Post-K cNMF Summary",
        "",
        "## K decisions used",
        "",
    ]
    for _, row in included_df.sort_values(["dataset", "sample_id_on_disk"]).iterrows():
        readme_lines.append(
            f"- {row['dataset']} / {row['sample_id_on_disk']}: "
            f"K* = {int(row['k_star'])}; "
            f"k_window_values = {row['k_window_values']}; "
            f"rationale = {row['rationale_short']}"
        )

    readme_lines.extend(["", "## Retained metaprograms", ""])
    if retained_df.empty:
        readme_lines.append("- No retained metaprograms passed the robustness rules.")
    else:
        for _, row in retained_df.sort_values("metaprogram_id").iterrows():
            overlap_record = overlap_lookup.get(str(row["metaprogram_id"]), {})
            readme_lines.append(
                f"- {row['metaprogram_id']}: {row['sample_coverage_tier']} "
                f"(samples={int(row['sample_coverage_n'])}, programs={int(row['n_programs'])}), "
                f"reason = {row['retained_reason']}, "
                f"best SNAI1-ac overlap = {overlap_record.get('best_signature_variant', 'NA')} "
                f"(best overlap coefficient = {float(overlap_record.get('best_overlap_coefficient', np.nan)):.3f})"
            )

    readme_lines.extend(["", "## Core metaprograms", ""])
    if core_df.empty:
        readme_lines.append("- No retained metaprograms currently meet the core criteria.")
    else:
        for _, row in core_df.sort_values("metaprogram_id").iterrows():
            readme_lines.append(
                f"- {row['metaprogram_id']}: {row['variance_partition_recommendation']} "
                f"({row['sample_coverage_tier']}, has_kstar_representative = {bool(row['has_kstar_representative'])})"
            )

    if not overlap_summary_df.empty:
        best_overlap_row = overlap_summary_df.sort_values(["best_fdr_bh", "best_overlap_coefficient"], ascending=[True, False]).iloc[0]
        readme_lines.extend(
            [
                "",
                "## Answers to the key questions",
                "",
                f"1. Which K* and k_window_values were used per sample? See the per-sample list above and the parsed manual K file at `{PARSED_MANUAL_K_PATH}`.",
                f"2. Which metaprograms were retained and why? {len(retained_df)} retained metaprograms passed the cross-metric robustness filter; the retained catalogue records within-sample recurrence, cross-sample recurrence, and the exact retained reason for each metaprogram.",
                f"3. Which are sample-specific, subset-specific, and cohort-recurrent? The retained catalogue includes `sample_coverage_tier` for every metaprogram, with {int((retained_df['sample_coverage_tier'] == 'sample_specific').sum())} sample-specific, {int((retained_df['sample_coverage_tier'] == 'subset_specific').sum())} subset-specific, and {int((retained_df['sample_coverage_tier'] == 'cohort_recurrent').sum())} cohort-recurrent metaprograms.",
                f"4. Is there a metaprogram that overlaps the SNAI1-ac signature variants? The strongest current overlap is {best_overlap_row['metaprogram_id']} against {best_overlap_row['best_signature_variant']} with overlap coefficient {best_overlap_row['best_overlap_coefficient']:.3f} and FDR {best_overlap_row['best_fdr_bh']:.3g}.",
                f"5. Which metaprograms are safe to carry into variance partition / downstream thesis interpretation? {', '.join(core_df['metaprogram_id'].astype(str).tolist()) if not core_df.empty else 'None yet; use the retained catalogue for discovery and the variance-partition recommendation column for filtering.'}",
            ]
        )

    README_POSTK_PATH.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    doc_lines = [
        "# cNMF post-K strategy and rationale",
        "",
        "## Manual K handling",
        "",
        "- `k_window_values` is treated as an explicit semicolon-separated set of reviewed K values, not as a contiguous range.",
        "- This allows disjoint stable modes to be represented faithfully when a sample shows more than one locally credible K regime.",
        "- `k_star` is the representative K used for per-spot usage export, while the full `k_window_values` set is retained for cross-K program discovery.",
        "",
        "## Representative usage versus consensus metaprogram scoring",
        "",
        "- Representative metaprogram usage uses the constituent program from a sample's manually selected `k_star` solution when that program is present in the retained metaprogram.",
        "- Consensus metaprogram scoring is separate: it scores the retained consensus gene list per spot and is intended for visualization and cross-sample interpretation.",
        "- Keeping those two quantities separate prevents us from conflating a sample-specific cNMF loading with a broader consensus gene program score.",
        "",
        "## Cross-metric robustness rule",
        "",
        "- A robust program edge requires support in the cosine clustering and in at least two of the three Jaccard top-N clustering families.",
        "- Rare programs are retained when they are internally robust across K within a sample, even if they are not broadly recurrent across the cohort.",
        "- Coverage tier governs interpretation strength rather than simple retention.",
        "",
        "## Current post-K outputs",
        "",
        f"- Parsed manual K decisions: `{PARSED_MANUAL_K_PATH}`",
        f"- Validation report: `{MANUAL_K_VALIDATION_PATH}`",
        f"- README summary: `{README_POSTK_PATH}`",
        f"- Clustering agreement table: `{CLUSTERING_DIR / 'clustering_agreement_summary.csv'}`",
    ]
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text("\n".join(doc_lines) + "\n", encoding="utf-8")

def run_post_k(manifest: pd.DataFrame, k_decisions_path: Path | None) -> None:
    included_df = parse_and_validate_manual_k(manifest, k_decisions_path)

    metadata_blocks: list[pd.DataFrame] = []
    score_blocks: list[pd.DataFrame] = []
    representative_usage_by_sample: dict[str, pd.DataFrame] = {}

    for _, sample_row in included_df.iterrows():
        metadata_df, score_df, usage_df = extract_programs_for_sample(sample_row, force=False)
        metadata_blocks.append(metadata_df)
        score_blocks.append(score_df)
        representative_usage_by_sample[str(sample_row["sample_label"])] = usage_df

    metadata_df = pd.concat(metadata_blocks, ignore_index=True)
    score_df = pd.concat(score_blocks, axis=0, sort=True).fillna(0.0)
    score_df = score_df.reindex(metadata_df["program_id"].astype(str).tolist())
    standardized_score_df = standardize_program_spectra(score_df)

    cosine_similarity_df = pd.DataFrame(
        cosine_similarity(standardized_score_df.to_numpy(dtype=float)),
        index=standardized_score_df.index,
        columns=standardized_score_df.index,
    )
    jaccard_top30_df = square_jaccard_matrix(build_top_gene_sets(score_df, 30))
    jaccard_top50_df = square_jaccard_matrix(build_top_gene_sets(score_df, 50))
    jaccard_top100_df = square_jaccard_matrix(build_top_gene_sets(score_df, 100))

    cosine_similarity_df.to_csv(SIMILARITY_DIR / "program_similarity_cosine.csv")
    jaccard_top30_df.to_csv(SIMILARITY_DIR / "program_similarity_jaccard_top30.csv")
    jaccard_top50_df.to_csv(SIMILARITY_DIR / "program_similarity_jaccard_top50.csv")
    jaccard_top100_df.to_csv(SIMILARITY_DIR / "program_similarity_jaccard_top100.csv")

    clustering_tables = {
        "cosine": threshold_cluster_labels(cosine_similarity_df, COSINE_CLUSTER_THRESHOLD),
        "jaccard_top30": threshold_cluster_labels(jaccard_top30_df, JACCARD_TOP30_THRESHOLD),
        "jaccard_top50": threshold_cluster_labels(jaccard_top50_df, JACCARD_TOP50_THRESHOLD),
        "jaccard_top100": threshold_cluster_labels(jaccard_top100_df, JACCARD_TOP100_THRESHOLD),
    }
    clustering_tables["cosine"].to_csv(CLUSTERING_DIR / "cosine_clusters.csv", index=False)
    clustering_tables["jaccard_top30"].to_csv(CLUSTERING_DIR / "jaccard_top30_clusters.csv", index=False)
    clustering_tables["jaccard_top50"].to_csv(CLUSTERING_DIR / "jaccard_top50_clusters.csv", index=False)
    clustering_tables["jaccard_top100"].to_csv(CLUSTERING_DIR / "jaccard_top100_clusters.csv", index=False)
    clustering_agreement_df = pairwise_clustering_agreement(clustering_tables)
    clustering_agreement_df.to_csv(CLUSTERING_DIR / "clustering_agreement_summary.csv", index=False)

    membership_df, retained_df, dropped_df, support_df = build_metaprogram_catalogue(metadata_df, clustering_tables)
    consensus_genes_df, composition_df, sample_contribution_df, consensus_gene_map = build_consensus_gene_tables(
        retained_df=retained_df,
        membership_df=membership_df,
        score_df=score_df,
        standardized_score_df=standardized_score_df,
        top50_sets=build_top_gene_sets(score_df, 50),
    )

    signature_gene_sets = load_signature_gene_sets()
    overlap_df, overlap_summary_df = fisher_overlap_rows(
        retained_df=retained_df,
        consensus_gene_map=consensus_gene_map,
        signature_gene_sets=signature_gene_sets,
        universe_genes=set(map(str, score_df.columns)),
    )

    representative_choices_df = choose_representative_programs(
        retained_membership_df=membership_df[membership_df["metaprogram_id"].notna()].copy(),
        cosine_similarity_df=cosine_similarity_df,
    )
    representative_usage_df, consensus_score_df, association_df, spatial_df = build_usage_and_association_outputs(
        included_df=included_df,
        representative_choices_df=representative_choices_df,
        representative_usage_by_sample=representative_usage_by_sample,
        consensus_gene_map=consensus_gene_map,
    )
    representative_choices_df = representative_choices_df.merge(
        retained_df[["metaprogram_id", "sample_coverage_n", "sample_coverage_tier", "has_kstar_representative"]],
        on="metaprogram_id",
        how="left",
    )

    association_summary_df = summarize_association_rows(association_df)
    spatial_summary_df = summarize_spatial_rows(spatial_df)
    retained_final_df, core_df = finalize_core_metaprograms(
        retained_df=retained_df,
        overlap_summary_df=overlap_summary_df,
        association_summary_df=association_summary_df,
        spatial_summary_df=spatial_summary_df,
    )

    retained_final_df.to_csv(RETENTION_DIR / "retained_metaprogram_catalogue.csv", index=False)
    core_df.to_csv(RETENTION_DIR / "core_metaprograms.csv", index=False)
    dropped_df.to_csv(RETENTION_DIR / "dropped_programs_and_reasons.csv", index=False)

    consensus_genes_df.to_csv(CONSENSUS_DIR / "metaprogram_consensus_genes.csv", index=False)
    composition_df.to_csv(CONSENSUS_DIR / "metaprogram_composition.csv", index=False)
    sample_contribution_df.to_csv(CONSENSUS_DIR / "metaprogram_sample_contribution.csv", index=False)

    overlap_df.to_csv(SIGNATURE_OVERLAP_DIR / "meta_program_signature_overlap.tsv", sep="\t", index=False)
    overlap_summary_df.to_csv(SIGNATURE_OVERLAP_DIR / "signature_overlap_summary.csv", index=False)

    representative_usage_df.to_csv(USAGE_DIR / "representative_metaprogram_usage_per_spot.csv", index=False)
    consensus_score_df.to_csv(USAGE_DIR / "consensus_metaprogram_scores_per_spot.csv", index=False)
    representative_choices_df.to_csv(USAGE_DIR / "representative_usage_availability.csv", index=False)

    spatial_df.to_csv(SPATIAL_DIR / "metaprogram_usage_moransI_per_sample.csv", index=False)
    spatial_summary_df.to_csv(SPATIAL_DIR / "metaprogram_usage_moransI_summary.csv", index=False)

    association_df.to_csv(META_DIR / "per_sample_metaprogram_association.csv", index=False)
    association_summary_df.to_csv(META_DIR / "metaprogram_association_summary.csv", index=False)
    support_df.to_csv(META_DIR / "metaprogram_membership_support_pairs.csv", index=False)
    membership_df.to_csv(META_DIR / "metaprogram_membership.csv", index=False)

    write_postk_documentation(
        included_df=included_df,
        retained_df=retained_final_df,
        core_df=core_df,
        overlap_summary_df=overlap_summary_df,
    )


def main() -> None:
    args = parse_args()
    ensure_dirs()

    manifest = build_cohort_manifest()
    selected = select_samples(manifest, requested=args.samples, max_samples=args.max_samples)

    if args.stage == "manifest":
        print(f"Wrote cohort manifest with {len(manifest)} candidate samples to {SAMPLE_MANIFEST_DIR / 'cohort_manifest.csv'}")
        print(f"Eligible for cNMF: {int(manifest['eligible_for_cnmf'].sum())}")
        return

    if args.stage == "pre_k":
        run_pre_k(selected, force=args.force)
        return

    if args.stage == "k_review":
        build_k_review_outputs(selected)
        print(f"Wrote K-review outputs to {K_SELECTION_DIR}")
        return

    if args.stage == "post_k":
        run_post_k(selected, k_decisions_path=args.k_decisions)
        return


if __name__ == "__main__":
    main()
