"""
First-pass ILR balance analysis of SpaCET broad lineage fractions versus
SNAI1-ac score.

Scope:
    - 23-sample downstream Visium cohort.
    - Per-sample computations only; no pooled spot-level computations.
    - Full malignant-retained composition after dropping Unidentifiable.
    - Correlation/reporting only. No regression and no publication figures.

Outputs:
    D:/HGSOC_Spatial_Atlas/05_analysis_ready/
        S2b_CellType_Composition_Correlation/ilr_first_pass/
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import skbio
from scipy import stats
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from skbio.stats.composition import closure, ilr, multi_replace, sbp_basis


BASE_DIR = Path(r"D:\HGSOC_Spatial_Atlas")
VISIUM_DIR = BASE_DIR / "05_analysis_ready" / "visium"
OUTPUT_DIR = (
    BASE_DIR
    / "05_analysis_ready"
    / "S2b_CellType_Composition_Correlation"
    / "ilr_first_pass"
)

SNAI1_COL = "SNAI1-ac_score"
SEED = 1
EXPECTED_HALLMARK_COUNT = 50
RSCRIPT = Path(r"C:\Program Files\R\R-4.4.3\bin\Rscript.exe")
FLOOR_CHECK_DATASET = "denisenko_2022"
FLOOR_CHECK_SAMPLE = "SP1"
FLOOR_CHECK_RDS = (
    BASE_DIR
    / "05_analysis_ready"
    / "visium"
    / FLOOR_CHECK_DATASET
    / f"{FLOOR_CHECK_SAMPLE}.rds"
)

COHORT_23 = {
    "denisenko_2022": ["SP1", "SP2", "SP3", "SP4", "SP5", "SP6", "SP7", "SP8"],
    "yamamoto_2025": [
        "Pt1-1",
        "Pt1-2",
        "Pt1-3",
        "Pt1-4",
        "Pt2-1",
        "Pt2-2",
        "Pt2-3",
        "Pt2-4",
    ],
    "ju_2024": [
        "CPS_OV19_LtOV1",
        "CPS_OV1RtOV3",
        "CPS_OV20RtOV4",
        "CPS_OV24RTOV4",
        "CPS_OV34RtOV1",
        "CPS_OV5LtOV4",
        "CPS_OV71_1",
    ],
}

EXPECTED_SPACET_COLUMNS = [
    "Malignant",
    "CAF",
    "Endothelial",
    "Plasma",
    "B cell",
    "T CD4",
    "T CD8",
    "NK",
    "cDC",
    "pDC",
    "Macrophage",
    "Mast",
    "Neutrophil",
    "Unidentifiable",
]

BIOLOGICAL_LINEAGES = [
    "Malignant",
    "CAF",
    "Endothelial",
    "Plasma",
    "B cell",
    "T CD4",
    "T CD8",
    "NK",
    "cDC",
    "pDC",
    "Macrophage",
    "Mast",
    "Neutrophil",
]

PRIMARY_BALANCES = {"b1", "b2", "b3", "b4", "b9"}
FULL_MODEL_PREDICTORS = ["b1", "b2", "b3", "b4", "b9"]
WITHOUT_B1_PREDICTORS = ["b2", "b3", "b4", "b9"]
CLEAN_SUBSET_PREDICTORS = ["b1", "b3", "b9"]
REGRESSION_SPECS = {
    "full_primary": FULL_MODEL_PREDICTORS,
    "without_b1": WITHOUT_B1_PREDICTORS,
    "clean_floor_robust": CLEAN_SUBSET_PREDICTORS,
}
JOINT_MODEL_SPECS = {
    "composition_only": FULL_MODEL_PREDICTORS,
    "hallmarks_only": "ALL_HALLMARKS",
    "combined": "BALANCES_PLUS_ALL_HALLMARKS",
}
ELASTIC_NET_L1_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]
CROSS_SAMPLE_ALPHA_GRID = np.logspace(-4, 0, 30).tolist()

BALANCE_TREE = [
    {
        "balance": "b1",
        "plus": ["Malignant"],
        "minus": [
            "CAF",
            "Endothelial",
            "Plasma",
            "B cell",
            "T CD4",
            "T CD8",
            "NK",
            "cDC",
            "pDC",
            "Macrophage",
            "Mast",
            "Neutrophil",
        ],
    },
    {
        "balance": "b2",
        "plus": ["CAF", "Endothelial"],
        "minus": [
            "Plasma",
            "B cell",
            "T CD4",
            "T CD8",
            "NK",
            "cDC",
            "pDC",
            "Macrophage",
            "Mast",
            "Neutrophil",
        ],
    },
    {"balance": "b3", "plus": ["CAF"], "minus": ["Endothelial"]},
    {
        "balance": "b4",
        "plus": ["Macrophage", "cDC", "pDC", "Mast", "Neutrophil"],
        "minus": ["Plasma", "B cell", "T CD4", "T CD8", "NK"],
    },
    {
        "balance": "b5",
        "plus": ["Macrophage", "cDC", "pDC"],
        "minus": ["Mast", "Neutrophil"],
    },
    {"balance": "b6", "plus": ["Macrophage"], "minus": ["cDC", "pDC"]},
    {"balance": "b7", "plus": ["cDC"], "minus": ["pDC"]},
    {"balance": "b8", "plus": ["Mast"], "minus": ["Neutrophil"]},
    {
        "balance": "b9",
        "plus": ["T CD4", "T CD8", "NK"],
        "minus": ["B cell", "Plasma"],
    },
    {"balance": "b10", "plus": ["T CD4", "T CD8"], "minus": ["NK"]},
    {"balance": "b11", "plus": ["T CD4"], "minus": ["T CD8"]},
    {"balance": "b12", "plus": ["B cell"], "minus": ["Plasma"]},
]


def h5ad_path_for(dataset: str, sample: str) -> Path:
    return VISIUM_DIR / dataset / sample / f"{sample}.h5ad"


def stop(message: str) -> None:
    raise SystemExit(f"STOP: {message}")


def write_and_read_tree() -> list[dict[str, list[str] | str]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tree_path = OUTPUT_DIR / "balance_tree.json"
    if not tree_path.exists():
        with tree_path.open("w", encoding="utf-8") as fh:
            json.dump(BALANCE_TREE, fh, indent=2)

    with tree_path.open("r", encoding="utf-8") as fh:
        tree = json.load(fh)

    realised_path = OUTPUT_DIR / "realised_partition.json"
    if not realised_path.exists():
        with realised_path.open("w", encoding="utf-8") as fh:
            json.dump(tree, fh, indent=2)
    return tree


def build_basis(tree: list[dict[str, list[str] | str]]) -> tuple[np.ndarray, list[str]]:
    lineage_set = set(BIOLOGICAL_LINEAGES)
    balance_names: list[str] = []
    sbp = np.zeros((len(tree), len(BIOLOGICAL_LINEAGES)), dtype=int)

    for row_idx, node in enumerate(tree):
        balance = str(node["balance"])
        plus = list(node["plus"])
        minus = list(node["minus"])
        balance_names.append(balance)

        overlap = sorted(set(plus).intersection(minus))
        missing = sorted((set(plus) | set(minus)).difference(lineage_set))
        if overlap:
            stop(f"{balance} has overlapping plus/minus lineages: {overlap}")
        if missing:
            stop(f"{balance} contains unknown lineages: {missing}")
        if not plus or not minus:
            stop(f"{balance} has an empty plus or minus side.")

        for lineage in plus:
            sbp[row_idx, BIOLOGICAL_LINEAGES.index(lineage)] = 1
        for lineage in minus:
            sbp[row_idx, BIOLOGICAL_LINEAGES.index(lineage)] = -1

    expected_names = [f"b{i}" for i in range(1, 13)]
    if balance_names != expected_names:
        stop(f"Balance order mismatch. Expected {expected_names}; got {balance_names}.")

    basis = sbp_basis(sbp)
    basis_df = pd.DataFrame(
        basis,
        index=balance_names,
        columns=BIOLOGICAL_LINEAGES,
    )
    basis_path = OUTPUT_DIR / "ilr_basis_matrix.csv"
    if not basis_path.exists():
        basis_df.to_csv(basis_path)
    return basis, balance_names


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running_max = 0.0
    m = len(ordered)
    for rank, (name, p_value) in enumerate(ordered, start=1):
        adj = min((m - rank + 1) * p_value, 1.0)
        running_max = max(running_max, adj)
        adjusted[name] = running_max
    return adjusted


def read_sample_obs(dataset: str, sample: str) -> pd.DataFrame:
    path = h5ad_path_for(dataset, sample)
    if not path.exists():
        stop(f"Missing h5ad for {dataset}/{sample}: {path}")

    adata = ad.read_h5ad(path, backed="r")
    obs = adata.obs.copy()
    missing = [col for col in [SNAI1_COL, *EXPECTED_SPACET_COLUMNS] if col not in obs.columns]
    if missing:
        present_spacet = [col for col in EXPECTED_SPACET_COLUMNS if col in obs.columns]
        stop(
            f"Missing expected columns in {dataset}/{sample}: {missing}. "
            f"Present expected SpaCET columns: {present_spacet}"
        )

    keep_cols = [SNAI1_COL, *EXPECTED_SPACET_COLUMNS]
    if "spot_key" in obs.columns:
        keep_cols.append("spot_key")
    subset = obs[keep_cols].copy()
    if subset.isna().any().any():
        bad = subset.columns[subset.isna().any()].tolist()
        stop(f"Missing values in {dataset}/{sample}: {bad}")

    if (subset[EXPECTED_SPACET_COLUMNS] < 0).any().any():
        bad = subset[EXPECTED_SPACET_COLUMNS].columns[
            (subset[EXPECTED_SPACET_COLUMNS] < 0).any()
        ].tolist()
        stop(f"Negative SpaCET fractions in {dataset}/{sample}: {bad}")

    return subset


def close_biological_lineages(raw_lineages: pd.DataFrame, dataset: str, sample: str) -> pd.DataFrame:
    row_sums = raw_lineages.sum(axis=1)
    if (row_sums <= 0).any():
        n_bad = int((row_sums <= 0).sum())
        stop(f"{dataset}/{sample} has {n_bad} spots with zero biological lineage total.")

    closed = closure(raw_lineages.to_numpy(dtype=float))
    return pd.DataFrame(closed, index=raw_lineages.index, columns=raw_lineages.columns)


def compute_balances(closed: pd.DataFrame, basis: np.ndarray, names: list[str], delta: float) -> pd.DataFrame:
    replaced = multi_replace(closed.to_numpy(dtype=float), delta=delta)
    balances = ilr(replaced, basis=basis)
    return pd.DataFrame(balances, index=closed.index, columns=names)


def spearman_by_balance(balances: pd.DataFrame, score: pd.Series) -> dict[str, float]:
    out: dict[str, float] = {}
    for balance in balances.columns:
        r, _ = stats.spearmanr(balances[balance].to_numpy(), score.to_numpy())
        out[balance] = float(r)
    return out


def sign_consistency(values: pd.Series) -> float:
    median = float(values.median())
    if median == 0:
        return float((values == 0).mean())
    return float((np.sign(values) == np.sign(median)).mean())


def summarize_correlations(per_sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    primary_p: dict[str, float] = {}

    for balance, group in per_sample.groupby("balance", sort=False):
        vals = group["spearman_r"].astype(float)
        row = {
            "balance": balance,
            "balance_type": "primary" if balance in PRIMARY_BALANCES else "exploratory",
            "median_r": float(vals.median()),
            "iqr_low": float(vals.quantile(0.25)),
            "iqr_high": float(vals.quantile(0.75)),
            "sign_consistency": sign_consistency(vals),
            "n_samples": int(vals.shape[0]),
            "wilcoxon_p": np.nan,
            "holm_p": np.nan,
        }
        if balance in PRIMARY_BALANCES:
            if np.allclose(vals.to_numpy(), 0):
                p_value = 1.0
            else:
                p_value = float(stats.wilcoxon(vals.to_numpy(), zero_method="wilcox").pvalue)
            row["wilcoxon_p"] = p_value
            primary_p[balance] = p_value
        rows.append(row)

    summary = pd.DataFrame(rows)
    adjusted = holm_adjust(primary_p)
    for balance, p_value in adjusted.items():
        summary.loc[summary["balance"] == balance, "holm_p"] = p_value
    return summary


def summarize_delta_sensitivity(delta_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (balance, setting, delta), group in delta_rows.groupby(
        ["balance", "delta_setting", "delta"], sort=False
    ):
        vals = group["spearman_r"].astype(float)
        rows.append(
            {
                "balance": balance,
                "delta_setting": setting,
                "delta": float(delta),
                "median_r": float(vals.median()),
                "iqr_low": float(vals.quantile(0.25)),
                "iqr_high": float(vals.quantile(0.75)),
                "sign_consistency": sign_consistency(vals),
                "median_sign": int(np.sign(vals.median())),
            }
        )
    out = pd.DataFrame(rows)

    base = out.loc[out["delta_setting"] == "base", ["balance", "median_r", "median_sign"]]
    base = base.rename(columns={"median_r": "base_median_r", "median_sign": "base_median_sign"})
    out = out.merge(base, on="balance", how="left")
    out["median_r_change_vs_base"] = out["median_r"] - out["base_median_r"]
    out["sign_changed_vs_base"] = out["median_sign"] != out["base_median_sign"]
    return out


def write_csv_if_missing(df: pd.DataFrame, path: Path) -> None:
    if not path.exists():
        df.to_csv(path, index=False)


def extract_source_prop_mat_to_csv() -> Path:
    if not RSCRIPT.exists():
        stop(f"Rscript not found for floor verification: {RSCRIPT}")
    if not FLOOR_CHECK_RDS.exists():
        stop(f"Raw SpaCET rds not found for floor verification: {FLOOR_CHECK_RDS}")

    out_csv = OUTPUT_DIR / f"floor_verification_{FLOOR_CHECK_SAMPLE}_source_propMat.csv"
    r_code = f"""
obj <- readRDS({json.dumps(str(FLOOR_CHECK_RDS).replace(chr(92), "/"))})
pm <- obj@results$deconvolution$propMat
lineages <- c({",".join(json.dumps(x) for x in BIOLOGICAL_LINEAGES)})
missing <- setdiff(lineages, rownames(pm))
if (length(missing) > 0) {{
  stop(paste("Missing source propMat lineages:", paste(missing, collapse=", ")))
}}
pm <- pm[lineages, , drop=FALSE]
write.csv(t(pm), file={json.dumps(str(out_csv).replace(chr(92), "/"))}, quote=TRUE)
"""
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False, encoding="utf-8") as fh:
        script_path = Path(fh.name)
        fh.write(r_code)
    try:
        result = subprocess.run(
            [str(RSCRIPT), str(script_path)],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        script_path.unlink(missing_ok=True)

    if result.returncode != 0:
        stop(
            "Could not extract source SpaCET propMat from rds. "
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    if not out_csv.exists():
        stop(f"R extraction finished but source propMat CSV is missing: {out_csv}")
    return out_csv


def verify_floor_structure() -> pd.DataFrame:
    source_csv = extract_source_prop_mat_to_csv()
    source = pd.read_csv(source_csv, index_col=0)
    obs = read_sample_obs(FLOOR_CHECK_DATASET, FLOOR_CHECK_SAMPLE)
    h5ad_values = obs[BIOLOGICAL_LINEAGES].astype(float)

    common_spots = source.index.intersection(h5ad_values.index)
    match_key = "h5ad_index"
    if common_spots.empty and "spot_key" in obs.columns:
        h5ad_values = h5ad_values.copy()
        h5ad_values.index = obs["spot_key"].astype(str)
        if h5ad_values.index.has_duplicates:
            stop(f"{FLOOR_CHECK_DATASET}/{FLOOR_CHECK_SAMPLE} has duplicated spot_key values.")
        common_spots = source.index.intersection(h5ad_values.index)
        match_key = "spot_key"
    if common_spots.empty:
        stop(
            f"No common spot IDs between source propMat and h5ad for "
            f"{FLOOR_CHECK_DATASET}/{FLOOR_CHECK_SAMPLE}."
        )
    source = source.loc[common_spots, BIOLOGICAL_LINEAGES].astype(float)
    h5ad_values = h5ad_values.loc[common_spots, BIOLOGICAL_LINEAGES].astype(float)

    rows = []
    all_zero_positions_match = True
    all_floor_values_present = True
    for lineage in BIOLOGICAL_LINEAGES:
        source_zero = source[lineage] == 0
        h5ad_zero = h5ad_values[lineage] == 0
        zero_positions_match = bool(source_zero.equals(h5ad_zero))
        all_zero_positions_match = all_zero_positions_match and zero_positions_match

        source_nonzero = np.sort(source.loc[source[lineage] > 0, lineage].to_numpy())
        h5ad_nonzero = np.sort(h5ad_values.loc[h5ad_values[lineage] > 0, lineage].to_numpy())
        source_floor_examples = source_nonzero[:5].tolist()
        h5ad_floor_examples = h5ad_nonzero[:5].tolist()
        source_has_tiny_floor = bool(source_nonzero.size > 0 and source_nonzero[0] < 1e-6)
        h5ad_has_tiny_floor = bool(h5ad_nonzero.size > 0 and h5ad_nonzero[0] < 1e-6)
        all_floor_values_present = all_floor_values_present and (
            source_has_tiny_floor == h5ad_has_tiny_floor
        )

        rows.append(
            {
                "dataset": FLOOR_CHECK_DATASET,
                "sample": FLOOR_CHECK_SAMPLE,
                "lineage": lineage,
                "match_key": match_key,
                "n_common_spots": int(common_spots.shape[0]),
                "n_source_spots": int(pd.read_csv(source_csv, usecols=[0]).shape[0]),
                "n_h5ad_spots": int(obs.shape[0]),
                "source_zero_count": int(source_zero.sum()),
                "h5ad_zero_count": int(h5ad_zero.sum()),
                "zero_counts_match": int(source_zero.sum()) == int(h5ad_zero.sum()),
                "zero_positions_match": zero_positions_match,
                "source_smallest_nonzero_examples": ";".join(
                    f"{x:.12g}" for x in source_floor_examples
                ),
                "h5ad_smallest_nonzero_examples": ";".join(
                    f"{x:.12g}" for x in h5ad_floor_examples
                ),
                "source_has_values_below_1e_minus_6": source_has_tiny_floor,
                "h5ad_has_values_below_1e_minus_6": h5ad_has_tiny_floor,
            }
        )

    verdict = "MATCH" if all_zero_positions_match and all_floor_values_present else "MISMATCH"
    out = pd.DataFrame(rows)
    out["verdict"] = verdict
    out["verdict_detail"] = (
        "Floor/zero structure matches source SpaCET propMat on common h5ad spots."
        if verdict == "MATCH"
        else "Floor/zero structure differs between source SpaCET propMat and h5ad obs."
    )
    out.to_csv(OUTPUT_DIR / "floor_verification.csv", index=False)
    return out


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0:
        stop("Cannot compute R2 because SNAI1-ac score is constant within a sample.")
    return 1.0 - ss_res / ss_tot


def zscore_matrix(x: np.ndarray, mean: np.ndarray | None = None, sd: np.ndarray | None = None):
    if mean is None:
        mean = x.mean(axis=0)
    if sd is None:
        sd = x.std(axis=0, ddof=0)
    if np.any(sd == 0):
        stop("A regression predictor has zero variance within a sample.")
    return (x - mean) / sd, mean, sd


def fit_ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    xz, _, _ = zscore_matrix(x)
    design = np.column_stack([np.ones(xz.shape[0]), xz])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    pred = design @ beta
    return beta, pred, r2_score(y, pred)


def cv_r2_5fold(x: np.ndarray, y: np.ndarray, seed: int = 1) -> float:
    rng = np.random.default_rng(seed)
    idx = np.arange(y.shape[0])
    rng.shuffle(idx)
    folds = np.array_split(idx, 5)
    pred = np.full(y.shape[0], np.nan, dtype=float)

    for fold in folds:
        train = np.setdiff1d(idx, fold, assume_unique=True)
        x_train_z, mean, sd = zscore_matrix(x[train])
        x_test_z, _, _ = zscore_matrix(x[fold], mean=mean, sd=sd)
        train_design = np.column_stack([np.ones(x_train_z.shape[0]), x_train_z])
        test_design = np.column_stack([np.ones(x_test_z.shape[0]), x_test_z])
        beta = np.linalg.lstsq(train_design, y[train], rcond=None)[0]
        pred[fold] = test_design @ beta

    if np.isnan(pred).any():
        stop("Cross-validation failed to produce predictions for all spots.")
    return r2_score(y, pred)


def outer_folds(n_obs: int, seed: int = SEED) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n_obs)
    rng.shuffle(idx)
    return [fold.astype(int) for fold in np.array_split(idx, 5)]


def run_regression(balances_by_sample: dict[tuple[str, str], pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_sample_rows = []
    full_coefficient_rows = []

    for dataset, samples in COHORT_23.items():
        for sample in samples:
            balances = balances_by_sample[(dataset, sample)]
            obs = read_sample_obs(dataset, sample)
            score = obs.loc[balances.index, SNAI1_COL].astype(float).to_numpy()

            sample_results: dict[str, dict[str, float]] = {}
            for spec, predictors in REGRESSION_SPECS.items():
                x = balances[predictors].to_numpy(dtype=float)
                beta, _, in_sample_r2 = fit_ols(x, score)
                cv_r2 = cv_r2_5fold(x, score, seed=1)
                sample_results[spec] = {
                    "in_sample_r2": in_sample_r2,
                    "cv_r2": cv_r2,
                }
                if spec == "full_primary":
                    for predictor, coef in zip(predictors, beta[1:]):
                        full_coefficient_rows.append(
                            {
                                "dataset": dataset,
                                "sample": sample,
                                "balance": predictor,
                                "standardized_coefficient": float(coef),
                            }
                        )

            delta_r2 = (
                sample_results["full_primary"]["in_sample_r2"]
                - sample_results["without_b1"]["in_sample_r2"]
            )
            delta_cv_r2 = (
                sample_results["full_primary"]["cv_r2"]
                - sample_results["without_b1"]["cv_r2"]
            )
            for spec, values in sample_results.items():
                per_sample_rows.append(
                    {
                        "dataset": dataset,
                        "sample": sample,
                        "specification": spec,
                        "predictors": "+".join(REGRESSION_SPECS[spec]),
                        "in_sample_r2": values["in_sample_r2"],
                        "cv_r2": values["cv_r2"],
                        "delta_r2_full_minus_without_b1": delta_r2,
                        "delta_cv_r2_full_minus_without_b1": delta_cv_r2,
                    }
                )

    per_sample = pd.DataFrame(per_sample_rows)
    summary_rows = []
    for spec, group in per_sample.groupby("specification", sort=False):
        for metric in ["in_sample_r2", "cv_r2"]:
            vals = group[metric].astype(float)
            summary_rows.append(
                {
                    "summary_type": "specification",
                    "specification": spec,
                    "metric": metric,
                    "median": float(vals.median()),
                    "iqr_low": float(vals.quantile(0.25)),
                    "iqr_high": float(vals.quantile(0.75)),
                    "n_samples": int(vals.shape[0]),
                }
            )
    delta_vals = (
        per_sample.loc[
            per_sample["specification"] == "full_primary",
            "delta_r2_full_minus_without_b1",
        ]
        .astype(float)
        .reset_index(drop=True)
    )
    summary_rows.append(
        {
            "summary_type": "delta",
            "specification": "full_minus_without_b1",
            "metric": "in_sample_delta_r2",
            "median": float(delta_vals.median()),
            "iqr_low": float(delta_vals.quantile(0.25)),
            "iqr_high": float(delta_vals.quantile(0.75)),
            "n_samples": int(delta_vals.shape[0]),
        }
    )
    delta_cv_vals = (
        per_sample.loc[
            per_sample["specification"] == "full_primary",
            "delta_cv_r2_full_minus_without_b1",
        ]
        .astype(float)
        .reset_index(drop=True)
    )
    summary_rows.append(
        {
            "summary_type": "delta",
            "specification": "full_minus_without_b1",
            "metric": "cv_delta_r2",
            "median": float(delta_cv_vals.median()),
            "iqr_low": float(delta_cv_vals.quantile(0.25)),
            "iqr_high": float(delta_cv_vals.quantile(0.75)),
            "n_samples": int(delta_cv_vals.shape[0]),
        }
    )
    regression_summary = pd.DataFrame(summary_rows)

    coefficients = pd.DataFrame(full_coefficient_rows)
    coefficient_summary_rows = []
    for balance, group in coefficients.groupby("balance", sort=False):
        vals = group["standardized_coefficient"].astype(float)
        median = float(vals.median())
        if median == 0:
            consistency = float((vals == 0).mean())
        else:
            consistency = float((np.sign(vals) == np.sign(median)).mean())
        coefficient_summary_rows.append(
            {
                "balance": balance,
                "median_standardized_coefficient": median,
                "iqr_low": float(vals.quantile(0.25)),
                "iqr_high": float(vals.quantile(0.75)),
                "sign_consistency": consistency,
                "n_samples": int(vals.shape[0]),
            }
        )
    regression_coefficients = pd.DataFrame(coefficient_summary_rows)

    per_sample.to_csv(OUTPUT_DIR / "regression_per_sample.csv", index=False)
    regression_summary.to_csv(OUTPUT_DIR / "regression_summary.csv", index=False)
    regression_coefficients.to_csv(OUTPUT_DIR / "regression_coefficients.csv", index=False)
    return per_sample, regression_summary, regression_coefficients


def cohort_sample_to_dataset() -> dict[str, str]:
    mapping = {}
    for dataset, samples in COHORT_23.items():
        for sample in samples:
            if sample in mapping:
                stop(f"Sample {sample} appears in more than one cohort dataset.")
            mapping[sample] = dataset
    return mapping


def discover_sample_hallmark_columns(obs: pd.DataFrame, dataset: str, sample: str) -> list[str]:
    hallmark_cols = sorted(
        [
            col
            for col in obs.columns
            if col.startswith("HALLMARK_")
            and col.endswith("_score")
            and "_hot" not in col
            and "_cold" not in col
        ]
    )
    if len(hallmark_cols) != EXPECTED_HALLMARK_COUNT:
        stop(
            f"{dataset}/{sample}: expected {EXPECTED_HALLMARK_COUNT} Hallmark score columns, "
            f"found {len(hallmark_cols)}: {hallmark_cols}"
        )
    return hallmark_cols


def validate_hallmark_columns_across_samples() -> list[str]:
    expected: list[str] | None = None
    rows = []
    for dataset, samples in COHORT_23.items():
        for sample in samples:
            obs = read_sample_obs_with_hallmarks(dataset, sample)
            hallmark_cols = discover_sample_hallmark_columns(obs, dataset, sample)
            if expected is None:
                expected = hallmark_cols
            elif hallmark_cols != expected:
                stop(
                    f"{dataset}/{sample}: Hallmark columns differ from first sample. "
                    f"Expected {expected}; got {hallmark_cols}"
                )
            for col in hallmark_cols:
                rows.append({"dataset": dataset, "sample": sample, "hallmark_column": col})
    if expected is None:
        stop("No Hallmark columns discovered.")
    write_csv_if_missing(
        pd.DataFrame({"hallmark_column": expected}),
        OUTPUT_DIR / "joint_predictability_hallmark_columns_used.csv",
    )
    write_csv_if_missing(
        pd.DataFrame(rows),
        OUTPUT_DIR / "joint_predictability_hallmark_columns_checked_by_sample.csv",
    )
    return expected


def read_sample_obs_with_hallmarks(dataset: str, sample: str) -> pd.DataFrame:
    path = h5ad_path_for(dataset, sample)
    if not path.exists():
        stop(f"Missing h5ad for {dataset}/{sample}: {path}")
    adata = ad.read_h5ad(path, backed="r")
    obs = adata.obs.copy()
    missing = [col for col in [SNAI1_COL, *EXPECTED_SPACET_COLUMNS] if col not in obs.columns]
    if missing:
        stop(f"Missing expected columns in {dataset}/{sample}: {missing}")
    hallmark_cols = discover_sample_hallmark_columns(obs, dataset, sample)
    check_cols = [SNAI1_COL, *EXPECTED_SPACET_COLUMNS, *hallmark_cols]
    if obs[check_cols].isna().any().any():
        bad = obs[check_cols].columns[obs[check_cols].isna().any()].tolist()
        stop(f"NaN values in {dataset}/{sample}: {bad}")
    return obs


def compute_balances_from_obs(
    obs: pd.DataFrame,
    basis: np.ndarray,
    balance_names: list[str],
    dataset: str,
    sample: str,
    base_delta: float,
) -> pd.DataFrame:
    missing = [col for col in EXPECTED_SPACET_COLUMNS if col not in obs.columns]
    if missing:
        stop(f"Missing SpaCET columns in concat sample {sample}: {missing}")
    raw = obs[BIOLOGICAL_LINEAGES].astype(float)
    if raw.isna().any().any():
        bad = raw.columns[raw.isna().any()].tolist()
        stop(f"NaN SpaCET fractions in concat sample {sample}: {bad}")
    if (raw < 0).any().any():
        bad = raw.columns[(raw < 0).any()].tolist()
        stop(f"Negative SpaCET fractions in concat sample {sample}: {bad}")
    closed = close_biological_lineages(raw, dataset, sample)
    return compute_balances(closed, basis, balance_names, delta=base_delta)


def nested_elastic_net_cv(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    outer: list[np.ndarray],
    sample: str,
    model_name: str,
) -> tuple[float, list[dict[str, float | str]]]:
    n = y.shape[0]
    all_idx = np.arange(n)
    pred = np.full(n, np.nan, dtype=float)
    fold_rows = []

    for fold_i, test_idx in enumerate(outer, start=1):
        train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=True)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])

        inner_cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
        model = ElasticNetCV(
            l1_ratio=ELASTIC_NET_L1_GRID,
            cv=inner_cv,
            random_state=SEED,
            max_iter=50000,
            n_jobs=None,
        )
        model.fit(x_train, y[train_idx])
        pred[test_idx] = model.predict(x_test)
        fold_rows.append(
            {
                "sample": sample,
                "model": model_name,
                "outer_fold": fold_i,
                "n_train": int(train_idx.shape[0]),
                "n_test": int(test_idx.shape[0]),
                "selected_alpha": float(model.alpha_),
                "selected_l1_ratio": float(model.l1_ratio_),
                "n_nonzero_coefficients": int(np.sum(np.abs(model.coef_) > 1e-12)),
                "n_features": int(len(feature_names)),
            }
        )

    if np.isnan(pred).any():
        stop(f"Nested CV failed to predict all spots for {sample}/{model_name}.")
    return r2_score(y, pred), fold_rows


def final_elastic_net_coefficients(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> tuple[pd.DataFrame, float, float]:
    scaler = StandardScaler()
    xz = scaler.fit_transform(x)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    model = ElasticNetCV(
        l1_ratio=ELASTIC_NET_L1_GRID,
        cv=cv,
        random_state=SEED,
        max_iter=50000,
        n_jobs=None,
    )
    model.fit(xz, y)
    coef = pd.DataFrame(
        {
            "feature": feature_names,
            "standardized_coefficient": model.coef_.astype(float),
            "selected": np.abs(model.coef_) > 1e-12,
        }
    )
    return coef, float(model.alpha_), float(model.l1_ratio_)


def run_joint_predictability(
    basis: np.ndarray,
    balance_names: list[str],
    base_delta: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hallmark_cols = validate_hallmark_columns_across_samples()
    sample_to_dataset = cohort_sample_to_dataset()
    per_sample_rows = []
    fold_rows = []
    coef_rows = []

    for sample, dataset in sample_to_dataset.items():
        sample_obs = read_sample_obs_with_hallmarks(dataset, sample)
        if sample_obs.empty:
            stop(f"No rows for sample {sample} in h5ad.")
        if sample_obs[SNAI1_COL].isna().any():
            stop(f"NaN SNAI1-ac score values in sample {sample}.")
        y = sample_obs[SNAI1_COL].astype(float).to_numpy()
        if np.std(y) == 0:
            stop(f"SNAI1-ac score is constant in sample {sample}.")

        balances = compute_balances_from_obs(
            sample_obs,
            basis,
            balance_names,
            dataset,
            sample,
            base_delta,
        )
        composition_x = balances[FULL_MODEL_PREDICTORS].to_numpy(dtype=float)
        hallmark_x = sample_obs[hallmark_cols].astype(float).to_numpy()
        combined_x = np.column_stack([composition_x, hallmark_x])

        model_inputs = {
            "composition_only": (composition_x, FULL_MODEL_PREDICTORS),
            "hallmarks_only": (hallmark_x, hallmark_cols),
            "combined": (combined_x, [*FULL_MODEL_PREDICTORS, *hallmark_cols]),
        }
        outer = outer_folds(y.shape[0], seed=SEED)

        cv_r2s: dict[str, float] = {}
        for model_name, (x, features) in model_inputs.items():
            if np.isnan(x).any():
                stop(f"NaN predictors in sample {sample}, model {model_name}.")
            if np.any(np.std(x, axis=0) == 0):
                zero_features = [
                    feature
                    for feature, sd in zip(features, np.std(x, axis=0))
                    if sd == 0
                ]
                stop(
                    f"Zero-variance predictors in sample {sample}, model {model_name}: "
                    f"{zero_features}"
                )
            cv_r2, model_fold_rows = nested_elastic_net_cv(
                x, y, features, outer, sample, model_name
            )
            cv_r2s[model_name] = cv_r2
            for row in model_fold_rows:
                row["dataset"] = dataset
                fold_rows.append(row)

            if model_name in {"hallmarks_only", "combined"}:
                coef_df, alpha, l1_ratio = final_elastic_net_coefficients(x, y, features)
                coef_df["dataset"] = dataset
                coef_df["sample"] = sample
                coef_df["model"] = model_name
                coef_df["final_alpha"] = alpha
                coef_df["final_l1_ratio"] = l1_ratio
                coef_rows.append(coef_df)

        add_hallmarks = cv_r2s["combined"] - cv_r2s["composition_only"]
        add_composition = cv_r2s["combined"] - cv_r2s["hallmarks_only"]
        prior_ols = pd.read_csv(OUTPUT_DIR / "regression_per_sample.csv")
        prior_ols = prior_ols.loc[
            (prior_ols["sample"] == sample)
            & (prior_ols["specification"] == "full_primary"),
            "cv_r2",
        ]
        if prior_ols.empty:
            stop(f"Missing prior OLS full-primary CV R2 anchor for sample {sample}.")
        prior_ols_cv_r2 = float(prior_ols.iloc[0])
        composition_anchor_delta = cv_r2s["composition_only"] - prior_ols_cv_r2

        for model_name, cv_r2 in cv_r2s.items():
            per_sample_rows.append(
                {
                    "dataset": dataset,
                    "sample": sample,
                    "model": model_name,
                    "n_spots": int(y.shape[0]),
                    "n_predictors": int(len(model_inputs[model_name][1])),
                    "cv_r2": cv_r2,
                    "combined_minus_composition_only": add_hallmarks,
                    "combined_minus_hallmarks_only": add_composition,
                    "prior_ols_full_primary_cv_r2": prior_ols_cv_r2,
                    "composition_elastic_net_minus_prior_ols_cv_r2": composition_anchor_delta,
                }
            )

    per_sample = pd.DataFrame(per_sample_rows)
    per_sample.to_csv(OUTPUT_DIR / "joint_predictability_elasticnet_per_sample.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(
        OUTPUT_DIR / "joint_predictability_elasticnet_outer_folds.csv",
        index=False,
    )

    summary_rows = []
    for model_name, group in per_sample.groupby("model", sort=False):
        vals = group["cv_r2"].astype(float)
        summary_rows.append(
            {
                "summary_type": "model_cv_r2",
                "model": model_name,
                "metric": "cv_r2",
                "median": float(vals.median()),
                "iqr_low": float(vals.quantile(0.25)),
                "iqr_high": float(vals.quantile(0.75)),
                "n_samples": int(vals.shape[0]),
            }
        )
    unique_samples = per_sample.drop_duplicates(["dataset", "sample"])
    for metric, label in [
        ("combined_minus_composition_only", "hallmarks_added_over_composition"),
        ("combined_minus_hallmarks_only", "composition_added_over_hallmarks"),
        (
            "composition_elastic_net_minus_prior_ols_cv_r2",
            "composition_elastic_net_minus_prior_ols_anchor",
        ),
    ]:
        vals = unique_samples[metric].astype(float)
        summary_rows.append(
            {
                "summary_type": "increment",
                "model": label,
                "metric": metric,
                "median": float(vals.median()),
                "iqr_low": float(vals.quantile(0.25)),
                "iqr_high": float(vals.quantile(0.75)),
                "n_samples": int(vals.shape[0]),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "joint_predictability_elasticnet_summary.csv", index=False)

    coef_all = pd.concat(coef_rows, ignore_index=True)
    coef_all.to_csv(OUTPUT_DIR / "joint_predictability_elasticnet_coefficients_all.csv", index=False)
    hallmark_coef = coef_all.loc[coef_all["feature"].isin(hallmark_cols)].copy()
    coef_summary_rows = []
    for (model_name, feature), group in hallmark_coef.groupby(["model", "feature"], sort=False):
        vals = group["standardized_coefficient"].astype(float)
        median = float(vals.median())
        if median == 0:
            sign_consistency_value = float((vals == 0).mean())
        else:
            sign_consistency_value = float((np.sign(vals) == np.sign(median)).mean())
        coef_summary_rows.append(
            {
                "model": model_name,
                "hallmark": feature,
                "median_standardized_coefficient": median,
                "iqr_low": float(vals.quantile(0.25)),
                "iqr_high": float(vals.quantile(0.75)),
                "sign_consistency": sign_consistency_value,
                "selection_frequency": float(group["selected"].mean()),
                "n_samples": int(group.shape[0]),
            }
        )
    coef_summary = pd.DataFrame(coef_summary_rows)
    coef_summary.to_csv(
        OUTPUT_DIR / "joint_predictability_elasticnet_hallmark_coefficients.csv",
        index=False,
    )

    report = {
        "input_h5ad_source": "23 per-sample h5ad files from h5ad_path_for(dataset, sample)",
        "score_column": SNAI1_COL,
        "random_seed": SEED,
        "n_samples": 23,
        "hallmark_count": len(hallmark_cols),
        "hallmark_columns": hallmark_cols,
        "models": JOINT_MODEL_SPECS,
        "estimator": "ElasticNetCV nested inside outer 5-fold CV per sample",
        "outer_cv": "same shuffled 5 folds per sample/model, generated with numpy.default_rng(seed=1)",
        "inner_cv": "KFold(n_splits=5, shuffle=True, random_state=1)",
        "l1_ratio_grid": ELASTIC_NET_L1_GRID,
    }
    (OUTPUT_DIR / "joint_predictability_elasticnet_manifest.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return per_sample, summary, coef_summary


def zscore_series_within_sample(values: pd.Series, dataset: str, sample: str, column: str) -> pd.Series:
    arr = values.astype(float)
    sd = float(arr.std(ddof=0))
    if sd == 0:
        stop(f"{dataset}/{sample}: zero standard deviation for {column}.")
    return (arr - float(arr.mean())) / sd


def build_cross_sample_design(
    basis: np.ndarray,
    balance_names: list[str],
    base_delta: float,
) -> tuple[pd.DataFrame, list[str]]:
    hallmark_cols = validate_hallmark_columns_across_samples()
    pd.DataFrame({"hallmark_column": hallmark_cols}).to_csv(
        OUTPUT_DIR / "cross_sample_generalization_hallmark_columns_used.csv",
        index=False,
    )
    rows = []
    for dataset, samples in COHORT_23.items():
        for sample in samples:
            obs = read_sample_obs_with_hallmarks(dataset, sample)
            balances = compute_balances_from_obs(
                obs,
                basis,
                balance_names,
                dataset,
                sample,
                base_delta,
            )
            sample_df = pd.DataFrame(
                {
                    "dataset": dataset,
                    "sample": sample,
                    "spot_id": obs.index.astype(str),
                    "SNAI1_ac_z": zscore_series_within_sample(
                        obs[SNAI1_COL], dataset, sample, SNAI1_COL
                    ).to_numpy(),
                },
                index=obs.index,
            )
            for col in FULL_MODEL_PREDICTORS:
                sample_df[col] = zscore_series_within_sample(
                    balances[col], dataset, sample, col
                ).to_numpy()
            for col in hallmark_cols:
                sample_df[col] = zscore_series_within_sample(
                    obs[col], dataset, sample, col
                ).to_numpy()
            if sample_df[["SNAI1_ac_z", *FULL_MODEL_PREDICTORS, *hallmark_cols]].isna().any().any():
                bad = sample_df.columns[sample_df.isna().any()].tolist()
                stop(f"{dataset}/{sample}: NaNs after within-sample standardization: {bad}")
            rows.append(sample_df.reset_index(drop=True))
    design = pd.concat(rows, ignore_index=True)
    design_manifest = (
        design.groupby(["dataset", "sample"], sort=True)
        .size()
        .reset_index(name="n_spots")
    )
    design_manifest["n_hallmark_columns"] = len(hallmark_cols)
    design_manifest["n_composition_predictors"] = len(FULL_MODEL_PREDICTORS)
    design_manifest["zscore_scope"] = "within_sample"
    design_manifest.to_csv(
        OUTPUT_DIR / "cross_sample_generalization_standardized_design_manifest.csv",
        index=False,
    )
    return design, hallmark_cols


def fit_grouped_elastic_net(
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_groups: np.ndarray,
) -> ElasticNetCV:
    n_groups = int(pd.Series(train_groups).nunique())
    if n_groups < 2:
        stop("Need at least two training groups for grouped inner CV.")
    cv = list(
        GroupKFold(n_splits=min(5, n_groups)).split(
            x_train,
            y_train,
            groups=train_groups,
        )
    )
    model = ElasticNetCV(
        l1_ratio=ELASTIC_NET_L1_GRID,
        alphas=np.array(CROSS_SAMPLE_ALPHA_GRID, dtype=float),
        cv=cv,
        random_state=SEED,
        max_iter=50000,
        n_jobs=None,
    )
    model.fit(x_train, y_train)
    return model


def run_cross_sample_generalization(
    basis: np.ndarray,
    balance_names: list[str],
    base_delta: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    design, hallmark_cols = build_cross_sample_design(basis, balance_names, base_delta)
    model_features = {
        "composition_only": FULL_MODEL_PREDICTORS,
        "hallmarks_only": hallmark_cols,
        "combined": [*FULL_MODEL_PREDICTORS, *hallmark_cols],
    }
    rows = []
    coef_rows = []

    # Scheme A: leave one sample out.
    for held_sample in sorted(design["sample"].unique()):
        held_mask = design["sample"].eq(held_sample)
        held_dataset = str(design.loc[held_mask, "dataset"].iloc[0])
        train = design.loc[~held_mask].copy()
        test = design.loc[held_mask].copy()
        for model_name, features in model_features.items():
            model = fit_grouped_elastic_net(
                train[features].to_numpy(dtype=float),
                train["SNAI1_ac_z"].to_numpy(dtype=float),
                train["sample"].to_numpy(),
            )
            pred = model.predict(test[features].to_numpy(dtype=float))
            score = r2_score(test["SNAI1_ac_z"].to_numpy(dtype=float), pred)
            rows.append(
                {
                    "scheme": "LOSO",
                    "held_out_dataset": held_dataset,
                    "held_out_sample": held_sample,
                    "model": model_name,
                    "n_train_spots": int(train.shape[0]),
                    "n_test_spots": int(test.shape[0]),
                    "n_train_samples": int(train["sample"].nunique()),
                    "n_test_samples": 1,
                    "n_features": int(len(features)),
                    "r2": score,
                    "selected_alpha": float(model.alpha_),
                    "selected_l1_ratio": float(model.l1_ratio_),
                    "n_nonzero_coefficients": int(np.sum(np.abs(model.coef_) > 1e-12)),
                }
            )
            if model_name == "hallmarks_only":
                for feature, coef in zip(features, model.coef_):
                    coef_rows.append(
                        {
                            "scheme": "LOSO",
                            "held_out_dataset": held_dataset,
                            "held_out_sample": held_sample,
                            "hallmark": feature,
                            "standardized_coefficient": float(coef),
                            "selected": bool(abs(coef) > 1e-12),
                            "selected_alpha": float(model.alpha_),
                            "selected_l1_ratio": float(model.l1_ratio_),
                        }
                    )

    # Scheme B: leave one dataset out; score is still recorded per held-out sample.
    for held_dataset in sorted(design["dataset"].unique()):
        train = design.loc[~design["dataset"].eq(held_dataset)].copy()
        test_dataset = design.loc[design["dataset"].eq(held_dataset)].copy()
        for model_name, features in model_features.items():
            model = fit_grouped_elastic_net(
                train[features].to_numpy(dtype=float),
                train["SNAI1_ac_z"].to_numpy(dtype=float),
                train["sample"].to_numpy(),
            )
            for held_sample, test in test_dataset.groupby("sample", sort=True):
                pred = model.predict(test[features].to_numpy(dtype=float))
                score = r2_score(test["SNAI1_ac_z"].to_numpy(dtype=float), pred)
                rows.append(
                    {
                        "scheme": "LODO",
                        "held_out_dataset": held_dataset,
                        "held_out_sample": held_sample,
                        "model": model_name,
                        "n_train_spots": int(train.shape[0]),
                        "n_test_spots": int(test.shape[0]),
                        "n_train_samples": int(train["sample"].nunique()),
                        "n_test_samples": int(test_dataset["sample"].nunique()),
                        "n_features": int(len(features)),
                        "r2": score,
                        "selected_alpha": float(model.alpha_),
                        "selected_l1_ratio": float(model.l1_ratio_),
                        "n_nonzero_coefficients": int(np.sum(np.abs(model.coef_) > 1e-12)),
                    }
                )

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "cross_sample_generalization_elasticnet_per_heldout_sample.csv", index=False)

    summary_rows = []
    for (scheme, model_name), group in results.groupby(["scheme", "model"], sort=False):
        vals = group["r2"].astype(float)
        summary_rows.append(
            {
                "summary_type": "overall",
                "scheme": scheme,
                "held_out_dataset": "all",
                "model": model_name,
                "median_r2": float(vals.median()),
                "iqr_low": float(vals.quantile(0.25)),
                "iqr_high": float(vals.quantile(0.75)),
                "n_heldout_samples": int(vals.shape[0]),
            }
        )
    for (scheme, dataset, model_name), group in results.groupby(
        ["scheme", "held_out_dataset", "model"], sort=False
    ):
        vals = group["r2"].astype(float)
        summary_rows.append(
            {
                "summary_type": "by_dataset",
                "scheme": scheme,
                "held_out_dataset": dataset,
                "model": model_name,
                "median_r2": float(vals.median()),
                "iqr_low": float(vals.quantile(0.25)),
                "iqr_high": float(vals.quantile(0.75)),
                "n_heldout_samples": int(vals.shape[0]),
            }
        )
    summary = pd.DataFrame(summary_rows)

    within = pd.read_csv(OUTPUT_DIR / "joint_predictability_elasticnet_per_sample.csv")
    ladder_rows = []
    for model_name in ["composition_only", "hallmarks_only", "combined"]:
        vals = within.loc[within["model"].eq(model_name), "cv_r2"].astype(float)
        ladder_rows.append(
            {
                "model": model_name,
                "rung": "within_sample_outer_cv",
                "median_r2": float(vals.median()),
                "iqr_low": float(vals.quantile(0.25)),
                "iqr_high": float(vals.quantile(0.75)),
                "n_units": int(vals.shape[0]),
            }
        )
        for scheme in ["LOSO", "LODO"]:
            vals = results.loc[
                results["scheme"].eq(scheme) & results["model"].eq(model_name),
                "r2",
            ].astype(float)
            ladder_rows.append(
                {
                    "model": model_name,
                    "rung": scheme,
                    "median_r2": float(vals.median()),
                    "iqr_low": float(vals.quantile(0.25)),
                    "iqr_high": float(vals.quantile(0.75)),
                    "n_units": int(vals.shape[0]),
                }
            )
    ladder = pd.DataFrame(ladder_rows)
    ladder.to_csv(OUTPUT_DIR / "cross_sample_generalization_ladder_summary.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "cross_sample_generalization_summary.csv", index=False)

    coef = pd.DataFrame(coef_rows)
    coef.to_csv(OUTPUT_DIR / "cross_sample_generalization_loso_hallmark_coefficients_all.csv", index=False)
    coef_summary_rows = []
    for hallmark, group in coef.groupby("hallmark", sort=False):
        vals = group["standardized_coefficient"].astype(float)
        median = float(vals.median())
        if median == 0:
            consistency = float((vals == 0).mean())
        else:
            consistency = float((np.sign(vals) == np.sign(median)).mean())
        coef_summary_rows.append(
            {
                "hallmark": hallmark,
                "median_standardized_coefficient": median,
                "iqr_low": float(vals.quantile(0.25)),
                "iqr_high": float(vals.quantile(0.75)),
                "sign_consistency": consistency,
                "selection_frequency": float(group["selected"].mean()),
                "n_loso_folds": int(group.shape[0]),
            }
        )
    coef_summary = pd.DataFrame(coef_summary_rows)
    coef_summary.to_csv(
        OUTPUT_DIR / "cross_sample_generalization_loso_hallmark_coefficients.csv",
        index=False,
    )

    synthesis = {
        "seed": SEED,
        "standardization": "outcome and predictors z-scored within each sample before pooling",
        "inner_cv": "GroupKFold by sample on training data only",
        "alpha_grid": CROSS_SAMPLE_ALPHA_GRID,
        "l1_ratio_grid": ELASTIC_NET_L1_GRID,
        "hallmark_columns": hallmark_cols,
        "outputs": [
            "cross_sample_generalization_elasticnet_per_heldout_sample.csv",
            "cross_sample_generalization_summary.csv",
            "cross_sample_generalization_ladder_summary.csv",
            "cross_sample_generalization_loso_hallmark_coefficients.csv",
        ],
    }
    (OUTPUT_DIR / "cross_sample_generalization_manifest.json").write_text(
        json.dumps(synthesis, indent=2),
        encoding="utf-8",
    )
    return results, ladder, coef_summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tree = write_and_read_tree()
    basis, balance_names = build_basis(tree)
    base_delta = (1 / len(BIOLOGICAL_LINEAGES)) ** 2
    delta_settings = {
        "base_div10": base_delta / 10,
        "base": base_delta,
        "base_x10": base_delta * 10,
    }

    per_sample_rows = []
    zero_rows = []
    closure_rows = []
    sanity_rows = []
    delta_rows = []
    balances_by_sample: dict[tuple[str, str], pd.DataFrame] = {}
    first_sample_seen = False

    for dataset, samples in COHORT_23.items():
        for sample in samples:
            obs = read_sample_obs(dataset, sample)
            raw = obs[BIOLOGICAL_LINEAGES].astype(float)
            score = obs[SNAI1_COL].astype(float)

            closed = close_biological_lineages(raw, dataset, sample)
            closed_sums = closed.sum(axis=1)
            before_sums = raw.sum(axis=1)
            closure_rows.append(
                {
                    "dataset": dataset,
                    "sample": sample,
                    "n_spots": int(closed.shape[0]),
                    "bio_sum_before_min": float(before_sums.min()),
                    "bio_sum_before_median": float(before_sums.median()),
                    "bio_sum_before_max": float(before_sums.max()),
                    "closed_sum_min": float(closed_sums.min()),
                    "closed_sum_median": float(closed_sums.median()),
                    "closed_sum_max": float(closed_sums.max()),
                }
            )

            zero_mask = closed <= 0
            total_replaced = int(zero_mask.to_numpy().sum())
            for lineage in BIOLOGICAL_LINEAGES:
                zero_rows.append(
                    {
                        "dataset": dataset,
                        "sample": sample,
                        "lineage": lineage,
                        "n_spots": int(closed.shape[0]),
                        "zero_count_lineage": int(zero_mask[lineage].sum()),
                        "zero_fraction_lineage": float(zero_mask[lineage].mean()),
                        "delta_base": float(base_delta),
                        "total_values_replaced_sample": total_replaced,
                    }
                )

            balances = compute_balances(closed, basis, balance_names, delta=base_delta)
            balances_by_sample[(dataset, sample)] = balances
            corr = spearman_by_balance(balances, score)
            for balance, r in corr.items():
                per_sample_rows.append(
                    {
                        "dataset": dataset,
                        "sample": sample,
                        "balance": balance,
                        "spearman_r": r,
                    }
                )

            for setting, delta in delta_settings.items():
                delta_balances = compute_balances(closed, basis, balance_names, delta=delta)
                delta_corr = spearman_by_balance(delta_balances[list(PRIMARY_BALANCES)], score)
                for balance, r in delta_corr.items():
                    delta_rows.append(
                        {
                            "dataset": dataset,
                            "sample": sample,
                            "balance": balance,
                            "delta_setting": setting,
                            "delta": float(delta),
                            "spearman_r": r,
                        }
                    )

            if not first_sample_seen:
                for spot_id in list(raw.index[:2]):
                    row = {
                        "dataset": dataset,
                        "sample": sample,
                        "spot_id": str(spot_id),
                    }
                    for lineage in BIOLOGICAL_LINEAGES:
                        row[f"raw_{lineage}"] = float(raw.loc[spot_id, lineage])
                    for balance in balance_names:
                        row[balance] = float(balances.loc[spot_id, balance])
                    sanity_rows.append(row)
                first_sample_seen = True

    per_sample = pd.DataFrame(per_sample_rows)
    if per_sample.groupby("balance")["sample"].count().min() != 23:
        counts = per_sample.groupby("balance")["sample"].count().to_dict()
        stop(f"Not all balances have 23 per-sample coefficients: {counts}")

    summary = summarize_correlations(per_sample)
    delta_sensitivity = summarize_delta_sensitivity(pd.DataFrame(delta_rows))

    write_csv_if_missing(
        per_sample,
        OUTPUT_DIR / "ilr_balance_correlations_per_sample.csv",
    )
    write_csv_if_missing(summary, OUTPUT_DIR / "ilr_balance_summary.csv")
    write_csv_if_missing(pd.DataFrame(zero_rows), OUTPUT_DIR / "zero_sparsity_report.csv")
    write_csv_if_missing(pd.DataFrame(closure_rows), OUTPUT_DIR / "closure_check.csv")
    write_csv_if_missing(pd.DataFrame(sanity_rows), OUTPUT_DIR / "sanity_two_spots.csv")
    write_csv_if_missing(
        delta_sensitivity,
        OUTPUT_DIR / "delta_sensitivity_primary_balances.csv",
    )

    floor_verification = verify_floor_structure()
    regression_per_sample, regression_summary, regression_coefficients = run_regression(
        balances_by_sample
    )
    joint_per_sample_path = OUTPUT_DIR / "joint_predictability_elasticnet_per_sample.csv"
    joint_summary_path = OUTPUT_DIR / "joint_predictability_elasticnet_summary.csv"
    joint_coefficients_path = (
        OUTPUT_DIR / "joint_predictability_elasticnet_hallmark_coefficients.csv"
    )
    if (
        joint_per_sample_path.exists()
        and joint_summary_path.exists()
        and joint_coefficients_path.exists()
    ):
        joint_per_sample = pd.read_csv(joint_per_sample_path)
        joint_summary = pd.read_csv(joint_summary_path)
        joint_hallmark_coefficients = pd.read_csv(joint_coefficients_path)
    else:
        joint_per_sample, joint_summary, joint_hallmark_coefficients = (
            run_joint_predictability(
                basis,
                balance_names,
                base_delta,
            )
        )
    cross_generalization, cross_ladder, cross_hallmark_coefficients = (
        run_cross_sample_generalization(
            basis,
            balance_names,
            base_delta,
        )
    )

    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(
            {
                "skbio_version": skbio.__version__,
                "composition_functions": {
                    "zero_replacement": "skbio.stats.composition.multi_replace",
                    "basis": "skbio.stats.composition.sbp_basis",
                    "transform": "skbio.stats.composition.ilr",
                },
                "base_delta": base_delta,
                "n_samples": 23,
                "score_column": SNAI1_COL,
                "lineages": BIOLOGICAL_LINEAGES,
                "primary_balances": sorted(PRIMARY_BALANCES),
                "regression": {
                    "specifications": REGRESSION_SPECS,
                    "ols": "within-sample; predictors z-scored within sample",
                    "cv": "within-sample 5-fold; shuffled with random seed 1; fold scaling fit on training spots only",
                },
                "joint_predictability": {
                    "input_h5ad_source": "23 per-sample h5ad files from h5ad_path_for(dataset, sample)",
                    "models": JOINT_MODEL_SPECS,
                    "estimator": "ElasticNetCV with nested within-sample outer 5-fold CV",
                    "seed": SEED,
                    "l1_ratio_grid": ELASTIC_NET_L1_GRID,
                    "outputs_prefix": "joint_predictability_elasticnet",
                },
                "cross_sample_generalization": {
                    "models": JOINT_MODEL_SPECS,
                    "schemes": ["LOSO", "LODO"],
                    "standardization": "outcome and predictors z-scored within each sample before any cross-sample training",
                    "estimator": "ElasticNet tuned by GridSearchCV with GroupKFold over training samples",
                    "seed": SEED,
                    "alpha_grid": CROSS_SAMPLE_ALPHA_GRID,
                    "l1_ratio_grid": ELASTIC_NET_L1_GRID,
                    "outputs_prefix": "cross_sample_generalization_elasticnet",
                },
                "floor_verification": {
                    "dataset": FLOOR_CHECK_DATASET,
                    "sample": FLOOR_CHECK_SAMPLE,
                    "source_rds": str(FLOOR_CHECK_RDS),
                    "source_matrix": "SpaCET_obj@results$deconvolution$propMat",
                },
                "out_of_scope": ["figures", "malignant_dropped_tme_only"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("ILR first-pass analysis complete.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"scikit-bio version: {skbio.__version__}")
    print(f"Base delta: {base_delta:.8f}")
    print("\nPrimary balance summary:")
    cols = ["balance", "median_r", "iqr_low", "iqr_high", "sign_consistency", "holm_p"]
    print(summary.loc[summary["balance_type"] == "primary", cols].to_string(index=False))
    print("\nExploratory balance summary:")
    cols = ["balance", "median_r", "iqr_low", "iqr_high", "sign_consistency"]
    print(summary.loc[summary["balance_type"] == "exploratory", cols].to_string(index=False))
    print("\nDelta sensitivity:")
    cols = [
        "balance",
        "delta_setting",
        "median_r",
        "median_r_change_vs_base",
        "sign_changed_vs_base",
    ]
    print(delta_sensitivity[cols].to_string(index=False))
    print("\nFloor verification:")
    verdict = floor_verification["verdict"].iloc[0]
    detail = floor_verification["verdict_detail"].iloc[0]
    print(f"{FLOOR_CHECK_DATASET}/{FLOOR_CHECK_SAMPLE}: {verdict} - {detail}")
    print("\nRegression R2 summary:")
    print(regression_summary.to_string(index=False))
    print("\nFull-model standardized coefficients:")
    print(regression_coefficients.to_string(index=False))
    print("\nJoint predictability nested elastic-net summary:")
    print(joint_summary.to_string(index=False))
    top = (
        joint_hallmark_coefficients.loc[
            joint_hallmark_coefficients["model"].eq("combined")
        ]
        .sort_values(
            ["selection_frequency", "sign_consistency", "median_standardized_coefficient"],
            ascending=[False, False, False],
        )
        .head(12)
    )
    print("\nTop combined-model Hallmark coefficient summary:")
    print(top.to_string(index=False))
    print("\nCross-sample generalization ladder:")
    print(cross_ladder.to_string(index=False))
    loso_top = (
        cross_hallmark_coefficients.sort_values(
            ["selection_frequency", "sign_consistency", "median_standardized_coefficient"],
            ascending=[False, False, False],
        )
        .head(12)
    )
    print("\nTop LOSO Hallmark coefficient summary:")
    print(loso_top.to_string(index=False))


if __name__ == "__main__":
    main()
