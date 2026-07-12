from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from scipy.stats import binomtest, hypergeom, norm
from sklearn.metrics import adjusted_rand_score


SEED = 20260622
ROOT = Path(r"D:\HGSOC_Spatial_Atlas")
PROJECT = Path(r"C:\Users\luchu\Documents\MSc\Master Thesis")
SCRIPT_DIR = PROJECT / "Code" / "HGSOC_Spatial_Atlas" / "scripts"
RUN_DIR = ROOT / "05_analysis_ready" / "20260424_definition3b_definition4_raw_geneNMF"
CNMF_ROOT = ROOT / "05_analysis_ready" / "S3_cNMF_Tumor_Programs"
CNMF_RUNS = CNMF_ROOT / "cnmf_runs"
MATRIX_DIR = CNMF_ROOT / "jaccard_raw_matrices"
MANUAL_DIR = MATRIX_DIR / "inspection_exports_average" / "variantB_nonjunk_manual_cut_v2"
FINAL_MEMBERS = MANUAL_DIR / "subcluster_signatures_scoring" / "signatures" / "manual_subcluster_program_members.csv"
FINAL_SIGNATURES = MANUAL_DIR / "subcluster_signatures_scoring" / "signatures" / "manual_subcluster_recurrent_gene_signatures_long.csv"
FINAL_SANITY = MANUAL_DIR / "subcluster_signatures_scoring" / "manual_subcluster_sanity_check.csv"
TOP50_FILE = MANUAL_DIR / "variantB_original_top50_positive_gene_spectra_score_long.csv"
VARIANT_B_MATRIX = MATRIX_DIR / "jaccard_variantB_keep_everything.csv"

HH_ROOT = RUN_DIR / "08_hh_programme_characterization"
HH_TABLES = HH_ROOT / "tables"
HH_LL_UNMATCHED = HH_ROOT / "hh_vs_ll_unmatched" / "tables"
HH_LL_MATCHED = HH_ROOT / "hh_vs_ll_malignant_matched" / "tables"
SPOT_TABLE_ROOT = RUN_DIR / "02_definition3b_mixture_programme_niches"

NATIVE_OUT = MANUAL_DIR / "native_usage_hh_meta_final_mp1_8"
RECURRENCE_OUT = MANUAL_DIR / "recurrence_specificity_diagnostics_final_mp1_8"
ROBUST_OUT = MANUAL_DIR / "robustness_assessment_final_mp1_8"
UMAP_OUT = MANUAL_DIR / "robustness_alllevels_umap_final_mp1_8"

MSIGDB_DIR = (
    PROJECT
    / "Code"
    / "HGSOC_Spatial_Atlas"
    / "00_documentation"
    / "kstar_sources_v0_2"
    / "msigdb_2025_1_Hs"
)
HALLMARK_GMT = MSIGDB_DIR / "h.all.v2025.1.Hs.symbols.gmt"
KEGG_GMT = MSIGDB_DIR / "c2.cp.kegg_legacy.v2025.1.Hs.symbols.gmt"
GO_RDS_DIR = (
    ROOT
    / "05_analysis_ready"
    / "S5plus_CSIDE_Alpha_Strengthening"
    / "runs"
    / "20260415_191521_kstar_niches_cside_alpha"
    / "01_kstar_niches"
    / "tmp"
)
GO_BP_RDS = GO_RDS_DIR / "go_bp_pathways.rds"
GO_CC_RDS = GO_RDS_DIR / "go_cc_pathways.rds"
RSCRIPT = Path(r"C:\Program Files\R\R-4.4.3\bin\x64\Rscript.exe")

PROGRAM_RE = re.compile(r"^(denisenko_2022|ju_2024|yamamoto_2025)__(.+)__K(\d+)__P(\d+)$")
GROUP_ORDER = [f"MP{i}" for i in range(1, 9)]
CONTEXT_ORDER = ["centre", "neighborhood", "spot_level"]
HEX_OFFSETS = [(-1, -1), (-1, 1), (0, -2), (0, 2), (1, -1), (1, 1)]
PSEUDOCOUNT = 1e-6

REPORT_LABELS = {
    "MP1": "angiogenic/vascular",
    "MP2": "iCAF-stress",
    "MP3": "complement-CAF",
    "MP4": "activated-myCAF",
    "MP5": "IFN/TLS immune",
    "MP6": "APC/TAM myeloid",
    "MP7": "malignant hypoxia",
    "MP8": "malignant acute-phase/secretory",
}


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_manifest(out_dir: Path, name: str, inputs: dict[str, object], outputs: list[Path], notes: list[str]) -> None:
    payload = {
        "analysis": name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__)),
        "final_mp_labels": REPORT_LABELS,
        "inputs": inputs,
        "outputs": [str(path) for path in outputs],
        "notes": notes,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    readme = [
        f"# {name}",
        "",
        f"Run timestamp: {payload['timestamp']}",
        "",
        "This rerun uses the final MP1-MP8 membership table and does not reuse the old MP/C1/C2 stale definitions.",
        "",
        "Final labels:",
    ]
    readme.extend(f"- {mp}: {label}" for mp, label in REPORT_LABELS.items())
    readme += ["", "Outputs:"]
    readme.extend(f"- {path.name}" for path in outputs)
    readme += ["", "Notes:"]
    readme.extend(f"- {note}" for note in notes)
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


def bh_adjust(values: pd.Series | np.ndarray | list[float]) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    out = np.full(vals.shape, np.nan, dtype=float)
    mask = np.isfinite(vals)
    if not mask.any():
        return out
    idx = np.where(mask)[0]
    order = idx[np.argsort(vals[idx])]
    ranked = vals[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out[order] = np.clip(adjusted, 0, 1)
    return out


def load_members() -> pd.DataFrame:
    require(FINAL_MEMBERS)
    members = pd.read_csv(FINAL_MEMBERS)
    needed = {"subcluster_id", "label", "coarse", "position", "program_id"}
    missing = needed - set(members.columns)
    if missing:
        raise ValueError(f"{FINAL_MEMBERS} lacks {sorted(missing)}")
    members = members.copy()
    members["group"] = members["subcluster_id"].astype(str)
    members["final_label"] = members["group"].map(REPORT_LABELS)
    if sorted(members["group"].unique()) != GROUP_ORDER:
        raise ValueError(f"Unexpected final MP groups: {sorted(members['group'].unique())}")
    if members["program_id"].duplicated().any():
        dup = members.loc[members["program_id"].duplicated(), "program_id"].head().tolist()
        raise ValueError(f"Duplicated final member programs: {dup}")
    return members


def program_columns(frame: pd.DataFrame) -> list[str]:
    return [str(c) for c in frame.columns if PROGRAM_RE.match(str(c))]


def parse_program(program_id: str) -> tuple[str, str, int, int]:
    match = PROGRAM_RE.match(str(program_id))
    if not match:
        raise ValueError(f"Cannot parse program id: {program_id}")
    return match.group(1), match.group(2), int(match.group(3)), int(match.group(4))


def spectra_path(program_id: str) -> tuple[Path, int]:
    dataset, sample, k_value, p_value = parse_program(program_id)
    sample_label = f"{dataset}__{sample}"
    return CNMF_RUNS / sample_label / f"{sample_label}.gene_spectra_score.k_{k_value}.dt_0_5.txt", p_value


def load_program_scores(program_ids: list[str], top_n: int = 100) -> tuple[dict[str, list[str]], dict[str, dict[str, float]]]:
    cache: dict[Path, pd.DataFrame] = {}
    top_genes: dict[str, list[str]] = {}
    top_scores: dict[str, dict[str, float]] = {}
    for program_id in program_ids:
        path, p_value = spectra_path(program_id)
        require(path)
        if path not in cache:
            df = pd.read_csv(path, sep="\t", index_col=0)
            df.index = [int(str(idx).replace("GEP", "").replace(".0", "")) for idx in df.index]
            cache[path] = df
        scores = cache[path].loc[p_value].astype(float).sort_values(ascending=False)
        top_genes[program_id] = [str(gene) for gene in scores.index[:top_n]]
        top_scores[program_id] = {str(gene): float(score) for gene, score in scores.iloc[:top_n].items()}
    return top_genes, top_scores


def load_gene_spectra_universe(program_ids: list[str]) -> set[str]:
    cache: dict[Path, pd.DataFrame] = {}
    universe: set[str] = set()
    for program_id in program_ids:
        path, _ = spectra_path(program_id)
        require(path)
        if path not in cache:
            cache[path] = pd.read_csv(path, sep="\t", index_col=0, nrows=1)
        universe.update(str(gene) for gene in cache[path].columns)
    return universe


def jaccard_matrix(program_ids: list[str], top_genes: dict[str, list[str]], n: int) -> pd.DataFrame:
    sets = {pid: set(top_genes[pid][:n]) for pid in program_ids}
    arr = np.zeros((len(program_ids), len(program_ids)), dtype=float)
    for i, left in enumerate(program_ids):
        for j, right in enumerate(program_ids):
            union = sets[left] | sets[right]
            arr[i, j] = len(sets[left] & sets[right]) / len(union) if union else 0.0
    return pd.DataFrame(arr, index=program_ids, columns=program_ids)


def mean_within(matrix: pd.DataFrame, members: list[str]) -> float:
    ids = [pid for pid in members if pid in matrix.index]
    if len(ids) < 2:
        return float("nan")
    sub = matrix.loc[ids, ids].to_numpy(float)
    vals = sub[np.triu_indices_from(sub, k=1)]
    return float(np.nanmean(vals)) if len(vals) else float("nan")


def mean_to_rest(matrix: pd.DataFrame, members: list[str], universe: list[str]) -> float:
    member_ids = [pid for pid in members if pid in matrix.index]
    rest_ids = [pid for pid in universe if pid not in set(member_ids) and pid in matrix.index]
    if not member_ids or not rest_ids:
        return float("nan")
    vals = matrix.loc[member_ids, rest_ids].to_numpy(float).ravel()
    return float(np.nanmean(vals)) if len(vals) else float("nan")


def group_definitions(members: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {"coarse": {}, "final_mp": {}}
    for coarse, sub in members.groupby("coarse", sort=True):
        out["coarse"][str(coarse)] = sub["program_id"].astype(str).tolist()
    for group, sub in members.groupby("group", sort=True):
        out["final_mp"][str(group)] = sub["program_id"].astype(str).tolist()
    return out


def ordered_groups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["group"] = pd.Categorical(out["group"], GROUP_ORDER, ordered=True)
    if "context" in out.columns:
        contexts = [x for x in ["centre", "neighborhood", "spot_level"] if x in set(out["context"].astype(str))]
        out["context"] = pd.Categorical(out["context"], contexts, ordered=True)
        out = out.sort_values([c for c in ["contrast", "context", "group"] if c in out.columns]).reset_index(drop=True)
        out["context"] = out["context"].astype(str)
    else:
        out = out.sort_values("group").reset_index(drop=True)
    out["group"] = out["group"].astype(str)
    return out


def load_native_member_contrast(path: Path, members: pd.DataFrame, contrast: str, context: str, effect_col: str) -> pd.DataFrame:
    require(path)
    df = pd.read_csv(path)
    df = df.merge(members[["program_id", "group", "final_label"]], on="program_id", how="inner")
    df["contrast"] = contrast
    df["context"] = context
    df["source"] = "cnmf_usage_native_program"
    df["dz"] = pd.to_numeric(df[effect_col], errors="coerce")
    if "n_pairs" in df.columns:
        df["n_weight"] = pd.to_numeric(df["n_pairs"], errors="coerce")
        df["n_weight_basis"] = "n_pairs"
    elif {"hh_n", "ll_n"}.issubset(df.columns):
        df["hh_n"] = pd.to_numeric(df["hh_n"], errors="coerce")
        df["ll_n"] = pd.to_numeric(df["ll_n"], errors="coerce")
        df["n_weight"] = np.minimum(df["hh_n"], df["ll_n"])
        df["n_weight_basis"] = "min_hh_ll_n"
    else:
        df["n_weight"] = 1.0
        df["n_weight_basis"] = "unit"
    keep = [
        "source",
        "contrast",
        "context",
        "dataset",
        "sample_id_on_disk",
        "sample_label",
        "group",
        "final_label",
        "program_id",
        "n_weight",
        "n_weight_basis",
        "dz",
        "hh_mean",
        "mean_difference_hh_minus_nonhh",
        "mean_difference_hh_minus_ll",
        "matched_nonhh_mean",
        "matched_ll_mean",
        "ll_mean",
        "direction",
        "p_value",
        "fdr_bh",
    ]
    for col in keep:
        if col not in df.columns:
            df[col] = np.nan
    return df[keep]


def collapse_native_per_sample(member_effects: pd.DataFrame, source: str) -> pd.DataFrame:
    rows = []
    keys = ["source", "contrast", "context", "dataset", "sample_id_on_disk", "sample_label", "group", "final_label"]
    for group_keys, sub in member_effects.groupby(keys, sort=False):
        row = dict(zip(keys, group_keys, strict=False))
        row["source"] = source
        row["dz"] = float(pd.to_numeric(sub["dz"], errors="coerce").mean())
        row["n_weight"] = int(pd.to_numeric(sub["n_weight"], errors="coerce").max())
        row["n_member_programs_in_sample"] = int(sub["program_id"].nunique())
        row["member_programs"] = ";".join(sub["program_id"].astype(str).tolist())
        row["n_weight_basis"] = str(sub["n_weight_basis"].dropna().iloc[0]) if sub["n_weight_basis"].notna().any() else ""
        rows.append(row)
    return pd.DataFrame(rows)


def meta_analyse(per_sample: pd.DataFrame, members: pd.DataFrame, variant: str, method: str, source: str) -> pd.DataFrame:
    member_counts = members.groupby("group")["program_id"].nunique().to_dict()
    rows = []
    for (contrast, context, group), sub in per_sample.groupby(["contrast", "context", "group"], sort=False):
        weights = pd.to_numeric(sub["n_weight"], errors="coerce").to_numpy(dtype=float)
        dz = pd.to_numeric(sub["dz"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(weights) & np.isfinite(dz) & (weights > 0)
        if not mask.any():
            continue
        weights = weights[mask]
        dz = dz[mask]
        total_weight = float(weights.sum())
        weighted_mean = float(np.sum(weights * dz) / total_weight)
        meta_z = float(weighted_mean * np.sqrt(total_weight))
        meta_p = float(2 * norm.sf(abs(meta_z)))
        n_enriched = int(np.sum(dz > 0))
        n_depleted = int(np.sum(dz < 0))
        directional_n = n_enriched + n_depleted
        dominant_n = max(n_enriched, n_depleted)
        rows.append(
            {
                "group": group,
                "variant": variant,
                "method": method,
                "source": source,
                "contrast": contrast,
                "context": context,
                "n_samples": int(sub["sample_label"].nunique()),
                "n_member_programs": int(member_counts.get(group, 0)),
                "final_label": REPORT_LABELS.get(group, ""),
                "total_weight": int(total_weight),
                "weight_basis": str(sub["n_weight_basis"].dropna().iloc[0]) if "n_weight_basis" in sub.columns and sub["n_weight_basis"].notna().any() else "",
                "weighted_mean_dz": weighted_mean,
                "meta_z": meta_z,
                "meta_p": meta_p,
                "n_HH_enriched": n_enriched,
                "n_HH_depleted": n_depleted,
                "direction_consistency_fraction": float(dominant_n / directional_n) if directional_n else np.nan,
                "dominant_direction": "HH_enriched" if n_enriched >= n_depleted else "HH_depleted",
                "binomial_direction_p": float(binomtest(dominant_n, directional_n, 0.5).pvalue) if directional_n else np.nan,
                "median_abs_dz": float(np.median(np.abs(dz))),
            }
        )
    meta = pd.DataFrame(rows)
    if meta.empty:
        return meta
    meta["meta_fdr_bh"] = np.nan
    meta["direction_fdr_bh"] = np.nan
    for _, idx in meta.groupby(["contrast", "context"]).groups.items():
        meta.loc[idx, "meta_fdr_bh"] = bh_adjust(meta.loc[idx, "meta_p"])
        meta.loc[idx, "direction_fdr_bh"] = bh_adjust(meta.loc[idx, "binomial_direction_p"])
    return ordered_groups(meta)


def make_valid_frame(frame: pd.DataFrame, programs: list[str]) -> pd.DataFrame:
    keep = ["spot_id", "dataset", "sample_id_on_disk", "sample_label", "LISA_category", "Malignant", "array_row", "array_col"]
    out = frame[keep + programs].copy()
    for col in ["Malignant", "array_row", "array_col", *programs]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    valid = np.isfinite(out[["Malignant", "array_row", "array_col", *programs]].to_numpy(dtype=float)).all(axis=1)
    return out.loc[valid].reset_index(drop=True)


def neighbor_frame(frame: pd.DataFrame, programs: list[str]) -> pd.DataFrame:
    coords = frame[["array_row", "array_col"]].astype(int).reset_index(drop=True)
    lookup = {(int(r.array_row), int(r.array_col)): i for i, r in coords.iterrows()}
    usage = frame[programs].to_numpy(dtype=float)
    keep = ["spot_id", "dataset", "sample_id_on_disk", "sample_label", "LISA_category", "Malignant", "array_row", "array_col"]
    rows = []
    for i, r in coords.iterrows():
        idx = [
            lookup[(int(r.array_row) + dr, int(r.array_col) + dc)]
            for dr, dc in HEX_OFFSETS
            if (int(r.array_row) + dr, int(r.array_col) + dc) in lookup
        ]
        if not idx:
            continue
        row = {col: frame.iloc[i][col] for col in keep}
        for program, value in zip(programs, usage[idx].mean(axis=0), strict=False):
            row[program] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def optimal_match(hh: pd.DataFrame, control: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cost = np.abs(hh["Malignant"].to_numpy(dtype=float)[:, None] - control["Malignant"].to_numpy(dtype=float)[None, :])
    hh_idx, control_idx = linear_sum_assignment(cost)
    order = np.argsort(hh_idx)
    return hh.iloc[hh_idx[order]].reset_index(drop=True), control.iloc[control_idx[order]].reset_index(drop=True)


def paired_dz(hh_vals: np.ndarray, control_vals: np.ndarray) -> tuple[float, float, float, float]:
    diff = hh_vals - control_vals
    sd = float(np.std(diff, ddof=1)) if len(diff) >= 2 else math.nan
    dz = float(np.mean(diff) / sd) if np.isfinite(sd) and sd > 0 else 0.0
    return dz, float(np.mean(hh_vals)), float(np.mean(control_vals)), float(np.mean(diff))


def unpaired_d(hh_vals: np.ndarray, control_vals: np.ndarray) -> tuple[float, float, float, float]:
    n1, n2 = len(hh_vals), len(control_vals)
    if n1 < 2 or n2 < 2:
        sd = math.nan
    else:
        sd = math.sqrt(((n1 - 1) * np.var(hh_vals, ddof=1) + (n2 - 1) * np.var(control_vals, ddof=1)) / (n1 + n2 - 2))
    d = float((np.mean(hh_vals) - np.mean(control_vals)) / sd) if np.isfinite(sd) and sd > 0 else 0.0
    return d, float(np.mean(hh_vals)), float(np.mean(control_vals)), float(np.mean(hh_vals) - np.mean(control_vals))


def family_score_effects(members: pd.DataFrame) -> pd.DataFrame:
    member_map = {group: sub["program_id"].astype(str).tolist() for group, sub in members.groupby("group", sort=True)}
    rows = []
    sample_paths = sorted(SPOT_TABLE_ROOT.glob("*\\tables\\spot_level_table.csv"))
    for path in sample_paths:
        frame = pd.read_csv(path)
        programs = program_columns(frame)
        if not programs:
            continue
        spot = make_valid_frame(frame, programs)
        contexts = {"centre": spot, "neighborhood": neighbor_frame(spot, programs)}
        print(f"[native-family-score] {spot['sample_label'].iat[0]}", flush=True)
        for context, data in contexts.items():
            hh = data[data["LISA_category"].astype(str).eq("High-High")].reset_index(drop=True)
            pools = {
                "hh_vs_nonhh_malignancy_matched": data[~data["LISA_category"].astype(str).eq("High-High")].reset_index(drop=True),
                "hh_vs_ll_malignancy_matched": data[data["LISA_category"].astype(str).eq("Low-Low")].reset_index(drop=True),
                "hh_vs_ll_unmatched": data[data["LISA_category"].astype(str).eq("Low-Low")].reset_index(drop=True),
            }
            if hh.empty:
                continue
            for contrast, control in pools.items():
                if control.empty:
                    continue
                matched = contrast in {"hh_vs_nonhh_malignancy_matched", "hh_vs_ll_malignancy_matched"}
                if matched:
                    left, right = optimal_match(hh, control)
                    n_weight = len(left)
                    n_weight_basis = "n_pairs"
                else:
                    left, right = hh, control
                    n_weight = min(len(left), len(right))
                    n_weight_basis = "min_hh_ll_n"
                info = left.iloc[0]
                local_programs = set(programs)
                for group in GROUP_ORDER:
                    local_members = [pid for pid in member_map[group] if pid in local_programs]
                    if not local_members:
                        continue
                    hh_sum = left[local_members].sum(axis=1).to_numpy(dtype=float)
                    cc_sum = right[local_members].sum(axis=1).to_numpy(dtype=float)
                    if matched:
                        dz, hh_mean, control_mean, diff = paired_dz(hh_sum, cc_sum)
                    else:
                        dz, hh_mean, control_mean, diff = unpaired_d(hh_sum, cc_sum)
                    rows.append(
                        {
                            "source": "cnmf_usage_native_family_score",
                            "contrast": contrast,
                            "context": context,
                            "dataset": str(info["dataset"]),
                            "sample_id_on_disk": str(info["sample_id_on_disk"]),
                            "sample_label": str(info["sample_label"]),
                            "group": group,
                            "final_label": REPORT_LABELS[group],
                            "n_weight": int(n_weight),
                            "n_weight_basis": n_weight_basis,
                            "n_HH": int(len(hh)),
                            "n_control": int(len(control)),
                            "dz": dz,
                            "n_member_programs_in_sample": int(len(local_members)),
                            "member_programs": ";".join(local_members),
                            "hh_mean": hh_mean,
                            "control_mean": control_mean,
                            "mean_difference_hh_minus_control": diff,
                        }
                    )
    return pd.DataFrame(rows)


def plot_native_meta(meta: pd.DataFrame, path: Path) -> None:
    if meta.empty:
        return
    plot = meta.copy()
    plot["group"] = pd.Categorical(plot["group"], GROUP_ORDER, ordered=True)
    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=plot,
        x="weighted_mean_dz",
        y="group",
        hue="context",
        col="contrast",
        kind="point",
        join=False,
        height=4.8,
        aspect=0.75,
        sharex=False,
    )
    for ax in g.axes.ravel():
        ax.axvline(0, color="black", lw=0.9)
        ax.set_xlabel("weighted mean effect size (HH - control)")
        ax.set_ylabel("final MP")
    g.fig.tight_layout()
    g.fig.savefig(path, dpi=250)
    plt.close(g.fig)


def rerun_native_usage(members: pd.DataFrame) -> list[Path]:
    ensure(NATIVE_OUT)
    direct_frames = [
        load_native_member_contrast(
            HH_TABLES / "hh_spot_level_programme_contrasts.csv",
            members,
            "hh_vs_nonhh_malignancy_matched",
            "centre",
            "cohens_dz_hh_minus_nonhh",
        ),
        load_native_member_contrast(
            HH_TABLES / "hh_neighborhood_programme_contrasts.csv",
            members,
            "hh_vs_nonhh_malignancy_matched",
            "neighborhood",
            "cohens_dz_hh_minus_nonhh",
        ),
        load_native_member_contrast(
            HH_LL_MATCHED / "hh_ll_matched_spot_level_programme_contrasts.csv",
            members,
            "hh_vs_ll_malignancy_matched",
            "centre",
            "cohens_dz_hh_minus_ll",
        ),
        load_native_member_contrast(
            HH_LL_MATCHED / "hh_ll_matched_neighborhood_programme_contrasts.csv",
            members,
            "hh_vs_ll_malignancy_matched",
            "neighborhood",
            "cohens_dz_hh_minus_ll",
        ),
        load_native_member_contrast(
            HH_LL_UNMATCHED / "hh_ll_spot_level_programme_contrasts.csv",
            members,
            "hh_vs_ll_unmatched",
            "centre",
            "cohens_d_hh_minus_ll",
        ),
        load_native_member_contrast(
            HH_LL_UNMATCHED / "hh_ll_neighborhood_programme_contrasts.csv",
            members,
            "hh_vs_ll_unmatched",
            "neighborhood",
            "cohens_d_hh_minus_ll",
        ),
    ]
    member_effects = pd.concat(direct_frames, ignore_index=True)
    per_sample = collapse_native_per_sample(member_effects, "cnmf_usage_native_program")
    meta = meta_analyse(
        per_sample,
        members,
        variant="native_usage_final_mp1_8",
        method="cnmf_usage_native_program_final_mp_average",
        source="cnmf_usage_native_program",
    )
    family_per_sample = family_score_effects(members)
    family_meta = meta_analyse(
        family_per_sample,
        members,
        variant="native_usage_family_score_final_mp1_8",
        method="cnmf_usage_native_family_score_sum",
        source="cnmf_usage_native_family_score",
    )
    outputs = [
        NATIVE_OUT / "mp_usage_native_member_effects.csv",
        NATIVE_OUT / "mp_usage_native_per_sample.csv",
        NATIVE_OUT / "mp_usage_native_meta_analysis.csv",
        NATIVE_OUT / "mp_usage_native_FAMILYSCORE_per_sample.csv",
        NATIVE_OUT / "mp_usage_native_FAMILYSCORE_meta_analysis.csv",
        NATIVE_OUT / "mp_usage_native_meta_summary_plot.png",
    ]
    member_effects.to_csv(outputs[0], index=False)
    per_sample.to_csv(outputs[1], index=False)
    meta.to_csv(outputs[2], index=False)
    family_per_sample.to_csv(outputs[3], index=False)
    family_meta.to_csv(outputs[4], index=False)
    combined_meta = pd.concat([meta, family_meta], ignore_index=True)
    plot_native_meta(combined_meta, outputs[5])
    coverage = {
        "final_mp_member_programs": int(members["program_id"].nunique()),
        "member_effect_rows": int(len(member_effects)),
        "family_score_rows": int(len(family_per_sample)),
    }
    (NATIVE_OUT / "coverage_summary.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    write_manifest(
        NATIVE_OUT,
        "final_mp1_8_native_usage_hh_meta",
        {
            "final_members": str(FINAL_MEMBERS),
            "hh_nonhh_tables": str(HH_TABLES),
            "hh_ll_matched_tables": str(HH_LL_MATCHED),
            "hh_ll_unmatched_tables": str(HH_LL_UNMATCHED),
            "spot_level_tables": str(SPOT_TABLE_ROOT),
        },
        outputs + [NATIVE_OUT / "coverage_summary.json"],
        [
            "Direct native member effects are collapsed to one sample-level MP effect before meta-analysis.",
            "HH vs non-HH and HH vs LL matched contrasts use malignant-fraction matching from existing contrast tables.",
            "HH vs LL unmatched contrasts are retained as a separate contrast and use min(HH_n, LL_n) as the meta-analysis weight.",
            "Family-score sensitivity recomputes summed MP member usage within each sample from spot-level tables.",
        ],
    )
    return outputs


def group_gene_stats(
    group: str,
    members_for_group: list[str],
    all_programs: list[str],
    groups: dict[str, list[str]],
    top_genes: dict[str, list[str]],
    top_scores: dict[str, dict[str, float]],
) -> pd.DataFrame:
    member_sets = {pid: set(top_genes[pid][:50]) for pid in members_for_group}
    outside = [pid for pid in all_programs if pid not in set(members_for_group)]
    outside_sets = {pid: set(top_genes[pid][:50]) for pid in outside}
    genes = sorted(set().union(*member_sets.values())) if member_sets else []
    rows = []
    for gene in genes:
        count_in = sum(gene in geneset for geneset in member_sets.values())
        frac_in = count_in / len(members_for_group) if members_for_group else np.nan
        loadings = [top_scores[pid][gene] for pid in members_for_group if gene in top_scores[pid]]
        count_outside = sum(gene in geneset for geneset in outside_sets.values())
        other_fracs = []
        for other_group, other_members in groups.items():
            if other_group == group:
                continue
            other_count = sum(gene in top_genes[pid][:50] for pid in other_members)
            other_fracs.append(other_count / len(other_members) if other_members else 0.0)
        max_other = max(other_fracs) if other_fracs else 0.0
        rows.append(
            {
                "level": "final_mp",
                "group": group,
                "final_label": REPORT_LABELS.get(group, ""),
                "n_programs_in_group": len(members_for_group),
                "gene": gene,
                "recurrence_count_in_MP": count_in,
                "recurrence_fraction_in_MP": frac_in,
                "mean_loading_in_MP": float(np.mean(loadings)) if loadings else np.nan,
                "median_loading_in_MP": float(np.median(loadings)) if loadings else np.nan,
                "recurrence_count_outside_MP": count_outside,
                "max_recurrence_fraction_other_MP": max_other,
                "specificity_delta": frac_in - max_other,
                "specificity_ratio": frac_in / (max_other + PSEUDOCOUNT),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["group", "recurrence_count_in_MP", "recurrence_fraction_in_MP", "mean_loading_in_MP", "gene"],
            ascending=[True, False, False, False, True],
        ).reset_index(drop=True)
        out["rank_within_group"] = out.groupby("group").cumcount() + 1
    return out


def recurrence_threshold_grid(full: pd.DataFrame) -> pd.DataFrame:
    rows = []
    thresholds = [
        ("frac_ge_1_5", 1 / 5, "inspection"),
        ("frac_ge_1_4", 1 / 4, "inspection"),
        ("frac_ge_1_3", 1 / 3, "inspection"),
        ("frac_ge_1_2", 1 / 2, "inspection"),
    ]
    for group, sub in full.groupby("group", sort=False):
        n_members = int(sub["n_programs_in_group"].iloc[0])
        for label, fraction, purpose in thresholds:
            passed = sub.loc[sub["recurrence_fraction_in_MP"] >= fraction]
            rows.append(
                {
                    "group": group,
                    "final_label": REPORT_LABELS[group],
                    "threshold_label": label,
                    "threshold_fraction": fraction,
                    "threshold_count": math.ceil(fraction * n_members),
                    "candidate_count_for_group_threshold": int(len(passed)),
                    "purpose": purpose,
                }
            )
        floor_count = n_members // 2
        passed = sub.loc[sub["recurrence_count_in_MP"] >= floor_count]
        rows.append(
            {
                "group": group,
                "final_label": REPORT_LABELS[group],
                "threshold_label": "final_signature_rule_floor_n_over_2",
                "threshold_fraction": np.nan,
                "threshold_count": floor_count,
                "candidate_count_for_group_threshold": int(min(len(passed), 50)),
                "purpose": "locked_final_mp_signature",
            }
        )
    return pd.DataFrame(rows)


def locked_signatures_from_recurrence(full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    checks = []
    for group in GROUP_ORDER:
        sub = full.loc[full["group"].eq(group)].copy()
        n_programs = int(sub["n_programs_in_group"].iloc[0])
        floor_count = n_programs // 2
        strict_all = sub.loc[sub["recurrence_count_in_MP"] >= floor_count].sort_values(
            ["recurrence_count_in_MP", "mean_loading_in_MP", "gene"], ascending=[False, False, True]
        )
        strict = strict_all.head(50).copy()
        checks.append(
            {
                "group": group,
                "final_label": REPORT_LABELS[group],
                "n_programs_in_group": n_programs,
                "final_floor_count": floor_count,
                "strict_uncapped_n": int(len(strict_all)),
                "strict_locked_n": int(len(strict)),
                "capped_at_50": bool(len(strict_all) > 50),
            }
        )
        for rank, row in enumerate(strict.itertuples(index=False), start=1):
            rows.append(
                {
                    "level": "final_mp",
                    "group": group,
                    "final_label": REPORT_LABELS[group],
                    "set_type": "strict",
                    "rank_within_set": rank,
                    "gene": row.gene,
                    "n_programs_in_group": n_programs,
                    "recurrence_count_in_MP": int(row.recurrence_count_in_MP),
                    "recurrence_fraction_in_MP": float(row.recurrence_fraction_in_MP),
                    "mean_loading_in_MP": float(row.mean_loading_in_MP),
                    "median_loading_in_MP": float(row.median_loading_in_MP),
                    "recurrence_count_outside_MP": int(row.recurrence_count_outside_MP),
                    "max_recurrence_fraction_other_MP": float(row.max_recurrence_fraction_other_MP),
                    "specificity_delta": float(row.specificity_delta),
                    "specificity_ratio": float(row.specificity_ratio),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(checks)


def plot_recurrence(full: pd.DataFrame, grid: pd.DataFrame, out_dir: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey=True)
    for ax, group in zip(axes.ravel(), GROUP_ORDER, strict=False):
        sub = full.loc[full["group"].eq(group)].sort_values(
            ["recurrence_fraction_in_MP", "mean_loading_in_MP"], ascending=False
        )
        ax.plot(np.arange(1, len(sub) + 1), sub["recurrence_fraction_in_MP"], color="#2C7FB8", lw=1.8)
        ax.set_title(f"{group} {REPORT_LABELS[group]}", fontsize=9)
        ax.set_xlabel("gene rank")
        ax.set_ylim(-0.02, 1.02)
    axes[0, 0].set_ylabel("member-program recurrence fraction")
    axes[1, 0].set_ylabel("member-program recurrence fraction")
    fig.tight_layout()
    fig.savefig(out_dir / "MP_recurrence_distribution_plots.png", dpi=250)
    plt.close(fig)

    size = grid.copy()
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(
        data=size,
        x="threshold_label",
        y="candidate_count_for_group_threshold",
        hue="group",
        marker="o",
        ax=ax,
    )
    ax.tick_params(axis="x", rotation=30)
    ax.set_xlabel("threshold")
    ax.set_ylabel("candidate genes")
    fig.tight_layout()
    fig.savefig(out_dir / "MP_signature_size_vs_threshold.png", dpi=250)
    plt.close(fig)

    top = full.loc[full["rank_within_group"] <= 80].copy()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=top,
        x="recurrence_fraction_in_MP",
        y="max_recurrence_fraction_other_MP",
        hue="group",
        s=26,
        alpha=0.75,
        ax=ax,
    )
    ax.plot([0, 1], [0, 1], color="0.4", lw=1, ls="--")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    fig.tight_layout()
    fig.savefig(out_dir / "MP_specificity_diagnostic.png", dpi=250)
    plt.close(fig)


def parse_gmt(path: Path, library: str) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    terms: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            term = parts[0]
            genes = {gene.strip() for gene in parts[2:] if gene.strip()}
            terms[f"{library}:{term}"] = {"library": library, "term_id": term, "term": term, "genes": genes}
    return terms


def parse_go_rds(path: Path, library: str) -> dict[str, dict[str, object]]:
    if not path.exists() or not RSCRIPT.exists():
        return {}
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "dump_pathways.R"
        output = Path(td) / "pathways.tsv"
        script.write_text(
            """
args <- commandArgs(trailingOnly=TRUE)
obj <- readRDS(args[[1]])
con <- file(args[[2]], open='wt')
on.exit(close(con))
nms <- names(obj)
if (is.null(nms)) nms <- as.character(seq_along(obj))
for (i in seq_along(obj)) {
  genes <- unique(as.character(obj[[i]]))
  genes <- genes[!is.na(genes) & nzchar(genes)]
  writeLines(paste(c(nms[[i]], genes), collapse='\\t'), con)
}
""",
            encoding="utf-8",
        )
        try:
            subprocess.run([str(RSCRIPT), str(script), str(path), str(output)], check=True, capture_output=True, text=True)
        except Exception:
            return {}
        terms: dict[str, dict[str, object]] = {}
        for line in output.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            term = parts[0]
            genes = {gene for gene in parts[1:] if gene}
            terms[f"{library}:{term}"] = {"library": library, "term_id": term, "term": term, "genes": genes}
        return terms


def run_ora(locked: pd.DataFrame, full: pd.DataFrame, universe: set[str]) -> pd.DataFrame:
    term_sets: dict[str, dict[str, object]] = {}
    term_sets.update(parse_gmt(HALLMARK_GMT, "HALLMARK"))
    term_sets.update(parse_gmt(KEGG_GMT, "KEGG_LEGACY"))
    term_sets.update(parse_go_rds(GO_BP_RDS, "GO_BP"))
    term_sets.update(parse_go_rds(GO_CC_RDS, "GO_CC"))
    if not term_sets:
        return pd.DataFrame()
    specificity = full.set_index(["group", "gene"])[["specificity_delta", "rank_within_group"]].to_dict("index")
    rows = []
    universe = set(universe)
    universe_n = len(universe)
    for group, sig in locked.groupby("group", sort=False):
        genes = set(sig["gene"].astype(str)) & universe
        if not genes:
            continue
        for term_info in term_sets.values():
            term_genes = set(term_info["genes"]) & universe
            if not term_genes:
                continue
            overlap = genes & term_genes
            k = len(overlap)
            if k == 0:
                continue
            n = len(genes)
            K = len(term_genes)
            pvalue = float(hypergeom.sf(k - 1, universe_n, K, n))
            ordered = sorted(
                overlap,
                key=lambda gene: specificity.get((group, gene), {}).get("rank_within_group", 10**9),
            )
            deltas = [specificity.get((group, gene), {}).get("specificity_delta", np.nan) for gene in ordered]
            rows.append(
                {
                    "level": "final_mp",
                    "group": group,
                    "final_label": REPORT_LABELS[group],
                    "set_type": "strict",
                    "library": term_info["library"],
                    "term_id": term_info["term_id"],
                    "term": term_info["term"],
                    "overlap_k": k,
                    "signature_size_n": n,
                    "term_size_K": K,
                    "universe_N": universe_n,
                    "fold_enrichment": (k / n) / (K / universe_n) if n and K and universe_n else np.nan,
                    "pvalue": pvalue,
                    "overlap_genes": ";".join(ordered),
                    "n_overlap_specific": int(np.sum(np.asarray(deltas, dtype=float) >= 0.3)),
                    "n_overlap_shared": int(np.sum(np.asarray(deltas, dtype=float) < 0.1)),
                    "frac_overlap_specific": float(np.mean(np.asarray(deltas, dtype=float) >= 0.3)) if deltas else np.nan,
                    "mean_overlap_specificity_delta": float(np.nanmean(deltas)) if deltas else np.nan,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["padj"] = np.nan
    for _, idx in result.groupby(["group", "library"]).groups.items():
        result.loc[idx, "padj"] = bh_adjust(result.loc[idx, "pvalue"])
    result["driven_by_shared_only"] = result["n_overlap_specific"].eq(0) & result["n_overlap_shared"].gt(0)
    return result.sort_values(["group", "library", "padj", "fold_enrichment"], ascending=[True, True, True, False])


def rerun_recurrence(members: pd.DataFrame) -> list[Path]:
    ensure(RECURRENCE_OUT)
    final_groups = {group: sub["program_id"].astype(str).tolist() for group, sub in members.groupby("group", sort=True)}
    top50 = pd.read_csv(TOP50_FILE)
    all_programs = sorted(top50["program_id"].astype(str).unique())
    top50["rank"] = pd.to_numeric(top50["rank"], errors="coerce")
    top50["gene_spectra_score"] = pd.to_numeric(top50["gene_spectra_score"], errors="coerce")
    top50 = top50.loc[top50["rank"].le(50)].copy()
    top_genes = {
        pid: sub.sort_values("rank")["gene"].astype(str).tolist()
        for pid, sub in top50.groupby("program_id", sort=False)
    }
    top_scores = {
        pid: sub.set_index("gene")["gene_spectra_score"].astype(float).to_dict()
        for pid, sub in top50.groupby("program_id", sort=False)
    }
    frames = []
    for group in GROUP_ORDER:
        frames.append(group_gene_stats(group, final_groups[group], all_programs, final_groups, top_genes, top_scores))
    full = pd.concat(frames, ignore_index=True)
    grid = recurrence_threshold_grid(full)
    locked, checks = locked_signatures_from_recurrence(full)
    summary = (
        full.groupby(["group", "final_label"], as_index=False)
        .agg(
            n_programs=("n_programs_in_group", "first"),
            n_genes_seen=("gene", "nunique"),
            max_recurrence_fraction=("recurrence_fraction_in_MP", "max"),
            n_genes_frac_ge_1_3=("recurrence_fraction_in_MP", lambda x: int((x >= 1 / 3).sum())),
            n_genes_frac_ge_1_2=("recurrence_fraction_in_MP", lambda x: int((x >= 1 / 2).sum())),
            n_locked_final_rule=("gene", lambda _: 0),
        )
    )
    locked_counts = locked.groupby("group")["gene"].nunique().to_dict()
    summary["n_locked_final_rule"] = summary["group"].map(locked_counts).fillna(0).astype(int)
    labels = pd.DataFrame(
        [
            {
                "group": group,
                "final_label": REPORT_LABELS[group],
                "member_label_from_final_table": str(members.loc[members["group"].eq(group), "label"].iloc[0]),
                "n_programs": int(members.loc[members["group"].eq(group), "program_id"].nunique()),
            }
            for group in GROUP_ORDER
        ]
    )
    cluster_summary = members[["group", "final_label", "label", "coarse", "position", "program_id"]].copy()
    cluster_summary = cluster_summary.merge(
        locked.groupby("group")["gene"].apply(lambda x: ";".join(x.head(15))).rename("top_recurrent_genes").reset_index(),
        on="group",
        how="left",
    )
    universe = load_gene_spectra_universe(all_programs)
    ora = run_ora(locked, full, universe)
    outputs = [
        RECURRENCE_OUT / "MP_gene_recurrence_full_table.csv",
        RECURRENCE_OUT / "MP_signature_candidates_threshold_grid.csv",
        RECURRENCE_OUT / "MP_recurrence_group_summary.csv",
        RECURRENCE_OUT / "MP_signatures_locked_long.csv",
        RECURRENCE_OUT / "MP_signatures_locked_size_check.csv",
        RECURRENCE_OUT / "MP_final_labels.csv",
        RECURRENCE_OUT / "MP_cluster_summary.csv",
        RECURRENCE_OUT / "MP_cluster_summary_labelled.csv",
        RECURRENCE_OUT / "MP_signature_ORA_long.csv",
        RECURRENCE_OUT / "MP_recurrence_distribution_plots.png",
        RECURRENCE_OUT / "MP_signature_size_vs_threshold.png",
        RECURRENCE_OUT / "MP_specificity_diagnostic.png",
    ]
    full.to_csv(outputs[0], index=False)
    grid.to_csv(outputs[1], index=False)
    summary.to_csv(outputs[2], index=False)
    locked.to_csv(outputs[3], index=False)
    checks.to_csv(outputs[4], index=False)
    labels.to_csv(outputs[5], index=False)
    cluster_summary.to_csv(outputs[6], index=False)
    cluster_summary.to_csv(outputs[7], index=False)
    ora.to_csv(outputs[8], index=False)
    with pd.ExcelWriter(RECURRENCE_OUT / "MP_signatures_locked_by_group.xlsx") as writer:
        for group, sub in locked.groupby("group", sort=False):
            sub.to_excel(writer, sheet_name=group, index=False)
    if not ora.empty:
        with pd.ExcelWriter(RECURRENCE_OUT / "MP_signature_ORA_by_group.xlsx") as writer:
            for group, sub in ora.groupby("group", sort=False):
                sub.head(30).to_excel(writer, sheet_name=group, index=False)
    plot_recurrence(full, grid, RECURRENCE_OUT)
    extra_outputs = [RECURRENCE_OUT / "MP_signatures_locked_by_group.xlsx"]
    if not ora.empty:
        extra_outputs.append(RECURRENCE_OUT / "MP_signature_ORA_by_group.xlsx")
    write_manifest(
        RECURRENCE_OUT,
        "final_mp1_8_recurrence_specificity_diagnostics",
        {
            "final_members": str(FINAL_MEMBERS),
            "top50_gene_spectra_score": str(TOP50_FILE),
            "spectra_score_universe_root": str(CNMF_RUNS),
            "hallmark_gmt": str(HALLMARK_GMT),
            "kegg_gmt": str(KEGG_GMT),
            "go_bp_rds": str(GO_BP_RDS),
            "go_cc_rds": str(GO_CC_RDS),
        },
        outputs + extra_outputs,
        [
            "Recurrence and specificity are computed at the final MP1-MP8 level only.",
            "The locked signature rule follows the final scoring script: occurrence >= floor(n_programs / 2), capped at 50 genes.",
            "Specificity is reported but not used as a filter.",
            "ORA uses the explicit cNMF gene_spectra_score universe and local MSigDB/GO resources when available.",
        ],
    )
    return outputs + extra_outputs


def separation_table(members: pd.DataFrame, matrices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    groups = group_definitions(members)
    rows = []
    for level, defs in groups.items():
        universe = sorted(set().union(*[set(v) for v in defs.values()]))
        for group, ids in defs.items():
            row = {"level": level, "group": group, "final_label": REPORT_LABELS.get(group, group), "n_programs": len(ids)}
            for label, matrix in matrices.items():
                within = mean_within(matrix, ids)
                to_rest = mean_to_rest(matrix, ids, universe)
                row[f"{label}_mean_within_jaccard"] = within
                row[f"{label}_mean_within_to_rest_jaccard"] = to_rest
                row[f"{label}_separation"] = within - to_rest if np.isfinite(within) and np.isfinite(to_rest) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def cluster_ari(matrix: pd.DataFrame, members: pd.DataFrame, k: int = 8) -> float:
    assigned = members["program_id"].astype(str).tolist()
    labels_true = members.set_index("program_id").loc[assigned, "group"].astype(str).tolist()
    sub = matrix.loc[assigned, assigned]
    dist = np.clip(1 - sub.to_numpy(float), 0, 1)
    np.fill_diagonal(dist, 0)
    z = linkage(squareform(dist, checks=False), method="average")
    labels_pred = fcluster(z, t=k, criterion="maxclust")
    return float(adjusted_rand_score(labels_true, labels_pred))


def topn_robustness(members: pd.DataFrame, top_genes: dict[str, list[str]], program_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame]]:
    matrices: dict[int, pd.DataFrame] = {}
    rows_flat = []
    rows_all = []
    groups = group_definitions(members)
    for n in [30, 50, 100]:
        matrix = jaccard_matrix(program_ids, top_genes, n=n)
        matrices[n] = matrix
        rows_flat.append({"N": n, "metric": "ARI_vs_final_MP8", "family": "all_assigned_final_mp", "value": cluster_ari(matrix, members)})
        for level, defs in groups.items():
            universe = sorted(set().union(*[set(v) for v in defs.values()]))
            for group, ids in defs.items():
                within = mean_within(matrix, ids)
                to_rest = mean_to_rest(matrix, ids, universe)
                sep = within - to_rest if np.isfinite(within) and np.isfinite(to_rest) else np.nan
                rows_all.append({"N": n, "level": level, "group": group, "n_programs": len(ids), "separation": sep})
                if level == "final_mp":
                    rows_flat.append({"N": n, "metric": "separation_within_minus_rest", "family": group, "value": sep})
    return pd.DataFrame(rows_flat), pd.DataFrame(rows_all), matrices


def bootstrap_robustness(members: pd.DataFrame, top_genes: dict[str, list[str]], n_iter: int = 250) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    assigned = members["program_id"].astype(str).tolist()
    groups = {group: sub["program_id"].astype(str).tolist() for group, sub in members.groupby("group", sort=True)}
    pair_counts = {group: [] for group in GROUP_ORDER}
    program_counts = {pid: [] for pid in assigned}
    sep_records = []
    all_levels = group_definitions(members)
    for iteration in range(n_iter):
        feature_sets = {}
        for pid in assigned:
            pool = np.asarray(top_genes[pid][:100], dtype=object)
            draw = rng.choice(pool, size=min(50, len(pool)), replace=True)
            feature_sets[pid] = set(str(x) for x in draw)
        matrix = jaccard_matrix_from_sets(assigned, feature_sets)
        dist = np.clip(1 - matrix.to_numpy(float), 0, 1)
        np.fill_diagonal(dist, 0)
        z = linkage(squareform(dist, checks=False), method="average")
        pred = pd.Series(fcluster(z, t=8, criterion="maxclust"), index=assigned)
        for group, ids in groups.items():
            ids = [pid for pid in ids if pid in pred.index]
            if len(ids) < 2:
                continue
            vals = []
            for i, left in enumerate(ids):
                own_vals = []
                for right in ids:
                    if left == right:
                        continue
                    same = float(pred[left] == pred[right])
                    own_vals.append(same)
                    if ids.index(right) > i:
                        vals.append(same)
                if own_vals:
                    program_counts[left].append(float(np.mean(own_vals)))
            if vals:
                pair_counts[group].append(float(np.mean(vals)))
        for level, defs in all_levels.items():
            universe = sorted(set().union(*[set(v) for v in defs.values()]))
            for group, ids in defs.items():
                within = mean_within(matrix, ids)
                to_rest = mean_to_rest(matrix, ids, universe)
                sep = within - to_rest if np.isfinite(within) and np.isfinite(to_rest) else np.nan
                sep_records.append({"iteration": iteration + 1, "level": level, "group": group, "separation": sep, "n_programs": len(ids)})
    rows = []
    for group in GROUP_ORDER:
        ids = groups[group]
        rows.append(
            {
                "record_type": "family",
                "family": group,
                "final_label": REPORT_LABELS[group],
                "program_id": "",
                "n_programs": len(ids),
                "bootstrap_within_family_coclustering": float(np.nanmean(pair_counts[group])) if pair_counts[group] else np.nan,
                "program_mean_coclustering_with_own_family": np.nan,
            }
        )
        for pid in ids:
            rows.append(
                {
                    "record_type": "program",
                    "family": group,
                    "final_label": REPORT_LABELS[group],
                    "program_id": pid,
                    "n_programs": len(ids),
                    "bootstrap_within_family_coclustering": np.nan,
                    "program_mean_coclustering_with_own_family": float(np.nanmean(program_counts[pid])) if program_counts[pid] else np.nan,
                }
            )
    boot = pd.DataFrame(rows)
    sep = pd.DataFrame(sep_records)
    summary = (
        sep.groupby(["level", "group", "n_programs"], as_index=False)
        .agg(
            bootstrap_iterations=("separation", "count"),
            separation_mean=("separation", "mean"),
            separation_std=("separation", "std"),
            separation_min=("separation", "min"),
            fraction_separation_gt0=("separation", lambda x: float(np.mean(pd.to_numeric(x, errors="coerce") > 0))),
        )
        .sort_values(["level", "group"])
    )
    return boot, summary


def jaccard_matrix_from_sets(program_ids: list[str], sets: dict[str, set[str]]) -> pd.DataFrame:
    arr = np.zeros((len(program_ids), len(program_ids)), dtype=float)
    for i, left in enumerate(program_ids):
        for j, right in enumerate(program_ids):
            union = sets[left] | sets[right]
            arr[i, j] = len(sets[left] & sets[right]) / len(union) if union else 0.0
    return pd.DataFrame(arr, index=program_ids, columns=program_ids)


def plot_robustness(topn_all: pd.DataFrame, boot_summary: pd.DataFrame, out_dir: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(
        data=topn_all.loc[topn_all["level"].eq("final_mp")],
        x="N",
        y="separation",
        hue="group",
        marker="o",
        ax=ax,
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("within minus rest Jaccard")
    fig.tight_layout()
    fig.savefig(out_dir / "robustness_topN_final_mp_separation.png", dpi=250)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    plot = boot_summary.loc[boot_summary["level"].eq("final_mp")].copy()
    sns.barplot(data=plot, x="group", y="fraction_separation_gt0", color="#2C7FB8", ax=ax)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("bootstrap fraction separation > 0")
    fig.tight_layout()
    fig.savefig(out_dir / "robustness_bootstrap_final_mp_separation.png", dpi=250)
    plt.close(fig)


def compute_embedding(matrix: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    dist = np.clip(1 - matrix.to_numpy(float), 0, 1)
    np.fill_diagonal(dist, 0)
    try:
        import umap

        reducer = umap.UMAP(metric="precomputed", random_state=SEED, n_neighbors=10, min_dist=0.15)
        coords = reducer.fit_transform(dist)
        method = "umap_precomputed_jaccard_distance"
    except Exception:
        from sklearn.manifold import MDS

        reducer = MDS(n_components=2, dissimilarity="precomputed", random_state=SEED, normalized_stress="auto")
        coords = reducer.fit_transform(dist)
        method = "mds_precomputed_jaccard_distance_fallback"
    return pd.DataFrame({"program_id": matrix.index.astype(str), "x": coords[:, 0], "y": coords[:, 1]}), method


def add_embedding_annotations(coords: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    out = coords.merge(members[["program_id", "group", "final_label", "coarse"]], on="program_id", how="left")
    out["group"] = out["group"].fillna("not_assigned_to_MP")
    out["final_label"] = out["final_label"].fillna("not assigned to final MP1-MP8")
    out["coarse"] = out["coarse"].fillna("not_assigned")
    parsed = out["program_id"].apply(lambda pid: pd.Series(parse_program(pid), index=["dataset", "sample_id_on_disk", "k", "program_number"]))
    out = pd.concat([out, parsed], axis=1)
    return out


def plot_embedding(coords: pd.DataFrame, out_dir: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    sns.scatterplot(data=coords, x="x", y="y", hue="group", s=30, linewidth=0, ax=axes[0])
    axes[0].set_title("Final MP assignment")
    sns.scatterplot(data=coords, x="x", y="y", hue="coarse", s=30, linewidth=0, ax=axes[1])
    axes[1].set_title("Final coarse branch")
    sns.scatterplot(data=coords, x="x", y="y", hue="dataset", s=30, linewidth=0, ax=axes[2])
    axes[2].set_title("Cohort")
    for ax in axes:
        ax.set_xlabel("component 1")
        ax.set_ylabel("component 2")
        ax.legend(loc="best", fontsize=7, frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "umap_companion_panels.png", dpi=250)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    plot = coords.loc[coords["group"].ne("not_assigned_to_MP")].copy()
    sns.scatterplot(data=coords, x="x", y="y", color="0.82", s=22, linewidth=0, ax=ax)
    sns.scatterplot(data=plot, x="x", y="y", hue="group", s=42, linewidth=0, ax=ax)
    ax.set_title("Final MP1-MP8 members on Variant B Jaccard embedding")
    fig.tight_layout()
    fig.savefig(out_dir / "robustness_alllevels_summary.png", dpi=250)
    plt.close(fig)


def rerun_robustness(members: pd.DataFrame) -> list[Path]:
    ensure(ROBUST_OUT)
    ensure(UMAP_OUT)
    variant_b = pd.read_csv(VARIANT_B_MATRIX, index_col=0)
    all_programs = variant_b.index.astype(str).tolist()
    top_genes, _ = load_program_scores(all_programs, top_n=100)
    recomputed50 = jaccard_matrix(all_programs, top_genes, n=50)
    cohesion = separation_table(members, {"saved_variantB": variant_b, "recomputed_top50": recomputed50})
    topn_flat, topn_all, topn_matrices = topn_robustness(members, top_genes, all_programs)
    boot, boot_all = bootstrap_robustness(members, top_genes, n_iter=250)
    coords, method = compute_embedding(variant_b)
    coords = add_embedding_annotations(coords, members)
    outputs_robust = [
        ROBUST_OUT / "robustness_cohesion.csv",
        ROBUST_OUT / "robustness_topN.csv",
        ROBUST_OUT / "robustness_bootstrap.csv",
        ROBUST_OUT / "robustness_topN_final_mp_separation.png",
        ROBUST_OUT / "robustness_bootstrap_final_mp_separation.png",
    ]
    cohesion.to_csv(outputs_robust[0], index=False)
    topn_flat.to_csv(outputs_robust[1], index=False)
    boot.to_csv(outputs_robust[2], index=False)
    plot_robustness(topn_all, boot_all, ROBUST_OUT)
    outputs_umap = [
        UMAP_OUT / "umap_coords.csv",
        UMAP_OUT / "robustness_topN_alllevels.csv",
        UMAP_OUT / "robustness_bootstrap_alllevels.csv",
        UMAP_OUT / "umap_companion_panels.png",
        UMAP_OUT / "robustness_alllevels_summary.png",
    ]
    coords.to_csv(outputs_umap[0], index=False)
    topn_all.to_csv(outputs_umap[1], index=False)
    boot_all.to_csv(outputs_umap[2], index=False)
    plot_embedding(coords, UMAP_OUT)
    (UMAP_OUT / "embedding_method.txt").write_text(method + "\n", encoding="utf-8")
    write_manifest(
        ROBUST_OUT,
        "final_mp1_8_robustness_assessment",
        {
            "final_members": str(FINAL_MEMBERS),
            "variantB_matrix": str(VARIANT_B_MATRIX),
            "cnmf_runs": str(CNMF_RUNS),
        },
        outputs_robust,
        [
            "Cohesion/separation are computed for final MP1-MP8 and their final coarse A/B/C branches.",
            "Top-N checks use full gene_spectra_score files for N=30, 50, and 100.",
            "Bootstrap coclustering samples top-100 genes with replacement, rebuilds Jaccard matrices, and clusters into k=8.",
        ],
    )
    write_manifest(
        UMAP_OUT,
        "final_mp1_8_robustness_alllevels_umap",
        {
            "final_members": str(FINAL_MEMBERS),
            "variantB_matrix": str(VARIANT_B_MATRIX),
            "cnmf_runs": str(CNMF_RUNS),
        },
        outputs_umap + [UMAP_OUT / "embedding_method.txt"],
        [
            f"Embedding method: {method}.",
            "Coordinates include all Variant B programs, with final MP1-MP8 assignments overlaid and unassigned programs retained.",
            "All-level robustness tables include final coarse A/B/C and final MP1-MP8 levels.",
        ],
    )
    return outputs_robust + outputs_umap


def main() -> None:
    for path in [FINAL_MEMBERS, TOP50_FILE, VARIANT_B_MATRIX, HH_TABLES, SPOT_TABLE_ROOT]:
        require(path)
    members = load_members()
    print("Loaded final MP1-MP8 membership:")
    print(members.groupby(["group", "final_label"], as_index=False)["program_id"].nunique().to_string(index=False))
    native_outputs = rerun_native_usage(members)
    recurrence_outputs = rerun_recurrence(members)
    robustness_outputs = rerun_robustness(members)
    summary = {
        "native_outputs": [str(path) for path in native_outputs],
        "recurrence_outputs": [str(path) for path in recurrence_outputs],
        "robustness_outputs": [str(path) for path in robustness_outputs],
    }
    summary_path = MANUAL_DIR / "final_mp1_8_rerun_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote rerun summary: {summary_path}")


if __name__ == "__main__":
    main()
