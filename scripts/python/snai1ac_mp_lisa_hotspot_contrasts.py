from __future__ import annotations

import json
import math
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import anndata as ad
import enrichmap as em
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu, skew, spearmanr


BASE = Path(r"D:\HGSOC_Spatial_Atlas")
VARIANT_DIR = BASE / r"05_analysis_ready\S3_cNMF_Tumor_Programs\jaccard_raw_matrices\inspection_exports_average\variantB_nonjunk_manual_cut_v2"
CORR_DIR = VARIANT_DIR / "subcluster_snai1ac_correlation"
SMOOTHED_DIR = CORR_DIR / "tumor_subset_scored_h5ad"
SCORING_DIR = VARIANT_DIR / "subcluster_signatures_scoring"
MP_SIGNATURES = SCORING_DIR / "signatures" / "manual_subcluster_recurrent_gene_signatures_long.csv"
SCORE_SUMMARY = SCORING_DIR / "manual_subcluster_enrichmap_score_summary.csv"
SNAI_WEIGHTS = BASE / r"05_analysis_ready\Signature\snai1_ac_weights.json"
OUT = VARIANT_DIR / "subcluster_snai1ac_lisa"
SCRIPT = Path(__file__).resolve()
R_SCRIPT = SCRIPT.parent / "R" / "snai1ac_mp_lisa_meta_analysis.R"
RSCRIPT = Path(r"C:\Program Files\R\R-4.4.3\bin\Rscript.exe")

SEED = 20260528
K = 6
N_PERM = 999
ALPHA = 0.05
SNAI_COL = "SNAI1-ac_score"
MAL_COL = "Malignant"
COMPARTMENT_COL = "interface"
TUMOUR_LABEL = "Tumor"
IS_TUMOR_COL = "is_tumor"
MP_COLS = [
    "MP1_angiogenic_vascular_score",
    "MP2_iCAF_stress_score",
    "MP3_complement_CAF_score",
    "MP4_activated_myCAF_score",
    "MP5_IFN_TLS_immune_score",
    "MP6_APC_TAM_myeloid_score",
    "MP7_malignant_hypoxia_score",
    "MP8_malignant_acute_phase_secretory_score",
]
PRIOR_LISA = {
    "script": r"C:\Users\luchu\Documents\MSc\Master Thesis\Code\HGSOC_Spatial_Atlas\scripts\spatial_signature_analysis.py",
    "notebook": r"C:\Users\luchu\Documents\MSc\Master Thesis\Code\HGSOC_Spatial_Atlas\notebooks\spatial_sign_analysis.ipynb",
    "HH_definition": "cats[sig & (z > 0) & (lag > 0)] = 'High-High'",
    "LL_definition": "cats[sig & (z < 0) & (lag < 0)] = 'Low-Low'",
    "significance": "padj = false_discovery_control(pvals); sig = padj < FDR_THRESHOLD; FDR_THRESHOLD = 0.05",
}


def stop(msg: str) -> None:
    raise SystemExit(f"STOP: {msg}")


def require(path: Path) -> None:
    if not path.exists():
        stop(f"Missing input: {path}")


def ensure_dirs() -> None:
    for d in [
        OUT,
        OUT / "tables",
        OUT / "plots" / "sample_lisa_maps",
        OUT / "variant_U_unsmoothed_h5ad",
        OUT / "logs",
        OUT / "scripts",
    ]:
        d.mkdir(parents=True, exist_ok=True)


def score_key(subcluster_id: str, label: str) -> str:
    return f"{subcluster_id}_{label.replace('/', '_').replace('-', '_')}"


def load_mp_summary() -> pd.DataFrame:
    require(SCORE_SUMMARY)
    mp = pd.read_csv(SCORE_SUMMARY)
    need = {"subcluster_id", "label", "score_key", "score_column"}
    if not need.issubset(mp.columns):
        stop(f"{SCORE_SUMMARY} lacks columns: {sorted(need - set(mp.columns))}")
    mp = mp[["subcluster_id", "label", "score_key", "score_column"]].drop_duplicates()
    mp = mp.sort_values("subcluster_id").reset_index(drop=True)
    if mp["score_column"].tolist() != MP_COLS:
        stop(f"MP score columns mismatch. Found {mp['score_column'].tolist()}")
    return mp


def load_signatures(mp: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, float]]:
    require(MP_SIGNATURES)
    require(SNAI_WEIGHTS)
    sig = pd.read_csv(MP_SIGNATURES)
    if not {"subcluster_id", "gene"}.issubset(sig.columns):
        stop(f"{MP_SIGNATURES} lacks subcluster_id/gene")
    mp_genes: dict[str, list[str]] = {}
    for row in mp.itertuples(index=False):
        genes = sig.loc[sig["subcluster_id"].eq(row.subcluster_id), "gene"].astype(str).tolist()
        if not genes:
            stop(f"No MP genes for {row.subcluster_id}")
        mp_genes[row.score_key] = genes
    weights = {str(g): float(w) for g, w in json.loads(SNAI_WEIGHTS.read_text(encoding="utf-8")).items()}
    if not weights:
        stop(f"No SNAI1-ac weights in {SNAI_WEIGHTS}")
    return mp_genes, weights


def smoothed_paths() -> list[tuple[str, str, Path]]:
    require(SMOOTHED_DIR)
    rows = []
    for path in sorted(SMOOTHED_DIR.rglob("*.h5ad")):
        dataset = path.parent.name
        sample = path.name.split(".tumor_subset_")[0]
        rows.append((dataset, sample, path))
    if not rows:
        stop(f"No smoothed tumour h5ads found under {SMOOTHED_DIR}")
    return rows


def ensure_hires_alias(adata) -> None:
    if "spatial" not in adata.uns:
        return
    for library in adata.uns["spatial"].values():
        images = library.get("images", {})
        if "hires" not in images and "lowres" in images:
            images["hires"] = images["lowres"]


def check_columns_and_malignant(paths: list[tuple[str, str, Path]], mp: pd.DataFrame) -> dict:
    all_needed = [SNAI_COL, MAL_COL, COMPARTMENT_COL, IS_TUMOR_COL, *mp["score_column"].tolist()]
    mal_min, mal_max, mal_vals, n = math.inf, -math.inf, [], 0
    bad = []
    for dataset, sample, path in paths:
        x = sc.read_h5ad(path, backed="r")
        missing = [c for c in all_needed if c not in x.obs.columns]
        if missing:
            stop(f"{dataset}__{sample} lacks columns: {missing}")
        comp = set(x.obs[COMPARTMENT_COL].astype(str).unique())
        if comp != {TUMOUR_LABEL}:
            stop(f"{dataset}__{sample} is not a pure Tumor subset: {sorted(comp)}")
        m = pd.to_numeric(x.obs[MAL_COL], errors="coerce")
        if m.isna().any():
            bad.append(f"{dataset}__{sample}: Malignant has NA/non-numeric values")
        mal_min = min(mal_min, float(m.min()))
        mal_max = max(mal_max, float(m.max()))
        mal_vals.extend(pd.unique(m.round(8)).tolist())
        t = pd.to_numeric(x.obs[IS_TUMOR_COL], errors="coerce")
        if not set(pd.unique(t.dropna())).issubset({0, 1}):
            bad.append(f"{dataset}__{sample}: is_tumor is not binary")
        n += x.n_obs
        x.file.close()
    if bad:
        stop("; ".join(bad))
    if mal_min < -1e-9 or mal_max > 1 + 1e-9 or set(mal_vals).issubset({0.0, 1.0}):
        stop(f"Malignant is not a continuous [0,1] SpaCET fraction: min={mal_min}, max={mal_max}, unique={sorted(set(mal_vals))[:10]}")
    return {"files": len(paths), "n_tumour_spots": n, "Malignant_min": mal_min, "Malignant_max": mal_max, "is_tumor": "binary 0/1"}


def score_unsmoothed(adata, mp: pd.DataFrame, mp_genes: dict[str, list[str]], snai_weights: dict[str, float]):
    snai_present = {g: w for g, w in snai_weights.items() if g in adata.var_names}
    if not snai_present:
        stop("No SNAI1-ac genes present for variant U re-score")
    em.tl.score(
        adata=adata,
        gene_set=list(snai_present.keys()),
        gene_weights={"SNAI1-ac": snai_present},
        score_key="SNAI1-ac",
        smoothing=False,
        correct_spatial_covariates=True,
        batch_key=None,
    )
    if SNAI_COL not in adata.obs:
        stop("Variant U SNAI1-ac re-score did not create SNAI1-ac_score")
    for row in mp.itertuples(index=False):
        genes = [g for g in mp_genes[row.score_key] if g in adata.var_names]
        if not genes:
            stop(f"No genes present for {row.subcluster_id} in variant U")
        em.tl.score(
            adata=adata,
            gene_set=genes,
            score_key=row.score_key,
            smoothing=False,
            correct_spatial_covariates=True,
            batch_key=None,
        )
        if row.score_column not in adata.obs:
            stop(f"Variant U MP re-score did not create {row.score_column}")


def knn_weights(coords: np.ndarray):
    import libpysal

    if coords.shape[0] <= K:
        stop(f"Too few tumour spots for k={K} KNN: n={coords.shape[0]}")
    w = libpysal.weights.KNN.from_array(coords, k=K)
    w.transform = "R"
    return w


def graph_stats_from_weights(w, n: int) -> tuple[int, float]:
    rows, cols = [], []
    for i, neigh in w.neighbors.items():
        rows.extend([i] * len(neigh))
        cols.extend(neigh)
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    graph = graph.maximum(graph.T).tocsr()
    n_comp, labels = connected_components(graph, directed=False)
    largest = int(np.bincount(labels).max()) if len(labels) else 0
    return int(n_comp), float(largest / n) if n else math.nan


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    vals = p[ok]
    order = np.argsort(vals)
    ranked = vals[order]
    n = len(vals)
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    tmp = np.empty(n)
    tmp[order] = np.clip(adj, 0, 1)
    out[ok] = tmp
    return out


def lisa_classes(adata, variant: str, dataset: str, sample: str):
    from esda.moran import Moran, Moran_Local

    coords = np.asarray(adata.obsm["spatial"])
    w = knn_weights(coords)
    n_comp, largest = graph_stats_from_weights(w, adata.n_obs)
    values = pd.to_numeric(adata.obs[SNAI_COL], errors="coerce").to_numpy()
    if not np.isfinite(values).all():
        stop(f"{variant} {dataset}__{sample}: SNAI1-ac_score contains non-finite values")
    lisa = Moran_Local(values, w, transformation="r", permutations=N_PERM, n_jobs=1, seed=SEED)
    qval = bh_fdr(lisa.p_sim)
    sig = qval <= ALPHA
    hh = sig & (lisa.q == 1)
    ll = sig & (lisa.q == 3)
    excluded = hh | ll
    hh_ring = np.zeros(adata.n_obs, dtype=bool)
    ll_ring = np.zeros(adata.n_obs, dtype=bool)
    for idx in np.where(hh)[0]:
        hh_ring[w.neighbors[int(idx)]] = True
    for idx in np.where(ll)[0]:
        ll_ring[w.neighbors[int(idx)]] = True
    hh_ring &= ~excluded
    ll_ring &= ~excluded
    moran_i = float(Moran(values, w, permutations=N_PERM).I)
    return {
        "w": w,
        "p_sim": lisa.p_sim,
        "p_fdr": qval,
        "q": lisa.q,
        "hh": hh,
        "ll": ll,
        "hh_ring": hh_ring,
        "ll_ring": ll_ring,
        "n_comp": n_comp,
        "largest_component_fraction": largest,
        "moran_i": moran_i,
    }


def rank_biserial(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    res = mannwhitneyu(x, y, alternative="two-sided", method="auto")
    rb = 2 * float(res.statistic) / (len(x) * len(y)) - 1
    return rb, float(res.pvalue)


def plot_sample_maps(dataset: str, sample: str, s_adata, u_adata, s_cls: dict, u_cls: dict):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    for ax, adata, cls, title in [(axes[0], s_adata, s_cls, "S smoothed"), (axes[1], u_adata, u_cls, "U unsmoothed")]:
        coords = np.asarray(adata.obsm["spatial"])
        ax.scatter(coords[:, 0], coords[:, 1], s=8, c="#C8C8C8", alpha=0.45, linewidths=0)
        ax.scatter(coords[cls["ll"], 0], coords[cls["ll"], 1], s=15, c="#2166AC", alpha=0.9, linewidths=0, label=f"LL n={int(cls['ll'].sum())}")
        ax.scatter(coords[cls["hh"], 0], coords[cls["hh"], 1], s=15, c="#B2182B", alpha=0.9, linewidths=0, label=f"HH n={int(cls['hh'].sum())}")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")
        ax.legend(frameon=False, loc="upper right", fontsize=7)
    fig.suptitle(f"{dataset} | {sample} | SNAI1-ac LISA FDR<=0.05")
    fig.tight_layout()
    out = OUT / "plots" / "sample_lisa_maps" / f"{dataset}__{sample}_S_vs_U_HH_LL.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def finite_pair(adata, col: str, mask: np.ndarray) -> np.ndarray:
    vals = pd.to_numeric(adata.obs[col], errors="coerce").to_numpy()
    return vals[mask & np.isfinite(vals)]


def process_all():
    ensure_dirs()
    for p in [MP_SIGNATURES, SCORE_SUMMARY, SNAI_WEIGHTS, R_SCRIPT, RSCRIPT]:
        require(p)
    mp = load_mp_summary()
    paths = smoothed_paths()
    col_check = check_columns_and_malignant(paths, mp)
    mp_genes, snai_weights = load_signatures(mp)

    class_rows, skew_rows, purity_rows, comp_rows, effect_rows, provenance = [], [], [], [], [], []
    figure_paths = []
    u_dir = OUT / "variant_U_unsmoothed_h5ad"

    for dataset, sample, path in paths:
        print(f"Processing {dataset}__{sample}")
        s_adata = sc.read_h5ad(path)
        ensure_hires_alias(s_adata)
        u_adata = s_adata.copy()
        score_unsmoothed(u_adata, mp, mp_genes, snai_weights)
        out_h5ad = u_dir / dataset / f"{sample}.tumor_subset_SNAI1ac_MP_unsmoothed_scores.h5ad"
        out_h5ad.parent.mkdir(parents=True, exist_ok=True)
        u_adata.write_h5ad(out_h5ad)
        provenance.append({"dataset": dataset, "sample": sample, "variant": "U", "score_source": "computed_fresh", "h5ad": str(out_h5ad)})
        provenance.append({"dataset": dataset, "sample": sample, "variant": "S", "score_source": "reused_arm3_smoothed", "h5ad": str(path)})

        variants = {"S": s_adata, "U": u_adata}
        classes = {}
        for variant, adata_obj in variants.items():
            cls = lisa_classes(adata_obj, variant, dataset, sample)
            classes[variant] = cls
            y = pd.to_numeric(adata_obj.obs[SNAI_COL], errors="coerce")
            m = pd.to_numeric(adata_obj.obs[MAL_COL], errors="coerce")
            ok = y.notna() & m.notna()
            purity_rows.append({
                "variant": variant, "dataset": dataset, "sample": sample,
                "n_tumour_spots": int(ok.sum()),
                "spearman_r": float(spearmanr(y[ok], m[ok]).statistic),
                "spearman_p": float(spearmanr(y[ok], m[ok]).pvalue),
            })
            masks = {"HH": cls["hh"], "LL": cls["ll"], "HH_ring": cls["hh_ring"], "LL_ring": cls["ll_ring"]}
            class_rows.append({
                "variant": variant, "dataset": dataset, "sample": sample,
                "n_tumour_spots": int(adata_obj.n_obs),
                "n_HH": int(cls["hh"].sum()), "n_LL": int(cls["ll"].sum()),
                "n_HH_ring": int(cls["hh_ring"].sum()), "n_LL_ring": int(cls["ll_ring"].sum()),
                "connected_components": cls["n_comp"],
                "largest_component_fraction": cls["largest_component_fraction"],
                "moran_i_snai1ac": cls["moran_i"],
                "lisa_k": K, "lisa_permutations": N_PERM, "fdr_method": "Benjamini-Hochberg", "alpha": ALPHA,
            })
            for class_name, mask in masks.items():
                vals = pd.to_numeric(adata_obj.obs.loc[mask, MAL_COL], errors="coerce")
                comp_rows.append({
                    "variant": variant, "dataset": dataset, "sample": sample, "class": class_name,
                    "n_spots": int(vals.notna().sum()),
                    "malignant_mean": float(vals.mean()) if vals.notna().any() else math.nan,
                    "malignant_median": float(vals.median()) if vals.notna().any() else math.nan,
                })
            for row in mp.itertuples(index=False):
                vals = pd.to_numeric(adata_obj.obs[row.score_column], errors="coerce").dropna().to_numpy()
                skew_rows.append({
                    "variant": variant, "dataset": dataset, "sample": sample,
                    "MP_id": row.subcluster_id, "MP_label": row.label,
                    "score_column": row.score_column,
                    "skewness": float(skew(vals, bias=False)) if len(vals) > 2 else math.nan,
                })
                for contrast, a_name, b_name in [("cluster_HH_vs_LL", "HH", "LL"), ("ring_HH_vs_LL", "HH_ring", "LL_ring")]:
                    x = finite_pair(adata_obj, row.score_column, masks[a_name])
                    yv = finite_pair(adata_obj, row.score_column, masks[b_name])
                    included = len(x) > 0 and len(yv) > 0
                    reason = "" if included else f"empty group: {a_name if len(x) == 0 else ''} {b_name if len(yv) == 0 else ''}".strip()
                    rb = pval = eff = math.nan
                    clipped = False
                    if included:
                        rb, pval = rank_biserial(x, yv)
                        eff = rb
                        if abs(eff) >= 1:
                            eff = math.copysign(0.999999, eff)
                            clipped = True
                    effect_rows.append({
                        "variant": variant, "contrast": contrast,
                        "group1": a_name, "group2": b_name,
                        "MP_id": row.subcluster_id, "MP_label": row.label, "score_column": row.score_column,
                        "dataset": dataset, "sample": sample,
                        "n_group1": int(len(x)), "n_group2": int(len(yv)),
                        "group1_mean": float(np.mean(x)) if len(x) else math.nan,
                        "group2_mean": float(np.mean(yv)) if len(yv) else math.nan,
                        "group1_median": float(np.median(x)) if len(x) else math.nan,
                        "group2_median": float(np.median(yv)) if len(yv) else math.nan,
                        "rank_biserial": rb,
                        "effect_for_meta": eff,
                        "mannwhitney_p": pval,
                        "included_in_meta": bool(included),
                        "exclusion_reason": reason,
                        "perfect_separation_clipped": clipped,
                    })
        figure_paths.append(str(plot_sample_maps(dataset, sample, s_adata, u_adata, classes["S"], classes["U"])))

    tables = OUT / "tables"
    class_df = pd.DataFrame(class_rows)
    skew_df = pd.DataFrame(skew_rows)
    purity_df = pd.DataFrame(purity_rows)
    comp_df = pd.DataFrame(comp_rows)
    effects_df = pd.DataFrame(effect_rows)
    prov_df = pd.DataFrame(provenance)
    class_df.to_csv(tables / "snai1ac_lisa_class_counts_diagnostics.csv", index=False)
    skew_df.to_csv(tables / "snai1ac_lisa_mp_skewness.csv", index=False)
    purity_df.to_csv(tables / "snai1ac_lisa_purity_diagnostic.csv", index=False)
    comp_df.to_csv(tables / "snai1ac_lisa_composition_readout.csv", index=False)
    effects_df.to_csv(tables / "snai1ac_lisa_per_sample_effects.csv", index=False)
    prov_df.to_csv(tables / "snai1ac_lisa_score_provenance.csv", index=False)

    shutil.copy2(SCRIPT, OUT / "scripts" / SCRIPT.name)
    shutil.copy2(R_SCRIPT, OUT / "scripts" / R_SCRIPT.name)
    r = subprocess.run([str(RSCRIPT), str(R_SCRIPT), str(OUT)], capture_output=True, text=True)
    if r.returncode != 0:
        stop(f"R meta-analysis failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")

    meta = pd.read_csv(tables / "snai1ac_lisa_meta_summary.csv")
    purity_meta = pd.read_csv(tables / "snai1ac_lisa_purity_meta_summary.csv") if (tables / "snai1ac_lisa_purity_meta_summary.csv").exists() else pd.DataFrame()
    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script_paths": {"python": str(SCRIPT), "R": str(R_SCRIPT)},
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "anndata": ad.__version__,
            "scanpy": sc.__version__,
            "scipy": scipy.__version__,
            "R_stdout": r.stdout.strip(),
        },
        "inputs": {
            "smoothed_tumour_h5ad_dir": str(SMOOTHED_DIR),
            "score_summary": str(SCORE_SUMMARY),
            "mp_signatures": str(MP_SIGNATURES),
            "snai1_ac_weights": str(SNAI_WEIGHTS),
        },
        "confirmed_columns": {
            "SNAI1-ac": SNAI_COL,
            "MP_scores": MP_COLS,
            "malignant_fraction": MAL_COL,
            "compartment": COMPARTMENT_COL,
            "tumour_label": TUMOUR_LABEL,
            "is_tumor": IS_TUMOR_COL,
            "malignant_check": col_check,
        },
        "prior_lisa_check": PRIOR_LISA,
        "lisa_settings": {
            "k": K,
            "permutations": N_PERM,
            "seed": SEED,
            "significance": "Benjamini-Hochberg FDR on p_sim per sample, alpha=0.05",
            "HH": "significant & Moran_Local q==1",
            "LL": "significant & Moran_Local q==3",
            "note": "This run uses BH-FDR on p_sim at alpha=0.05 per sample, as requested. The located Visium spatial-signature prior also used FDR; the user checkpoint described the intended threshold as an upgrade relative to an uncorrected p_sim<0.05 precedent, so both facts are retained in the audit trail.",
        },
        "variants": {
            "S": "arm-3 tumour subset scores reused; smoothing=True",
            "U": "computed fresh on same tumour subset; smoothing=False; correct_spatial_covariates=True; SNAI1-ac weighted; MPs uniform",
        },
        "checkpoint_2_decisions": {
            "minimum_group_n_floor": "none; include whenever both groups are non-empty",
            "effect_size": "rank-biserial correlation from Mann-Whitney U",
            "meta_transform": "Fisher z; variance 1/(n_group1+n_group2-3); ±1 clipped to ±0.999999",
        },
        "output_audit": {
            "class_rows": int(len(class_df)),
            "skewness_rows": int(len(skew_df)),
            "purity_rows": int(len(purity_df)),
            "composition_rows": int(len(comp_df)),
            "effect_rows": int(len(effects_df)),
            "meta_rows": int(len(meta)),
            "purity_meta_rows": int(len(purity_meta)),
            "figures": figure_paths,
            "perfect_separation_clipped_rows": int(effects_df["perfect_separation_clipped"].sum()),
            "contrast_exclusions": effects_df.loc[~effects_df["included_in_meta"], ["variant", "contrast", "MP_id", "dataset", "sample", "exclusion_reason"]].to_dict("records"),
        },
    }
    (OUT / "snai1ac_lisa_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log_lines = [
        "SNAI1-ac within-tumour LISA hotspot/coldspot MP contrast run",
        f"Timestamp: {manifest['timestamp']}",
        f"Python script: {SCRIPT}",
        f"R script: {R_SCRIPT}",
        f"Output: {OUT}",
        "",
        "Prior Visium LISA check:",
        f"- HH: {PRIOR_LISA['HH_definition']}",
        f"- LL: {PRIOR_LISA['LL_definition']}",
        f"- significance in located Visium script/notebook: {PRIOR_LISA['significance']}",
        "",
        "This run:",
        "- S: reused arm-3 smoothed tumour-subset scores.",
        "- U: computed fresh raw EnrichMap unsmoothed scores on the same Tumor subset.",
        "- FDR: Benjamini-Hochberg on p_sim per sample at alpha=0.05; HH=q==1, LL=q==3.",
        "- Threshold note: the located Visium spatial-signature prior also used FDR; the checkpoint described this as stricter than an uncorrected p_sim<0.05 precedent, so this run records the deliberate strict BH-FDR rule explicitly.",
        "- Contrast rule: no n floor; skip only empty-group sample contrasts.",
        "- Effect size/meta: rank-biserial; Fisher-z REML with vi=1/(n1+n2-3).",
        "",
        f"Malignant check: {col_check}",
        f"Class-count rows: {len(class_df)}",
        f"Skewness rows: {len(skew_df)}",
        f"Effect rows: {len(effects_df)}",
        f"Meta rows: {len(meta)}",
        f"Contrast exclusions: {(~effects_df['included_in_meta']).sum()}",
        f"Perfect-separation clips: {effects_df['perfect_separation_clipped'].sum()}",
        "",
        "R output:",
        r.stdout.strip(),
        r.stderr.strip(),
    ]
    (OUT / "logs" / "snai1ac_lisa_run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    print("\n".join(log_lines))


if __name__ == "__main__":
    np.random.seed(SEED)
    process_all()
