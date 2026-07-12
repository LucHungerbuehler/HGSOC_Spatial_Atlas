from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"D:\HGSOC_Spatial_Atlas")
RUN = ROOT / "05_analysis_ready" / "20260424_definition3b_definition4_raw_geneNMF"
HH = RUN / "08_hh_programme_characterization"
FAM = RUN / "08_snaI1_tme_family_definitions"
SUM = RUN / "06_summary"
RIDGE = RUN / "07_snai1ac_program_prediction"
CSIDE = ROOT / "05_analysis_ready" / "S5plus_CSIDE_Alpha_Strengthening" / "runs" / "20260415_191521_kstar_niches_cside_alpha"
XEN = ROOT / "05_analysis_ready" / "Xenium_signature"
CLIN = ROOT / "03_metadata" / "clinical_annotations" / "clinical annotations.xlsx"
OUT = RUN / "11_research_synthesis"


PRIMARY_ROLES = {"primary_tumor_intrinsic", "primary_tumor_spot_context", "context_primary_or_sensitivity"}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def clean_sample_name(x: str) -> str:
    x = str(x).strip()
    x = re.sub(r"\s*\([^)]*\)\s*$", "", x).strip()
    return x


def dataset_key(x: str) -> str:
    x = str(x).strip().lower()
    return x.replace("_2022", "").replace("_2024", "").replace("_2025", "")


def load_clinical() -> pd.DataFrame:
    if not CLIN.exists():
        return pd.DataFrame()
    df = pd.read_excel(CLIN)
    df["dataset_key"] = df["dataset"].map(dataset_key)
    df["sample_id_on_disk"] = df["sample"].map(clean_sample_name)
    keep = [
        "dataset_key",
        "sample_id_on_disk",
        "GEO",
        "sections source",
        "collection",
        "treatment status",
        "treatment response (CRS, 3 tier)",
        "tissue preservation",
        "PARPi sensitivity",
        "BRCA1/2",
    ]
    return df[[c for c in keep if c in df.columns]].copy()


def boolish(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def load_program_annotations() -> pd.DataFrame:
    ann = read_csv(FAM / "program_to_snaI1_tme_family_v1.csv")
    if ann.empty:
        return ann
    ann["include_primary_snaI1_tme_bool"] = boolish(ann["include_primary_snaI1_tme"])
    ann["technical_flag_bool"] = boolish(ann["technical_flag"])
    ann["primary_family_bool"] = (
        ann["include_primary_snaI1_tme_bool"]
        & ann["analysis_role"].isin(PRIMARY_ROLES)
        & ~ann["technical_flag_bool"]
        & ~ann["family_id"].fillna("").str.startswith("F90")
    )
    return ann


def dominant_row(g: pd.DataFrame) -> pd.Series:
    idx = g["mean_difference_hh_minus_nonhh"].abs().idxmax()
    return g.loc[idx]


def family_sample_effects() -> tuple[pd.DataFrame, pd.DataFrame]:
    ann = load_program_annotations()
    contrasts = pd.concat(
        [
            read_csv(HH / "tables" / "hh_spot_level_programme_contrasts.csv"),
            read_csv(HH / "tables" / "hh_neighborhood_programme_contrasts.csv"),
        ],
        ignore_index=True,
    )
    df = contrasts.merge(
        ann[
            [
                "program_id",
                "family_id",
                "family_label",
                "analysis_role",
                "primary_family_bool",
                "alignment_category_draft",
                "program_identity_draft",
                "top20_genes",
                "top_functional_terms_v0_2",
            ]
        ],
        on="program_id",
        how="left",
    )
    df = df[df["primary_family_bool"].fillna(False)].copy()
    df["dataset_key"] = df["dataset"].map(dataset_key)
    df["sig_fdr_0_05"] = df["fdr_bh"] <= 0.05
    df["program_direction"] = np.where(df["mean_difference_hh_minus_nonhh"] >= 0, "HH_enriched", "HH_depleted")

    rows = []
    for keys, g in df.groupby(["context_type", "dataset", "sample_id_on_disk", "sample_label", "family_id", "family_label", "analysis_role"], dropna=False):
        dom = dominant_row(g)
        n_pos = int((g["mean_difference_hh_minus_nonhh"] > 0).sum())
        n_neg = int((g["mean_difference_hh_minus_nonhh"] < 0).sum())
        n_sig_pos = int(((g["mean_difference_hh_minus_nonhh"] > 0) & g["sig_fdr_0_05"]).sum())
        n_sig_neg = int(((g["mean_difference_hh_minus_nonhh"] < 0) & g["sig_fdr_0_05"]).sum())
        mean_diff = float(g["mean_difference_hh_minus_nonhh"].mean())
        mean_dz = float(g["cohens_dz_hh_minus_nonhh"].mean())
        support = (abs(mean_dz) >= 0.2) or (n_sig_pos + n_sig_neg > 0)
        if n_pos and n_neg:
            internal = "mixed_programs"
        elif n_pos:
            internal = "all_programs_HH_enriched"
        else:
            internal = "all_programs_HH_depleted"
        rows.append(
            {
                "context_type": keys[0],
                "dataset": keys[1],
                "sample_id_on_disk": keys[2],
                "sample_label": keys[3],
                "family_id": keys[4],
                "family_label": keys[5],
                "analysis_role": keys[6],
                "n_programs": int(len(g)),
                "mean_difference_hh_minus_nonhh": mean_diff,
                "median_difference_hh_minus_nonhh": float(g["median_difference_hh_minus_nonhh"].median()),
                "mean_cohens_dz_hh_minus_nonhh": mean_dz,
                "max_abs_cohens_dz": float(g["cohens_dz_hh_minus_nonhh"].abs().max()),
                "n_pos_programs": n_pos,
                "n_neg_programs": n_neg,
                "n_sig_fdr_0_05": int(g["sig_fdr_0_05"].sum()),
                "n_sig_pos_fdr_0_05": n_sig_pos,
                "n_sig_neg_fdr_0_05": n_sig_neg,
                "family_direction_by_mean": "HH_enriched" if mean_diff >= 0 else "HH_depleted",
                "family_direction_supported": bool(support),
                "internal_program_direction": internal,
                "dominant_program_id": dom["program_id"],
                "dominant_program_mean_difference": float(dom["mean_difference_hh_minus_nonhh"]),
                "dominant_program_cohens_dz": float(dom["cohens_dz_hh_minus_nonhh"]),
                "dominant_program_fdr_bh": float(dom["fdr_bh"]),
                "dominant_program_alignment_category": dom.get("alignment_category_draft", ""),
                "dominant_program_identity": dom.get("program_identity_draft", ""),
                "dominant_program_top20_genes": dom.get("top20_genes", ""),
                "dominant_program_terms": dom.get("top_functional_terms_v0_2", ""),
            }
        )
    fam_sample = pd.DataFrame(rows)
    clin = load_clinical()
    if not clin.empty:
        fam_sample["dataset_key"] = fam_sample["dataset"].map(dataset_key)
        fam_sample = fam_sample.merge(clin, on=["dataset_key", "sample_id_on_disk"], how="left")

    flip_rows = []
    for keys, g in fam_sample.groupby(["context_type", "family_id", "family_label", "analysis_role"], dropna=False):
        supported = g[g["family_direction_supported"]].copy()
        if supported.empty:
            supported = g.copy()
        n_enr = int((supported["family_direction_by_mean"] == "HH_enriched").sum())
        n_dep = int((supported["family_direction_by_mean"] == "HH_depleted").sum())
        flip = n_enr > 0 and n_dep > 0
        flip_rows.append(
            {
                "context_type": keys[0],
                "family_id": keys[1],
                "family_label": keys[2],
                "analysis_role": keys[3],
                "n_samples_present": int(g["sample_label"].nunique()),
                "n_supported_samples": int(supported["sample_label"].nunique()),
                "n_HH_enriched_samples": n_enr,
                "n_HH_depleted_samples": n_dep,
                "n_internal_mixed_samples": int((g["internal_program_direction"] == "mixed_programs").sum()),
                "n_samples_with_any_significant_program": int((g["n_sig_fdr_0_05"] > 0).sum()),
                "median_mean_dz": float(g["mean_cohens_dz_hh_minus_nonhh"].median()),
                "mean_abs_mean_dz": float(g["mean_cohens_dz_hh_minus_nonhh"].abs().mean()),
                "flip_evidence": bool(flip),
                "interpretation": (
                    "bidirectional_across_samples"
                    if flip
                    else ("mostly_HH_enriched" if n_enr > n_dep else "mostly_HH_depleted" if n_dep > n_enr else "weak_or_balanced")
                ),
            }
        )
    flip = pd.DataFrame(flip_rows)
    return fam_sample, flip


def summarize_continuous_family_correlations(ann: pd.DataFrame) -> pd.DataFrame:
    corr = read_csv(SUM / "programme_snai1ac_correlations_all_samples.csv")
    if corr.empty or ann.empty:
        return pd.DataFrame()
    df = corr.merge(
        ann[["program_id", "family_id", "family_label", "analysis_role", "primary_family_bool"]],
        on="program_id",
        how="left",
    )
    df = df[df["primary_family_bool"].fillna(False)].copy()
    rows = []
    for keys, g in df.groupby(["family_id", "family_label", "analysis_role"], dropna=False):
        rows.append(
            {
                "family_id": keys[0],
                "family_label": keys[1],
                "analysis_role": keys[2],
                "n_programs": int(g["program_id"].nunique()),
                "n_samples": int(g["sample_label"].nunique()),
                "median_spearman_rho": float(g["spearman_rho"].median()),
                "mean_spearman_rho": float(g["spearman_rho"].mean()),
                "n_positive_sig": int(((g["spearman_rho"] > 0) & g["significant_fdr_0_05"]).sum()),
                "n_negative_sig": int(((g["spearman_rho"] < 0) & g["significant_fdr_0_05"]).sum()),
                "n_bidirectional_sig_samples": int(
                    g[g["significant_fdr_0_05"]]
                    .assign(sign=np.where(g[g["significant_fdr_0_05"]]["spearman_rho"] > 0, "pos", "neg"))
                    .groupby("sample_label")["sign"]
                    .nunique()
                    .gt(1)
                    .sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def collect_external_summaries() -> dict:
    out: dict = {}
    ridge = read_csv(RIDGE / "tables" / "per_sample_ridge_cv_performance.csv")
    if not ridge.empty:
        out["ridge"] = {
            "n_samples": int(len(ridge)),
            "median_cv_r2": float(ridge["cv_r2"].median()) if "cv_r2" in ridge else None,
            "mean_cv_r2": float(ridge["cv_r2"].mean()) if "cv_r2" in ridge else None,
            "max_cv_r2": float(ridge["cv_r2"].max()) if "cv_r2" in ridge else None,
            "median_cv_spearman": float(ridge["cv_spearman"].median()) if "cv_spearman" in ridge else None,
            "top_samples": ridge.sort_values("cv_r2", ascending=False).head(6).to_dict("records") if "cv_r2" in ridge else [],
        }
    cside_lodo = read_csv(CSIDE / "02_cside_strengthening" / "leave_one_dataset_out_summary.csv")
    cside_loso = read_csv(CSIDE / "02_cside_strengthening" / "leave_one_sample_out_summary.csv")
    leading = read_csv(CSIDE / "02_cside_strengthening" / "leading_edge_stability_summary.csv")
    out["cside"] = {
        "leave_one_dataset_shape": list(cside_lodo.shape),
        "leave_one_sample_shape": list(cside_loso.shape),
        "leading_edge_shape": list(leading.shape),
    }
    sig = read_csv(XEN / "tables" / "pairwise_ifit2_slc2a1_shared_gene_sensitivity_stats.csv")
    if not sig.empty:
        out["snai1_signature_stress"] = sig.to_dict("records")
    return out


def plot_heatmap(df: pd.DataFrame, context: str, out: Path) -> None:
    sub = df[df["context_type"] == context].copy()
    if sub.empty:
        return
    sample_order = clinical_sample_order(sub)
    family_order = (
        sub[["analysis_role", "family_label", "family_id"]]
        .drop_duplicates()
        .sort_values(["analysis_role", "family_label"])["family_id"]
        .tolist()
    )
    labels = sub.drop_duplicates("family_id").set_index("family_id")["family_label"].to_dict()
    mat = sub.pivot_table(
        index="family_id",
        columns="sample_label",
        values="mean_cohens_dz_hh_minus_nonhh",
        aggfunc="mean",
    ).reindex(index=family_order, columns=sample_order)
    plt.figure(figsize=(14, 7.8))
    vmax = np.nanpercentile(np.abs(mat.values), 95)
    vmax = max(vmax, 0.2)
    im = plt.imshow(mat.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, fraction=0.024, pad=0.02, label="Mean Cohen dz (HH minus matched non-HH)")
    plt.yticks(range(len(mat.index)), [labels.get(i, i).replace("/", "/\n") for i in mat.index], fontsize=8)
    plt.xticks(range(len(mat.columns)), [c.replace("denisenko_2022__", "D:").replace("yamamoto_2025__", "Y:").replace("ju_2024__", "J:") for c in mat.columns], rotation=60, ha="right", fontsize=8)
    plt.title(f"Primary program-family HH effect map ({context.replace('_', ' ')})")
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def short_clinical_label(value: object, field: str) -> str:
    if pd.isna(value):
        return "NA"
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return "NA"
    if field == "dataset":
        return {"denisenko_2022": "Den", "yamamoto_2025": "Yam", "ju_2024": "Ju"}.get(s, s[:3])
    if field == "collection":
        if "recurrence" in s.lower():
            return "REC"
        return s.upper()[:3]
    if field == "treatment status":
        low = s.lower()
        if "naive" in low:
            return "naive"
        if "nact" in low:
            return "NACT"
        if "bev" in low and "tc" in low:
            return "TC+Bev"
        if "tc" in low:
            return "TC"
        return "multi" if len(s) > 10 else s
    if field == "sections source":
        low = s.lower()
        if "omentum" in low or "ometum" in low:
            return "omentum"
        if "peritoneum" in low:
            return "perit."
        if "vaginal" in low:
            return "vaginal"
        if "fallopian" in low:
            return "tube"
        if "ovary" in low:
            return "ovary"
        return s[:8]
    if field == "PARPi sensitivity":
        try:
            f = float(s)
            return "sens" if f == 1 else "res"
        except Exception:
            return "NA"
    if field == "BRCA1/2":
        low = s.lower()
        if "brca1" in low:
            return "BRCA1"
        if "brca2" in low:
            return "BRCA2"
        if "mut" in low:
            return "BRCA mut"
        if "negative" in low:
            return "WT/neg"
        return s[:8]
    if field == "treatment response (CRS, 3 tier)":
        try:
            return f"CRS{int(float(s))}"
        except Exception:
            return "NA"
    return s


def clinical_sort_token(value: object, field: str) -> str:
    label = short_clinical_label(value, field)
    orders = {
        "dataset": {"Den": "0_Den", "Ju": "1_Ju", "Yam": "2_Yam", "NA": "9_NA"},
        "treatment response (CRS, 3 tier)": {"CRS1": "1_CRS1", "CRS2": "2_CRS2", "CRS3": "3_CRS3", "NA": "9_NA"},
        "PARPi sensitivity": {"sens": "1_sens", "res": "2_res", "NA": "9_NA"},
        "BRCA1/2": {"BRCA1": "1_BRCA1", "BRCA2": "2_BRCA2", "BRCA mut": "3_BRCA_mut", "WT/neg": "4_WTneg", "NA": "9_NA"},
        "collection": {"PDS": "1_PDS", "IDS": "2_IDS", "REC": "3_REC", "NA": "9_NA"},
        "sections source": {"ovary": "1_ovary", "tube": "2_tube", "omentum": "3_omentum", "perit.": "4_peritoneum", "vaginal": "5_vaginal", "NA": "9_NA"},
    }
    return orders.get(field, {}).get(label, label)


def clinical_sample_order(df: pd.DataFrame) -> list[str]:
    fields = [
        "dataset",
        "treatment response (CRS, 3 tier)",
        "PARPi sensitivity",
        "BRCA1/2",
        "collection",
        "sections source",
        "sample_id_on_disk",
    ]
    meta_cols = ["sample_label"] + [field for field in fields if field in df.columns]
    meta = df[meta_cols].drop_duplicates("sample_label").copy()
    for field in fields:
        if field not in meta.columns:
            meta[field] = ""
        meta[f"__sort_{field}"] = meta[field].map(lambda value, f=field: clinical_sort_token(value, f))
    sort_cols = [f"__sort_{field}" for field in fields]
    return meta.sort_values(sort_cols)["sample_label"].tolist()


def plot_heatmap_with_clinical_tracks(df: pd.DataFrame, context: str, out: Path) -> None:
    sub = df[df["context_type"] == context].copy()
    if sub.empty:
        return
    sample_meta = (
        sub[
            [
                "dataset",
                "sample_id_on_disk",
                "sample_label",
                "collection",
                "treatment status",
                "sections source",
                "treatment response (CRS, 3 tier)",
                "PARPi sensitivity",
                "BRCA1/2",
            ]
        ]
        .drop_duplicates("sample_label")
    )
    order_lookup = {sample: idx for idx, sample in enumerate(clinical_sample_order(sub))}
    sample_meta["__order"] = sample_meta["sample_label"].map(order_lookup)
    sample_meta = sample_meta.sort_values("__order")
    sample_order = sample_meta["sample_label"].tolist()
    family_order = (
        sub[["analysis_role", "family_label", "family_id"]]
        .drop_duplicates()
        .sort_values(["analysis_role", "family_label"])["family_id"]
        .tolist()
    )
    labels = sub.drop_duplicates("family_id").set_index("family_id")["family_label"].to_dict()
    mat = sub.pivot_table(
        index="family_id",
        columns="sample_label",
        values="mean_cohens_dz_hh_minus_nonhh",
        aggfunc="mean",
    ).reindex(index=family_order, columns=sample_order)
    vmax = np.nanpercentile(np.abs(mat.values), 95)
    vmax = max(vmax, 0.2)

    tracks = [
        ("dataset", "cohort"),
        ("collection", "collection"),
        ("treatment status", "treatment"),
        ("sections source", "site"),
        ("treatment response (CRS, 3 tier)", "CRS"),
        ("PARPi sensitivity", "PARPi"),
        ("BRCA1/2", "BRCA"),
    ]
    palettes = {
        "cohort": {"Den": "#4c78a8", "Yam": "#f58518", "Ju": "#54a24b"},
        "collection": {"PDS": "#8dd3c7", "IDS": "#fb8072", "REC": "#bc80bd", "NA": "#eeeeee"},
        "treatment": {"naive": "#b3de69", "NACT": "#fb8072", "TC": "#fdb462", "TC+Bev": "#fccde5", "multi": "#bc80bd", "NA": "#eeeeee"},
        "site": {"ovary": "#80b1d3", "tube": "#bebada", "omentum": "#fdb462", "perit.": "#b3de69", "vaginal": "#fb8072", "NA": "#eeeeee"},
        "CRS": {"CRS1": "#d73027", "CRS2": "#fee08b", "CRS3": "#1a9850", "NA": "#eeeeee"},
        "PARPi": {"sens": "#1a9850", "res": "#d73027", "NA": "#eeeeee"},
        "BRCA": {"BRCA1": "#7570b3", "BRCA2": "#e7298a", "BRCA mut": "#66a61e", "WT/neg": "#999999", "NA": "#eeeeee"},
    }

    fig = plt.figure(figsize=(16.5, 9.4))
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1.25, 7.8], width_ratios=[18, 0.42], hspace=0.04, wspace=0.04)
    ax_tracks = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])

    im = ax.imshow(mat.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, cax=cax, label="Mean Cohen dz\n(HH minus matched non-HH)")
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels([labels.get(i, i).replace("/", "/\n") for i in mat.index], fontsize=8)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(
        [
            c.replace("denisenko_2022__", "D:")
            .replace("yamamoto_2025__", "Y:")
            .replace("ju_2024__", "J:")
            for c in mat.columns
        ],
        rotation=60,
        ha="right",
        fontsize=8,
    )
    fig.suptitle(f"Primary program-family HH effect map with clinical tracks ({context.replace('_', ' ')})", fontsize=16, y=0.995)

    ax_tracks.set_xlim(-0.5, len(sample_order) - 0.5)
    ax_tracks.set_ylim(-0.5, len(tracks) - 0.5)
    ax_tracks.set_xticks([])
    ax_tracks.set_yticks(range(len(tracks)))
    ax_tracks.set_yticklabels([label for _, label in tracks], fontsize=8)
    for i, sample in enumerate(sample_order):
        meta = sample_meta[sample_meta["sample_label"] == sample].iloc[0]
        for j, (field, label) in enumerate(tracks):
            val = short_clinical_label(meta.get(field, np.nan), field)
            color = palettes[label].get(val, "#dddddd")
            ax_tracks.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1, facecolor=color, edgecolor="white", lw=0.4))
            if label in {"CRS", "PARPi", "BRCA"} and val != "NA":
                ax_tracks.text(i, j, val.replace("BRCA", "B"), ha="center", va="center", fontsize=5, color="black")
    ax_tracks.invert_yaxis()
    ax_tracks.spines[:].set_visible(False)

    legend_lines = []
    for label in ["cohort", "collection", "treatment", "site", "CRS", "PARPi", "BRCA"]:
        vals = []
        for field, lab in tracks:
            if lab == label:
                vals = [short_clinical_label(v, field) for v in sample_meta[field].drop_duplicates().tolist()]
        vals = [v for v in vals if v != "NA"]
        if vals:
            legend_lines.append(f"{label}: " + ", ".join(sorted(set(vals))))
    fig.text(0.03, 0.01, " | ".join(legend_lines), fontsize=7, color="#444444")
    plt.subplots_adjust(top=0.93, bottom=0.18, left=0.13, right=0.93)
    plt.savefig(out, dpi=220)
    plt.close()


def clinical_stratified_summary(fam_sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    df = fam_sample[fam_sample["context_type"].isin(["spot_level", "neighborhood"])].copy()
    fields = ["dataset", "collection", "treatment status", "sections source", "PARPi sensitivity", "BRCA1/2", "treatment response (CRS, 3 tier)"]
    for context, g0 in df.groupby("context_type"):
        for family, g1 in g0.groupby(["family_id", "family_label"], dropna=False):
            for field in fields:
                if field not in g1.columns:
                    continue
                for val, g in g1.groupby(field, dropna=False):
                    if len(g) < 2 and field not in {"PARPi sensitivity", "BRCA1/2", "treatment response (CRS, 3 tier)"}:
                        continue
                    rows.append(
                        {
                            "context_type": context,
                            "family_id": family[0],
                            "family_label": family[1],
                            "clinical_field": field,
                            "clinical_value": "NA" if pd.isna(val) else str(val),
                            "n_samples": int(g["sample_label"].nunique()),
                            "n_HH_enriched": int((g["family_direction_by_mean"] == "HH_enriched").sum()),
                            "n_HH_depleted": int((g["family_direction_by_mean"] == "HH_depleted").sum()),
                            "median_mean_dz": float(g["mean_cohens_dz_hh_minus_nonhh"].median()),
                            "mean_abs_mean_dz": float(g["mean_cohens_dz_hh_minus_nonhh"].abs().mean()),
                            "n_any_significant_program": int((g["n_sig_fdr_0_05"] > 0).sum()),
                        }
                    )
    return pd.DataFrame(rows)


def plot_flip_summary(flip: pd.DataFrame, context: str, out: Path) -> None:
    sub = flip[flip["context_type"] == context].copy()
    if sub.empty:
        return
    sub = sub.sort_values("mean_abs_mean_dz", ascending=True)
    y = np.arange(len(sub))
    plt.figure(figsize=(10.5, 6.8))
    plt.barh(y - 0.18, sub["n_HH_enriched_samples"], height=0.34, color="#3d8c84", label="HH enriched")
    plt.barh(y + 0.18, -sub["n_HH_depleted_samples"], height=0.34, color="#a95059", label="HH depleted")
    plt.axvline(0, color="#333333", lw=0.8)
    plt.yticks(y, sub["family_label"], fontsize=8)
    plt.xlabel("Supported samples (enriched right, depleted left)")
    plt.title(f"Direction flips by family ({context.replace('_', ' ')})")
    plt.legend(loc="lower right", frameon=False)
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def aggregation_audit(fam_sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for context, g in fam_sample.groupby("context_type"):
        total_cells = int(len(g))
        single_cells = int((g["n_programs"] == 1).sum())
        two_cells = int((g["n_programs"] == 2).sum())
        gt2_cells = int((g["n_programs"] > 2).sum())
        rows.append(
            {
                "context_type": context,
                "family_sample_cells": total_cells,
                "single_program_cells": single_cells,
                "two_program_cells": two_cells,
                "gt2_program_cells": gt2_cells,
                "fraction_single_program_cells": single_cells / total_cells if total_cells else np.nan,
                "fraction_aggregated_cells": (total_cells - single_cells) / total_cells if total_cells else np.nan,
                "represented_program_rows": int(g["n_programs"].sum()),
            }
        )
    context_summary = pd.DataFrame(rows)

    family_summary = (
        fam_sample.groupby(["context_type", "family_id", "family_label"], as_index=False)
        .agg(
            family_sample_cells=("sample_label", "nunique"),
            represented_program_rows=("n_programs", "sum"),
            mean_programs_per_cell=("n_programs", "mean"),
            median_programs_per_cell=("n_programs", "median"),
            max_programs_per_cell=("n_programs", "max"),
            aggregated_cells=("n_programs", lambda s: int((s > 1).sum())),
        )
        .sort_values(["context_type", "aggregated_cells", "represented_program_rows"], ascending=[True, False, False])
    )
    return context_summary, family_summary


def plot_aggregation_audit(context_summary: pd.DataFrame, out: Path) -> None:
    if context_summary.empty:
        return
    sub = context_summary.set_index("context_type").loc[[c for c in ["spot_level", "neighborhood"] if c in context_summary["context_type"].values]]
    labels = [idx.replace("_", " ") for idx in sub.index]
    single = sub["single_program_cells"].to_numpy(dtype=float)
    two = sub["two_program_cells"].to_numpy(dtype=float)
    gt2 = sub["gt2_program_cells"].to_numpy(dtype=float)
    y = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ax.barh(y, single, color="#3d8c84", label="1 programme")
    ax.barh(y, two, left=single, color="#d98c4a", label="2 programmes")
    ax.barh(y, gt2, left=single + two, color="#a95059", label=">2 programmes")
    for i, row in enumerate(sub.itertuples()):
        ax.text(row.family_sample_cells + 1, i, f"{int(row.single_program_cells)}/{int(row.family_sample_cells)} single", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Sample-family cells")
    ax.set_title("How much aggregation is in the family-labeled programme summary?")
    ax.legend(frameon=False, ncol=3, loc="lower right")
    ax.set_xlim(0, max(sub["family_sample_cells"]) * 1.28)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def plot_clinical_context(fam_sample: pd.DataFrame, out: Path) -> None:
    cols = ["sample_label", "dataset", "sections source", "collection", "treatment status", "BRCA1/2", "PARPi sensitivity"]
    clin = fam_sample[cols].drop_duplicates("sample_label").sort_values(["dataset", "sample_label"])
    if clin.empty:
        return
    fig, ax = plt.subplots(figsize=(13.5, 5.5))
    ax.axis("off")
    display = clin.copy()
    display["sample_label"] = display["sample_label"].str.replace("denisenko_2022__", "D:", regex=False).str.replace("yamamoto_2025__", "Y:", regex=False).str.replace("ju_2024__", "J:", regex=False)
    for c in display.columns:
        display[c] = display[c].fillna("").astype(str).str.slice(0, 34)
    table = ax.table(cellText=display.values, colLabels=display.columns, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.25)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d3d5d5")
        if r == 0:
            cell.set_facecolor("#1e3552")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f2f2ef")
    plt.title("Clinical context available for sample-level interpretation", pad=18)
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)

    ann = load_program_annotations()
    fam_sample, flip = family_sample_effects()
    cont = summarize_continuous_family_correlations(ann)
    ext = collect_external_summaries()

    fam_sample.to_csv(OUT / "tables" / "family_sample_hh_effects_with_clinical_context.csv", index=False)
    flip.to_csv(OUT / "tables" / "family_hh_flip_summary.csv", index=False)
    cont.to_csv(OUT / "tables" / "family_continuous_snai1ac_correlation_summary.csv", index=False)
    ann.to_csv(OUT / "tables" / "program_family_annotation_snapshot.csv", index=False)
    aggregation_context, aggregation_family = aggregation_audit(fam_sample)
    aggregation_context.to_csv(OUT / "tables" / "family_labelled_programme_aggregation_audit_by_context.csv", index=False)
    aggregation_family.to_csv(OUT / "tables" / "family_labelled_programme_aggregation_audit_by_family.csv", index=False)
    (OUT / "tables" / "external_branch_summary.json").write_text(json.dumps(ext, indent=2, default=str), encoding="utf-8")

    for context in ["spot_level", "neighborhood"]:
        plot_heatmap(fam_sample, context, OUT / "figures" / f"primary_family_hh_effect_heatmap_{context}.png")
        plot_heatmap_with_clinical_tracks(fam_sample, context, OUT / "figures" / f"primary_family_hh_effect_heatmap_clinical_tracks_{context}.png")
        plot_flip_summary(flip, context, OUT / "figures" / f"family_direction_flip_summary_{context}.png")
    plot_clinical_context(fam_sample, OUT / "figures" / "clinical_context_table.png")
    plot_aggregation_audit(aggregation_context, OUT / "figures" / "family_labelled_programme_aggregation_audit.png")
    clinical_stratified_summary(fam_sample).to_csv(OUT / "tables" / "clinical_stratified_family_hh_direction_summary.csv", index=False)

    print("Wrote synthesis outputs to", OUT)
    print("Family-sample rows:", len(fam_sample))
    print("Flip rows:", len(flip))
    print("Continuous summary rows:", len(cont))


if __name__ == "__main__":
    main()
