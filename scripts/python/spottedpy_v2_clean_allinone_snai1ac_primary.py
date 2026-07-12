from __future__ import annotations

import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ANALYSIS_ROOT = Path(r"D:\HGSOC_Spatial_Atlas\05_analysis_ready")
RUN_ROOT = ANALYSIS_ROOT / "SpottedPy_v2_paper_aligned"
SOURCE_ROOT = RUN_ROOT / "02_neighborhood_enrichment" / "consensus_source_full"
RAW_TABLE = SOURCE_ROOT / "tables" / "consensus_source_allinone_pearson_correlations.csv"
OUT_ROOT = SOURCE_ROOT / "allinone_snai1ac_primary_clean"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
QC_DIR = OUT_ROOT / "qc"
SCRIPT_DIR = OUT_ROOT / "scripts_used"

SNAI1_PRIMARY = "SNAI1_ac"
SOURCE_ORDER = ["snai1ac_consensus_hot", "snai1ac_consensus_cold", "snai12r_hot"]
SOURCE_LABELS = {
    "snai1ac_consensus_hot": "SNAI1-ac hot",
    "snai1ac_consensus_cold": "SNAI1-ac cold",
    "snai12r_hot": "SNAI1-2R hot",
}
SOURCE_FILE_LABELS = {
    "snai1ac_consensus_hot": "snai1ac_hot",
    "snai1ac_consensus_cold": "snai1ac_cold",
    "snai12r_hot": "snai12r_hot",
}

CORE_COMPARATORS = ["SNAI1_scoregenes", "SNAI1_2R_scoregenes"]
SPACET_TARGETS = [
    "SpaCET_Malignant",
    "SpaCET_CAF",
    "SpaCET_Endothelial",
    "SpaCET_Macrophage",
    "SpaCET_B_cell",
    "SpaCET_T_CD4",
    "SpaCET_T_CD8",
    "SpaCET_NK",
    "SpaCET_Plasma",
    "SpaCET_Unidentifiable",
]
MP_TARGETS = [
    "MP1_angiogenic_vascular",
    "MP2_iCAF_stress",
    "MP3_complement_CAF",
    "MP4_activated_myCAF",
    "MP5_IFN_TLS_immune",
    "MP6_APC_TAM_myeloid",
    "MP7_malignant_hypoxia",
    "MP8_malignant_acute_phase_secretory",
]
HALLMARK_GROUPS_RAW = {
    "Proliferation": [
        "HALLMARK_E2F_TARGETS",
        "HALLMARK_G2M_CHECKPOINT",
        "HALLMARK_MYC_TARGETS_V1",
        "HALLMARK_MYC_TARGETS_V2",
    ],
    "Metabolism": [
        "HALLMARK_GLYCOLYSIS",
        "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
        "HALLMARK_FATTY_ACID_METABOLISM",
        "HALLMARK_CHOLESTEROL_HOMEOSTASIS",
        "HALLMARK_ADIPOGENESIS",
    ],
    "Immune Response & Inflammation": [
        "HALLMARK_INFLAMMATORY_RESPONSE",
        "HALLMARK_INTERFERON_ALPHA_RESPONSE",
        "HALLMARK_INTERFERON_GAMMA_RESPONSE",
        "HALLMARK_COMPLEMENT",
        "HALLMARK_IL2_STAT5_SIGNALING",
        "HALLMARK_IL6_JAK_STAT3_SIGNALING",
        "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    ],
    "Cellular Stress & Apoptosis": [
        "HALLMARK_APOPTOSIS",
        "HALLMARK_DNA_REPAIR",
        "HALLMARK_HYPOXIA",
        "HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY",
        "HALLMARK_UNFOLDED_PROTEIN_RESPONSE",
    ],
    "Signaling & Development": [
        "HALLMARK_WNT_BETA_CATENIN_SIGNALING",
        "HALLMARK_NOTCH_SIGNALING",
        "HALLMARK_HEDGEHOG_SIGNALING",
        "HALLMARK_TGF_BETA_SIGNALING",
        "HALLMARK_PI3K_AKT_MTOR_SIGNALING",
        "HALLMARK_MTORC1_SIGNALING",
        "HALLMARK_KRAS_SIGNALING_DN",
        "HALLMARK_KRAS_SIGNALING_UP",
        "HALLMARK_ANGIOGENESIS",
    ],
    "Structure, Adhesion & Cellular Components": [
        "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
        "HALLMARK_APICAL_JUNCTION",
        "HALLMARK_APICAL_SURFACE",
    ],
    "Other Biological States": [
        "HALLMARK_XENOBIOTIC_METABOLISM",
        "HALLMARK_PROTEIN_SECRETION",
        "HALLMARK_ANDROGEN_RESPONSE",
        "HALLMARK_ESTROGEN_RESPONSE_EARLY",
        "HALLMARK_ESTROGEN_RESPONSE_LATE",
        "HALLMARK_PEROXISOME",
    ],
}
HALLMARK_GROUPS = {
    group: [name.replace("HALLMARK_", "") for name in names]
    for group, names in HALLMARK_GROUPS_RAW.items()
}
HALLMARK_TARGETS = [name for names in HALLMARK_GROUPS.values() for name in names]
TARGET_ORDER = {
    "core_comparators": CORE_COMPARATORS,
    "spacet": SPACET_TARGETS,
    "mp": MP_TARGETS,
    "hallmark": HALLMARK_TARGETS,
}
CLASS_ORDER = ["core_comparators", "spacet", "mp", "hallmark"]
CLASS_LABELS = {
    "core_comparators": "Core Comparators",
    "spacet": "SpaCET",
    "mp": "MP1-MP8",
    "hallmark": "Hallmark",
}
TARGET_CLASS_BY_VAR: dict[str, str] = {}
for class_name, targets in TARGET_ORDER.items():
    for target in targets:
        TARGET_CLASS_BY_VAR[target] = class_name

DISPLAY_LABELS = {
    "SNAI1_scoregenes": "SNAI1",
    "SNAI1_2R_scoregenes": "SNAI1-2R",
    "SpaCET_Malignant": "Malignant",
    "SpaCET_CAF": "CAF",
    "SpaCET_Endothelial": "Endothelial",
    "SpaCET_Macrophage": "Macrophage",
    "SpaCET_B_cell": "B cell",
    "SpaCET_T_CD4": "T CD4",
    "SpaCET_T_CD8": "T CD8",
    "SpaCET_NK": "NK",
    "SpaCET_Plasma": "Plasma",
    "SpaCET_Unidentifiable": "Unidentifiable",
    "MP1_angiogenic_vascular": "MP1 angiogenic/vascular",
    "MP2_iCAF_stress": "MP2 iCAF-stress",
    "MP3_complement_CAF": "MP3 complement-CAF",
    "MP4_activated_myCAF": "MP4 activated-myCAF",
    "MP5_IFN_TLS_immune": "MP5 IFN/TLS immune",
    "MP6_APC_TAM_myeloid": "MP6 APC/TAM myeloid",
    "MP7_malignant_hypoxia": "MP7 malignant hypoxia",
    "MP8_malignant_acute_phase_secretory": "MP8 malignant acute-phase/secretory",
}


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def safe_name(value: str) -> str:
    text = str(value).replace("*", "star")
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)


def display_label(variable: str) -> str:
    if variable in DISPLAY_LABELS:
        return DISPLAY_LABELS[variable]
    return variable.replace("_", " ")


def hallmark_group_for(variable: str) -> str:
    for group, variables in HALLMARK_GROUPS.items():
        if variable in variables:
            return group
    return ""


def bh_qvalues(pvals: pd.Series) -> pd.Series:
    values = pd.to_numeric(pvals, errors="coerce")
    q = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    n = len(valid)
    if n == 0:
        return q
    ranks = np.arange(1, n + 1)
    adjusted = (valid.to_numpy(dtype=float) * n) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    q.loc[valid.index] = adjusted
    return q


def significance_marker(qval: float) -> str:
    if not np.isfinite(qval):
        return ""
    if qval <= 1e-4:
        return "****"
    if qval <= 1e-3:
        return "***"
    if qval <= 1e-2:
        return "**"
    if qval <= 0.05:
        return "*"
    return ""


def panel_priority(row: pd.Series) -> int:
    target_class = row["target_class"]
    variable_class = row["variable_class"]
    if target_class == "core_comparators" and variable_class == "all_variables":
        return 0
    if target_class == "spacet" and variable_class == "spacet":
        return 0
    if target_class == "mp" and variable_class == "mp_kstar":
        return 0
    if target_class == "hallmark" and variable_class == "hallmark":
        return 0
    if variable_class == "all_variables":
        return 1
    return 2


def canonicalize_snai1ac_pairs(raw: pd.DataFrame) -> pd.DataFrame:
    no_kstar = raw[
        ~raw["variable_a"].astype(str).str.startswith("Kstar_")
        & ~raw["variable_b"].astype(str).str.startswith("Kstar_")
    ].copy()
    snai = no_kstar[
        (no_kstar["variable_a"] == SNAI1_PRIMARY) ^ (no_kstar["variable_b"] == SNAI1_PRIMARY)
    ].copy()
    snai["target_variable"] = np.where(
        snai["variable_a"] == SNAI1_PRIMARY, snai["variable_b"], snai["variable_a"]
    )
    snai["target_class"] = snai["target_variable"].map(TARGET_CLASS_BY_VAR)
    snai = snai[snai["target_class"].notna()].copy()
    snai["hallmark_group"] = snai["target_variable"].map(hallmark_group_for)
    snai["target_label"] = snai["target_variable"].map(display_label)
    snai["source_label"] = snai["source_group"].map(SOURCE_LABELS)
    snai["panel_priority"] = snai.apply(panel_priority, axis=1)
    snai = snai.sort_values(
        [
            "sample_label",
            "source_group",
            "target_class",
            "target_variable",
            "panel_priority",
            "variable_class",
        ]
    )
    snai = snai.drop_duplicates(
        ["sample_label", "source_group", "target_class", "target_variable"], keep="first"
    ).copy()
    snai["corr"] = pd.to_numeric(snai["corr"], errors="coerce")
    snai["pval"] = pd.to_numeric(snai["pval"], errors="coerce")
    snai["bh_scope"] = (
        "per_sample__source_group__target_class"
    )
    snai["q_class_per_sample"] = (
        snai.groupby(["sample_label", "source_group", "target_class"], group_keys=False)["pval"]
        .apply(bh_qvalues)
        .reindex(snai.index)
    )
    snai["sig_class_per_sample"] = snai["q_class_per_sample"] <= 0.05
    snai["sig_marker_class_per_sample"] = snai["q_class_per_sample"].map(significance_marker)
    snai["fisher_z"] = np.arctanh(snai["corr"].clip(-0.999999, 0.999999))
    out_cols = [
        "sample_label",
        "source_group",
        "source_label",
        "target_class",
        "target_variable",
        "target_label",
        "hallmark_group",
        "mode",
        "n_source_nodes",
        "corr",
        "fisher_z",
        "pval",
        "q_class_per_sample",
        "sig_class_per_sample",
        "sig_marker_class_per_sample",
        "bh_scope",
        "variable_class",
        "variable_a",
        "variable_b",
    ]
    return snai[out_cols].sort_values(["sample_label", "source_group", "target_class", "target_variable"])


def safe_wilcoxon(values: np.ndarray) -> float:
    if len(values) == 0 or np.allclose(values, 0, equal_nan=False):
        return math.nan
    try:
        return float(wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return math.nan


def cross_sample_summary(per_sample: pd.DataFrame) -> pd.DataFrame:
    records = []
    for keys, group in per_sample.groupby(["source_group", "target_class", "target_variable"], sort=False):
        source_group, target_class, target_variable = keys
        corr = pd.to_numeric(group["corr"], errors="coerce").dropna()
        z = np.arctanh(corr.clip(-0.999999, 0.999999))
        if len(z) >= 2 and float(np.nanstd(z, ddof=1)) > 0:
            p_t = float(ttest_1samp(z, popmean=0.0, nan_policy="omit").pvalue)
        else:
            p_t = math.nan
        p_w = safe_wilcoxon(z.to_numpy(dtype=float))
        records.append(
            {
                "source_group": source_group,
                "source_label": SOURCE_LABELS.get(source_group, source_group),
                "target_class": target_class,
                "target_variable": target_variable,
                "target_label": display_label(target_variable),
                "hallmark_group": hallmark_group_for(target_variable),
                "n_samples": int(corr.shape[0]),
                "n_positive": int((corr > 0).sum()),
                "n_negative": int((corr < 0).sum()),
                "corr_mean": float(corr.mean()) if len(corr) else math.nan,
                "corr_median": float(corr.median()) if len(corr) else math.nan,
                "fisher_z_mean": float(z.mean()) if len(z) else math.nan,
                "fisher_z_median": float(z.median()) if len(z) else math.nan,
                "corr_from_mean_fisher_z": float(np.tanh(z.mean())) if len(z) else math.nan,
                "corr_from_median_fisher_z": float(np.tanh(z.median())) if len(z) else math.nan,
                "p_cross_sample_fisher_t": p_t,
                "p_cross_sample_wilcoxon": p_w,
            }
        )
    summary = pd.DataFrame.from_records(records)
    if summary.empty:
        return summary
    summary["q_cross_sample_class_fisher_t"] = (
        summary.groupby(["source_group", "target_class"], group_keys=False)["p_cross_sample_fisher_t"]
        .apply(bh_qvalues)
        .reindex(summary.index)
    )
    summary["q_cross_sample_class_wilcoxon"] = (
        summary.groupby(["source_group", "target_class"], group_keys=False)["p_cross_sample_wilcoxon"]
        .apply(bh_qvalues)
        .reindex(summary.index)
    )
    summary["sig_cross_sample_class"] = summary["q_cross_sample_class_fisher_t"] <= 0.05
    summary["sig_marker_cross_sample_class"] = summary["q_cross_sample_class_fisher_t"].map(significance_marker)
    summary["cross_sample_bh_scope"] = "source_group__target_class"
    return summary.sort_values(["source_group", "target_class", "target_variable"])


def ordered_targets(class_name: str, present: set[str]) -> list[str]:
    return [target for target in TARGET_ORDER[class_name] if target in present]


def heatmap(
    matrix: pd.DataFrame,
    qvals: pd.DataFrame,
    out_path: Path,
    title: str,
    cbar_label: str,
    q_legend: str,
    width_per_col: float = 0.38,
) -> None:
    if matrix.empty:
        return
    row_labels = [SOURCE_LABELS.get(idx, idx) for idx in matrix.index]
    col_labels = [display_label(col) for col in matrix.columns]
    width = max(6.2, min(24, 2.4 + width_per_col * len(col_labels)))
    height = max(3.8, 1.35 + 0.72 * len(row_labels))
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(matrix.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=90, ha="center", fontsize=7)
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(title, fontsize=11, pad=10)
    for i, row_name in enumerate(matrix.index):
        for j, col_name in enumerate(matrix.columns):
            marker = significance_marker(qvals.loc[row_name, col_name]) if col_name in qvals.columns else ""
            if marker:
                ax.text(j, i, marker, ha="center", va="center", color="black", fontsize=8, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(cbar_label, fontsize=8)
    fig.text(0.01, 0.01, q_legend, fontsize=7, ha="left", va="bottom", color="#374151")
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.38, top=0.86)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_cross_sample_heatmaps(summary: pd.DataFrame) -> None:
    out_dir = FIG_DIR / "cross_sample"
    out_dir.mkdir(parents=True, exist_ok=True)
    for class_name in CLASS_ORDER:
        class_df = summary[summary["target_class"] == class_name].copy()
        targets = ordered_targets(class_name, set(class_df["target_variable"]))
        if not targets:
            continue
        matrix = (
            class_df.pivot(index="source_group", columns="target_variable", values="corr_from_median_fisher_z")
            .reindex(index=SOURCE_ORDER, columns=targets)
            .dropna(axis=0, how="all")
        )
        qvals = (
            class_df.pivot(index="source_group", columns="target_variable", values="q_cross_sample_class_fisher_t")
            .reindex(index=matrix.index, columns=matrix.columns)
        )
        heatmap(
            matrix,
            qvals,
            out_dir / f"heatmap_{class_name}.png",
            f"SNAI1-ac all-in-one correlation, {CLASS_LABELS[class_name]}",
            "Median Pearson r across samples",
            "Stars: class-wise cross-sample BH q from Fisher-z one-sample t-test.",
            width_per_col=0.42 if class_name != "hallmark" else 0.33,
        )
    hallmark = summary[summary["target_class"] == "hallmark"].copy()
    for group_name, targets_all in HALLMARK_GROUPS.items():
        targets = [target for target in targets_all if target in set(hallmark["target_variable"])]
        if not targets:
            continue
        matrix = (
            hallmark.pivot(index="source_group", columns="target_variable", values="corr_from_median_fisher_z")
            .reindex(index=SOURCE_ORDER, columns=targets)
            .dropna(axis=0, how="all")
        )
        qvals = (
            hallmark.pivot(index="source_group", columns="target_variable", values="q_cross_sample_class_fisher_t")
            .reindex(index=matrix.index, columns=matrix.columns)
        )
        heatmap(
            matrix,
            qvals,
            out_dir / f"heatmap_hallmark_{safe_name(group_name).lower()}.png",
            f"SNAI1-ac all-in-one correlation, Hallmark: {group_name}",
            "Median Pearson r across samples",
            "Stars: full Hallmark-class cross-sample BH q, not subgroup-specific BH.",
            width_per_col=0.46,
        )


def plot_per_sample_heatmaps(per_sample: pd.DataFrame) -> None:
    out_dir = FIG_DIR / "per_sample"
    out_dir.mkdir(parents=True, exist_ok=True)
    for sample_label, sample_df in per_sample.groupby("sample_label", sort=False):
        sample_dir = out_dir / safe_name(sample_label)
        sample_dir.mkdir(parents=True, exist_ok=True)
        for class_name in CLASS_ORDER:
            class_df = sample_df[sample_df["target_class"] == class_name].copy()
            targets = ordered_targets(class_name, set(class_df["target_variable"]))
            if not targets:
                continue
            matrix = (
                class_df.pivot(index="source_group", columns="target_variable", values="corr")
                .reindex(index=SOURCE_ORDER, columns=targets)
                .dropna(axis=0, how="all")
            )
            qvals = (
                class_df.pivot(index="source_group", columns="target_variable", values="q_class_per_sample")
                .reindex(index=matrix.index, columns=matrix.columns)
            )
            heatmap(
                matrix,
                qvals,
                sample_dir / f"heatmap_{class_name}.png",
                f"{sample_label}: SNAI1-ac all-in-one correlation, {CLASS_LABELS[class_name]}",
                "Pearson r",
                "Stars: class-wise BH q within this sample and source group.",
                width_per_col=0.42 if class_name != "hallmark" else 0.33,
            )


def select_ranked_rows(class_df: pd.DataFrame) -> pd.DataFrame:
    frame = class_df.sort_values("corr_from_median_fisher_z").copy()
    if frame["target_class"].iloc[0] == "hallmark" and len(frame) > 12:
        neg = frame.head(6)
        pos = frame.tail(6)
        frame = pd.concat([neg, pos], axis=0).drop_duplicates("target_variable")
    return frame.sort_values("corr_from_median_fisher_z")


def plot_ranked_bars(summary: pd.DataFrame) -> None:
    out_dir = FIG_DIR / "cross_sample"
    out_dir.mkdir(parents=True, exist_ok=True)
    for (source_group, class_name), class_df in summary.groupby(["source_group", "target_class"], sort=False):
        if class_name not in CLASS_ORDER or class_df.empty:
            continue
        frame = select_ranked_rows(class_df)
        height = max(3.6, 1.2 + 0.34 * len(frame))
        fig, ax = plt.subplots(figsize=(10, height))
        values = frame["corr_from_median_fisher_z"].to_numpy(dtype=float)
        labels = [display_label(v) for v in frame["target_variable"]]
        colors = np.where(values >= 0, "#b2182b", "#2166ac")
        bars = ax.barh(labels, values, color=colors)
        ax.axvline(0, color="#111827", linewidth=0.8)
        ax.set_xlabel("Median Pearson r across samples")
        ax.set_title(f"{SOURCE_LABELS.get(source_group, source_group)}: {CLASS_LABELS[class_name]}", fontsize=11)
        for bar, value, marker in zip(bars, values, frame["sig_marker_cross_sample_class"]):
            marker_text = "" if pd.isna(marker) else str(marker).strip()
            if not marker_text:
                continue
            xpos = value / 2 if abs(value) > 0.04 else value + (0.035 if value >= 0 else -0.035)
            ax.text(
                xpos,
                bar.get_y() + bar.get_height() / 2,
                marker_text,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="#ffffff" if abs(value) > 0.04 else "#111827",
            )
        ax.tick_params(axis="y", labelsize=8)
        fig.text(
            0.01,
            0.01,
            "Stars: class-wise cross-sample BH q from Fisher-z one-sample t-test.",
            fontsize=7,
            ha="left",
            va="bottom",
            color="#374151",
        )
        fig.subplots_adjust(left=0.34, right=0.98, bottom=0.18, top=0.88)
        out_path = out_dir / f"ranked_{SOURCE_FILE_LABELS.get(source_group, safe_name(source_group))}_{class_name}.png"
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)


def build_qc(raw: pd.DataFrame, per_sample: pd.DataFrame, summary: pd.DataFrame) -> None:
    QC_DIR.mkdir(parents=True, exist_ok=True)
    raw_kstar = (
        raw["variable_a"].astype(str).str.startswith("Kstar_")
        | raw["variable_b"].astype(str).str.startswith("Kstar_")
    )
    raw_snai = (raw["variable_a"] == SNAI1_PRIMARY) ^ (raw["variable_b"] == SNAI1_PRIMARY)
    metrics = [
        ("raw_rows_total", len(raw)),
        ("raw_rows_kstar_involving", int(raw_kstar.sum())),
        ("raw_rows_non_kstar", int((~raw_kstar).sum())),
        ("raw_rows_snai1ac_pairs_any_class", int(raw_snai.sum())),
        ("clean_rows_snai1ac_primary", len(per_sample)),
        ("clean_rows_with_kstar", int(per_sample["target_variable"].astype(str).str.startswith("Kstar_").sum())),
        ("n_samples", int(per_sample["sample_label"].nunique())),
        ("n_source_groups", int(per_sample["source_group"].nunique())),
        ("n_targets", int(per_sample["target_variable"].nunique())),
        ("cross_sample_rows", len(summary)),
    ]
    pd.DataFrame(metrics, columns=["metric", "value"]).to_csv(QC_DIR / "audit_counts.csv", index=False)
    (
        per_sample.groupby(["sample_label", "source_group", "target_class"], as_index=False)
        .agg(
            n_tests=("target_variable", "nunique"),
            n_nominal_p_lt_0_05=("pval", lambda x: int((pd.to_numeric(x, errors="coerce") < 0.05).sum())),
            n_class_bh_q_lt_0_05=("q_class_per_sample", lambda x: int((pd.to_numeric(x, errors="coerce") <= 0.05).sum())),
        )
        .to_csv(QC_DIR / "per_sample_bh_scope_counts.csv", index=False)
    )
    (
        summary.groupby(["source_group", "target_class"], as_index=False)
        .agg(
            n_tests=("target_variable", "nunique"),
            n_cross_sample_q_lt_0_05=("q_cross_sample_class_fisher_t", lambda x: int((pd.to_numeric(x, errors="coerce") <= 0.05).sum())),
        )
        .to_csv(QC_DIR / "cross_sample_bh_scope_counts.csv", index=False)
    )
    if per_sample["target_variable"].astype(str).str.startswith("Kstar_").any():
        raise RuntimeError("Kstar variables survived cleaned per-sample table.")
    if summary["target_variable"].astype(str).str.startswith("Kstar_").any():
        raise RuntimeError("Kstar variables survived cleaned cross-sample summary.")


def build_manifest(per_sample: pd.DataFrame, summary: pd.DataFrame) -> dict:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": (
            "Clean SpottedPy all-in-one neighborhood correlation layer restricted to the "
            "SNAI1-ac primary row/column with class-wise BH correction."
        ),
        "input_raw_table": str(RAW_TABLE),
        "output_root": str(OUT_ROOT),
        "raw_matrix_preserved": True,
        "snai1ac_primary_variable": SNAI1_PRIMARY,
        "source_groups": SOURCE_LABELS,
        "excluded_variables": "Any variable with prefix Kstar_ before cleaned summaries and BH correction.",
        "primary_per_sample_bh_scope": "sample_label x source_group x target_class",
        "primary_cross_sample_bh_scope": "source_group x target_class",
        "no_all_target_panels": True,
        "target_classes": {
            "core_comparators": CORE_COMPARATORS,
            "spacet": SPACET_TARGETS,
            "mp": MP_TARGETS,
            "hallmark": HALLMARK_GROUPS,
        },
        "statistics": {
            "per_sample": "Pearson r and raw p-value inherited from all-in-one table; class-wise BH recalculated after Kstar removal and SNAI1-ac-pair restriction.",
            "cross_sample_primary": "One-sample t-test of Fisher-z transformed per-sample Pearson r values against 0, with class-wise BH correction.",
            "cross_sample_sensitivity": "Wilcoxon signed-rank test of Fisher-z transformed correlations against 0, with class-wise BH correction.",
        },
        "n_samples": int(per_sample["sample_label"].nunique()),
        "n_per_sample_rows": int(len(per_sample)),
        "n_cross_sample_rows": int(len(summary)),
        "tables": {
            "per_sample": str(TABLE_DIR / "allinone_snai1ac_primary_per_sample.csv"),
            "cross_sample_summary": str(TABLE_DIR / "allinone_snai1ac_primary_cross_sample_summary.csv"),
            "target_manifest": str(TABLE_DIR / "allinone_snai1ac_primary_target_manifest.csv"),
        },
    }


def write_readme(manifest: dict) -> None:
    text = f"""# SpottedPy V2 All-In-One SNAI1-ac Primary Clean

This branch preserves the raw SpottedPy-style all-in-one matrix and creates a
clean interpretation layer for the paper-aligned SNAI1-ac question.

## Question

Within neighborhoods centered on SNAI1-ac consensus hot spots, SNAI1-ac
consensus cold spots, or SNAI1-2R hot spots, which class-specific target
features co-vary with local SNAI1-ac?

## What Changed Relative To The Raw All-In-One Matrix

- The raw table was not rewritten: `{RAW_TABLE}`.
- Kstar sample-specific programs were excluded before all cleaned summaries.
- Only pairs involving `{SNAI1_PRIMARY}` were retained.
- TME-vs-TME, MP-vs-Hallmark, Hallmark-vs-Hallmark and other non-primary pairs
  are provenance/QC only, not biological results in this branch.
- No all-target panels are produced.

## Primary BH Scope

Per-sample q-values are calculated within:

`sample_label x source_group x target_class`

Cross-sample q-values are calculated within:

`source_group x target_class`

Hallmark subgroup plots use the full Hallmark-class q-values, not
subgroup-specific correction.

## Tables

- `tables/allinone_snai1ac_primary_per_sample.csv`
- `tables/allinone_snai1ac_primary_cross_sample_summary.csv`
- `tables/allinone_snai1ac_primary_target_manifest.csv`
- `qc/audit_counts.csv`
- `qc/per_sample_bh_scope_counts.csv`
- `qc/cross_sample_bh_scope_counts.csv`

## Figures

- `figures/cross_sample/heatmap_*.png`: class-specific cross-sample heatmaps.
- `figures/cross_sample/ranked_*.png`: class-specific ranked SNAI1-ac bars.
- `figures/per_sample/<sample>/heatmap_*.png`: sample-specific class heatmaps.

## Statistics

Per-sample Pearson r and raw p-values come from the existing all-in-one table.
Cross-sample primary p-values use a one-sample t-test of Fisher-z transformed
correlations against zero. Wilcoxon signed-rank p-values are exported as a
sensitivity column.

## Run Manifest

See `run_manifest.json`.

Created: {manifest["created_at"]}
"""
    (OUT_ROOT / "README.md").write_text(text, encoding="utf-8")


def target_manifest() -> pd.DataFrame:
    rows = []
    for class_name in CLASS_ORDER:
        for target in TARGET_ORDER[class_name]:
            rows.append(
                {
                    "target_class": class_name,
                    "target_variable": target,
                    "target_label": display_label(target),
                    "hallmark_group": hallmark_group_for(target),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    log("Preparing output directories")
    for path in [TABLE_DIR, FIG_DIR, QC_DIR, SCRIPT_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    log(f"Reading raw all-in-one table: {RAW_TABLE}")
    raw = pd.read_csv(RAW_TABLE)
    log(f"Raw rows: {len(raw):,}")
    per_sample = canonicalize_snai1ac_pairs(raw)
    log(f"Clean SNAI1-ac primary rows: {len(per_sample):,}")
    summary = cross_sample_summary(per_sample)
    log(f"Cross-sample rows: {len(summary):,}")

    per_sample.to_csv(TABLE_DIR / "allinone_snai1ac_primary_per_sample.csv", index=False)
    summary.to_csv(TABLE_DIR / "allinone_snai1ac_primary_cross_sample_summary.csv", index=False)
    target_manifest().to_csv(TABLE_DIR / "allinone_snai1ac_primary_target_manifest.csv", index=False)

    log("Writing QC tables")
    build_qc(raw, per_sample, summary)

    log("Generating class-specific figures")
    plot_cross_sample_heatmaps(summary)
    plot_ranked_bars(summary)
    plot_per_sample_heatmaps(per_sample)

    manifest = build_manifest(per_sample, summary)
    (OUT_ROOT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_readme(manifest)
    shutil.copy2(Path(__file__), SCRIPT_DIR / Path(__file__).name)
    log(f"Done: {OUT_ROOT}")


if __name__ == "__main__":
    main()
