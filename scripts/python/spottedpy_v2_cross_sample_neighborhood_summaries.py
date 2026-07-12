from __future__ import annotations

import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_1samp, wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


RUN_ROOT = Path(r"D:\HGSOC_Spatial_Atlas\05_analysis_ready\SpottedPy_v2_paper_aligned")
NEIGHBORHOOD_ROOT = RUN_ROOT / "02_neighborhood_enrichment"

RUN_MODE = os.environ.get("SPOTTEDPY_V2_NEIGHBORHOOD_RUN_MODE", "preflight").strip().lower()
DEFAULT_CONSENSUS_SUBDIR = "consensus_source_full" if RUN_MODE == "full" else "consensus_source_preflight"
DEFAULT_CONTRAST_SUBDIR = (
    "source_group_correlation_contrast_full"
    if RUN_MODE == "full"
    else "source_group_correlation_contrast_preflight"
)
CONSENSUS_SUBDIR = os.environ.get("SPOTTEDPY_V2_CONSENSUS_OUT_SUBDIR", DEFAULT_CONSENSUS_SUBDIR).strip()
CONTRAST_SUBDIR = os.environ.get("SPOTTEDPY_V2_CONTRAST_OUT_SUBDIR", DEFAULT_CONTRAST_SUBDIR).strip()

CONSENSUS_ROOT = NEIGHBORHOOD_ROOT / CONSENSUS_SUBDIR
CONSENSUS_INPUT_PAPERSTYLE = CONSENSUS_ROOT / "tables" / "consensus_source_sourcegroup_comparison_long_paperstyle_v2.csv"
CONSENSUS_INPUT_NATIVE = CONSENSUS_ROOT / "tables" / "consensus_source_sourcegroup_comparison_long.csv"
CONSENSUS_VARIABLE_MANIFEST = CONSENSUS_ROOT / "tables" / "consensus_source_variable_manifest.csv"
CONSENSUS_OUT = CONSENSUS_ROOT / "cross_sample_summary"

CONTRAST_ROOT = NEIGHBORHOOD_ROOT / CONTRAST_SUBDIR
CONTRAST_INPUT = CONTRAST_ROOT / "tables" / "source_group_correlation_contrasts.csv"
CONTRAST_OUT = CONTRAST_ROOT / "cross_sample_summary"

SOURCE_ORDER = [
    "snai1ac_consensus_hot",
    "snai1ac_consensus_cold",
    "snai12r_hot",
]

SOURCE_LABELS = {
    "snai1ac_consensus_hot": "SNAI1-ac hot",
    "snai1ac_consensus_cold": "SNAI1-ac cold",
    "snai12r_hot": "SNAI1-2R hot",
}

CONTRAST_ORDER = [
    "hot_vs_cold",
    "hot_vs_snai12r_hot",
    "hot_vs_background",
]

CONTRAST_LABELS = {
    "hot_vs_cold": "Hot vs cold",
    "hot_vs_snai12r_hot": "Hot vs SNAI1-2R hot",
    "hot_vs_background": "Hot vs background",
}

FAMILY_ORDER = ["core", "spacet", "mp", "hallmark"]
CORE_ORDER = ["SNAI1_ac", "SNAI1_scoregenes", "SNAI1_2R_scoregenes"]
FAMILY_LABELS = {
    "core": "core",
    "spacet": "SpaCET",
    "mp": "MP",
    "hallmark": "Hallmark",
}

CLASS_PLOT_ORDER = ["all_variables", "core", "spacet", "mp", "hallmark"]
CLASS_LABELS = {
    "all_variables": "all variables",
    "core": "core",
    "spacet": "SpaCET",
    "mp": "MP",
    "hallmark": "Hallmark",
}

CONSENSUS_EXCLUDED_FAMILIES = {"kstar"}
CONTRAST_EXCLUDED_FAMILIES = {"kstar"}


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))


def bh_qvalues(pvals: pd.Series) -> pd.Series:
    values = pd.to_numeric(pvals, errors="coerce")
    out = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    n = len(valid)
    if n == 0:
        return out
    ranks = np.arange(1, n + 1)
    adjusted = valid.to_numpy() * n / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out.loc[valid.index] = np.clip(adjusted, 0, 1)
    return out


def sig_marker(qval: float) -> str:
    if pd.isna(qval):
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


def fisher_z(r: float) -> float:
    if pd.isna(r):
        return math.nan
    return float(np.arctanh(np.clip(float(r), -0.999999, 0.999999)))


def inverse_fisher_z(z: float) -> float:
    if pd.isna(z):
        return math.nan
    return float(np.tanh(float(z)))


def safe_iqr(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return math.nan
    return float(values.quantile(0.75) - values.quantile(0.25))


def safe_wilcoxon(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    values = values[values != 0]
    if len(values) < 2:
        return math.nan
    try:
        return float(wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return math.nan


def safe_ttest(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 2:
        return math.nan
    try:
        return float(ttest_1samp(values, 0.0, nan_policy="omit").pvalue)
    except ValueError:
        return math.nan


def safe_sign_test(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    n_pos = int((values > 0).sum())
    n_neg = int((values < 0).sum())
    n = n_pos + n_neg
    if n == 0:
        return math.nan
    return float(binomtest(max(n_pos, n_neg), n=n, p=0.5, alternative="two-sided").pvalue)


def variable_family(variable: str, manifest_lookup: dict[str, str] | None = None) -> str:
    variable = str(variable)
    if variable in {"SNAI1_ac", "SNAI1_scoregenes", "SNAI1_2R_scoregenes"}:
        return "core"
    if variable.startswith("SpaCET_"):
        return "spacet"
    if variable.startswith("MP"):
        return "mp"
    if variable.startswith("Kstar_"):
        return "kstar"
    if manifest_lookup and variable in manifest_lookup:
        family = str(manifest_lookup[variable])
        if family == "mp_kstar":
            return "mp"
        return family
    return "hallmark"


def pretty_variable(variable: str) -> str:
    variable = str(variable)
    if variable == "SNAI1_ac":
        return "SNAI1-ac"
    if variable == "SNAI1_scoregenes":
        return "SNAI1"
    if variable == "SNAI1_2R_scoregenes":
        return "SNAI1-2R"
    if variable.startswith("SpaCET_"):
        return variable.replace("SpaCET_", "").replace("_", " ")
    if variable.startswith("MP"):
        return variable
    return variable.replace("_", " ")


def ordered_variables(frame: pd.DataFrame, selected_class: str) -> list[str]:
    subset = frame if selected_class == "all_variables" else frame.loc[frame["variable_family"] == selected_class]
    variables = list(dict.fromkeys(subset["outer_variable"].dropna().astype(str)))
    core_priority = {value: idx for idx, value in enumerate(CORE_ORDER)}
    return sorted(
        variables,
        key=lambda value: (
            FAMILY_ORDER.index(variable_family(value, frame.set_index("outer_variable")["variable_family"].to_dict()))
            if variable_family(value, frame.set_index("outer_variable")["variable_family"].to_dict()) in FAMILY_ORDER
            else 999,
            core_priority.get(value, 999),
            pretty_variable(value).lower(),
        ),
    )


def group_spans(variables: list[str], frame: pd.DataFrame) -> list[tuple[str, int, int]]:
    family_lookup = frame.drop_duplicates("outer_variable").set_index("outer_variable")["variable_family"].to_dict()
    spans: list[tuple[str, int, int]] = []
    current = None
    start = 0
    for idx, variable in enumerate(variables):
        family = family_lookup.get(variable, variable_family(variable))
        if current is None:
            current = family
            start = idx
        elif family != current:
            spans.append((FAMILY_LABELS.get(current, current), start, idx - 1))
            current = family
            start = idx
    if current is not None:
        spans.append((FAMILY_LABELS.get(current, current), start, len(variables) - 1))
    return spans


def add_family_ticks(ax: plt.Axes, variables: list[str], frame: pd.DataFrame) -> None:
    for label, start, end in group_spans(variables, frame):
        midpoint = (start + end) / 2
        ax.text(
            midpoint,
            -1.35,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            transform=ax.transData,
            clip_on=False,
        )
        if end < len(variables) - 1:
            ax.axvline(end + 0.5, color="#111827", linewidth=0.8)


def prep_dirs(root: Path, figure_dirs: list[str]) -> dict[str, Path]:
    table_dir = root / "tables"
    script_dir = root / "scripts_used"
    table_dir.mkdir(parents=True, exist_ok=True)
    script_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = {}
    for figure_dir in figure_dirs:
        path = root / "figures" / figure_dir
        path.mkdir(parents=True, exist_ok=True)
        figure_paths[figure_dir] = path
    shutil.copy2(Path(__file__), script_dir / Path(__file__).name)
    return {"tables": table_dir, "scripts": script_dir, **figure_paths}


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_readme(path: Path, title: str, body: str) -> None:
    path.write_text(f"# {title}\n\n{body.strip()}\n", encoding="utf-8")


def summarize_values(
    group: pd.DataFrame,
    value_col: str,
    sample_cols: list[str] | None = None,
) -> dict[str, object]:
    values = pd.to_numeric(group[value_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    out: dict[str, object] = {
        "n_samples_available": int(values.shape[0]),
        "sample_labels": ";".join(sorted(group.loc[values.index, "sample_label"].astype(str).unique())),
        "median_value": float(values.median()) if len(values) else math.nan,
        "mean_value": float(values.mean()) if len(values) else math.nan,
        "iqr_value": safe_iqr(values),
        "n_positive": int((values > 0).sum()),
        "n_negative": int((values < 0).sum()),
        "n_zero": int((values == 0).sum()),
        "fraction_positive": float((values > 0).mean()) if len(values) else math.nan,
        "fraction_negative": float((values < 0).mean()) if len(values) else math.nan,
        "wilcoxon_pval": safe_wilcoxon(values),
        "ttest_pval": safe_ttest(values),
        "sign_test_pval": safe_sign_test(values),
    }
    if sample_cols:
        for col in sample_cols:
            if col in group:
                col_values = pd.to_numeric(group[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                out[f"median_{col}"] = float(col_values.median()) if len(col_values) else math.nan
    return out


def plot_source_heatmap(
    summary: pd.DataFrame,
    out_path: Path,
    selected_class: str,
    title: str,
    value_col: str,
    q_col: str,
    y_col: str,
    y_order: list[str],
    y_labels: dict[str, str],
    color_label: str,
    vlim: float,
) -> None:
    variables = ordered_variables(summary, selected_class)
    if not variables:
        return
    subset = summary if selected_class == "all_variables" else summary.loc[summary["variable_family"] == selected_class]
    rows = [value for value in y_order if value in set(subset[y_col])]
    if not rows:
        return
    matrix = pd.DataFrame(np.nan, index=rows, columns=variables, dtype=float)
    qmatrix = matrix.copy()
    for row in subset.itertuples(index=False):
        row_id = getattr(row, y_col)
        variable = row.outer_variable
        if row_id in matrix.index and variable in matrix.columns:
            matrix.loc[row_id, variable] = getattr(row, value_col)
            qmatrix.loc[row_id, variable] = getattr(row, q_col)
    width = max(8, min(24, 0.33 * len(variables) + 3.2))
    height = max(2.8, 0.46 * len(rows) + 2.4)
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap="coolwarm", vmin=-vlim, vmax=vlim, aspect="auto")
    ax.set_xticks(range(len(variables)))
    ax.set_xticklabels([pretty_variable(v) for v in variables], rotation=90, fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([y_labels.get(row, row) for row in rows], fontsize=9)
    ax.set_title(title, fontsize=11)
    for i, row in enumerate(rows):
        for j, variable in enumerate(variables):
            mark = sig_marker(qmatrix.loc[row, variable])
            if mark:
                ax.text(j, i, mark, ha="center", va="center", fontsize=7, color="#111827")
    if selected_class == "all_variables":
        add_family_ticks(ax, variables, summary)
    colorbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    colorbar.set_label(color_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_ranked_bars(
    summary: pd.DataFrame,
    out_path: Path,
    selected_class: str,
    title: str,
    value_col: str,
    q_col: str,
    id_col: str,
    id_value: str,
    max_each_direction: int = 6,
) -> None:
    subset = summary.loc[summary[id_col] == id_value].copy()
    if selected_class != "all_variables":
        subset = subset.loc[subset["variable_family"] == selected_class].copy()
    if subset.empty:
        return
    subset["rank_abs"] = subset[value_col].abs()
    pos = subset.loc[subset[value_col] > 0].nlargest(max_each_direction, value_col)
    neg = subset.loc[subset[value_col] < 0].nsmallest(max_each_direction, value_col)
    plot_df = pd.concat([neg, pos], ignore_index=True).drop_duplicates("outer_variable")
    if plot_df.empty:
        plot_df = subset.nlargest(max_each_direction * 2, "rank_abs")
    plot_df = plot_df.sort_values(value_col)
    height = max(4, 0.36 * len(plot_df) + 1.8)
    fig, ax = plt.subplots(figsize=(8.8, height))
    colors = np.where(plot_df[value_col] >= 0, "#b2182b", "#2166ac")
    bars = ax.barh([pretty_variable(v) for v in plot_df["outer_variable"]], plot_df[value_col], color=colors)
    ax.axvline(0, color="#111827", linewidth=0.8)
    ax.set_xlabel(value_col.replace("_", " "))
    ax.set_title(title, fontsize=11)
    x_values = plot_df[value_col].to_numpy(dtype=float)
    if len(x_values):
        span = max(abs(np.nanmin(x_values)), abs(np.nanmax(x_values)), 0.05)
        ax.set_xlim(-span * 1.25, span * 1.25)
        offset = span * 0.035
    else:
        offset = 0.01
    for y, (bar, row) in enumerate(zip(bars, plot_df.itertuples(index=False))):
        mark = sig_marker(getattr(row, q_col))
        if not mark:
            continue
        x = getattr(row, value_col)
        ax.text(
            x / 2,
            bar.get_y() + bar.get_height() / 2,
            mark,
            va="center",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#ffffff",
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_sample_strips(
    canonical: pd.DataFrame,
    ranked: pd.DataFrame,
    out_path: Path,
    id_col: str,
    id_value: str,
    value_col: str,
    title: str,
    max_variables: int = 12,
) -> None:
    top_variables = ranked.loc[ranked[id_col] == id_value].copy()
    if top_variables.empty:
        return
    top_variables["rank_abs"] = top_variables["median_value"].abs()
    top_variables = top_variables.nlargest(max_variables, "rank_abs")["outer_variable"].tolist()
    subset = canonical.loc[(canonical[id_col] == id_value) & (canonical["outer_variable"].isin(top_variables))].copy()
    if subset.empty:
        return
    order = sorted(top_variables, key=lambda v: pretty_variable(v).lower())
    x_lookup = {variable: idx for idx, variable in enumerate(order)}
    fig_width = max(7.5, min(14, 0.45 * len(order) + 3))
    fig, ax = plt.subplots(figsize=(fig_width, 4.2))
    for sample_idx, sample in enumerate(sorted(subset["sample_label"].unique())):
        sample_df = subset.loc[subset["sample_label"] == sample]
        x = [x_lookup[v] + (sample_idx - 1) * 0.08 for v in sample_df["outer_variable"]]
        ax.scatter(x, sample_df[value_col], s=30, alpha=0.82, label=sample)
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([pretty_variable(v) for v in order], rotation=90, fontsize=7)
    ax.set_ylabel(value_col.replace("_", " "))
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_hot_cold_mp_boxplots(canonical: pd.DataFrame, out_path: Path) -> None:
    subset = canonical.loc[
        (canonical["contrast"] == "hot_vs_cold")
        & (canonical["variable_family"] == "mp")
    ].copy()
    if subset.empty:
        return

    mp_order = sorted(subset["outer_variable"].dropna().unique(), key=pretty_variable)
    records = []
    for row in subset.itertuples(index=False):
        records.append(
            {
                "sample_label": row.sample_label,
                "outer_variable": row.outer_variable,
                "group": "SNAI1-ac hot",
                "pearson_r": row.r_hot,
            }
        )
        records.append(
            {
                "sample_label": row.sample_label,
                "outer_variable": row.outer_variable,
                "group": "SNAI1-ac cold",
                "pearson_r": row.r_reference,
            }
        )
    long_df = pd.DataFrame(records)
    long_df["pearson_r"] = pd.to_numeric(long_df["pearson_r"], errors="coerce")
    long_df = long_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["pearson_r"])
    if long_df.empty:
        return

    x_positions = np.arange(len(mp_order), dtype=float)
    offsets = {"SNAI1-ac hot": -0.18, "SNAI1-ac cold": 0.18}
    colors = {"SNAI1-ac hot": "#b2182b", "SNAI1-ac cold": "#2166ac"}

    fig_width = max(9.5, 0.85 * len(mp_order) + 3)
    fig, ax = plt.subplots(figsize=(fig_width, 5.0))

    for group, offset in offsets.items():
        box_data = [
            long_df.loc[
                (long_df["outer_variable"] == variable) & (long_df["group"] == group),
                "pearson_r",
            ].to_numpy(dtype=float)
            for variable in mp_order
        ]
        positions = x_positions + offset
        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.28,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111827", "linewidth": 1.1},
            whiskerprops={"color": "#374151", "linewidth": 0.8},
            capprops={"color": "#374151", "linewidth": 0.8},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[group])
            patch.set_alpha(0.42)
            patch.set_edgecolor(colors[group])

        for idx, variable in enumerate(mp_order):
            vals = long_df.loc[
                (long_df["outer_variable"] == variable) & (long_df["group"] == group),
                ["sample_label", "pearson_r"],
            ].sort_values("sample_label")
            if vals.empty:
                continue
            jitter = np.linspace(-0.035, 0.035, len(vals)) if len(vals) > 1 else np.array([0.0])
            ax.scatter(
                np.full(len(vals), positions[idx]) + jitter,
                vals["pearson_r"].to_numpy(dtype=float),
                s=28,
                color=colors[group],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )

    # Connect paired sample values within each MP to make the sample-level unit visible.
    for idx, variable in enumerate(mp_order):
        wide = subset.loc[subset["outer_variable"] == variable, ["sample_label", "r_hot", "r_reference"]].copy()
        wide["r_hot"] = pd.to_numeric(wide["r_hot"], errors="coerce")
        wide["r_reference"] = pd.to_numeric(wide["r_reference"], errors="coerce")
        wide = wide.dropna(subset=["r_hot", "r_reference"])
        for row in wide.itertuples(index=False):
            ax.plot(
                [x_positions[idx] + offsets["SNAI1-ac hot"], x_positions[idx] + offsets["SNAI1-ac cold"]],
                [row.r_hot, row.r_reference],
                color="#6b7280",
                alpha=0.35,
                linewidth=0.8,
                zorder=2,
            )

    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([pretty_variable(variable) for variable in mp_order], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Inner-outer Pearson r")
    ax.set_title("SNAI1-ac hot vs cold: MP neighborhood correlations across samples", fontsize=11)
    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=colors[group], alpha=0.65, markersize=9)
        for group in offsets
    ]
    ax.legend(handles, list(offsets), frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def canonicalize_consensus() -> tuple[pd.DataFrame, pd.DataFrame]:
    log("Reading consensus source table")
    consensus_input = CONSENSUS_INPUT_PAPERSTYLE if CONSENSUS_INPUT_PAPERSTYLE.exists() else CONSENSUS_INPUT_NATIVE
    if not consensus_input.exists():
        raise FileNotFoundError(f"Missing consensus source table: {CONSENSUS_INPUT_PAPERSTYLE} or {CONSENSUS_INPUT_NATIVE}")
    log(f"Consensus input: {consensus_input}")
    raw = pd.read_csv(consensus_input)
    manifest = pd.read_csv(CONSENSUS_VARIABLE_MANIFEST) if CONSENSUS_VARIABLE_MANIFEST.exists() else pd.DataFrame()
    manifest_lookup = (
        manifest.drop_duplicates("variable_id").set_index("variable_id")["family"].astype(str).to_dict()
        if not manifest.empty
        else {}
    )
    canonical = raw.loc[raw["variable_class"] == "all_variables"].copy()
    canonical["corr"] = pd.to_numeric(canonical["corr"], errors="coerce")
    canonical["pval"] = pd.to_numeric(canonical["pval"], errors="coerce")
    canonical["qval"] = pd.to_numeric(canonical["qval"], errors="coerce")
    canonical["n_pairs"] = pd.to_numeric(canonical["n_pairs"], errors="coerce")
    canonical["variable_family"] = canonical["outer_variable"].map(lambda value: variable_family(value, manifest_lookup))
    canonical = canonical.loc[~canonical["variable_family"].isin(CONSENSUS_EXCLUDED_FAMILIES)].copy()
    canonical["outer_variable_display"] = canonical["outer_variable"].map(pretty_variable)
    canonical["fisher_z"] = canonical["corr"].map(fisher_z)
    canonical = canonical.drop_duplicates(["sample_label", "source_group", "outer_variable"])
    return canonical, manifest


def consensus_summary(canonical: pd.DataFrame) -> pd.DataFrame:
    records = []
    grouped = canonical.groupby(["source_group", "outer_variable", "variable_family"], dropna=False)
    for (source_group, outer_variable, family), group in grouped:
        base = summarize_values(group, "fisher_z", ["corr", "n_pairs"])
        base.update(
            {
                "source_group": source_group,
                "source_group_display": SOURCE_LABELS.get(source_group, source_group),
                "outer_variable": outer_variable,
                "outer_variable_display": pretty_variable(outer_variable),
                "variable_family": family,
                "median_fisher_z": base.pop("median_value"),
                "mean_fisher_z": base.pop("mean_value"),
                "iqr_fisher_z": base.pop("iqr_value"),
            }
        )
        base["median_corr_from_fisher_z"] = inverse_fisher_z(base["median_fisher_z"])
        base["mean_corr_from_fisher_z"] = inverse_fisher_z(base["mean_fisher_z"])
        records.append(base)
    summary = pd.DataFrame(records)
    summary = summary.sort_values(["source_group", "variable_family", "outer_variable"]).reset_index(drop=True)
    summary["wilcoxon_qval"] = np.nan
    summary["ttest_qval"] = np.nan
    summary["sign_test_qval"] = np.nan
    for (source_group, family), idx in summary.groupby(["source_group", "variable_family"]).groups.items():
        idx = list(idx)
        summary.loc[idx, "wilcoxon_qval"] = bh_qvalues(summary.loc[idx, "wilcoxon_pval"])
        summary.loc[idx, "ttest_qval"] = bh_qvalues(summary.loc[idx, "ttest_pval"])
        summary.loc[idx, "sign_test_qval"] = bh_qvalues(summary.loc[idx, "sign_test_pval"])
    summary["sig_marker"] = summary["wilcoxon_qval"].map(sig_marker)
    return summary


def run_consensus() -> dict[str, object]:
    out_dirs = prep_dirs(
        CONSENSUS_OUT,
        [
            "01_cross_sample_source_group_heatmaps",
            "02_cross_sample_ranked_correlation_barplots",
            "03_cross_sample_sample_value_strips",
        ],
    )
    canonical, manifest = canonicalize_consensus()
    summary = consensus_summary(canonical)
    ranked = summary.copy()
    ranked["rank_abs"] = ranked["median_corr_from_fisher_z"].abs()
    ranked = ranked.sort_values(["source_group", "rank_abs"], ascending=[True, False])

    canonical.to_csv(out_dirs["tables"] / "consensus_source_cross_sample_canonical_input.csv", index=False)
    summary.to_csv(out_dirs["tables"] / "consensus_source_cross_sample_summary.csv", index=False)
    ranked.to_csv(out_dirs["tables"] / "consensus_source_cross_sample_ranked_for_plots.csv", index=False)
    manifest.to_csv(out_dirs["tables"] / "consensus_source_cross_sample_variable_manifest.csv", index=False)

    for selected_class in CLASS_PLOT_ORDER:
        plot_source_heatmap(
            summary,
            out_dirs["01_cross_sample_source_group_heatmaps"]
            / f"{selected_class}__cross_sample_source_group_median_pearson.png",
            selected_class,
            f"Cross-sample source-group neighborhood correlations ({CLASS_LABELS[selected_class]})",
            "median_corr_from_fisher_z",
            "wilcoxon_qval",
            "source_group",
            SOURCE_ORDER,
            SOURCE_LABELS,
            "median Fisher-z Pearson r",
            1.0,
        )
    for source_group in SOURCE_ORDER:
        for selected_class in CLASS_PLOT_ORDER:
            plot_ranked_bars(
                summary,
                out_dirs["02_cross_sample_ranked_correlation_barplots"]
                / f"{safe_name(source_group)}__{selected_class}__ranked_median_pearson.png",
                selected_class,
                f"{SOURCE_LABELS.get(source_group, source_group)}: recurrent neighborhood correlations ({CLASS_LABELS[selected_class]})",
                "median_corr_from_fisher_z",
                "wilcoxon_qval",
                "source_group",
                source_group,
            )
        plot_sample_strips(
            canonical,
            ranked.rename(columns={"median_corr_from_fisher_z": "median_value"}),
            out_dirs["03_cross_sample_sample_value_strips"] / f"{safe_name(source_group)}__top12_sample_values.png",
            "source_group",
            source_group,
            "corr",
            f"{SOURCE_LABELS.get(source_group, source_group)}: sample-level Pearson r for top variables",
        )

    readme = f"""
This layer summarizes the per-sample `{CONSENSUS_SUBDIR}` neighborhood-correlation results across samples.

Canonical input is `tables/consensus_source_cross_sample_canonical_input.csv`, derived from the `all_variables` rows of the per-sample preflight table.
Exact sample-specific K* program variables are intentionally omitted from this cross-sample summary.

The sample/patient remains the biological unit. Pearson correlations are converted to Fisher z per sample before cross-sample summarization.
Figures show the median Fisher-z Pearson correlation, back-transformed to r for display. Stars use Wilcoxon signed-rank p-values BH-adjusted within each source group and variable family.

Run mode: `{RUN_MODE}`.
"""
    write_readme(CONSENSUS_OUT / "README.md", "SpottedPy v2 Consensus Source Cross-Sample Summary", readme)
    return {
        "root": str(CONSENSUS_OUT),
        "canonical_rows": int(len(canonical)),
        "summary_rows": int(len(summary)),
        "samples": sorted(canonical["sample_label"].unique().tolist()),
        "excluded_families": sorted(CONSENSUS_EXCLUDED_FAMILIES),
    }


def canonicalize_contrast() -> pd.DataFrame:
    log("Reading source-group correlation contrast table")
    raw = pd.read_csv(CONTRAST_INPUT)
    canonical = raw.loc[raw["variable_class"] == "all_variables"].copy()
    numeric_cols = [
        "r_hot",
        "r_reference",
        "delta_r",
        "z_hot",
        "z_reference",
        "delta_z",
        "n_hot_pairs",
        "n_reference_pairs",
        "source_node_overlap",
        "fisher_pval",
        "permutation_pval",
        "fisher_qval",
        "permutation_qval",
    ]
    for col in numeric_cols:
        if col in canonical:
            canonical[col] = pd.to_numeric(canonical[col], errors="coerce")
    canonical["variable_family"] = canonical["outer_variable"].map(lambda value: variable_family(value))
    canonical = canonical.loc[~canonical["variable_family"].isin(CONTRAST_EXCLUDED_FAMILIES)].copy()
    canonical["outer_variable_display"] = canonical["outer_variable"].map(pretty_variable)
    canonical = canonical.drop_duplicates(["sample_label", "contrast", "outer_variable"])
    return canonical


def sign_pattern(r_hot: float, r_reference: float, delta_z: float) -> str:
    if pd.isna(r_hot) or pd.isna(r_reference) or pd.isna(delta_z):
        return "not_available"
    if r_hot >= 0 and r_reference >= 0:
        return "both_positive_hot_more_positive" if delta_z > 0 else "both_positive_hot_less_positive"
    if r_hot < 0 and r_reference < 0:
        return "both_negative_hot_less_negative" if delta_z > 0 else "both_negative_hot_more_negative"
    if r_hot >= 0 and r_reference < 0:
        return "hot_positive_reference_negative"
    return "hot_negative_reference_positive"


def contrast_summary(canonical: pd.DataFrame) -> pd.DataFrame:
    records = []
    grouped = canonical.groupby(["contrast", "outer_variable", "variable_family"], dropna=False)
    for (contrast, outer_variable, family), group in grouped:
        base = summarize_values(group, "delta_z", ["r_hot", "r_reference", "delta_r", "n_hot_pairs", "n_reference_pairs"])
        base.update(
            {
                "contrast": contrast,
                "contrast_display": CONTRAST_LABELS.get(contrast, contrast),
                "outer_variable": outer_variable,
                "outer_variable_display": pretty_variable(outer_variable),
                "variable_family": family,
                "median_delta_z": base.pop("median_value"),
                "mean_delta_z": base.pop("mean_value"),
                "iqr_delta_z": base.pop("iqr_value"),
            }
        )
        base["median_sign_pattern"] = sign_pattern(
            base.get("median_r_hot", math.nan),
            base.get("median_r_reference", math.nan),
            base["median_delta_z"],
        )
        records.append(base)
    summary = pd.DataFrame(records)
    summary = summary.sort_values(["contrast", "variable_family", "outer_variable"]).reset_index(drop=True)
    summary["wilcoxon_qval"] = np.nan
    summary["ttest_qval"] = np.nan
    summary["sign_test_qval"] = np.nan
    for (contrast, family), idx in summary.groupby(["contrast", "variable_family"]).groups.items():
        idx = list(idx)
        summary.loc[idx, "wilcoxon_qval"] = bh_qvalues(summary.loc[idx, "wilcoxon_pval"])
        summary.loc[idx, "ttest_qval"] = bh_qvalues(summary.loc[idx, "ttest_pval"])
        summary.loc[idx, "sign_test_qval"] = bh_qvalues(summary.loc[idx, "sign_test_pval"])
    summary["sig_marker"] = summary["wilcoxon_qval"].map(sig_marker)
    return summary


def run_contrast() -> dict[str, object]:
    out_dirs = prep_dirs(
        CONTRAST_OUT,
        [
            "01_cross_sample_delta_z_heatmaps",
            "02_cross_sample_ranked_delta_z_barplots",
            "03_cross_sample_sample_value_strips",
            "04_cross_sample_hot_cold_mp_boxplots",
        ],
    )
    canonical = canonicalize_contrast()
    summary = contrast_summary(canonical)
    ranked = summary.copy()
    ranked["rank_abs"] = ranked["median_delta_z"].abs()
    ranked = ranked.sort_values(["contrast", "rank_abs"], ascending=[True, False])

    canonical.to_csv(out_dirs["tables"] / "source_group_contrast_cross_sample_canonical_input.csv", index=False)
    summary.to_csv(out_dirs["tables"] / "source_group_contrast_cross_sample_summary.csv", index=False)
    ranked.to_csv(out_dirs["tables"] / "source_group_contrast_cross_sample_ranked_for_plots.csv", index=False)
    variable_manifest = (
        canonical[["outer_variable", "outer_variable_display", "variable_family"]]
        .drop_duplicates()
        .sort_values(["variable_family", "outer_variable"])
    )
    variable_manifest.to_csv(out_dirs["tables"] / "source_group_contrast_cross_sample_variable_manifest.csv", index=False)

    finite = summary["median_delta_z"].replace([np.inf, -np.inf], np.nan).dropna()
    vlim = float(max(0.2, min(2.5, finite.abs().max() if len(finite) else 1.0)))
    for selected_class in CLASS_PLOT_ORDER:
        plot_source_heatmap(
            summary,
            out_dirs["01_cross_sample_delta_z_heatmaps"] / f"{selected_class}__cross_sample_delta_z.png",
            selected_class,
            f"Cross-sample SNAI1-ac-hot relative neighborhood-correlation shifts ({CLASS_LABELS[selected_class]})",
            "median_delta_z",
            "wilcoxon_qval",
            "contrast",
            CONTRAST_ORDER,
            CONTRAST_LABELS,
            "median delta Fisher z",
            vlim,
        )
    for contrast in CONTRAST_ORDER:
        for selected_class in CLASS_PLOT_ORDER:
            plot_ranked_bars(
                summary,
                out_dirs["02_cross_sample_ranked_delta_z_barplots"]
                / f"{safe_name(contrast)}__{selected_class}__ranked_median_delta_z.png",
                selected_class,
                f"{CONTRAST_LABELS.get(contrast, contrast)}: recurrent signed shifts ({CLASS_LABELS[selected_class]})",
                "median_delta_z",
                "wilcoxon_qval",
                "contrast",
                contrast,
            )
        plot_sample_strips(
            canonical,
            ranked.rename(columns={"median_delta_z": "median_value"}),
            out_dirs["03_cross_sample_sample_value_strips"] / f"{safe_name(contrast)}__top12_sample_delta_z.png",
            "contrast",
            contrast,
            "delta_z",
            f"{CONTRAST_LABELS.get(contrast, contrast)}: sample-level delta Fisher z for top variables",
        )

    plot_hot_cold_mp_boxplots(
        canonical,
        out_dirs["04_cross_sample_hot_cold_mp_boxplots"] / "hot_vs_cold__mp__r_hot_vs_r_cold_boxplots.png",
    )

    readme = f"""
This layer summarizes the per-sample `{CONTRAST_SUBDIR}` results across samples.

Canonical input is `tables/source_group_contrast_cross_sample_canonical_input.csv`, derived from the `all_variables` rows of the per-sample contrast table.
Exact sample-specific K* program variables are intentionally omitted from the main cross-sample summary.

The sample/patient remains the biological unit. The primary effect is `delta_z = atanh(r_hot) - atanh(r_reference)`.
Positive delta-z values mean the SNAI1-ac-hot source group is shifted toward a more positive correlation than the reference group. Negative values mean it is shifted toward a more negative correlation.

The summary table includes `median_r_hot`, `median_r_reference`, and `median_sign_pattern` to support precise interpretation and avoid collapsing "more positive", "less negative", "weaker positive", and "more negative" into one vague coupling statement.

Stars use Wilcoxon signed-rank p-values BH-adjusted within each contrast and variable family. No global all-variable BH option is applied, and no minimum `n_hot_pairs`/`n_reference_pairs` threshold is used.

Run mode: `{RUN_MODE}`.
"""
    write_readme(CONTRAST_OUT / "README.md", "SpottedPy v2 Source-Group Contrast Cross-Sample Summary", readme)
    return {
        "root": str(CONTRAST_OUT),
        "canonical_rows": int(len(canonical)),
        "summary_rows": int(len(summary)),
        "samples": sorted(canonical["sample_label"].unique().tolist()),
        "excluded_families": sorted(CONTRAST_EXCLUDED_FAMILIES),
    }


def main() -> None:
    log("Starting SpottedPy v2 cross-sample neighborhood summaries")
    consensus_result = run_consensus()
    contrast_result = run_contrast()
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "run_mode": RUN_MODE,
        "run_root": str(RUN_ROOT),
        "status": "completed",
        "consensus_source_preflight": consensus_result,
        "source_group_correlation_contrast_preflight": contrast_result,
        "design_notes": [
            "Preflight layer only; regenerate after the full cohort run.",
            "Sample/patient is the cross-sample unit.",
            "K* programs are omitted from primary cross-sample summaries because exact K* programs are sample-specific.",
            "Source-group contrast summaries use no minimum pair-count threshold.",
            "BH correction is within source/contrast and variable family; no stricter all-variable-per-contrast option is applied.",
        ],
    }
    write_json(CONSENSUS_OUT / "run_manifest.json", manifest)
    write_json(CONTRAST_OUT / "run_manifest.json", manifest)
    log("Completed cross-sample summaries")
    log(f"Consensus summary root: {CONSENSUS_OUT}")
    log(f"Contrast summary root: {CONTRAST_OUT}")


if __name__ == "__main__":
    main()
