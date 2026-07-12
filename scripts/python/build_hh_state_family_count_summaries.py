from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"D:\HGSOC_Spatial_Atlas\05_analysis_ready")
HH_ROOT = ROOT / "20260424_definition3b_definition4_raw_geneNMF" / "08_hh_programme_characterization"
FAMILY_TABLE = (
    ROOT
    / "20260424_definition3b_definition4_raw_geneNMF"
    / "11_research_synthesis"
    / "tables"
    / "program_family_annotation_snapshot.csv"
)
OUT_ROOT = HH_ROOT / "report_assets"
OUT_TABLES = OUT_ROOT / "tables"
OUT_FIGURES = OUT_ROOT / "figures"


CONTRASTS = [
    {
        "contrast_id": "hh_vs_matched_nonhh",
        "contrast_label": "HH versus matched non-HH",
        "control_label": "non-HH",
        "spot_table": HH_ROOT / "tables" / "hh_spot_level_programme_contrasts.csv",
        "neighborhood_table": HH_ROOT / "tables" / "hh_neighborhood_programme_contrasts.csv",
    },
    {
        "contrast_id": "hh_vs_matched_ll",
        "contrast_label": "HH versus matched LL",
        "control_label": "LL",
        "spot_table": HH_ROOT
        / "hh_vs_ll_malignant_matched"
        / "tables"
        / "hh_ll_matched_spot_level_programme_contrasts.csv",
        "neighborhood_table": HH_ROOT
        / "hh_vs_ll_malignant_matched"
        / "tables"
        / "hh_ll_matched_neighborhood_programme_contrasts.csv",
    },
    {
        "contrast_id": "hh_vs_ll_unmatched",
        "contrast_label": "HH versus LL unmatched",
        "control_label": "LL",
        "spot_table": HH_ROOT / "hh_vs_ll_unmatched" / "tables" / "hh_ll_spot_level_programme_contrasts.csv",
        "neighborhood_table": HH_ROOT
        / "hh_vs_ll_unmatched"
        / "tables"
        / "hh_ll_neighborhood_programme_contrasts.csv",
    },
]

FAMILY_ORDER = [
    "ECM-remodelling myCAF",
    "Inflammatory/hypoxic CAF-stress",
    "Angiogenic vascular/pericyte",
    "IFN/TLS chemokine immune",
    "APC/TAM myeloid",
    "Ciliated epithelial context",
    "Malignant hypoxia/stress",
    "Malignant OXPHOS/metabolic",
    "Malignant proliferation/biosynthesis",
    "Malignant EMT/interface/secretory",
    "Technical/low-quality",
]

CONTEXT_LABELS = {
    "spot_level": "Centre spots",
    "neighborhood": "Immediate neighbours",
}


def ensure_dirs() -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)


def load_family_table() -> pd.DataFrame:
    family = pd.read_csv(FAMILY_TABLE)
    keep = [
        "program_id",
        "family_id",
        "family_label",
        "analysis_role",
        "alignment_category_draft",
        "technical_flag_bool",
    ]
    return family[keep].drop_duplicates("program_id")


def load_one_table(path: Path, context: str, contrast: dict[str, str], family: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["fdr_bh"] = pd.to_numeric(df["fdr_bh"], errors="coerce")
    df = df.loc[df["fdr_bh"] < 0.05].copy()
    df["context_type"] = context
    df["context_label"] = CONTEXT_LABELS[context]
    df["contrast_id"] = contrast["contrast_id"]
    df["contrast_label"] = contrast["contrast_label"]
    df["control_label"] = contrast["control_label"]
    df["enriched_state"] = np.where(df["direction"].eq("HH_enriched"), "HH", contrast["control_label"])
    df["signed_count"] = np.where(df["enriched_state"].eq("HH"), 1, -1)
    merged = df.merge(family, on="program_id", how="left")
    merged["family_label"] = merged["family_label"].fillna("Unmapped")
    merged["family_id"] = merged["family_id"].fillna("Unmapped")
    return merged


def build_significant_program_table() -> pd.DataFrame:
    family = load_family_table()
    rows = []
    for contrast in CONTRASTS:
        rows.append(load_one_table(contrast["spot_table"], "spot_level", contrast, family))
        rows.append(load_one_table(contrast["neighborhood_table"], "neighborhood", contrast, family))
    out = pd.concat(rows, ignore_index=True)
    sort_cols = ["contrast_id", "context_type", "family_label", "enriched_state", "sample_label", "program_id"]
    return out.sort_values(sort_cols).reset_index(drop=True)


def make_counts(sig: pd.DataFrame) -> pd.DataFrame:
    counts = (
        sig.groupby(
            ["contrast_id", "contrast_label", "control_label", "context_type", "context_label", "family_label", "enriched_state"],
            dropna=False,
        )
        .size()
        .reset_index(name="n_significant_program_contrasts")
    )
    return counts


def plot_contrast(counts: pd.DataFrame, contrast_id: str, out_path: Path) -> None:
    sub = counts.loc[counts["contrast_id"].eq(contrast_id)].copy()
    if sub.empty:
        raise ValueError(f"No rows for contrast {contrast_id}")

    contrast_label = sub["contrast_label"].iloc[0]
    control_label = sub["control_label"].iloc[0]
    families = [x for x in FAMILY_ORDER if x in set(sub["family_label"])]
    extra = sorted(set(sub["family_label"]) - set(families))
    families = families + extra
    contexts = ["spot_level", "neighborhood"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, max(5.8, 0.42 * len(families))), sharey=True)
    colors = {"HH": "#bf3f3f", control_label: "#3f6fb5"}

    max_count = 1
    for ax, context in zip(axes, contexts, strict=True):
        ctx = sub.loc[sub["context_type"].eq(context)]
        y = np.arange(len(families))
        hh_counts = []
        control_counts = []
        for fam in families:
            fam_ctx = ctx.loc[ctx["family_label"].eq(fam)]
            hh = fam_ctx.loc[fam_ctx["enriched_state"].eq("HH"), "n_significant_program_contrasts"].sum()
            ctl = fam_ctx.loc[fam_ctx["enriched_state"].eq(control_label), "n_significant_program_contrasts"].sum()
            hh_counts.append(int(hh))
            control_counts.append(int(ctl))
        max_count = max(max_count, *(hh_counts or [0]), *(control_counts or [0]))
        ax.barh(y, [-x for x in control_counts], color=colors[control_label], label=f"{control_label} enriched")
        ax.barh(y, hh_counts, color=colors["HH"], label="HH enriched")
        ax.axvline(0, color="#222222", linewidth=0.8)
        ax.set_title(CONTEXT_LABELS[context], fontsize=12)
        ax.set_xlabel("Significant local program contrasts")
        ax.grid(axis="x", color="#dddddd", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_yticks(y)
        ax.set_yticklabels(families, fontsize=9)

    limit = max_count + 1
    for ax in axes:
        ax.set_xlim(-limit, limit)
        ticks = np.arange(-limit, limit + 1, max(1, int(np.ceil(limit / 5))))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(abs(int(t))) for t in ticks])

    axes[0].invert_yaxis()
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(contrast_label, fontsize=13, y=0.98)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    sig = build_significant_program_table()
    counts = make_counts(sig)

    sig.to_csv(OUT_TABLES / "hh_state_significant_local_programs_by_family.csv", index=False)
    counts.to_csv(OUT_TABLES / "hh_state_family_direction_counts.csv", index=False)

    for contrast in CONTRASTS:
        plot_contrast(
            counts,
            contrast["contrast_id"],
            OUT_FIGURES / f"{contrast['contrast_id']}_significant_program_family_counts.png",
        )

    print(f"Wrote {len(sig)} significant local program rows")
    print(OUT_TABLES / "hh_state_significant_local_programs_by_family.csv")
    print(OUT_TABLES / "hh_state_family_direction_counts.csv")
    for contrast in CONTRASTS:
        print(OUT_FIGURES / f"{contrast['contrast_id']}_significant_program_family_counts.png")


if __name__ == "__main__":
    main()
