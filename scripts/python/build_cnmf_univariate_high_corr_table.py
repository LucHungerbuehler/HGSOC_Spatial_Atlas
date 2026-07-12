"""Build report table for high univariate cNMF usage correlations."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


ROOT = Path(r"D:\HGSOC_Spatial_Atlas\05_analysis_ready")
CORR = (
    ROOT
    / "S3_cNMF_Tumor_Programs"
    / "snai1ac_kstar_usage_univariate_and_model_diagnostics_unsmoothed_uncorrected_v1"
    / "tables"
    / "kstar_usage_univariate_correlations.csv"
)
FAMILY = (
    ROOT
    / "20260424_definition3b_definition4_raw_geneNMF"
    / "11_research_synthesis"
    / "tables"
    / "program_family_annotation_snapshot.csv"
)
OUT_DIR = (
    ROOT
    / "S3_cNMF_Tumor_Programs"
    / "snai1ac_program_decomposition_unsmoothed_uncorrected_v1"
    / "07_report_examples"
    / "tables"
)


def latex_escape(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.replace("\u00a0", " ")
    text = text.replace("γ", "GAMMA_PLACEHOLDER")
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = "".join(replacements.get(char, char) for char in text)
    return escaped.replace("GAMMA\\_PLACEHOLDER", r"\(\gamma\)")


def remove_parenthetical_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    while re.search(r"\([^()]*\)", text):
        text = re.sub(r"\([^()]*\)", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" / ", "/")
    text = text.replace("/", " / ")
    return text.strip()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    corr = pd.read_csv(CORR)
    family = pd.read_csv(FAMILY)

    keep = corr.loc[corr["abs_spearman_r"].ge(0.3)].copy()
    table = keep.merge(
        family[
            [
                "program_id",
                "family_id",
                "family_label",
                "alignment_category_draft",
            ]
        ],
        on="program_id",
        how="left",
        validate="one_to_one",
    )

    if table["family_label"].isna().any():
        missing = table.loc[table["family_label"].isna(), "program_id"].tolist()
        raise RuntimeError(f"Missing family assignments for {missing}")

    table = table.assign(
        sample=table["sample_id_on_disk"],
        alignment_category_display=table["alignment_category_draft"].map(remove_parenthetical_text),
        pearson_r=table["pearson_r"].astype(float),
        spearman_rho=table["spearman_r"].astype(float),
        abs_spearman_rho=table["abs_spearman_r"].astype(float),
    )
    table = table.sort_values(
        ["family_id", "sample", "program_short"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    out_cols = [
        "sample",
        "program_short",
        "alignment_category_display",
        "family_id",
        "family_label",
        "pearson_r",
        "spearman_rho",
        "abs_spearman_rho",
        "program_id",
    ]
    csv_path = OUT_DIR / "univariate_program_spearman_abs_ge_0p3.csv"
    table[out_cols].to_csv(csv_path, index=False)

    rows = [
        r"\begin{table}[p]",
        r"    \centering",
        r"    \scriptsize",
        r"    \begin{tabular}{llp{0.34\textwidth}p{0.20\textwidth}rr}",
        r"        \toprule",
        r"        Sample & Program & Alignment category & Family & Pearson \(r\) & Spearman \(\rho_s\) \\",
        r"        \midrule",
    ]

    previous_family = None
    for _, row in table.iterrows():
        family_label = str(row["family_label"])
        if previous_family is not None and family_label != previous_family:
            rows.append(r"        \addlinespace")
        previous_family = family_label
        rows.append(
            "        "
            + " & ".join(
                [
                    latex_escape(row["sample"]),
                    latex_escape(row["program_short"]),
                    latex_escape(row["alignment_category_display"]),
                    latex_escape(row["family_label"]),
                    f"{float(row['pearson_r']):.3f}",
                    f"{float(row['spearman_rho']):.3f}",
                ]
            )
            + r" \\"
        )

    rows.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"    \caption{Local cNMF programs with \(|\rho_s| \ge 0.3\) against the unsmoothed, uncorrected SNAI1-ac score. Correlations were calculated within each sample using raw program usage. Programs are grouped by the report-facing family assignment.}",
            r"    \label{tab:cnmf_univariate_high_corr_programs}",
            r"\end{table}",
            "",
        ]
    )

    tex_path = OUT_DIR / "univariate_program_spearman_abs_ge_0p3_latex.tex"
    tex_path.write_text("\n".join(rows), encoding="utf-8")

    print(csv_path)
    print(tex_path)


if __name__ == "__main__":
    main()
