# Signature_Weights_prep.py
"""
Prepare signature weights for EnrichMap scoring.
Produces three weight JSON files from the Excel DEG table:

  1. snai1_ac_weights.json       — SNAI1-2R vs SNAI1  (acetylation-specific)
  2. snai1_vs_gfp_weights.json   — SNAI1 vs GFP       (full SNAI1 program)
  3. snai12r_vs_gfp_weights.json — SNAI1-2R vs GFP    (non-acetylatable SNAI1 program)

Sign convention:
  - All three scores are oriented so that a HIGH score = HIGH activity
    of the respective biological program.
  - SNAI1-2R vs SNAI1: sign IS flipped   (negative FC = activated by acetylation)
  - SNAI1 vs GFP:      sign NOT flipped  (positive FC = activated by SNAI1)
  - SNAI1-2R vs GFP:   sign NOT flipped  (positive FC = activated by SNAI1-2R)

Filtering thresholds (same as signature derivation):
  - BH-corrected padj < 0.05
  - |log2FC| > 1

Usage:
    python Signature_Weights_prep.py

Input:  tt_PEO4-SNAI1-2R_Analysis.xlsx  (Cmpr sheet)
Output: 05_analysis_ready/Signature/snai1_ac_weights.json
        05_analysis_ready/Signature/snai1_vs_gfp_weights.json
        05_analysis_ready/Signature/snai12r_vs_gfp_weights.json
"""

import sys
import json
from pathlib import Path
from datetime import datetime

try:
    import pandas as pd
    import numpy as np
    PACKAGES_AVAILABLE = True
except ImportError as e:
    print(f"Missing package: {e}")
    PACKAGES_AVAILABLE = False

# --- Configuration ---
BASE_DIR       = Path(r"D:\HGSOC_Spatial_Atlas")
EXCEL_FILE     = BASE_DIR / "05_analysis_ready" / "Signature" / "tt_PEO4-SNAI1-2R_Analysis.xlsx"
OUTPUT_DIR     = BASE_DIR / "05_analysis_ready" / "Signature"

GENE_COL       = 'Gene'
CAP_THRESHOLD  = 3       # cap log2FC at ±3
FDR_THRESHOLD  = 0.05
LOG2FC_THRESHOLD = 1.0

# Three comparisons to process
# Fields: label, ppee_col, fc_col, flip_sign, output_filename
COMPARISONS = [
    {
        'label':       'SNAI1-2R vs SNAI1 (acetylation-specific)',
        'ppee_col':    'PEO4-2R_SNAI1vsSNAI1_PPEE',
        'fc_col':      'PEO4-2R_lg2fc (SNAI1-SNAI1)',
        'flip_sign':   True,   # negative FC = activated by acetylation -> flip to positive weight
        'score_key':   'SNAI1-ac',
        'score_col':   'SNAI1-ac_score',
        'output_file': 'snai1_ac_weights.json',
    },
    {
        'label':       'SNAI1 vs GFP (full SNAI1 program)',
        'ppee_col':    'PEO4_SNAI1vsGFP_PPEE',
        'fc_col':      'PEO4_lg2fc (SNAI1-GFP)',
        'flip_sign':   False,  # positive FC = activated by SNAI1 -> already correct direction
        'score_key':   'SNAI1-vs-GFP',
        'score_col':   'SNAI1_vs_GFP_score',
        'output_file': 'snai1_vs_gfp_weights.json',
    },
    {
        'label':       'SNAI1-2R vs GFP (non-acetylatable SNAI1 program)',
        'ppee_col':    'PEO4-2R_SNAI1vsGFP_PPEE',
        'fc_col':      'PEO4-2R_lg2fc (SNAI1-GFP)',
        'flip_sign':   False,  # positive FC = activated by SNAI1-2R -> already correct direction
        'score_key':   'SNAI1-2R-vs-GFP',
        'score_col':   'SNAI1_2R_vs_GFP_score',
        'output_file': 'snai12r_vs_gfp_weights.json',
    },
]


def bh_correct(df, ppee_col):
    """Apply Benjamini-Hochberg correction to a PPEE column across all genes."""
    df_sorted = df.sort_values(ppee_col).reset_index(drop=True)
    n = len(df_sorted)
    df_sorted['rank'] = range(1, n + 1)
    df_sorted['padj'] = (df_sorted[ppee_col] * n / df_sorted['rank']).clip(upper=1.0)
    # Enforce monotonicity
    df_sorted['padj'] = df_sorted['padj'][::-1].cummin()[::-1]
    return df_sorted.set_index(df.index if hasattr(df, 'index') else None)


def filter_degs(df, ppee_col, fc_col):
    """Filter to significant DEGs using BH correction."""
    df_corrected = bh_correct(df.copy().reset_index(drop=True), ppee_col)
    sig = df_corrected[
        (df_corrected['padj'] < FDR_THRESHOLD) &
        (df_corrected[fc_col].abs() > LOG2FC_THRESHOLD)
    ].copy()
    return sig


def create_weights(sig, fc_col, flip_sign):
    """
    Create weights dictionary from filtered DEGs.

    Weighting strategy (same as existing acetylation signature):
      1. Cap log2FC at ±CAP_THRESHOLD
      2. Divide by CAP_THRESHOLD to normalise to [-1, +1]
      3. Optionally flip sign so that high weight = high program activity
    """
    capped_fc = sig[fc_col].clip(lower=-CAP_THRESHOLD, upper=CAP_THRESHOLD)

    weights = {}
    for _, row in sig.iterrows():
        gene   = row[GENE_COL]
        weight = capped_fc[row.name] / CAP_THRESHOLD
        if flip_sign:
            weight = -weight
        weights[gene] = round(float(weight), 6)

    return weights


def save_weights(weights, output_path):
    """Save weights as JSON."""
    with open(output_path, 'w') as f:
        json.dump(weights, f, indent=2)
    print(f"    Saved: {output_path.name}")


def process_comparison(df, comp):
    """Process one comparison: filter DEGs, create weights, save JSON."""
    print(f"\n  {comp['label']}")
    print(f"  {'-' * 60}")

    # Filter DEGs
    sig = filter_degs(df, comp['ppee_col'], comp['fc_col'])
    n_up   = (sig[comp['fc_col']] > 0).sum()
    n_down = (sig[comp['fc_col']] < 0).sum()
    print(f"  DEGs passing threshold: {len(sig)} (up: {n_up}, down: {n_down})")

    # Create weights
    weights = create_weights(sig, comp['fc_col'], comp['flip_sign'])
    w_vals  = list(weights.values())
    n_pos   = sum(1 for w in w_vals if w > 0)
    n_neg   = sum(1 for w in w_vals if w < 0)
    print(f"  Weights: {len(weights)} genes "
          f"(positive: {n_pos}, negative: {n_neg})")
    print(f"  Weight range: [{min(w_vals):.3f}, {max(w_vals):.3f}]")
    print(f"  Sign flipped: {comp['flip_sign']}")
    print(f"  High score means: high {comp['score_key']} activity")

    # Save
    out_path = OUTPUT_DIR / comp['output_file']
    save_weights(weights, out_path)

    return weights


def main():
    if not PACKAGES_AVAILABLE:
        sys.exit(1)

    print("\n" + "=" * 70)
    print("Signature Weights Preparation — All Three Comparisons")
    print("=" * 70)
    print(f"Input:  {EXCEL_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Thresholds: padj < {FDR_THRESHOLD}, |log2FC| > {LOG2FC_THRESHOLD}")
    print("=" * 70)

    # Load Excel
    if not EXCEL_FILE.exists():
        print(f"\nERROR: Excel file not found: {EXCEL_FILE}")
        print("Please ensure tt_PEO4-SNAI1-2R_Analysis.xlsx is in the Signature directory.")
        sys.exit(1)

    print(f"\nLoading: {EXCEL_FILE.name}")
    df = pd.read_excel(EXCEL_FILE, sheet_name='Cmpr')
    print(f"Loaded {len(df)} genes")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process all three comparisons
    for comp in COMPARISONS:
        process_comparison(df, comp)

    print("\n" + "=" * 70)
    print("All three weight files prepared successfully.")
    print("Next step: run enrichmap_scoring.py --all")
    print("=" * 70)


if __name__ == "__main__":
    main()