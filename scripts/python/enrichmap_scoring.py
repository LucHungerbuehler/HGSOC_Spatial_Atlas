# enrichmap_scoring.py
"""
EnrichMap scoring for all three SNAI1 signatures.

Scores each Visium sample with three signatures:
  1. SNAI1-ac_score         — acetylation-specific (SNAI1-2R vs SNAI1)
  2. SNAI1_vs_GFP_score     — full SNAI1 program (SNAI1 vs GFP)
  3. SNAI1_2R_vs_GFP_score  — non-acetylatable SNAI1 program (SNAI1-2R vs GFP)

Usage:
    python enrichmap_scoring.py visium/yamamoto_2025
    python enrichmap_scoring.py visium/yamamoto_2025 --sample Pt1-1
    python enrichmap_scoring.py --all
    python enrichmap_scoring.py --list

Input:  02_processed_data/visium/<dataset>/<sample>.h5ad
        05_analysis_ready/Signature/snai1_ac_weights.json
        05_analysis_ready/Signature/snai1_vs_gfp_weights.json
        05_analysis_ready/Signature/snai12r_vs_gfp_weights.json
Output: Overwrites h5ad with all three scores added to adata.obs
        05_analysis_ready/visium/<dataset>/<sample>/EnrichMap/*.png
"""

import sys
import json
from pathlib import Path

try:
    import scanpy as sc
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import enrichmap as em
    PACKAGES_AVAILABLE = True
except ImportError as e:
    print(f"Missing package: {e}")
    PACKAGES_AVAILABLE = False

# --- Configuration ---
BASE_DIR           = Path(r"D:\HGSOC_Spatial_Atlas")
PROCESSED_DATA_DIR = BASE_DIR / "02_processed_data"
OUTPUT_DIR         = BASE_DIR / "05_analysis_ready"
SIGNATURE_DIR      = OUTPUT_DIR / "Signature"

# Three signatures to score — order, names, and files
SIGNATURES = [
    {
        'score_key': 'SNAI1-ac',
        'score_col': 'SNAI1-ac_score',
        'weights_file': SIGNATURE_DIR / 'snai1_ac_weights.json',
    },
    {
        'score_key': 'SNAI1-vs-GFP',
        'score_col': 'SNAI1_vs_GFP_score',
        'weights_file': SIGNATURE_DIR / 'snai1_vs_gfp_weights.json',
    },
    {
        'score_key': 'SNAI1-2R-vs-GFP',
        'score_col': 'SNAI1_2R_vs_GFP_score',
        'weights_file': SIGNATURE_DIR / 'snai12r_vs_gfp_weights.json',
    },
]

UNS_SIGNATURE_METADATA = {
    'SNAI1-ac': {
        'genes_key': 'SNAI1_ac_signature_genes',
        'weights_key': 'SNAI1_ac_signature_weights',
        'detected_key': 'SNAI1_ac_signature_genes_detected',
    },
    'SNAI1-vs-GFP': {
        'genes_key': 'SNAI1_vs_GFP_signature_genes',
        'weights_key': 'SNAI1_vs_GFP_signature_weights',
        'detected_key': 'SNAI1_vs_GFP_signature_genes_detected',
    },
    'SNAI1-2R-vs-GFP': {
        'genes_key': 'SNAI1_2R_vs_GFP_signature_genes',
        'weights_key': 'SNAI1_2R_vs_GFP_signature_weights',
        'detected_key': 'SNAI1_2R_vs_GFP_signature_genes_detected',
    },
}

KNOWN_DATASETS = [
    'visium/yamamoto_2025',
    'visium/ju_2024',
    'visium/denisenko_2022',
    'visium/10X_ov_standard',
    'visium/10X_ov_11mm',
]


def load_all_weights():
    """Load all three weight dictionaries. Returns None if any file is missing."""
    weights = {}
    for sig in SIGNATURES:
        wf = sig['weights_file']
        if not wf.exists():
            print(f"ERROR: Weights file not found: {wf}")
            print("Run Signature_Weights_prep.py first.")
            return None
        with open(wf, 'r') as f:
            weights[sig['score_key']] = json.load(f)
        print(f"  Loaded {len(weights[sig['score_key']])} weights: {wf.name}")
    return weights


def find_h5ad_samples(dataset_path):
    """Find all .h5ad files in a dataset directory."""
    return sorted(
        [{'path': p, 'name': p.stem} for p in dataset_path.glob("*.h5ad")],
        key=lambda x: x['name']
    )


def has_images(adata):
    """Check if sample has histology images."""
    if 'spatial' not in adata.uns:
        return False
    library_id = list(adata.uns['spatial'].keys())[0]
    return len(adata.uns['spatial'][library_id].get('images', [])) > 0


def store_signature_metadata(adata, all_weights):
    """Persist signature gene sets and weights for downstream provenance checks."""
    for sig in SIGNATURES:
        key = sig['score_key']
        meta = UNS_SIGNATURE_METADATA[key]
        weights = all_weights.get(key, {})

        adata.uns[meta['genes_key']] = sorted(str(g) for g in weights.keys())
        adata.uns[meta['weights_key']] = {
            str(g): float(w) for g, w in weights.items()
        }


def score_sample(sample_info, all_weights, dataset_id):
    """Run EnrichMap scoring for all three signatures on a single sample."""
    sample_path = sample_info['path']
    sample_name = sample_info['name']

    print(f"\n{'-' * 60}")
    print(f"Processing: {sample_name}")
    print(f"{'-' * 60}")

    adata = sc.read_h5ad(sample_path)
    print(f"  Spots: {adata.n_obs}, Genes: {adata.n_vars}")
    store_signature_metadata(adata, all_weights)

    # Score each signature
    for sig in SIGNATURES:
        key      = sig['score_key']
        col      = sig['score_col']
        weights  = all_weights[key]
        meta     = UNS_SIGNATURE_METADATA[key]

        # Filter weights to genes present in this sample
        weights_filtered = {g: w for g, w in weights.items() if g in adata.var_names}
        n_found   = len(weights_filtered)
        n_total   = len(weights)
        print(f"\n  [{key}]")
        print(f"  Genes found: {n_found}/{n_total}")
        adata.uns[meta['detected_key']] = sorted(str(g) for g in weights_filtered.keys())
        adata.uns[f"{meta['detected_key']}_count"] = int(n_found)
        adata.uns[f"{meta['genes_key']}_count"] = int(n_total)

        if n_found == 0:
            print(f"  WARNING: No signature genes found — skipping this signature")
            continue

        em.tl.score(
            adata=adata,
            gene_set=list(weights_filtered.keys()),
            gene_weights={key: weights_filtered},
            score_key=key,
            smoothing=True,
            correct_spatial_covariates=True
        )

        if col not in adata.obs.columns:
            print(f"  WARNING: Score column '{col}' not created")
            continue

        scores = adata.obs[col]
        print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}], "
              f"mean: {scores.mean():.3f}")

    # Save h5ad (overwrites existing)
    print(f"\n  Saving h5ad...")
    adata.write_h5ad(sample_path)

    # Create plots
    create_plots(adata, dataset_id, sample_name)

    print(f"  Done: {sample_name}")
    return True


def create_plots(adata, dataset_id, sample_name):
    """Create spatial plots for all three scores."""
    plot_dir = OUTPUT_DIR / dataset_id / sample_name / "EnrichMap"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for sig in SIGNATURES:
        col = sig['score_col']
        key = sig['score_key']

        if col not in adata.obs.columns:
            continue

        # Spatial plot
        if has_images(adata):
            try:
                fig, ax = plt.subplots(figsize=(8, 8))
                em.pl.spatial_enrichmap(
                    adata,
                    score_key=col,
                    size=1,
                    img_alpha=1,
                    alpha=0.7,
                    ax=ax
                )
                fname = f"spatial_enrichmap_{key.replace('/', '_')}.png"
                plt.savefig(plot_dir / fname, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"    Saved: {fname}")
            except Exception as e:
                print(f"    Warning: spatial_enrichmap failed for {key}: {e}")

    print(f"    Plot directory: {plot_dir}")


def score_dataset(dataset_id, sample_name=None):
    """Score all samples in a dataset."""
    print(f"\nDataset: {dataset_id}")

    dataset_path = PROCESSED_DATA_DIR / dataset_id
    if not dataset_path.exists():
        print(f"  ERROR: Dataset not found: {dataset_path}")
        return []

    all_weights = load_all_weights()
    if all_weights is None:
        return []

    samples = find_h5ad_samples(dataset_path)
    if not samples:
        print(f"  No h5ad files found")
        return []

    if sample_name:
        samples = [s for s in samples if s['name'] == sample_name]
        if not samples:
            print(f"  ERROR: Sample not found: {sample_name}")
            return []

    print(f"Found {len(samples)} sample(s)")

    results = []
    for sample in samples:
        success = score_sample(sample, all_weights, dataset_id)
        results.append({'sample': sample['name'], 'success': success})

    successful = sum(1 for r in results if r['success'])
    print(f"\nCompleted: {successful}/{len(results)} samples")
    return results


def score_all():
    """Score all known datasets."""
    all_results = {}
    for dataset_id in KNOWN_DATASETS:
        dataset_path = PROCESSED_DATA_DIR / dataset_id
        if dataset_path.exists():
            all_results[dataset_id] = score_dataset(dataset_id)
        else:
            print(f"\nSkipping {dataset_id} (not found)")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for dataset_id, results in all_results.items():
        if results:
            successful = sum(1 for r in results if r['success'])
            print(f"  {dataset_id}: {successful}/{len(results)} samples")
        else:
            print(f"  {dataset_id}: skipped")
    return all_results


def list_status():
    """List scoring status for all datasets."""
    score_cols = [s['score_col'] for s in SIGNATURES]
    header = f"{'Dataset':<30} {'h5ad':<8} " + \
             " ".join(f"{c:<25}" for c in score_cols)
    print(f"\n{header}")
    print("-" * (30 + 8 + 25 * len(score_cols)))

    for dataset_id in KNOWN_DATASETS:
        dataset_path = PROCESSED_DATA_DIR / dataset_id
        if not dataset_path.exists():
            print(f"{dataset_id:<30} {'–':<8}")
            continue

        h5ad_files  = list(dataset_path.glob("*.h5ad"))
        scored_counts = {col: 0 for col in score_cols}

        for h5ad_file in h5ad_files:
            try:
                adata = sc.read_h5ad(h5ad_file, backed='r')
                for col in score_cols:
                    if col in adata.obs.columns:
                        scored_counts[col] += 1
                adata.file.close()
            except Exception:
                pass

        n = len(h5ad_files)
        row = f"{dataset_id:<30} {str(n):<8} "
        row += " ".join(f"{scored_counts[c]}/{n:<22}" for c in score_cols)
        print(row)


def main():
    if not PACKAGES_AVAILABLE:
        sys.exit(1)

    print("\n" + "=" * 70)
    print("EnrichMap Scoring — Three SNAI1 Signatures")
    print("=" * 70)

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command in ('--list', '-l'):
        list_status()

    elif command in ('--all', '-a'):
        score_all()

    elif command in ('--help', '-h'):
        print(__doc__)

    else:
        dataset_id  = command
        sample_name = None
        if '--sample' in sys.argv or '-s' in sys.argv:
            try:
                idx = sys.argv.index('--sample') if '--sample' in sys.argv \
                      else sys.argv.index('-s')
                sample_name = sys.argv[idx + 1]
            except (IndexError, ValueError):
                print("ERROR: --sample requires a sample name")
                sys.exit(1)

        results = score_dataset(dataset_id, sample_name)
        if not results or not all(r['success'] for r in results):
            print("\nSome samples failed — check output above")
            sys.exit(1)


if __name__ == "__main__":
    main()
