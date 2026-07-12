"""
Generate figures for SNAI1-ac spatial signature analysis.

Standalone plotting script that reads analyzed h5ad files and generates
publication-quality figures.

Usage:
    python plot_signature_figures.py visium/denisenko_2022 --sample SP6
    python plot_signature_figures.py visium/denisenko_2022 --all
    python plot_signature_figures.py --list

Requires:
    - Analyzed h5ad from spatial_signature_analysis.py
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import scanpy as sc

from analysis_utils import ensure_dir, save_figure


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(r"D:\HGSOC_Spatial_Atlas")
ANALYSIS_DIR = BASE_DIR / "05_analysis_ready"

# Cell types for stacked bar (order matters for visual)
CELLTYPE_COLS = [
    'Malignant', 'CAF', 'Endothelial', 'Plasma', 'B cell', 'T CD4', 'T CD8',
    'NK', 'cDC', 'pDC', 'Macrophage', 'Mast', 'Neutrophil', 'Unidentifiable'
]

HISTOLOGY_BACKGROUND_ALPHA = 0.50

KNOWN_DATASETS = [
    'visium/denisenko_2022',
    'visium/yamamoto_2025',
    'visium/ju_2024',
    'visium/stur_2021',
    'visium/10X_ov_standard',
   # 'visium/10X_ov_11mm', doesn't have SpaGCN
]

THESIS_COHORT = {
    'visium/denisenko_2022': ['SP1', 'SP2', 'SP3', 'SP4', 'SP5', 'SP6', 'SP7', 'SP8'],
    'visium/yamamoto_2025': [
        'Pt1-1', 'Pt1-2', 'Pt1-3', 'Pt1-4',
        'Pt2-1', 'Pt2-2', 'Pt2-3', 'Pt2-4',
    ],
    'visium/ju_2024': [
        'CPS_OV1RtOV3', 'CPS_OV5LtOV4', 'CPS_OV19_LtOV1',
        'CPS_OV20RtOV4', 'CPS_OV24RTOV4', 'CPS_OV34RtOV1', 'CPS_OV71_1',
    ],
}


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_density_by_region(adata, sample_name, figures_dir):
    """
    KDE density plot of SNAI1-ac score by interface region.
    Shows Tumor, Interface, Stroma as overlapping density curves.
    """
    score = adata.obs['SNAI1-ac_score']
    interface = adata.obs['interface']
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    colors = {'Tumor': 'red', 'Interface': 'orange', 'Stroma': 'blue'}
    
    for region in ['Tumor', 'Interface', 'Stroma']:
        mask = interface == region
        if mask.sum() == 0:
            continue
        
        region_scores = score[mask]
        mean_val = region_scores.mean()
        n = mask.sum()
        
        sns.kdeplot(region_scores, ax=ax, color=colors[region], linewidth=2,
                    label=f'{region} (mean={mean_val:.2f}, n={n})')
    
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('SNAI1-ac Score')
    ax.set_ylabel('Density')
    ax.set_title(f'{sample_name}: SNAI1-ac Score Density by Region')
    ax.legend()
    
    save_figure(fig, figures_dir / 'density_by_region.png')
    print(f"   Saved: density_by_region.png")


def plot_spatial_sanity_check(adata, sample_name, n_clusters, figures_dir):
    """
    Side-by-side spatial plot: SpaGCN domains vs Malignant fraction.
    Visual sanity check that domains correspond to biological regions.
    """
    domain_key = f'spagcn_{n_clusters}_refined'
    
    if domain_key not in adata.obs.columns:
        print(f"   Skipping sanity check: {domain_key} not found")
        return
    
    if 'Malignant' not in adata.obs.columns:
        print(f"   Skipping sanity check: Malignant not found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: SpaGCN domains
    sc.pl.spatial(adata, color=domain_key, ax=axes[0], show=False, 
                  title=f'SpaGCN Domains ({n_clusters} clusters)', size=1.3)
    
    # Right: Malignant fraction
    sc.pl.spatial(adata, color='Malignant', ax=axes[1], show=False,
                  title='Malignant Fraction', size=1.3, cmap='Reds', vmin=0, vmax=1)
    
    plt.tight_layout()
    save_figure(fig, figures_dir / f'spatial_sanity_spagcn{n_clusters}.png')
    print(f"   Saved: spatial_sanity_spagcn{n_clusters}.png")


def get_spatial_image_and_coords(adata):
    """Return image, scaled coordinates, and image label for spatial overlays."""
    if 'spatial' not in adata.obsm:
        return None, None, None

    if 'spatial' not in adata.uns or not adata.uns['spatial']:
        return None, adata.obsm['spatial'], 'no_image'

    library_id = list(adata.uns['spatial'].keys())[0]
    spatial = adata.uns['spatial'][library_id]
    images = spatial.get('images', {})
    scalefactors = spatial.get('scalefactors', {})

    if 'hires' in images:
        image_key = 'hires'
        scale_key = 'tissue_hires_scalef'
    elif 'lowres' in images:
        image_key = 'lowres'
        scale_key = 'tissue_lowres_scalef'
    else:
        return None, adata.obsm['spatial'], 'no_image'

    scale = scalefactors.get(scale_key, 1.0)
    return images[image_key], adata.obsm['spatial'] * scale, image_key


def fade_histology_background(img, alpha=HISTOLOGY_BACKGROUND_ALPHA):
    """Fade H&E toward white so hotspot/domain colors remain dominant."""
    img = np.asarray(img)
    if img.size == 0:
        return img

    image_alpha = None
    rgb = img.astype(float)
    if rgb.max() > 1:
        rgb = rgb / 255.0
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        image_alpha = rgb[:, :, 3:4]
        rgb = rgb[:, :, :3]

    rgb = np.clip(alpha * rgb + (1 - alpha), 0, 1)

    if image_alpha is not None:
        return np.concatenate([rgb, image_alpha], axis=2)
    return rgb


def plot_lisa_hotspots_by_domain(adata, sample_name, figures_dir, n_clusters=9):
    """
    H&E background with only LISA High-High and Low-Low spots,
    colored by SpaGCN domain (same colors as original domain plot).
    """
    domain_key = f'spagcn_{n_clusters}_refined'
    color_key = f'{domain_key}_colors'
    output_name = f'lisa_hotspots_by_domain_spagcn{n_clusters}.png'

    if 'LISA_category' not in adata.obs.columns:
        print(f"   Skipping LISA overlay: LISA_category not found")
        return
    
    if domain_key not in adata.obs.columns:
        print(f"   Skipping LISA overlay: {domain_key} not found")
        return
    
    if 'spatial' not in adata.obsm:
        print(f"   Skipping LISA overlay: no spatial coordinates")
        return
    
    # Identify hotspots
    lisa = adata.obs['LISA_category']
    hotspot_mask = (lisa == 'High-High') | (lisa == 'Low-Low')
    
    n_hh = (lisa == 'High-High').sum()
    n_ll = (lisa == 'Low-Low').sum()
    
    if n_hh + n_ll == 0:
        print(f"   Skipping LISA overlay: no hotspots found")
        return
    
    # Get domain info
    domains = adata.obs[domain_key].astype('category')
    n_domains = len(domains.cat.categories)
    
    # Get the existing color palette (same one used for original plot)
    if color_key in adata.uns:
        palette = adata.uns[color_key]
    else:
        palette = sc.pl.palettes.default_20[:n_domains]
    
    # Build RGBA color array - same colors, but alpha=0 for non-hotspots
    colors = np.zeros((len(adata), 4))
    for i, cat in enumerate(domains.cat.categories):
        mask = domains == cat
        rgb = mcolors.to_rgba(palette[i])
        colors[mask] = rgb
    
    # Make non-hotspots transparent
    colors[~hotspot_mask, 3] = 0
    
    # Get spatial info
    img, coords, image_key = get_spatial_image_and_coords(adata)
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    if img is not None:
        ax.imshow(fade_histology_background(img))
    else:
        ax.set_facecolor('white')
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=3)
    
    # Legend - only domains present in hotspots
    hotspot_domains = domains[hotspot_mask].unique()
    for i, cat in enumerate(domains.cat.categories):
        if cat in hotspot_domains:
            ax.scatter([], [], c=[palette[i]], label=cat, s=30)
    
    ax.legend(title='Domain', loc='upper right', fontsize=8)
    if img is None:
        ax.set_aspect('equal')
        ax.invert_yaxis()
    ax.axis('off')
    plt.tight_layout()
    
    output_path = figures_dir / output_name
    save_figure(fig, output_path)
    print(f"   Saved: {output_name}")

    if n_clusters == 9:
        legacy_path = figures_dir / 'lisa_hotspots_by_domain.png'
        shutil.copyfile(output_path, legacy_path)
        print(f"   Saved: lisa_hotspots_by_domain.png")


def plot_spatial_lisa(adata, sample_name, figures_dir):
    """Spatial scatter of current LISA labels with legend counts."""
    if 'LISA_category' not in adata.obs.columns:
        print(f"   Skipping spatial LISA: LISA_category not found")
        return

    if 'spatial' not in adata.obsm:
        print(f"   Skipping spatial LISA: no spatial coordinates")
        return

    lisa = adata.obs['LISA_category'].astype(str)
    coords = adata.obsm['spatial']
    colors = {
        'Not significant': 'lightgray',
        'High-Low': 'orange',
        'Low-High': 'lightblue',
        'Low-Low': 'blue',
        'High-High': 'red',
    }

    fig, ax = plt.subplots(figsize=(7, 7))
    for cat in ['Not significant', 'High-Low', 'Low-High', 'Low-Low', 'High-High']:
        mask = lisa == cat
        n = int(mask.sum())
        if n:
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=colors[cat],
                label=f"{cat} ({n})",
                s=8,
                alpha=0.7,
            )

    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.legend(loc='best', fontsize=8)
    ax.axis('off')
    plt.tight_layout()
    save_figure(fig, figures_dir / 'spatial_lisa.png')
    print(f"   Saved: spatial_lisa.png")


def jaccard(set_a, set_b):
    """Jaccard index for two index sets."""
    if len(set_a) == 0 and len(set_b) == 0:
        return np.nan
    return len(set_a & set_b) / len(set_a | set_b)


def refresh_hotspot_comparison(adata, figures_dir):
    """Rewrite hotspot comparison CSV from current LISA and threshold labels."""
    if 'LISA_category' not in adata.obs.columns or 'hotspot_strict' not in adata.obs.columns:
        print(f"   Skipping hotspot comparison refresh: required columns not found")
        return

    lisa = adata.obs['LISA_category'].astype(str).to_numpy()
    hotspot = adata.obs['hotspot_strict'].astype(str).to_numpy()

    lisa_hh = set(np.where(lisa == 'High-High')[0])
    lisa_ll = set(np.where(lisa == 'Low-Low')[0])
    thresh_hot = set(np.where(hotspot == 'Hotspot')[0])
    thresh_cold = set(np.where(hotspot == 'Coldspot')[0])

    comparison = {
        'Hotspot_jaccard': jaccard(lisa_hh, thresh_hot),
        'Coldspot_jaccard': jaccard(lisa_ll, thresh_cold),
        'LISA_HH_count': len(lisa_hh),
        'LISA_LL_count': len(lisa_ll),
        'Threshold_hot_count': len(thresh_hot),
        'Threshold_cold_count': len(thresh_cold),
    }

    csv_dir = figures_dir.parent / 'csvs'
    ensure_dir(csv_dir)
    pd.DataFrame([comparison]).to_csv(csv_dir / 'hotspot_comparison.csv', index=False)
    print(f"   Saved: hotspot_comparison.csv")


def plot_spatial_hotspots_threshold(adata, sample_name, figures_dir):
    """Spatial scatter of current threshold hotspot labels with legend counts."""
    if 'hotspot_strict' not in adata.obs.columns:
        print(f"   Skipping threshold hotspots: hotspot_strict not found")
        return

    if 'spatial' not in adata.obsm:
        print(f"   Skipping threshold hotspots: no spatial coordinates")
        return

    hotspot = adata.obs['hotspot_strict'].astype(str)
    coords = adata.obsm['spatial']
    colors = {'None': 'lightgray', 'Coldspot': 'blue', 'Hotspot': 'red'}

    fig, ax = plt.subplots(figsize=(7, 7))
    for cat in ['None', 'Coldspot', 'Hotspot']:
        mask = hotspot == cat
        n = int(mask.sum())
        if n:
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=colors[cat],
                label=f"{cat} ({n})",
                s=8,
                alpha=0.7,
            )

    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.legend(loc='best', fontsize=8)
    ax.axis('off')
    plt.tight_layout()
    save_figure(fig, figures_dir / 'spatial_hotspots_threshold.png')
    print(f"   Saved: spatial_hotspots_threshold.png")


def plot_stacked_bar_composition(adata, sample_name, n_clusters, figures_dir):
    """
    Stacked bar plot of cell type composition by SpaGCN domain.
    """
    domain_key = f'spagcn_{n_clusters}_refined'
    
    if domain_key not in adata.obs.columns:
        print(f"   Skipping stacked bar: {domain_key} not found")
        return
    
    # Get available cell types
    celltype_cols = [c for c in CELLTYPE_COLS if c in adata.obs.columns]
    
    # Calculate mean composition per domain
    domains = adata.obs[domain_key].astype(str)
    unique_domains = sorted(domains.unique())
    
    composition = {}
    for domain in unique_domains:
        mask = domains == domain
        composition[domain] = adata.obs.loc[mask, celltype_cols].mean()
    
    comp_df = pd.DataFrame(composition)
    
    # Plot stacked bar
    fig, ax = plt.subplots(figsize=(10, 7))
    
    comp_df.T.plot(kind='bar', stacked=True, ax=ax, width=0.8)
    
    ax.set_xlabel('SpaGCN Domain')
    ax.set_ylabel('Fraction')
    ax.set_title(f'{sample_name}: Domain Composition')
    ax.legend(title='Cell Type', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_ylim(0, 1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    plt.tight_layout()
    save_figure(fig, figures_dir / f'stacked_bar_spagcn{n_clusters}.png')
    print(f"   Saved: stacked_bar_spagcn{n_clusters}.png")


# =============================================================================
# MAIN
# =============================================================================

def generate_plots(sample_path, sample_name, figures_dir, lisa_only=False):
    """Generate all plots for a single sample."""
    
    print(f"\n--- Generating plots for {sample_name} ---")
    
    # Load data
    adata = sc.read_h5ad(sample_path)
    print(f"   Loaded: {adata.n_obs} spots")
    
    ensure_dir(figures_dir)

    if lisa_only:
        refresh_hotspot_comparison(adata, figures_dir)
        plot_spatial_lisa(adata, sample_name, figures_dir)
        plot_spatial_hotspots_threshold(adata, sample_name, figures_dir)
        for n_clusters in [5, 9]:
            plot_lisa_hotspots_by_domain(adata, sample_name, figures_dir, n_clusters)
        print(f"   Done")
        return
    
    # Plot 1: Density by region
    if 'interface' in adata.obs.columns:
        plot_density_by_region(adata, sample_name, figures_dir)
    
    # Plot 2 & 3: Spatial sanity check and stacked bar (for 5 and 9 clusters)
    for n_clusters in [5, 9]:
        plot_spatial_sanity_check(adata, sample_name, n_clusters, figures_dir)
        plot_stacked_bar_composition(adata, sample_name, n_clusters, figures_dir)
    
    # Plot 4: LISA hotspots colored by SpaGCN domain
    for n_clusters in [5, 9]:
        plot_lisa_hotspots_by_domain(adata, sample_name, figures_dir, n_clusters)
    
    print(f"   Done")


def find_analyzed_samples(dataset_path):
    """Find samples with completed analysis."""
    samples = []
    
    if not dataset_path.exists():
        return samples
    
    for sample_dir in dataset_path.iterdir():
        if not sample_dir.is_dir():
            continue
        
        h5ad_file = sample_dir / f"{sample_dir.name}.h5ad"
        if h5ad_file.exists():
            samples.append({
                'path': h5ad_file,
                'name': sample_dir.name,
                'figures_dir': sample_dir / 'signature_analysis' / 'figures'
            })
    
    return sorted(samples, key=lambda x: x['name'])


def process_dataset(dataset_id, sample_name=None, thesis_cohort=False, lisa_only=False):
    """Process all samples in a dataset."""
    dataset_path = ANALYSIS_DIR / dataset_id
    
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return
    
    samples = find_analyzed_samples(dataset_path)
    if thesis_cohort:
        keep = set(THESIS_COHORT.get(dataset_id, []))
        samples = [s for s in samples if s['name'] in keep]
    
    if not samples:
        print(f"No analyzed samples found in {dataset_path}")
        return
    
    if sample_name:
        samples = [s for s in samples if s['name'] == sample_name]
        if not samples:
            print(f"Sample not found: {sample_name}")
            return
    
    print(f"Found {len(samples)} sample(s)")
    
    for sample in samples:
        generate_plots(sample['path'], sample['name'], sample['figures_dir'], lisa_only=lisa_only)


def list_status(thesis_cohort=False):
    """List available samples."""
    print(f"\n{'Dataset':<30} {'Samples':<10}")
    print("-" * 40)
    
    dataset_ids = THESIS_COHORT.keys() if thesis_cohort else KNOWN_DATASETS
    for dataset_id in dataset_ids:
        dataset_path = ANALYSIS_DIR / dataset_id
        samples = find_analyzed_samples(dataset_path)
        if thesis_cohort:
            keep = set(THESIS_COHORT.get(dataset_id, []))
            samples = [s for s in samples if s['name'] in keep]
        n = len(samples)
        status = str(n) if n > 0 else "-"
        print(f"{dataset_id:<30} {status:<10}")


def main():
    parser = argparse.ArgumentParser(description='Generate signature analysis figures')
    parser.add_argument('dataset', nargs='?', help='Dataset ID')
    parser.add_argument('--sample', '-s', help='Specific sample')
    parser.add_argument('--all', '-a', action='store_true', help='Process all datasets')
    parser.add_argument('--list', '-l', action='store_true', help='List available samples')
    parser.add_argument('--thesis-cohort', action='store_true', help='Restrict to the 23 thesis samples')
    parser.add_argument('--lisa-only', action='store_true', help='Only regenerate LISA hotspot/domain overlays')
    
    args = parser.parse_args()
    
    if args.list:
        list_status(thesis_cohort=args.thesis_cohort)
        return
    
    if args.all:
        dataset_ids = THESIS_COHORT.keys() if args.thesis_cohort else KNOWN_DATASETS
        for dataset_id in dataset_ids:
            process_dataset(
                dataset_id,
                thesis_cohort=args.thesis_cohort,
                lisa_only=args.lisa_only,
            )
        return
    
    if args.dataset:
        process_dataset(
            args.dataset,
            args.sample,
            thesis_cohort=args.thesis_cohort,
            lisa_only=args.lisa_only,
        )
        return
    
    print("Usage:")
    print("  python plot_signature_figures.py visium/denisenko_2022 --sample SP6")
    print("  python plot_signature_figures.py visium/denisenko_2022 --all")
    print("  python plot_signature_figures.py --list")


if __name__ == "__main__":
    main()
