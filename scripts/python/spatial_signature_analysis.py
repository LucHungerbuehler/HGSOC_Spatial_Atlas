"""
Spatial Signature Analysis for SNAI1-ac in HGSOC.

Analyzes the spatial distribution and relationships of SNAI1-ac signature scores
across tumor microenvironment regions.

Usage:
    python spatial_signature_analysis.py visium/denisenko_2022
    python spatial_signature_analysis.py visium/denisenko_2022 --sample SP6
    python spatial_signature_analysis.py --all
    python spatial_signature_analysis.py --list

Prerequisites:
    - Preprocessed h5ad with SNAI1-ac_score, SpaGCN domains
    - SpaCET results: {sample}_celltypes.csv, {sample}_interface.csv

Outputs:
    - Updated h5ad with merged SpaCET data and analysis results
    - CSV files with statistics
    - Figures for visualization
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures

import os
import sys
import argparse
import json
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from scipy.stats import spearmanr, mannwhitneyu, kruskal, chi2_contingency, fisher_exact
from scipy.stats import skew, kurtosis

# Import libpysal and esda for spatial statistics
try:
    import libpysal
    from esda.moran import Moran
    SPATIAL_STATS_AVAILABLE = True
except ImportError:
    SPATIAL_STATS_AVAILABLE = False
    print("Warning: libpysal/esda not available. Moran's I will be skipped.")

# Import shared utilities
from analysis_utils import (
    cohens_d, bimodality_coefficient, partial_spearman,
    get_neighbor_composition, get_neighbor_composition_excluding_group,
    get_ring_composition, classify_by_neighbor_threshold,
    jaccard, convert_keys_to_str, ensure_dir, save_figure
)


# =============================================================================
# CONFIGURATION - Modify these parameters as needed
# =============================================================================

# Directories
BASE_DIR = Path(r"D:\HGSOC_Spatial_Atlas")
PROCESSED_DIR = BASE_DIR / "02_processed_data"
ANALYSIS_DIR = BASE_DIR / "05_analysis_ready"
SIGNATURE_DIR = ANALYSIS_DIR / "Signature"

# Spatial analysis parameters
MORAN_K_NEIGHBORS = 6           # k for spatial weights matrix
SNAI1_GROUP_SD_THRESHOLD = 1.0  # +/- SD for HIGH/MID/LOW classification

# Neighborhood analysis
N_RINGS = 3                     # Number of rings for ring analysis
N_PERMUTATIONS = 1000           # Permutations for significance testing

# Hotspot detection
HOTSPOT_MIN_NEIGHBORS = 5       # Minimum high neighbors (out of 6) for strict hotspot
LISA_N_PERMUTATIONS = 999       # Permutations for LISA
FDR_THRESHOLD = 0.05            # FDR threshold for significance

# SpaCET cell type columns (standard output from SpaCET)
CELLTYPE_COLS = [
    'Malignant', 'CAF', 'Endothelial', 'Plasma', 'B cell', 'T CD4', 'T CD8',
    'NK', 'cDC', 'pDC', 'Macrophage', 'Mast', 'Neutrophil', 'Unidentifiable'
]

# Major cell types for focused analyses
MAJOR_CELLTYPES = [
    'Malignant', 'CAF', 'Macrophage', 'Endothelial', 'B cell', 
    'T CD4', 'T CD8', 'NK', 'Plasma','Unidentifiable'
]

# Known datasets
KNOWN_DATASETS = [
    'visium/denisenko_2022',
    'visium/yamamoto_2025',
    'visium/ju_2024',
    'visium/stur_2021',
    'visium/10X_ov_standard',
   # 'visium/10X_ov_11mm', doesn't have SpaGCN
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_header(dataset_id=None, sample=None):
    """Print script header."""
    print("\n" + "=" * 70)
    print("SNAI1-ac Spatial Signature Analysis")
    print("=" * 70)
    if dataset_id:
        print(f"Dataset: {dataset_id}")
    if sample:
        print(f"Sample: {sample}")
    print(f"Input: {PROCESSED_DIR}")
    print(f"Output: {ANALYSIS_DIR}")
    print("=" * 70)


def find_h5ad_samples(dataset_path):
    """Find all .h5ad files in a dataset directory."""
    samples = []
    for h5ad_file in dataset_path.glob("*.h5ad"):
        samples.append({
            'path': h5ad_file,
            'name': h5ad_file.stem
        })
    return sorted(samples, key=lambda x: x['name'])


def find_spacet_files(sample_name, dataset_id):
    """Find SpaCET result files for a sample."""
    analysis_sample_dir = ANALYSIS_DIR / dataset_id / sample_name
    
    celltypes_file = analysis_sample_dir / f"{sample_name}_celltypes.csv"
    interface_file = analysis_sample_dir / f"{sample_name}_interface.csv"
    
    return celltypes_file, interface_file


def load_signature_genes_from_weights():
    """Fallback loader for the primary SNAI1-ac signature gene set."""
    weights_path = SIGNATURE_DIR / 'snai1_ac_weights.json'
    if not weights_path.exists():
        return []

    try:
        with open(weights_path, 'r') as f:
            weights = json.load(f)
    except Exception as e:
        print(f"   Warning: could not load signature weights from {weights_path.name}: {e}")
        return []

    return sorted({str(g).strip() for g in weights.keys() if str(g).strip()})


# =============================================================================
# SECTION 0: SPACET MERGE
# =============================================================================

def merge_spacet_data(adata, celltypes_file, interface_file):
    """
    Merge SpaCET deconvolution results into adata.obs.
    
    Creates spot_key from array_row and array_col, then merges cell type
    fractions and interface labels.
    """
    print("\n--- Merging SpaCET data ---")
    
    # Create spot key
    adata.obs['spot_key'] = (
        adata.obs['array_row'].astype(str) + 'x' + 
        adata.obs['array_col'].astype(str)
    )
    
    # Load SpaCET files
    celltypes = pd.read_csv(celltypes_file, index_col=0)
    interface = pd.read_csv(interface_file)
    
    print(f"   h5ad spots: {len(adata)}")
    print(f"   SpaCET celltypes rows: {len(celltypes)}")
    print(f"   SpaCET interface rows: {len(interface)}")
    
    # Check overlap
    h5ad_keys = set(adata.obs['spot_key'])
    spacet_keys = set(celltypes.index)
    overlap = len(h5ad_keys & spacet_keys)
    missing = len(h5ad_keys - spacet_keys)
    
    print(f"   Overlap: {overlap} spots")
    if missing > 0:
        print(f"   Warning: {missing} h5ad spots not in SpaCET")
    
    # Set index for merging
    celltypes.index.name = 'spot_key'
    interface = interface.set_index('spot_id')
    
    # Merge cell type fractions
    celltypes_matched = celltypes.reindex(adata.obs['spot_key'])
    for col in celltypes_matched.columns:
        adata.obs[col] = celltypes_matched[col].values
    
    # Merge interface labels
    adata.obs['interface'] = interface.reindex(adata.obs['spot_key'])['interface'].values
    
    # Report
    n_celltypes = len(celltypes.columns)
    interface_counts = adata.obs['interface'].value_counts()
    print(f"   Added {n_celltypes} cell type columns")
    print(f"   Interface: {dict(interface_counts)}")
    
    return adata


# =============================================================================
# SECTION 1.1: DISTRIBUTION ANALYSIS
# =============================================================================

def analyze_distribution(adata, output_dir, figures_dir):
    """
    Analyze SNAI1-ac score distribution.
    
    Computes: basic stats, skewness, kurtosis, bimodality coefficient, Moran's I.
    Stores results in adata.uns['SNAI1_ac_stats'].
    """
    print("\n--- Phase 1.1: Distribution Analysis ---")
    
    score = adata.obs['SNAI1-ac_score'].values
    
    # Basic statistics
    stats = {
        'mean': float(np.mean(score)),
        'std': float(np.std(score)),
        'median': float(np.median(score)),
        'min': float(np.min(score)),
        'max': float(np.max(score)),
        'skewness': float(skew(score)),
        'kurtosis': float(kurtosis(score, fisher=True)),
    }
    
    # Bimodality coefficient
    bc, is_bimodal = bimodality_coefficient(score)
    stats['bimodality_coefficient'] = float(bc)
    stats['is_bimodal'] = bool(is_bimodal)
    
    print(f"   Mean: {stats['mean']:.3f}, SD: {stats['std']:.3f}")
    print(f"   Skewness: {stats['skewness']:.3f}, Kurtosis: {stats['kurtosis']:.3f}")
    print(f"   Bimodality coefficient: {bc:.3f} ({'bimodal' if is_bimodal else 'unimodal'})")
    
    # Moran's I (spatial autocorrelation)
    if SPATIAL_STATS_AVAILABLE and 'spatial' in adata.obsm:
        coords = adata.obsm['spatial']
        w = libpysal.weights.KNN.from_array(coords, k=MORAN_K_NEIGHBORS)
        w.transform = 'r'  # Row-standardize
        
        moran = Moran(score, w)
        stats['morans_I'] = {
            'I': float(moran.I),
            'EI': float(moran.EI),
            'p_value': float(moran.p_sim)
        }
        print(f"   Moran's I: {moran.I:.3f} (p={moran.p_sim:.4f})")

        #Moran's I for all Hallmark scores (reuse w)
        hallmark_cols = [c for c in adata.obs.columns
                         if c.startswith('HALLMARK_') and c.endswith('_score')]
        hallmark_morans = {}
        for col in hallmark_cols:
            m = Moran(adata.obs[col].values, w)
            hallmark_morans[col] = {'I': float(m.I),'EI': float(m.EI) ,'p_value': float(m.p_sim)}
        adata.uns['hallmark_morans_I'] = hallmark_morans
        print(f"   Computed Moran's I for {len(hallmark_cols)} Hallmark scores")

        #save CSV
        hallmark_morans_df = pd.DataFrame([{'pathway': col, 'morans_I': v['I'], 'morans_EI': v['EI'], 'p_value': v['p_value']} for col, v in hallmark_morans.items()])
        hallmark_morans_df.to_csv(output_dir / 'hallmark_morans_I.csv', index=False)
    else:
        stats['morans_I'] = {'I': np.nan, 'EI': np.nan, 'p_value': np.nan}
        print("   Moran's I: skipped (spatial stats not available)")
        adata.uns['hallmark_morans_I'] = {}
    
    # Stats by interface region
    stats['by_region'] = {}
    for region in ['Tumor', 'Interface', 'Stroma']:
        mask = adata.obs['interface'] == region
        if mask.sum() > 0:
            region_scores = score[mask]
            stats['by_region'][region] = {
                'mean': float(np.mean(region_scores)),
                'std': float(np.std(region_scores)),
                'n': int(mask.sum())
            }
    
    # Store in adata.uns
    adata.uns['SNAI1_ac_stats'] = stats
    
    # Save CSV
    stats_df = pd.DataFrame([{
        'mean': stats['mean'],
        'std': stats['std'],
        'median': stats['median'],
        'min': stats['min'],
        'max': stats['max'],
        'skewness': stats['skewness'],
        'kurtosis': stats['kurtosis'],
        'bimodality_coefficient': stats['bimodality_coefficient'],
        'is_bimodal': stats['is_bimodal'],
        'morans_I': stats['morans_I']['I'],
        'morans_I_pvalue': stats['morans_I']['p_value'],
    }])
    stats_df.to_csv(output_dir / 'distribution_stats.csv', index=False)
    
    # --- Figures ---
    
    # Histogram + KDE
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(score, bins=50, density=True, alpha=0.7, label='Histogram')
    sns.kdeplot(score, ax=ax, color='darkblue', linewidth=2, label='KDE')
    ax.axvline(stats['mean'], color='red', linestyle='-', linewidth=2, 
               label=f"Mean: {stats['mean']:.3f}")
    ax.axvline(stats['mean'] - stats['std'], color='red', linestyle='--', linewidth=1.5)
    ax.axvline(stats['mean'] + stats['std'], color='red', linestyle='--', linewidth=1.5,
               label=f"+/-1 SD: {stats['std']:.3f}")
    ax.set_xlabel('SNAI1-ac Score')
    ax.set_ylabel('Density')
    ax.set_title('SNAI1-ac Score Distribution')
    ax.legend()
    save_figure(fig, figures_dir / 'histogram_kde.png')
    
    # Spatial plot
    if 'spatial' in adata.obsm:
        fig, ax = plt.subplots(figsize=(7, 7))
        sc.pl.spatial(
            adata,
            color='SNAI1-ac_score',
            ax=ax,
            show=False,
            cmap='RdBu_r',
            vcenter=0,
            alpha_img=0.5,
            size=1.0,
            title='SNAI1-ac Score',
        )
        save_figure(fig, figures_dir / 'spatial_snai1ac.png')
    
    print(f"   Saved: distribution_stats.csv, histogram_kde.png, spatial_snai1ac.png")
    
    return stats


# =============================================================================
# SECTION 1.2: SPAGCN DOMAIN ANALYSIS
# =============================================================================

def analyze_spagcn_domains(adata, n_clusters, output_dir, figures_dir):
    """
    Analyze SNAI1-ac and cell type composition by SpaGCN domains.
    
    Parameters
    ----------
    n_clusters : int
        Number of SpaGCN clusters (5 or 9)
    """
    domain_key = f'spagcn_{n_clusters}_refined'
    prefix = f'spagcn{n_clusters}'
    
    print(f"\n--- Phase 1.2: SpaGCN {n_clusters}-cluster Analysis ---")
    
    if domain_key not in adata.obs.columns:
        print(f"   Warning: {domain_key} not found, skipping")
        return None
    
    domains = adata.obs[domain_key].astype(str)
    unique_domains = sorted(domains.unique())
    n_domains = len(unique_domains)
    print(f"   Found {n_domains} domains")
    
    # Get cell type columns present in data
    celltype_cols = [c for c in CELLTYPE_COLS if c in adata.obs.columns]
    
    # --- Domain composition ---
    composition = {}
    counts = {}
    for domain in unique_domains:
        mask = domains == domain
        counts[domain] = int(mask.sum())
        composition[domain] = {}
        for ct in celltype_cols:
            composition[domain][ct] = float(adata.obs.loc[mask, ct].mean())
    
    # Save composition CSV
    comp_df = pd.DataFrame(composition).T
    comp_df.index.name = 'domain'
    comp_df.to_csv(output_dir / f'{prefix}_composition.csv')
    
    # Store in adata.uns
    adata.uns[f'spagcn_{n_clusters}_composition'] = convert_keys_to_str(composition)
    adata.uns[f'spagcn_{n_clusters}_counts'] = convert_keys_to_str(counts)
    
    # --- SNAI1-ac stats by domain ---
    snai1_stats = {}
    score = adata.obs['SNAI1-ac_score'].values
    
    for domain in unique_domains:
        mask = domains == domain
        domain_scores = score[mask]
        snai1_stats[domain] = {
            'mean': float(np.mean(domain_scores)),
            'std': float(np.std(domain_scores)),
            'median': float(np.median(domain_scores)),
            'count': int(mask.sum())
        }
    
    # Save SNAI1-ac stats CSV
    snai1_df = pd.DataFrame(snai1_stats).T
    snai1_df.index.name = 'domain'
    snai1_df.to_csv(output_dir / f'{prefix}_snai1ac_stats.csv')
    
    adata.uns[f'spagcn_{n_clusters}_snai1_stats'] = convert_keys_to_str(snai1_stats)
    
    # --- Pairwise comparisons ---
    pairwise_results = []
    domain_pairs = list(combinations(unique_domains, 2))
    
    for d1, d2 in domain_pairs:
        scores1 = score[domains == d1]
        scores2 = score[domains == d2]
        
        stat, pval = mannwhitneyu(scores1, scores2, alternative='two-sided')
        d = cohens_d(scores1, scores2)
        
        pairwise_results.append({
            'domain1': d1,
            'domain2': d2,
            'mean1': np.mean(scores1),
            'mean2': np.mean(scores2),
            'cohens_d': d,
            'U_statistic': stat,
            'p_value': pval
        })
    
    pairwise_df = pd.DataFrame(pairwise_results)
    # Bonferroni correction
    pairwise_df['p_adjusted'] = pairwise_df['p_value'] * len(pairwise_df)
    pairwise_df['p_adjusted'] = pairwise_df['p_adjusted'].clip(upper=1.0)
    pairwise_df.to_csv(output_dir / f'{prefix}_pairwise_effects.csv', index=False)
    
    # --- Variance decomposition ---
    # Between-domain variance: variance of domain means
    domain_means = [snai1_stats[d]['mean'] for d in unique_domains]
    between_var = float(np.var(domain_means))
    
    # Within-domain variance: mean of domain variances
    within_vars = {}
    for domain in unique_domains:
        domain_scores = score[domains == domain]
        within_vars[domain] = {
            'variance': float(np.var(domain_scores)),
            'std': float(np.std(domain_scores)),
            'n_spots': int(len(domain_scores))
        }
    mean_within_var = float(np.mean([v['variance'] for v in within_vars.values()]))
    
    variance_stats = {
        'within_domain': within_vars,
        'between_domain_var': between_var,
        'mean_within_var': mean_within_var,
        'between_within_ratio': between_var / mean_within_var if mean_within_var > 0 else np.nan
    }
    
    variance_df = pd.DataFrame([{
        'between_domain_var': between_var,
        'mean_within_var': mean_within_var,
        'ratio': variance_stats['between_within_ratio']
    }])
    variance_df.to_csv(output_dir / f'{prefix}_variance.csv', index=False)
    
    adata.uns[f'spagcn_{n_clusters}_variance'] = convert_keys_to_str(variance_stats)
    
    print(f"   Between/within variance ratio: {variance_stats['between_within_ratio']:.2f}")
    
    # --- Figures ---
    
    # Composition heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(comp_df[MAJOR_CELLTYPES if set(MAJOR_CELLTYPES).issubset(comp_df.columns) else comp_df.columns], 
                annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
    ax.set_title(f'Cell Type Composition by SpaGCN {n_clusters} Domains')
    ax.set_xlabel('Cell Type')
    ax.set_ylabel('Domain')
    save_figure(fig, figures_dir / f'heatmap_{prefix}_composition.png')
    
    # Violin plot of SNAI1-ac by domain
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_data = pd.DataFrame({
        'domain': domains,
        'SNAI1-ac': score
    })
    sns.violinplot(data=plot_data, x='domain', y='SNAI1-ac', ax=ax, 
                   order=unique_domains, inner='box')
    ax.set_xlabel('SpaGCN Domain')
    ax.set_ylabel('SNAI1-ac Score')
    ax.set_title(f'SNAI1-ac by SpaGCN {n_clusters} Domains')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    save_figure(fig, figures_dir / f'violin_{prefix}_snai1ac.png')
    
    print(f"   Saved: {prefix}_composition.csv, {prefix}_snai1ac_stats.csv, "
          f"{prefix}_pairwise_effects.csv, {prefix}_variance.csv")
    
    return snai1_stats


# =============================================================================
# SECTION 1.3: INTERFACE ANALYSIS
# =============================================================================

def analyze_interface(adata, output_dir, figures_dir):
    """
    Analyze SNAI1-ac by SpaCET interface regions and concordance with SpaGCN.
    """
    print("\n--- Phase 1.3: Interface Analysis ---")
    
    score = adata.obs['SNAI1-ac_score'].values
    interface = adata.obs['interface']
    regions = ['Tumor', 'Interface', 'Stroma']
    
    # --- Stats by region ---
    region_stats = []
    for region in regions:
        mask = interface == region
        if mask.sum() == 0:
            continue
        region_scores = score[mask]
        region_stats.append({
            'region': region,
            'n': int(mask.sum()),
            'mean': float(np.mean(region_scores)),
            'std': float(np.std(region_scores)),
            'median': float(np.median(region_scores))
        })
    
    stats_df = pd.DataFrame(region_stats)
    
    # Pairwise Cohen's d
    effect_sizes = []
    for r1, r2 in [('Tumor', 'Interface'), ('Interface', 'Stroma'), ('Tumor', 'Stroma')]:
        scores1 = score[interface == r1]
        scores2 = score[interface == r2]
        d = cohens_d(scores1, scores2)
        effect_sizes.append({
            'comparison': f'{r1}_vs_{r2}',
            'cohens_d': d
        })
    
    effects_df = pd.DataFrame(effect_sizes)
    
    # Combine and save
    stats_df.to_csv(output_dir / 'interface_stats.csv', index=False)
    effects_df.to_csv(output_dir / 'interface_effects.csv', index=False)
    
    print(f"   Region counts: {dict(zip(stats_df['region'], stats_df['n']))}")
    for row in effects_df.itertuples():
        print(f"   {row.comparison}: Cohen's d = {row.cohens_d:.3f}")
    
    # --- SpaGCN concordance ---
    for n_clusters in [5, 9]:
        domain_key = f'spagcn_{n_clusters}_refined'
        if domain_key not in adata.obs.columns:
            continue
        
        domains = adata.obs[domain_key].astype(str)
        unique_domains = sorted(domains.unique())
        
        # Contingency table
        contingency = {}
        for domain in unique_domains:
            contingency[domain] = {}
            domain_mask = domains == domain
            for region in regions:
                region_mask = interface == region
                contingency[domain][region] = int((domain_mask & region_mask).sum())
        
        cont_df = pd.DataFrame(contingency).T
        cont_df.index.name = 'domain'
        
        # Row and column percentages
        row_pct = cont_df.div(cont_df.sum(axis=1), axis=0) * 100
        col_pct = cont_df.div(cont_df.sum(axis=0), axis=1) * 100
        
        # Jaccard index for each domain-region pair
        jaccard_matrix = {}
        for domain in unique_domains:
            jaccard_matrix[domain] = {}
            domain_set = set(np.where(domains == domain)[0])
            for region in regions:
                region_set = set(np.where(interface == region)[0])
                jaccard_matrix[domain][region] = jaccard(domain_set, region_set)
        
        jaccard_df = pd.DataFrame(jaccard_matrix).T
        
        # Save
        cont_df.to_csv(output_dir / f'interface_spagcn{n_clusters}_concordance.csv')
        
        # Store in adata.uns
        adata.uns[f'interface_vs_spagcn_{n_clusters}'] = convert_keys_to_str({
            'contingency': contingency,
            'jaccard': jaccard_matrix
        })
        
        print(f"   SpaGCN {n_clusters} concordance saved")
    
    # --- Violin plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_data = pd.DataFrame({
        'interface': interface,
        'SNAI1-ac': score
    })
    sns.violinplot(data=plot_data, x='interface', y='SNAI1-ac', ax=ax,
                   order=regions, inner=None, color='lightgray')
    sns.boxplot(data=plot_data, x='interface', y='SNAI1-ac', ax=ax,
                order=regions, width=0.15, showfliers=False,
                boxprops={'facecolor': 'white'}, medianprops={'color': 'red'})
    ax.set_xlabel('Spatial Region')
    ax.set_ylabel('SNAI1-ac Score')
    ax.set_title('SNAI1-ac by Interface Region')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Add effect size annotations
    y_max = score.max()
    for i, row in enumerate(effects_df.itertuples()):
        ax.text(1, y_max * (0.95 - i*0.08), 
                f"{row.comparison}: d={row.cohens_d:.2f}", fontsize=9)
    
    save_figure(fig, figures_dir / 'violin_interface_snai1ac.png')
    
    print(f"   Saved: interface_stats.csv, interface_effects.csv, violin_interface_snai1ac.png")
    
    return stats_df


# =============================================================================
# SECTION 2.1: CORRELATION ANALYSIS
# =============================================================================

def analyze_correlations(adata, output_dir, figures_dir):
    """
    Analyze correlations between SNAI1-ac and cell type fractions.
    
    Computes raw Spearman and partial correlations (controlling for Malignant).
    """
    print("\n--- Phase 2.1: Correlation Analysis ---")
    
    score = adata.obs['SNAI1-ac_score'].values
    celltype_cols = [c for c in CELLTYPE_COLS if c in adata.obs.columns]
    
    # Raw correlations
    raw_results = []
    for ct in celltype_cols:
        ct_values = adata.obs[ct].values
        r, p = spearmanr(score, ct_values)
        raw_results.append({
            'cell_type': ct,
            'spearman_r': r,
            'p_value': p
        })
    
    raw_df = pd.DataFrame(raw_results)
    raw_df.to_csv(output_dir / 'correlations_raw.csv', index=False)
    
    # Partial correlations (controlling for Malignant)
    if 'Malignant' in celltype_cols:
        malignant = adata.obs['Malignant'].values
        partial_results = []
        
        for ct in celltype_cols:
            if ct == 'Malignant':
                partial_results.append({
                    'cell_type': ct,
                    'partial_r': np.nan,
                    'p_value': np.nan
                })
            else:
                ct_values = adata.obs[ct].values
                r, p = partial_spearman(score, ct_values, malignant)
                partial_results.append({
                    'cell_type': ct,
                    'partial_r': r,
                    'p_value': p
                })
        
        partial_df = pd.DataFrame(partial_results)
        partial_df.to_csv(output_dir / 'correlations_partial.csv', index=False)
        
        # Comparison table
        comparison = raw_df.merge(partial_df, on='cell_type', suffixes=('_raw', '_partial'))
        comparison['r_change'] = comparison['partial_r'] - comparison['spearman_r']
        comparison.to_csv(output_dir / 'correlations_comparison.csv', index=False)
        
        print("   Top raw correlations:")
        top_raw = raw_df.nlargest(5, 'spearman_r', keep='first')
        for _, row in top_raw.iterrows():
            print(f"      {row['cell_type']}: r = {row['spearman_r']:.3f}")
    
    # Scatter plot: Malignant vs SNAI1-ac
    if 'Malignant' in adata.obs.columns:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(adata.obs['Malignant'], score, alpha=0.3, s=10)
        r = raw_df.loc[raw_df['cell_type'] == 'Malignant', 'spearman_r'].values[0]
        ax.set_xlabel('Malignant Fraction')
        ax.set_ylabel('SNAI1-ac Score')
        ax.set_title(f'SNAI1-ac vs Malignant (r = {r:.3f})')
        save_figure(fig, figures_dir / 'scatter_malignant_snai1ac.png')
    
    print(f"   Saved: correlations_raw.csv, correlations_partial.csv, "
          f"correlations_comparison.csv, scatter_malignant_snai1ac.png")
    
    return raw_df


# =============================================================================
# SECTION 3.2: NEIGHBORHOOD ANALYSIS
# =============================================================================

def analyze_neighborhoods(adata, output_dir, figures_dir):
    """
    Analyze neighborhood composition around HIGH/MID/LOW SNAI1-ac spots.
    
    Defines groups using +/- 1 SD thresholds, computes neighbor composition,
    and runs permutation tests for significance.
    """
    print("\n--- Phase 3.2: Neighborhood Analysis ---")
    
    score = adata.obs['SNAI1-ac_score'].values
    mean_score = np.mean(score)
    std_score = np.std(score)
    
    # Define groups
    high_thresh = mean_score + SNAI1_GROUP_SD_THRESHOLD * std_score
    low_thresh = mean_score - SNAI1_GROUP_SD_THRESHOLD * std_score
    
    groups = np.where(score >= high_thresh, 'HIGH',
                      np.where(score <= low_thresh, 'LOW', 'MID'))
    adata.obs['SNAI1_ac_group'] = groups
    
    n_high = (groups == 'HIGH').sum()
    n_mid = (groups == 'MID').sum()
    n_low = (groups == 'LOW').sum()
    
    print(f"   HIGH (>{high_thresh:.3f}): {n_high} spots")
    print(f"   MID: {n_mid} spots")
    print(f"   LOW (<{low_thresh:.3f}): {n_low} spots")
    
    # Build spatial neighbor graph
    if 'spatial_neighbors' not in adata.uns:
        sc.pp.neighbors(adata, use_rep='spatial', n_neighbors=MORAN_K_NEIGHBORS + 1,
                        key_added='spatial_neighbors')
    
    conn = adata.obsp['spatial_neighbors_connectivities']
    
    # Get cell type fractions
    celltype_cols = [c for c in CELLTYPE_COLS if c in adata.obs.columns]
    fractions = adata.obs[celltype_cols].values
    
    # Neighbor composition for each group
    high_idx = np.where(groups == 'HIGH')[0]
    mid_idx = np.where(groups == 'MID')[0]
    low_idx = np.where(groups == 'LOW')[0]
    
    high_comp = get_neighbor_composition(high_idx, conn, fractions)
    mid_comp = get_neighbor_composition(mid_idx, conn, fractions)
    low_comp = get_neighbor_composition(low_idx, conn, fractions)
    
    # Save neighbor composition
    comp_df = pd.DataFrame({
        'cell_type': celltype_cols,
        'HIGH_neighbors': high_comp,
        'MID_neighbors': mid_comp,
        'LOW_neighbors': low_comp
    })
    comp_df['HIGH_minus_LOW'] = comp_df['HIGH_neighbors'] - comp_df['LOW_neighbors']
    comp_df.to_csv(output_dir / 'neighbor_composition.csv', index=False)
    
    # Permutation test for enrichment
    # IMPORTANT: Loop order is permutations (outer) -> cell types (inner)
    # This way we only call get_neighbor_composition once per permutation
    print(f"   Running {N_PERMUTATIONS} permutations...")
    
    # Store null distributions for all cell types
    null_distributions = {ct: [] for ct in celltype_cols}
    
    np.random.seed(42)  # For reproducibility
    
    for perm in range(N_PERMUTATIONS):
        # Random "high" spots
        random_idx = np.random.choice(len(score), size=n_high, replace=False)
        
        # Get neighbor composition (computes ALL cell types at once)
        null_comp = get_neighbor_composition(random_idx, conn, fractions)
        
        # Store result for each cell type
        for j, ct in enumerate(celltype_cols):
            null_distributions[ct].append(null_comp[j])
        
        # Progress feedback
        if (perm + 1) % 200 == 0:
            print(f"      Permutation {perm + 1}/{N_PERMUTATIONS}")
    
    print("      Done")
    
    # Calculate enrichment statistics for each cell type
    enrichment_results = []
    
    for i, ct in enumerate(celltype_cols):
        observed = high_comp[i]
        null = np.array(null_distributions[ct])
        null_mean = null.mean()
        
        # Enrichment ratio
        enrichment = observed / null_mean if null_mean > 0 else np.nan
        
        # Two-tailed p-value (standard permutation test formula)
        p_value = (np.sum(np.abs(null - null_mean) >= np.abs(observed - null_mean)) + 1) / (N_PERMUTATIONS + 1)
        
        # 95% CI from null distribution
        ci_lower = np.percentile(null, 2.5)
        ci_upper = np.percentile(null, 97.5)
        
        enrichment_results.append({
            'cell_type': ct,
            'observed_high': observed,
            'null_mean': null_mean,
            'null_std': null.std(),
            'enrichment_ratio': enrichment,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value
        })
    
    enrich_df = pd.DataFrame(enrichment_results)
    
    # FDR correction
    from scipy.stats import false_discovery_control
    enrich_df['p_adjusted'] = false_discovery_control(enrich_df['p_value'].fillna(1))
    enrich_df['significant'] = enrich_df['p_adjusted'] < FDR_THRESHOLD
    
    enrich_df.to_csv(output_dir / 'neighbor_enrichment.csv', index=False)
    
    n_sig = enrich_df['significant'].sum()
    print(f"   {n_sig} cell types significantly enriched/depleted (FDR < {FDR_THRESHOLD})")
    
    # Ring analysis with permutation tests
    print(f"   Computing {N_RINGS}-ring composition with permutation tests...")
    high_ring_comp = get_ring_composition(high_idx, conn, fractions, N_RINGS)
    low_ring_comp = get_ring_composition(low_idx, conn, fractions, N_RINGS)
    
    # Permutation test for rings
    null_ring_distributions = {r: {ct: [] for ct in celltype_cols} for r in range(1, N_RINGS + 1)}
    
    for perm in range(N_PERMUTATIONS):
        # Random "high" spots
        random_idx = np.random.choice(len(score), size=n_high, replace=False)
        
        # Get ring composition for random spots
        null_ring_comp = get_ring_composition(random_idx, conn, fractions, N_RINGS)
        
        # Store for each ring and cell type
        for r in range(1, N_RINGS + 1):
            for j, ct in enumerate(celltype_cols):
                null_ring_distributions[r][ct].append(null_ring_comp[r][j])
        
        if (perm + 1) % 200 == 0:
            print(f"      Ring permutation {perm + 1}/{N_PERMUTATIONS}")
    
    print("      Done")
    
    # Calculate ring enrichment statistics
    ring_results = []
    for r in range(1, N_RINGS + 1):
        for i, ct in enumerate(celltype_cols):
            observed = high_ring_comp[r][i]
            null = np.array(null_ring_distributions[r][ct])
            null_mean = null.mean()
            
            # Enrichment ratio
            enrichment = observed / null_mean if null_mean > 0 else np.nan
            
            # Two-tailed p-value
            p_value = (np.sum(np.abs(null - null_mean) >= np.abs(observed - null_mean)) + 1) / (N_PERMUTATIONS + 1)
            
            ring_results.append({
                'ring': r,
                'cell_type': ct,
                'HIGH': observed,
                'LOW': low_ring_comp[r][i],
                'diff': observed - low_ring_comp[r][i],
                'null_mean': null_mean,
                'enrichment_ratio': enrichment,
                'p_value': p_value
            })
    
    ring_df = pd.DataFrame(ring_results)
    
    # FDR correction per ring
    for r in range(1, N_RINGS + 1):
        mask = ring_df['ring'] == r
        ring_df.loc[mask, 'p_adjusted'] = false_discovery_control(ring_df.loc[mask, 'p_value'].fillna(1))
    
    ring_df['significant'] = ring_df['p_adjusted'] < FDR_THRESHOLD
    ring_df.to_csv(output_dir / 'ring_composition.csv', index=False)
    
    # --- Figures ---
    
    # Spatial plot of groups
    if 'spatial' in adata.obsm:
        fig, ax = plt.subplots(figsize=(7, 7))
        colors = {'HIGH': 'red', 'MID': 'lightgray', 'LOW': 'blue'}
        for group in ['MID', 'HIGH', 'LOW']:  # Plot MID first so HIGH/LOW are on top
            mask = groups == group
            coords = adata.obsm['spatial'][mask]
            ax.scatter(coords[:, 0], coords[:, 1], c=colors[group], 
                       label=group, s=8, alpha=0.7)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title('SNAI1-ac Groups (HIGH/MID/LOW)')
        ax.axis('off')
        save_figure(fig, figures_dir / 'spatial_snai1ac_groups.png')
    
    # Heatmap of neighbor composition
    fig, ax = plt.subplots(figsize=(10, 6))
    heatmap_data = comp_df.set_index('cell_type')[['HIGH_neighbors', 'MID_neighbors', 'LOW_neighbors']]
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
    ax.set_title('Neighbor Composition by SNAI1-ac Group')
    save_figure(fig, figures_dir / 'heatmap_neighbor_composition.png')
    
    # Bar chart with significance
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(celltype_cols))
    width = 0.35
    ax.bar(x - width/2, high_comp, width, label='HIGH neighbors', color='red', alpha=0.7)
    ax.bar(x + width/2, low_comp, width, label='LOW neighbors', color='blue', alpha=0.7)
    
    # Add significance stars
    for i, row in enrich_df.iterrows():
        if row['significant']:
            ax.text(i, max(high_comp[i], low_comp[i]) + 0.01, '*', ha='center', fontsize=12)
    
    ax.set_xticks(x)
    ax.set_xticklabels(celltype_cols, rotation=45, ha='right')
    ax.set_ylabel('Mean Neighbor Fraction')
    ax.set_title('Neighbor Composition: HIGH vs LOW SNAI1-ac')
    ax.legend()
    plt.tight_layout()
    save_figure(fig, figures_dir / 'bar_enrichment.png')
    
    print(f"   Saved: neighbor_composition.csv, neighbor_enrichment.csv, ring_composition.csv")
    
    return comp_df


# =============================================================================
# SECTION 3.3: HOTSPOT DETECTION
# =============================================================================

def detect_hotspots(adata, output_dir, figures_dir):
    """
    Detect SNAI1-ac hotspots using LISA and threshold-based methods.
    """
    print("\n--- Phase 3.3: Hotspot Detection ---")
    
    score = adata.obs['SNAI1-ac_score'].values

    # Build spatial weights from array coordinates (standardised 100um Visium grid,
    # comparable across samples regardless of imaging resolution)
    array_coords = adata.obs[['array_row', 'array_col']].values.astype(float)
    w_array = libpysal.weights.KNN.from_array(array_coords, k=MORAN_K_NEIGHBORS)
    w_array.transform = 'r'

    # Convert to dense row-normalised matrix (reused for all variables)
    W_normalized = np.array(w_array.full()[0])
    W_normalized = W_normalized / W_normalized.sum(axis=1, keepdims=True)
    W_normalized = np.nan_to_num(W_normalized, nan=0.0)

    # Also build sparse connectivity matrix for threshold-based hotspot methods below
    # (kept for backward compatibility with classify_by_neighbor_threshold)
    if 'spatial_neighbors' not in adata.uns:
        sc.pp.neighbors(adata, use_rep='spatial', n_neighbors=MORAN_K_NEIGHBORS + 1,
                        key_added='spatial_neighbors')
    conn = adata.obsp['spatial_neighbors_connectivities']

    from scipy.stats import false_discovery_control

    def run_lisa(values, n_perms):
        """Run Local Moran's I and return category array."""
        z = (values - values.mean()) / values.std()
        lag = W_normalized @ z
        local_I = z * lag

        null_I = np.zeros((len(z), n_perms))
        for p in range(n_perms):
            pz = np.random.permutation(z)
            null_I[:, p] = pz * (W_normalized @ pz)

        pvals = np.array([
            (np.sum(np.abs(null_I[i] - null_I[i].mean()) >=
                    np.abs(local_I[i] - null_I[i].mean())) + 1) / (n_perms + 1)
            for i in range(len(z))
        ])
        padj = false_discovery_control(pvals)
        sig = padj < FDR_THRESHOLD

        cats = np.array(['Not significant'] * len(z), dtype=object)
        cats[sig & (z > 0) & (lag > 0)] = 'High-High'
        cats[sig & (z < 0) & (lag < 0)] = 'Low-Low'
        cats[sig & (z > 0) & (lag < 0)] = 'High-Low'
        cats[sig & (z < 0) & (lag > 0)] = 'Low-High'
        return cats, z, lag, local_I, pvals, padj

    # --- SNAI1-ac LISA (999 permutations — reported result) ---
    print(f"   Running LISA for SNAI1-ac ({LISA_N_PERMUTATIONS} permutations)...")
    np.random.seed(42)
    lisa_category, z_scores, spatial_lag, local_moran_I, p_values, p_adjusted = \
        run_lisa(score, LISA_N_PERMUTATIONS)

    adata.obs['LISA_category'] = lisa_category

    # Counts
    lisa_counts = pd.Series(lisa_category).value_counts()
    print(f"   LISA results: {dict(lisa_counts)}")

    # --- Multi-variable LISA (99 permutations — intermediate for downstream analyses) ---
    hallmark_cols = [c for c in adata.obs.columns
                     if c.startswith('HALLMARK_') and c.endswith('_score')]
    celltype_vars = [c for c in MAJOR_CELLTYPES if c in adata.obs.columns]

    lisa_dict = {'SNAI1-ac': lisa_category}

    print(f"   Running LISA for {len(hallmark_cols)} Hallmark pathways "
          f"and {len(celltype_vars)} cell types (99 permutations each)...")

    for col in hallmark_cols:
        cats, _, _, _, _, _ = run_lisa(adata.obs[col].values, 99)
        lisa_dict[col] = cats

    for col in celltype_vars:
        cats, _, _, _, _, _ = run_lisa(adata.obs[col].values, 99)
        lisa_dict[col] = cats

    print(f"   Multi-variable LISA complete ({len(lisa_dict)} variables total)")

    # Store in adata.uns
    adata.uns['LISA'] = {k: v.tolist() for k, v in lisa_dict.items()}

    # Save wide CSV: rows = spots, columns = variables
    lisa_wide = pd.DataFrame(lisa_dict, index=adata.obs_names)
    lisa_wide.index.name = 'spot'
    lisa_wide.to_csv(output_dir / 'lisa_all_variables.csv')
    print(f"   Saved: lisa_all_variables.csv ({lisa_wide.shape[0]} spots x {lisa_wide.shape[1]} variables)")
    
    # Save LISA results
    lisa_df = pd.DataFrame({
        'spot': adata.obs_names,
        'z_score': z_scores,
        'spatial_lag': spatial_lag,
        'local_moran_I': local_moran_I,
        'p_value': p_values,
        'p_adjusted': p_adjusted,
        'LISA_category': lisa_category
    })
    lisa_df.to_csv(output_dir / 'lisa_results.csv', index=False)
    
    # --- Threshold-based hotspots ---
    mean_score = score.mean()
    std_score = score.std()
    high_mask = score >= (mean_score + std_score)
    low_mask = score <= (mean_score - std_score)
    
    hotspot_strict = np.array(['None'] * len(score), dtype=object)
    
    for i in range(len(score)):
        if classify_by_neighbor_threshold(i, conn, high_mask, HOTSPOT_MIN_NEIGHBORS):
            hotspot_strict[i] = 'Hotspot'
        elif classify_by_neighbor_threshold(i, conn, low_mask, HOTSPOT_MIN_NEIGHBORS):
            hotspot_strict[i] = 'Coldspot'
    
    adata.obs['hotspot_strict'] = hotspot_strict
    
    hotspot_counts = pd.Series(hotspot_strict).value_counts()
    print(f"   Threshold hotspots: {dict(hotspot_counts)}")
    
    # Method comparison (Jaccard)
    lisa_hh = set(np.where(lisa_category == 'High-High')[0])
    lisa_ll = set(np.where(lisa_category == 'Low-Low')[0])
    thresh_hot = set(np.where(hotspot_strict == 'Hotspot')[0])
    thresh_cold = set(np.where(hotspot_strict == 'Coldspot')[0])
    
    comparison = {
        'Hotspot_jaccard': jaccard(lisa_hh, thresh_hot),
        'Coldspot_jaccard': jaccard(lisa_ll, thresh_cold),
        'LISA_HH_count': len(lisa_hh),
        'LISA_LL_count': len(lisa_ll),
        'Threshold_hot_count': len(thresh_hot),
        'Threshold_cold_count': len(thresh_cold)
    }
    
    pd.DataFrame([comparison]).to_csv(output_dir / 'hotspot_comparison.csv', index=False)
    print(f"   Hotspot Jaccard: {comparison['Hotspot_jaccard']:.3f}")
    print(f"   Coldspot Jaccard: {comparison['Coldspot_jaccard']:.3f}")
    
    # --- Figures ---
    
    # LISA spatial plot
    if 'spatial' in adata.obsm:
        fig, ax = plt.subplots(figsize=(7, 7))
        colors = {
            'High-High': 'red',
            'Low-Low': 'blue',
            'High-Low': 'orange',
            'Low-High': 'lightblue',
            'Not significant': 'lightgray'
        }
        for cat in ['Not significant', 'High-Low', 'Low-High', 'Low-Low', 'High-High']:
            mask = lisa_category == cat
            if mask.sum() > 0:
                coords = adata.obsm['spatial'][mask]
                ax.scatter(coords[:, 0], coords[:, 1], c=colors[cat], 
                           label=f"{cat} ({mask.sum()})", s=8, alpha=0.7)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.legend(loc='best', fontsize=8)
        ax.set_title('LISA Hotspot Analysis')
        ax.axis('off')
        save_figure(fig, figures_dir / 'spatial_lisa.png')
        
        # Threshold-based spatial plot
        fig, ax = plt.subplots(figsize=(7, 7))
        colors_thresh = {'Hotspot': 'red', 'Coldspot': 'blue', 'None': 'lightgray'}
        for cat in ['None', 'Coldspot', 'Hotspot']:
            mask = hotspot_strict == cat
            if mask.sum() > 0:
                coords = adata.obsm['spatial'][mask]
                ax.scatter(coords[:, 0], coords[:, 1], c=colors_thresh[cat],
                           label=f"{cat} ({mask.sum()})", s=8, alpha=0.7)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.legend(loc='best')
        ax.set_title('Threshold-based Hotspots')
        ax.axis('off')
        save_figure(fig, figures_dir / 'spatial_hotspots_threshold.png')
    
    print(f"   Saved: lisa_results.csv, hotspot_comparison.csv, spatial_lisa.png, "
          f"spatial_hotspots_threshold.png")
    
    return lisa_df

# =============================================================================
# SECTION 3.4: ENRICHMENT ANALYSIS
# =============================================================================

def analyze_enrichment(adata, output_dir, figures_dir):
    """
    Analyze interface enrichment and halo composition.
    """
    print("\n--- Phase 3.4: Enrichment Analysis ---")
    
    groups = adata.obs['SNAI1_ac_group'].values
    interface = adata.obs['interface'].values
    
    # Contingency table
    contingency = pd.crosstab(groups, interface)
    
    # Chi-square test
    chi2, p_chi2, dof, expected = chi2_contingency(contingency)
    print(f"   Chi-square: {chi2:.2f}, p = {p_chi2:.4f}")
    
    # Fisher's exact for HIGH at Interface vs Tumor
    high_interface = ((groups == 'HIGH') & (interface == 'Interface')).sum()
    high_tumor = ((groups == 'HIGH') & (interface == 'Tumor')).sum()
    not_high_interface = ((groups != 'HIGH') & (interface == 'Interface')).sum()
    not_high_tumor = ((groups != 'HIGH') & (interface == 'Tumor')).sum()
    
    fisher_table = [[high_interface, high_tumor],
                    [not_high_interface, not_high_tumor]]
    odds_ratio, p_fisher = fisher_exact(fisher_table)
    
    print(f"   Fisher (HIGH Interface vs Tumor): OR = {odds_ratio:.3f}, p = {p_fisher:.4f}")
    
    enrichment_results = {
        'chi2': chi2,
        'chi2_pvalue': p_chi2,
        'dof': dof,
        'fisher_odds_ratio': odds_ratio,
        'fisher_pvalue': p_fisher
    }
    
    pd.DataFrame([enrichment_results]).to_csv(output_dir / 'interface_enrichment.csv', index=False)
    
    # --- Halo analysis ---
    if 'LISA_category' in adata.obs.columns:
        lisa_category = adata.obs['LISA_category'].values
        conn = adata.obsp['spatial_neighbors_connectivities']
        
        celltype_cols = [c for c in CELLTYPE_COLS if c in adata.obs.columns]
        fractions = adata.obs[celltype_cols].values
        
        # Find halo spots (neighbors of HH hotspots that aren't hotspots themselves)
        hh_idx = np.where(lisa_category == 'High-High')[0]
        hh_set = set(hh_idx)
        
        halo_spots = set()
        for idx in hh_idx:
            neighbors = set(conn[idx].nonzero()[1])
            halo_spots.update(neighbors - hh_set)
        
        halo_idx = np.array(list(halo_spots))
        
        if len(halo_idx) > 0:
            # Composition comparison
            hotspot_comp = fractions[hh_idx].mean(axis=0)
            halo_comp = fractions[halo_idx].mean(axis=0)
            overall_comp = fractions.mean(axis=0)
            
            halo_df = pd.DataFrame({
                'cell_type': celltype_cols,
                'hotspot': hotspot_comp,
                'halo': halo_comp,
                'overall': overall_comp
            })
            halo_df['hotspot_vs_overall'] = halo_df['hotspot'] - halo_df['overall']
            halo_df['halo_vs_overall'] = halo_df['halo'] - halo_df['overall']
            
            halo_df.to_csv(output_dir / 'halo_composition.csv', index=False)
            print(f"   Halo analysis: {len(halo_idx)} halo spots around {len(hh_idx)} hotspots")
            
            # Boxplot
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(celltype_cols))
            width = 0.25
            ax.bar(x - width, hotspot_comp, width, label='Hotspot', color='red', alpha=0.7)
            ax.bar(x, halo_comp, width, label='Halo', color='orange', alpha=0.7)
            ax.bar(x + width, overall_comp, width, label='Overall', color='gray', alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(celltype_cols, rotation=45, ha='right')
            ax.set_ylabel('Mean Fraction')
            ax.set_title('Cell Type Composition: Hotspot vs Halo vs Overall')
            ax.legend()
            plt.tight_layout()
            save_figure(fig, figures_dir / 'boxplot_halo.png')
        else:
            print("   No halo spots found")
    
    print(f"   Saved: interface_enrichment.csv, halo_composition.csv, boxplot_halo.png")
    
    return enrichment_results


# =============================================================================
# SECTION 4.1: CIRCULARITY CHECK
# =============================================================================

def check_circularity(adata, output_dir):
    """
    Check for circularity between SNAI1-ac signature and domain markers.
    """
    print("\n--- Phase 4.1: Circularity Check ---")
    
    # SNAI1-ac signature genes (from the signature file)
    # These would ideally be loaded from the signature file
    # For now, we'll check what's stored or use a placeholder
    
    snai1_genes = set()
    signature_source = None
    if 'SNAI1_ac_signature_genes' in adata.uns:
        snai1_genes = {
            str(g).strip() for g in adata.uns['SNAI1_ac_signature_genes']
            if str(g).strip()
        }
        signature_source = 'adata.uns'
    
    if len(snai1_genes) == 0:
        fallback_genes = load_signature_genes_from_weights()
        if fallback_genes:
            snai1_genes = set(fallback_genes)
            adata.uns['SNAI1_ac_signature_genes'] = sorted(snai1_genes)
            signature_source = 'weights_json'
            print(f"   Loaded {len(snai1_genes)} signature genes from snai1_ac_weights.json")
    
    if len(snai1_genes) == 0:
        print("   No signature genes found in adata.uns, skipping circularity check")
        return None
    
    results = []
    
    # Check overlap with domain markers
    for n_clusters in [5, 9]:
        domain_key = f'spagcn_{n_clusters}_refined'
        if domain_key not in adata.obs.columns:
            continue
        
        # Run rank_genes_groups if not already done
        key = f'rank_genes_{domain_key}'
        if key not in adata.uns:
            sc.tl.rank_genes_groups(adata, groupby=domain_key, method='wilcoxon',
                                    key_added=key, n_genes=50)
        
        # Get top markers for each domain
        domain_markers = set()
        for group in adata.uns[key]['names'].dtype.names:
            genes = adata.uns[key]['names'][group][:50]
            domain_markers.update(genes)
        
        # Calculate overlap
        overlap = snai1_genes & domain_markers
        overlap_pct = len(overlap) / len(snai1_genes) * 100 if len(snai1_genes) > 0 else 0
        
        results.append({
            'comparison': f'SNAI1_ac vs SpaGCN_{n_clusters}_markers',
            'signature_source': signature_source,
            'n_signature_genes': len(snai1_genes),
            'n_marker_genes': len(domain_markers),
            'n_overlap': len(overlap),
            'overlap_pct': overlap_pct,
            'overlap_genes': ','.join(sorted(overlap)) if overlap else ''
        })
        
        print(f"   SpaGCN {n_clusters}: {len(overlap)}/{len(snai1_genes)} genes overlap "
              f"({overlap_pct:.1f}%)")
    
    # Check HVG overlap
    if 'highly_variable' in adata.var.columns:
        hvg_genes = set(adata.var_names[adata.var['highly_variable']])
        overlap = snai1_genes & hvg_genes
        overlap_pct = len(overlap) / len(snai1_genes) * 100 if len(snai1_genes) > 0 else 0
        
        results.append({
            'comparison': 'SNAI1_ac vs HVG',
            'signature_source': signature_source,
            'n_signature_genes': len(snai1_genes),
            'n_marker_genes': len(hvg_genes),
            'n_overlap': len(overlap),
            'overlap_pct': overlap_pct,
            'overlap_genes': ','.join(sorted(overlap)) if overlap else ''
        })
        
        print(f"   HVG: {len(overlap)}/{len(snai1_genes)} genes overlap ({overlap_pct:.1f}%)")
    
    if results:
        pd.DataFrame(results).to_csv(output_dir / 'circularity_check.csv', index=False)
    
    return results


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_single_sample(sample_info, dataset_id):
    """
    Run complete spatial signature analysis on a single sample.
    """
    sample_name = sample_info['name']
    sample_path = sample_info['path']
    
    print(f"\n{'='*70}")
    print(f"Analyzing: {dataset_id} / {sample_name}")
    print(f"{'='*70}")
    
    # Find SpaCET files
    celltypes_file, interface_file = find_spacet_files(sample_name, dataset_id)
    
    if not celltypes_file.exists() or not interface_file.exists():
        print(f"   ERROR: SpaCET files not found")
        print(f"      Expected: {celltypes_file}")
        print(f"      Expected: {interface_file}")
        print(f"   Skipping sample")
        return None
    
    # Create output directories
    output_base = ANALYSIS_DIR / dataset_id / sample_name
    output_dir = output_base / 'signature_analysis' / 'csvs'
    figures_dir = output_base / 'signature_analysis' / 'figures'
    ensure_dir(output_dir)
    ensure_dir(figures_dir)
    
    # Load data
    print(f"\n   Loading: {sample_path.name}")
    adata = sc.read_h5ad(sample_path)
    print(f"   Spots: {adata.n_obs}, Genes: {adata.n_vars}")
    
    # Check prerequisites
    if 'SNAI1-ac_score' not in adata.obs.columns:
        print("   ERROR: SNAI1-ac_score not found in adata.obs")
        return None
    
    # Section 0: Merge SpaCET data
    adata = merge_spacet_data(adata, celltypes_file, interface_file)
    
    # Section 1.1: Distribution analysis
    analyze_distribution(adata, output_dir, figures_dir)
    
    # Section 1.2: SpaGCN domain analysis (5 and 9 clusters)
    for n_clusters in [5, 9]:
        analyze_spagcn_domains(adata, n_clusters, output_dir, figures_dir)
    
    # Section 1.3: Interface analysis
    analyze_interface(adata, output_dir, figures_dir)
    
    # Section 2.1: Correlation analysis
    analyze_correlations(adata, output_dir, figures_dir)
    
    # Section 3.2: Neighborhood analysis
    analyze_neighborhoods(adata, output_dir, figures_dir)
    
    # Section 3.3: Hotspot detection
    detect_hotspots(adata, output_dir, figures_dir)
    
    # Section 3.4: Enrichment analysis
    analyze_enrichment(adata, output_dir, figures_dir)
    
    # Section 4.1: Circularity check
    check_circularity(adata, output_dir)
    
    # Save updated h5ad
    # Convert any integer keys to strings for h5ad compatibility
    for key in list(adata.uns.keys()):
        if isinstance(adata.uns[key], dict):
            adata.uns[key] = convert_keys_to_str(adata.uns[key])
    
    output_h5ad = output_base / f"{sample_name}.h5ad"
    adata.write_h5ad(output_h5ad)
    print(f"\n   Saved: {output_h5ad}")
    
    print(f"\n   Analysis complete for {sample_name}")
    
    return {
        'sample': sample_name,
        'dataset': dataset_id,
        'n_spots': adata.n_obs,
        'output_dir': str(output_base)
    }


def analyze_dataset(dataset_id, sample_name=None):
    """Analyze all samples in a dataset."""
    dataset_path = PROCESSED_DIR / dataset_id
    
    if not dataset_path.exists():
        print(f"   ERROR: Dataset not found: {dataset_path}")
        return []
    
    # Find samples
    samples = find_h5ad_samples(dataset_path)
    
    if not samples:
        print(f"   No .h5ad files found in {dataset_path}")
        return []
    
    # Filter to specific sample if requested
    if sample_name:
        samples = [s for s in samples if s['name'] == sample_name]
        if not samples:
            print(f"   Sample not found: {sample_name}")
            return []
    
    print(f"   Found {len(samples)} sample(s)")
    
    # Analyze each sample
    results = []
    for sample in samples:
        result = analyze_single_sample(sample, dataset_id)
        if result:
            results.append(result)
    
    return results


def find_analyzed_samples(dataset_id):
    """Find analysis-ready h5ad outputs under 05_analysis_ready."""
    dataset_path = ANALYSIS_DIR / dataset_id
    if not dataset_path.exists():
        return []

    samples = []
    for sample_dir in sorted(dataset_path.iterdir()):
        if not sample_dir.is_dir():
            continue
        h5ad_path = sample_dir / f"{sample_dir.name}.h5ad"
        if h5ad_path.exists():
            samples.append({
                'name': sample_dir.name,
                'path': h5ad_path,
                'output_dir': sample_dir / 'signature_analysis' / 'csvs',
            })
    return samples


def run_circularity_only(dataset_id, sample_name=None):
    """Backfill and run only the circularity check on analysis-ready h5ads."""
    samples = find_analyzed_samples(dataset_id)
    if not samples:
        print(f"   No analysis-ready h5ad files found for {dataset_id}")
        return []

    if sample_name:
        samples = [s for s in samples if s['name'] == sample_name]
        if not samples:
            print(f"   Sample not found in analysis-ready outputs: {sample_name}")
            return []

    print(f"   Running circularity-only mode for {len(samples)} sample(s)")
    results = []

    for sample in samples:
        print(f"\n{'='*70}")
        print(f"Circularity only: {dataset_id} / {sample['name']}")
        print(f"{'='*70}")

        ensure_dir(sample['output_dir'])
        adata = sc.read_h5ad(sample['path'])
        circularity = check_circularity(adata, sample['output_dir'])
        adata.write_h5ad(sample['path'])

        results.append({
            'dataset': dataset_id,
            'sample': sample['name'],
            'n_checks': len(circularity) if circularity else 0,
            'path': str(sample['path']),
        })

    return results


def list_status():
    """List analysis status for all datasets."""
    print(f"\n{'Dataset':<30} {'Samples':<10} {'SpaCET':<10} {'Analyzed':<10}")
    print("-" * 60)
    
    for dataset_id in KNOWN_DATASETS:
        processed_path = PROCESSED_DIR / dataset_id
        analysis_path = ANALYSIS_DIR / dataset_id
        
        # Count samples
        if processed_path.exists():
            h5ad_files = list(processed_path.glob("*.h5ad"))
            n_samples = len(h5ad_files)
        else:
            n_samples = 0
        
        # Check SpaCET
        spacet_count = 0
        if analysis_path.exists():
            for sample_dir in analysis_path.iterdir():
                if sample_dir.is_dir():
                    celltypes = sample_dir / f"{sample_dir.name}_celltypes.csv"
                    if celltypes.exists():
                        spacet_count += 1
        
        # Check analyzed
        analyzed_count = 0
        if analysis_path.exists():
            for sample_dir in analysis_path.iterdir():
                if sample_dir.is_dir():
                    sig_dir = sample_dir / 'signature_analysis'
                    if sig_dir.exists():
                        analyzed_count += 1
        
        samples_status = f"{n_samples}" if n_samples > 0 else "-"
        spacet_status = f"{spacet_count}" if spacet_count > 0 else "-"
        analyzed_status = f"{analyzed_count}" if analyzed_count > 0 else "-"
        
        print(f"{dataset_id:<30} {samples_status:<10} {spacet_status:<10} {analyzed_status:<10}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Spatial signature analysis for SNAI1-ac in HGSOC'
    )
    parser.add_argument(
        'dataset',
        nargs='?',
        help='Dataset ID (e.g., visium/denisenko_2022)'
    )
    parser.add_argument(
        '--sample', '-s',
        help='Analyze specific sample only'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Analyze all datasets'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List analysis status'
    )
    parser.add_argument(
        '--circularity-only',
        action='store_true',
        help='Run only the circularity check on existing analysis-ready h5ad outputs'
    )
    
    args = parser.parse_args()
    
    if args.list:
        print_header()
        list_status()
        return
    
    if args.all:
        print_header()
        all_results = []
        for dataset_id in KNOWN_DATASETS:
            if args.circularity_only:
                results = run_circularity_only(dataset_id)
            else:
                results = analyze_dataset(dataset_id)
            all_results.extend(results)
        
        print(f"\n{'='*70}")
        print(f"COMPLETE: Analyzed {len(all_results)} samples")
        print(f"{'='*70}")
        return
    
    if args.dataset:
        print_header(args.dataset, args.sample)
        if args.circularity_only:
            results = run_circularity_only(args.dataset, sample_name=args.sample)
        else:
            results = analyze_dataset(args.dataset, sample_name=args.sample)
        
        print(f"\n{'='*70}")
        print(f"COMPLETE: Analyzed {len(results)} sample(s)")
        print(f"{'='*70}")
        return
    
    # No arguments - print usage
    print_header()
    print("\nUsage:")
    print("  python spatial_signature_analysis.py visium/denisenko_2022")
    print("  python spatial_signature_analysis.py visium/denisenko_2022 --sample SP6")
    print("  python spatial_signature_analysis.py visium/denisenko_2022 --circularity-only")
    print("  python spatial_signature_analysis.py --all")
    print("  python spatial_signature_analysis.py --list")


if __name__ == "__main__":
    main()
