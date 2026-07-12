"""
Preprocessing Pipeline for HGSOC Spatial Transcriptomics Data.

Usage:
    python preprocessing.py <platform/dataset_id>
    python preprocessing.py visium/yamamoto_2025
    python preprocessing.py visium/yamamoto_2025 --sample Pt1-1
    python preprocessing.py visium/yamamoto_2025 --genes snai1_signature.txt
    python preprocessing.py visium/yamamoto_2025 --n-pcs 10 --leiden-resolution 0.8
    python preprocessing.py --all
    python preprocessing.py --list

Processes .h5ad files from 02_processed_data/
Outputs plots to 05_analysis_ready/
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import random
import torch
from scipy.sparse import issparse

# Check for required packages
try:
    import scanpy as sc
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    PACKAGES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Missing package: {e}")
    print("   Install with: pip install scanpy pandas matplotlib seaborn")
    PACKAGES_AVAILABLE = False

# Check for SpaGCN (optional)
try:
    import SpaGCN as spg
    SPAGCN_AVAILABLE = True
except ImportError:
    SPAGCN_AVAILABLE = False
    print("⚠️  SpaGCN not available. Spatial clustering will be skipped.")
    print("   Install with: pip install SpaGCN")

# Configuration
BASE_DIR = Path(r"D:\HGSOC_Spatial_Atlas")
PROCESSED_DATA_DIR = BASE_DIR / "02_processed_data"
OUTPUT_DIR = BASE_DIR / "05_analysis_ready"

# Default preprocessing parameters
PREPROCESS_PARAMS = {
    # Filtering
    'min_genes_per_spot': 500,
    'min_counts_per_spot': 1000,
    'max_mito_pct': 20,
    'min_cells_per_gene': 3,
    
    # Normalization
    'target_sum': 1e4,  # normalize_total target
    
    # HVG selection
    'n_top_genes': 2000,
    
    # PCA
    'n_pcs': 30,
    
    # Clustering
    'n_neighbors': 15,
    'leiden_resolution': 0.5,
    
    # UMAP
    'umap_min_dist': 0.3,

    # SPaGCN 
    'spagcn_target_clusters': [5, 9],
    'spagcn_p': 0.5,
    'spagcn_seeds': 42,

}

# Known datasets
KNOWN_DATASETS = [
    'stur_2021',
    'denisenko_2022',
    'yamamoto_2025',
    'ju_2024',
    'mcfawn_2024',
    'xu_2024',
    '10x_ov_standard',
    '10x_ov_11mm',
]


def print_header(dataset_id=None):
    """Print script header"""
    print("\n" + "=" * 70)
    print("🧬 HGSOC Spatial Atlas - Preprocessing Pipeline")
    print("=" * 70)
    if dataset_id:
        print(f"📦 Dataset: {dataset_id}")
    print(f"📁 Input: {PROCESSED_DATA_DIR}")
    print(f"📁 Output: {OUTPUT_DIR}")
    print(f"\n⚙️  Parameters:")
    print(f"   Min genes/spot: {PREPROCESS_PARAMS['min_genes_per_spot']}")
    print(f"   Min counts/spot: {PREPROCESS_PARAMS['min_counts_per_spot']}")
    print(f"   Max mito %: {PREPROCESS_PARAMS['max_mito_pct']}")
    print(f"   Min cells/gene: {PREPROCESS_PARAMS['min_cells_per_gene']}")
    print(f"   HVGs: {PREPROCESS_PARAMS['n_top_genes']}")
    print(f"   PCs: {PREPROCESS_PARAMS['n_pcs']}")
    print(f"   n_neighbors: {PREPROCESS_PARAMS['n_neighbors']}")
    print(f"   Leiden resolution: {PREPROCESS_PARAMS['leiden_resolution']}")
    print(f"   SpaGCN targets: {PREPROCESS_PARAMS['spagcn_target_clusters']}")
    print("=" * 70)


def load_custom_genes(gene_file):
    """Load custom gene list from file"""
    if gene_file is None:
        return None
    
    gene_path = Path(gene_file)
    if not gene_path.exists():
        print(f"⚠️  Gene file not found: {gene_file}")
        return None
    
    with open(gene_path, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    
    print(f"📋 Loaded {len(genes)} custom genes from {gene_file}")
    return genes


def find_h5ad_samples(dataset_path):
    """Find all .h5ad files in a dataset directory"""
    samples = []
    
    for h5ad_file in dataset_path.glob("*.h5ad"):
        samples.append({
            'path': str(h5ad_file),
            'name': h5ad_file.stem
        })
    
    return sorted(samples, key=lambda x: x['name'])


def preprocess_single_sample(sample_info, dataset_id, custom_genes=None):
    """
    Run preprocessing pipeline on a single sample.
    
    Steps:
    1. Load data
    2. Store raw counts in layers
    3. Filter spots and genes
    4. Normalize and log-transform
    5. Select HVGs (+ custom genes)
    6. PCA
    7. Clustering (Leiden + SpaGCN)
    8. UMAP
    9. Save and create plots
    """
    sample_path = sample_info['path']
    sample_name = sample_info['name']
    
    print(f"\n{'='*60}")
    print(f"🔬 Processing: {dataset_id} / {sample_name}")
    print(f"{'='*60}")
    
    try:
        # =====================================================================
        # Step 1: Load data
        # =====================================================================
        print(f"\n📂 Loading: {Path(sample_path).name}")
        adata = sc.read_h5ad(sample_path)
        adata.var_names_make_unique()
        
        print(f"   Spots: {adata.n_obs}")
        print(f"   Genes: {adata.n_vars}")
        
        # =====================================================================
        # Step 2: Store raw counts
        # =====================================================================
        if 'counts' not in adata.layers:
            adata.layers['counts'] = adata.X.copy()
            print(f"   ✅ Stored raw counts in adata.layers['counts']")
        else:
            print(f"   ℹ️  Raw counts already in adata.layers['counts']")
            adata.X = adata.layers['counts'].copy()
        
        # =====================================================================
        # Step 3: Calculate QC metrics (if not present)
        # =====================================================================
        if 'pct_counts_mt' not in adata.obs.columns:
            print(f"\n📊 Calculating QC metrics...")
            adata.var['mt'] = adata.var_names.str.startswith('MT-')
            sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
        
        # =====================================================================
        # Step 4: Filtering
        # =====================================================================
        print(f"\n🔧 Filtering...")
        n_spots_before = adata.n_obs
        n_genes_before = adata.n_vars
        
        # Filter spots
        sc.pp.filter_cells(adata, min_genes=PREPROCESS_PARAMS['min_genes_per_spot'])
        sc.pp.filter_cells(adata, min_counts=PREPROCESS_PARAMS['min_counts_per_spot'])
        
        # Filter by mito percentage
        adata = adata[adata.obs['pct_counts_mt'] < PREPROCESS_PARAMS['max_mito_pct'], :].copy()
        
        # Filter genes
        sc.pp.filter_genes(adata, min_cells=PREPROCESS_PARAMS['min_cells_per_gene'])
        
        n_spots_after = adata.n_obs
        n_genes_after = adata.n_vars
        
        print(f"   Spots: {n_spots_before} → {n_spots_after} ({n_spots_before - n_spots_after} removed)")
        print(f"   Genes: {n_genes_before} → {n_genes_after} ({n_genes_before - n_genes_after} removed)")
        
        # =====================================================================
        # Step 5: Normalization
        # =====================================================================
        print(f"\n📐 Normalizing...")
        sc.pp.normalize_total(adata, target_sum=PREPROCESS_PARAMS['target_sum'])
        sc.pp.log1p(adata)
        
        print(f"   ✅ normalize_total (target={PREPROCESS_PARAMS['target_sum']:.0f})")
        print(f"   ✅ log1p transformation")
        
        # =====================================================================
        # Step 6: HVG Selection
        # =====================================================================
        print(f"\n🧬 Selecting highly variable genes...")
        sc.pp.highly_variable_genes(
            adata, 
            n_top_genes=PREPROCESS_PARAMS['n_top_genes'],
            flavor='seurat_v3',
            layer='counts'  # Use raw counts for HVG selection
        )
        
        n_hvg = adata.var['highly_variable'].sum()
        print(f"   ✅ {n_hvg} HVGs selected")
        
        # Add custom genes if provided
        if custom_genes:
            genes_found = [g for g in custom_genes if g in adata.var_names]
            genes_not_found = [g for g in custom_genes if g not in adata.var_names]
            
            if genes_found:
                adata.var.loc[genes_found, 'highly_variable'] = True
                n_hvg_after = adata.var['highly_variable'].sum()
                print(f"   ✅ Added {len(genes_found)} custom genes → {n_hvg_after} total HVGs")
            
            if genes_not_found:
                print(f"   ⚠️  {len(genes_not_found)} custom genes not found in data")
        
        # =====================================================================
        # Step 7: PCA
        # =====================================================================
        print(f"\n📉 Running PCA...")
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, n_comps=PREPROCESS_PARAMS['n_pcs'], use_highly_variable=True)
        
        # Calculate variance explained
        var_explained = adata.uns['pca']['variance_ratio']
        cumvar = np.cumsum(var_explained)
        
        print(f"   ✅ {PREPROCESS_PARAMS['n_pcs']} PCs computed")
        print(f"   Variance explained (PC1-5): {var_explained[:5].sum()*100:.1f}%")
        print(f"   Variance explained (all): {cumvar[-1]*100:.1f}%")
        
        # =====================================================================
        # Step 8: Neighbors and Clustering
        # =====================================================================
        print(f"\n🔗 Building neighborhood graph...")
        sc.pp.neighbors(
            adata, 
            n_neighbors=PREPROCESS_PARAMS['n_neighbors'], 
            n_pcs=PREPROCESS_PARAMS['n_pcs']
        )
        
        print(f"\n🎯 Leiden clustering...")
        sc.tl.leiden(adata, resolution=PREPROCESS_PARAMS['leiden_resolution'], key_added='leiden')
        n_clusters = adata.obs['leiden'].nunique()
        print(f"   ✅ {n_clusters} clusters found (resolution={PREPROCESS_PARAMS['leiden_resolution']})")
        
        # SpaGCN clustering (if available)
        if SPAGCN_AVAILABLE:
            print(f"\n🌐 SpaGCN spatial clustering...")
            try:
                adata = run_spagcn_clustering(adata)
            except Exception as e:
                print(f"   ⚠️  SpaGCN failed: {e}")
        
        # =====================================================================
        # Step 9: UMAP
        # =====================================================================
        print(f"\n🗺️  Computing UMAP...")
        sc.tl.umap(adata, min_dist=PREPROCESS_PARAMS['umap_min_dist'])
        print(f"   ✅ UMAP computed")
        
        # =====================================================================
        # Step 10: Save and Plot
        # =====================================================================
        print(f"\n💾 Saving results...")
        
        # Save h5ad (overwrite)
        adata.write_h5ad(sample_path)
        print(f"   ✅ Saved: {sample_path}")
        
        # Create output directory for plots
        sample_output_dir = OUTPUT_DIR / dataset_id / sample_name
        sample_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate plots
        create_preprocessing_plots(adata, dataset_id, sample_name, sample_output_dir)
        
        # Create summary
        summary = {
            'dataset': dataset_id,
            'sample_id': sample_name,
            'n_spots_raw': n_spots_before,
            'n_spots_filtered': n_spots_after,
            'n_genes_raw': n_genes_before,
            'n_genes_filtered': n_genes_after,
            'n_hvgs': adata.var['highly_variable'].sum(),
            'n_pcs': PREPROCESS_PARAMS['n_pcs'],
            'n_neighbors': PREPROCESS_PARAMS['n_neighbors'],
            'n_leiden_clusters': n_clusters,
            'leiden_resolution': PREPROCESS_PARAMS['leiden_resolution'],
            'processed_date': datetime.now().isoformat()
        }
        
        if SPAGCN_AVAILABLE:
            for n_target in PREPROCESS_PARAMS['spagcn_target_clusters']:
                key = f'spagcn_{n_target}_refined'
                if key in adata.obs.columns:
                    summary[f'n_{key}'] = adata.obs[key].nunique()

        print(f"\n✅ Preprocessing complete: {sample_name}")
        
        return summary
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_spagcn_clustering(adata):
    """
    Run SpaGCN spatial clustering with histology integration.
    Requires spatial coordinates in adata.obsm['spatial'].
    """
    import SpaGCN as spg
    
    # Check prerequisites
    if 'spatial' not in adata.obsm:
        raise ValueError("No spatial coordinates found in adata.obsm['spatial']")
    if 'counts' not in adata.layers:
        raise ValueError("Raw counts not found in adata.layers['counts']")
    
    # === 1. Separate preprocessing path ===
    adata_spagcn = adata.copy()
    adata_spagcn.X = adata_spagcn.layers['counts'].copy()
    
    spg.prefilter_genes(adata_spagcn, min_cells=3)
    spg.prefilter_specialgenes(adata_spagcn)
    sc.pp.normalize_total(adata_spagcn)
    sc.pp.log1p(adata_spagcn)
    
    # Convert to dense (SpaGCN compatibility)
    if issparse(adata_spagcn.X):
        adata_spagcn.X = adata_spagcn.X.toarray()
    
    print(f"   Genes after SpaGCN filtering: {adata_spagcn.n_vars}")
    
    # === 2. Get coordinates ===
    x_pixel = adata_spagcn.obsm['spatial'][:, 0]
    y_pixel = adata_spagcn.obsm['spatial'][:, 1]
    x_array = adata_spagcn.obs['array_row'].values
    y_array = adata_spagcn.obs['array_col'].values
    
    # === 3. Calculate adjacency matrix with histology ===
    sample_key = list(adata_spagcn.uns['spatial'].keys())[0]
    
    if 'hires' in adata_spagcn.uns['spatial'][sample_key]['images']:
        print("   Using histology image...")
        img = adata_spagcn.uns['spatial'][sample_key]['images']['hires']
        
        # Scale coordinates to match hires image dimensions
        scale = adata_spagcn.uns['spatial'][sample_key]['scalefactors']['tissue_hires_scalef']
        x_pixel_scaled = (x_pixel * scale).astype(int)
        y_pixel_scaled = (y_pixel * scale).astype(int)
        
        # Convert image: scanpy stores as RGB float 0-1, SpaGCN expects BGR uint8 0-255
        img_for_spagcn = (img * 255).astype(np.uint8)
        img_for_spagcn = img_for_spagcn[..., ::-1].copy()  # RGB -> BGR
        
        s = 1   # weight for histology
        b = 49  # area for color extraction
        adj = spg.calculate_adj_matrix(
            x=x_pixel_scaled, y=y_pixel_scaled,
            x_pixel=x_pixel_scaled, y_pixel=y_pixel_scaled,
            image=img_for_spagcn, beta=b, alpha=s, histology=True
        )
    else:
        print("   No histology image, using coordinates only...")
        adj = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, histology=False)
    
    # === 4. Find l parameter ===
    p = PREPROCESS_PARAMS['spagcn_p']
    l = spg.search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)
    print(f"   Optimal l: {l}")
    
    # === 5. Adjacency for refinement (uses array coordinates) ===
    adj_2d = spg.calculate_adj_matrix(x=x_array, y=y_array, histology=False)
    
    # === 6. Loop through target cluster counts ===
    seed = PREPROCESS_PARAMS['spagcn_seeds']
    
    for n_target in PREPROCESS_PARAMS['spagcn_target_clusters']:
        print(f"   Finding resolution for {n_target} clusters...")
        
        # Find resolution for target cluster count
        res = spg.search_res(
            adata_spagcn, adj, l, n_target,
            start=0.3, step=0.1, tol=5e-3, lr=0.05, max_epochs=25,
            r_seed=seed, t_seed=seed, n_seed=seed
        )
        print(f"   Resolution for {n_target} clusters: {res}")
        
        # Set seeds for reproducibility
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Train SpaGCN
        clf = spg.SpaGCN()
        clf.set_l(l)
        clf.train(
            adata_spagcn, adj,
            init_spa=True, init="louvain", res=res,
            tol=5e-3, lr=0.05, max_epochs=200
        )
        
        y_pred, prob = clf.predict()
        
        # Refinement
        refined_pred = spg.refine(
            sample_id=adata_spagcn.obs.index.tolist(),
            pred=y_pred.tolist(),
            dis=adj_2d,
            shape="hexagon"
        )
        
        # Store results in original adata
        key_unrefined = f'spagcn_{n_target}'
        key_refined = f'spagcn_{n_target}_refined'
        
        adata.obs[key_unrefined] = y_pred
        adata.obs[key_unrefined] = adata.obs[key_unrefined].astype('category')
        adata.obs[key_refined] = refined_pred
        adata.obs[key_refined] = adata.obs[key_refined].astype('category')
        
        print(f"   ✅ spagcn_{n_target}: {adata.obs[key_unrefined].nunique()} domains")
        print(f"   ✅ spagcn_{n_target}_refined: {adata.obs[key_refined].nunique()} domains")
    
    return adata


def create_preprocessing_plots(adata, dataset_id, sample_name, output_dir):
    """Create visualization plots for preprocessed data"""
    
    print(f"   📊 Creating plots...")
    
    # =========================================================================
    # Plot 1: Elbow plot (variance explained)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    
    var_ratio = adata.uns['pca']['variance_ratio']
    cumvar = np.cumsum(var_ratio)
    
    ax.bar(range(1, len(var_ratio) + 1), var_ratio, alpha=0.7, label='Individual')
    ax.plot(range(1, len(var_ratio) + 1), cumvar, 'ro-', label='Cumulative')
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
    
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained')
    ax.set_title(f'PCA Elbow Plot: {sample_name}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'elbow_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Plot 2: HVG plot
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sc.pl.highly_variable_genes(adata, show=False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hvg_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Plot 3: UMAP with Leiden clusters
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 7))
    
    sc.pl.umap(adata, color='leiden', ax=ax, show=False, title=f'UMAP - Leiden Clusters: {sample_name}')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'umap_leiden.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Plot 4: Domain plots (Leiden + SpaGCN) in domains/ subdirectory
    # =========================================================================
    domains_dir = output_dir / 'domains'
    domains_dir.mkdir(parents=True, exist_ok=True)
    
    # Leiden spatial
    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        sc.pl.spatial(adata, color='leiden', ax=ax, show=False, size=0.8,
                      title=f'Leiden ({adata.obs["leiden"].nunique()} clusters)')
        plt.tight_layout()
        plt.savefig(domains_dir / 'leiden.png', dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"   ⚠️  Leiden spatial plot failed: {e}")
    
    # SpaGCN refined plots
    for n_target in PREPROCESS_PARAMS['spagcn_target_clusters']:
        key = f'spagcn_{n_target}_refined'
        if key in adata.obs.columns:
            try:
                fig, ax = plt.subplots(figsize=(6, 6))
                sc.pl.spatial(adata, color=key, ax=ax, show=False, size=0.8,
                              title=f'SpaGCN {n_target} refined ({adata.obs[key].nunique()} domains)')
                plt.tight_layout()
                plt.savefig(domains_dir / f'{key}.png', dpi=150, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"   ⚠️  {key} plot failed: {e}")
    
    print(f"   ✅ Plots saved to: {output_dir}")

def preprocess_dataset(dataset_id, sample_name=None, custom_genes=None):
    """Preprocess all samples in a dataset"""
    print(f"\n{'='*70}")
    print(f"📊 Preprocessing: {dataset_id}")
    print(f"{'='*70}")
    
    dataset_path = PROCESSED_DATA_DIR / dataset_id
    
    if not dataset_path.exists():
        print(f"   ❌ Dataset not found: {dataset_path}")
        return []
    
    # Find samples
    samples = find_h5ad_samples(dataset_path)
    
    if not samples:
        print(f"   ⚠️  No .h5ad files found")
        return []
    
    # Filter to specific sample if requested
    if sample_name:
        samples = [s for s in samples if s['name'] == sample_name]
        if not samples:
            print(f"   ❌ Sample not found: {sample_name}")
            return []
    
    print(f"   Found {len(samples)} sample(s)")
    
    # Process each sample
    all_summaries = []
    for sample in samples:
        summary = preprocess_single_sample(sample, dataset_id, custom_genes)
        if summary:
            all_summaries.append(summary)
    
    # Save dataset summary
    if all_summaries:
        dataset_output_dir = OUTPUT_DIR / dataset_id
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(all_summaries)
        summary_file = dataset_output_dir / 'preprocessing_summary.csv'
        df.to_csv(summary_file, index=False)
        print(f"\n📄 Summary saved: {summary_file}")
    
    return all_summaries


def preprocess_all(custom_genes=None):
    """Preprocess all known datasets"""
    all_results = []
    
    for dataset_id in KNOWN_DATASETS:
        dataset_path = PROCESSED_DATA_DIR / dataset_id
        if dataset_path.exists():
            results = preprocess_dataset(dataset_id, custom_genes=custom_genes)
            all_results.extend(results)
    
    # Create master summary
    if all_results:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(all_results)
        master_file = OUTPUT_DIR / 'master_preprocessing_summary.csv'
        df.to_csv(master_file, index=False)
        print(f"\n📄 Master summary saved: {master_file}")
    
    return all_results


def list_status():
    """List preprocessing status for all datasets"""
    print(f"\n{'Dataset':<30} {'Raw h5ad':<12} {'Preprocessed':<15} {'Samples':<8}")
    print("-" * 70)
    
    for dataset_id in KNOWN_DATASETS:
        dataset_path = PROCESSED_DATA_DIR / dataset_id
        output_path = OUTPUT_DIR / dataset_id
        
        # Check raw data
        if dataset_path.exists():
            h5ad_files = list(dataset_path.glob("*.h5ad"))
            raw_status = "✅" if h5ad_files else "❌"
            n_samples = len(h5ad_files)
        else:
            raw_status = "❌"
            n_samples = 0
        
        # Check preprocessing output
        if output_path.exists() and (output_path / 'preprocessing_summary.csv').exists():
            preproc_status = "✅"
        else:
            preproc_status = "❌" if n_samples > 0 else "-"
        
        print(f"{dataset_id:<30} {raw_status:<12} {preproc_status:<15} {n_samples:<8}")


def main():
    """Main entry point"""
    if not PACKAGES_AVAILABLE:
        sys.exit(1)
    
    parser = argparse.ArgumentParser(
        description='Preprocessing pipeline for HGSOC spatial transcriptomics data'
    )
    parser.add_argument(
        'dataset',
        nargs='?',
        help='Dataset ID (e.g., visium/yamamoto_2025)'
    )
    parser.add_argument(
        '--sample',
        '-s',
        help='Process specific sample only'
    )
    parser.add_argument(
        '--genes',
        '-g',
        help='Path to custom gene list file (one gene per line)'
    )
    parser.add_argument(
        '--all',
        '-a',
        action='store_true',
        help='Process all datasets'
    )
    parser.add_argument(
        '--list',
        '-l',
        action='store_true',
        help='List dataset status'
    )
    # Parameters that can be overridden via config.yaml / main.py
    parser.add_argument(
        '--n-top-genes',
        type=int,
        default=PREPROCESS_PARAMS['n_top_genes'],
        help='Number of highly variable genes to select'
    )
    parser.add_argument(
        '--n-pcs',
        type=int,
        default=PREPROCESS_PARAMS['n_pcs'],
        help='Number of principal components to use'
    )
    parser.add_argument(
        '--n-neighbors',
        type=int,
        default=PREPROCESS_PARAMS['n_neighbors'],
        help='Number of neighbors for graph construction'
    )
    parser.add_argument(
        '--leiden-resolution',
        type=float,
        default=PREPROCESS_PARAMS['leiden_resolution'],
        help='Resolution for Leiden clustering'
    )
    
    args = parser.parse_args()
    
    # Override defaults with command-line args
    PREPROCESS_PARAMS['n_top_genes'] = args.n_top_genes
    PREPROCESS_PARAMS['n_pcs'] = args.n_pcs
    PREPROCESS_PARAMS['n_neighbors'] = args.n_neighbors
    PREPROCESS_PARAMS['leiden_resolution'] = args.leiden_resolution
    
    # Load custom genes if provided
    custom_genes = load_custom_genes(args.genes) if args.genes else None
    
    if args.list:
        print_header()
        list_status()
    
    elif args.all:
        print_header()
        preprocess_all(custom_genes=custom_genes)
    
    elif args.dataset:
        print_header(args.dataset)
        preprocess_dataset(args.dataset, sample_name=args.sample, custom_genes=custom_genes)
    
    else:
        print_header()
        print("\nUsage:")
        print("  python preprocessing.py <platform/dataset_id>")
        print("  python preprocessing.py visium/yamamoto_2025")
        print("  python preprocessing.py visium/yamamoto_2025 --sample Pt1-1")
        print("  python preprocessing.py visium/yamamoto_2025 --genes snai1_signature.txt")
        print("  python preprocessing.py visium/yamamoto_2025 --n-pcs 10 --leiden-resolution 0.8")
        print("  python preprocessing.py --all")
        print("  python preprocessing.py --list")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print("✅ PREPROCESSING PIPELINE COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()