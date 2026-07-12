"""
Quality Control Pipeline for HGSOC Spatial Transcriptomics Data.

Usage:
    python run_qc_pipeline.py <platform/dataset_id>
    python run_qc_pipeline.py visium/yamamoto_2025
    python run_qc_pipeline.py --all
    python run_qc_pipeline.py --list

Loads processed .h5ad files from 02_processed_data/
"""

import os
import sys
from pathlib import Path

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

# Configuration
BASE_DIR = Path(r"D:\HGSOC_Spatial_Atlas")
PROCESSED_DATA_DIR = BASE_DIR / "02_processed_data"
QC_OUTPUT_DIR = BASE_DIR / "04_quality_control"

# QC thresholds
QC_THRESHOLDS = {
    'min_genes_per_spot': 500,
    'min_counts_per_spot': 1000,
    'max_mito_pct': 20
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
    print("🧬 HGSOC Spatial Atlas - QC Pipeline")
    print("=" * 70)
    if dataset_id:
        print(f"📦 Dataset: {dataset_id}")
    print(f"📁 Processed data: {PROCESSED_DATA_DIR}")
    print(f"📁 QC output: {QC_OUTPUT_DIR}")
    print(f"\n⚙️  Thresholds:")
    print(f"   Min genes/spot: {QC_THRESHOLDS['min_genes_per_spot']}")
    print(f"   Min counts/spot: {QC_THRESHOLDS['min_counts_per_spot']}")
    print(f"   Max mito %: {QC_THRESHOLDS['max_mito_pct']}")
    print("=" * 70)


def find_h5ad_samples(dataset_path):
    """Find all .h5ad files in a dataset directory"""
    samples = []
    
    for h5ad_file in dataset_path.glob("*.h5ad"):
        samples.append({
            'path': str(h5ad_file),
            'name': h5ad_file.stem
        })
    
    return sorted(samples, key=lambda x: x['name'])


def run_qc_single_sample(sample_info, dataset_name):
    """Run QC on a single sample"""
    sample_path = sample_info['path']
    sample_name = sample_info['name']
    
    print(f"\n🔬 QC: {dataset_name} / {sample_name}")
    print(f"   📁 File: {Path(sample_path).name}")
    
    try:
        # Load h5ad file
        adata = sc.read_h5ad(sample_path)
        adata.var_names_make_unique()
        
        # Calculate QC metrics
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
        
        # Extract metrics
        metrics = {
            'dataset': dataset_name,
            'sample_id': sample_name,
            'n_spots': adata.n_obs,
            'n_genes': adata.n_vars,
            'mean_counts_per_spot': adata.obs['total_counts'].mean(),
            'median_counts_per_spot': adata.obs['total_counts'].median(),
            'mean_genes_per_spot': adata.obs['n_genes_by_counts'].mean(),
            'median_genes_per_spot': adata.obs['n_genes_by_counts'].median(),
            'mean_mito_pct': adata.obs['pct_counts_mt'].mean(),
            'median_mito_pct': adata.obs['pct_counts_mt'].median()
        }
        
        # QC pass/fail
        pass_genes = metrics['mean_genes_per_spot'] >= QC_THRESHOLDS['min_genes_per_spot']
        pass_counts = metrics['mean_counts_per_spot'] >= QC_THRESHOLDS['min_counts_per_spot']
        pass_mito = metrics['mean_mito_pct'] <= QC_THRESHOLDS['max_mito_pct']
        
        metrics['qc_status'] = 'PASS' if (pass_genes and pass_counts and pass_mito) else 'CHECK'
        
        # Create QC plots
        create_qc_plots(adata, dataset_name, sample_name)
        
        # Print summary
        print(f"   ✅ Spots: {metrics['n_spots']}")
        print(f"   ✅ Genes: {metrics['n_genes']}")
        print(f"   ✅ Mean genes/spot: {metrics['mean_genes_per_spot']:.0f}")
        print(f"   ✅ Mean counts/spot: {metrics['mean_counts_per_spot']:.0f}")
        print(f"   ✅ Mean mito %: {metrics['mean_mito_pct']:.1f}")
        print(f"   📊 QC Status: {metrics['qc_status']}")
        
        return metrics
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_qc_plots(adata, dataset_name, sample_name):
    """Create QC visualization plots"""
    
    # Create sample-specific output directory
    sample_qc_dir = QC_OUTPUT_DIR / dataset_name / sample_name
    sample_qc_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: QC metrics distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'QC Metrics: {dataset_name} / {sample_name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Total counts distribution
    adata.obs['total_counts'].hist(bins=50, ax=axes[0, 0], color='steelblue', edgecolor='black')
    axes[0, 0].axvline(QC_THRESHOLDS['min_counts_per_spot'], color='red', linestyle='--', label='Threshold')
    axes[0, 0].set_xlabel('Total UMI Counts')
    axes[0, 0].set_ylabel('Number of Spots')
    axes[0, 0].set_title('UMI Count Distribution')
    axes[0, 0].legend()
    
    # Plot 2: Genes detected distribution
    adata.obs['n_genes_by_counts'].hist(bins=50, ax=axes[0, 1], color='forestgreen', edgecolor='black')
    axes[0, 1].axvline(QC_THRESHOLDS['min_genes_per_spot'], color='red', linestyle='--', label='Threshold')
    axes[0, 1].set_xlabel('Genes Detected')
    axes[0, 1].set_ylabel('Number of Spots')
    axes[0, 1].set_title('Genes Detected Distribution')
    axes[0, 1].legend()
    
    # Plot 3: Mitochondrial percentage
    adata.obs['pct_counts_mt'].hist(bins=50, ax=axes[0, 2], color='coral', edgecolor='black')
    axes[0, 2].axvline(QC_THRESHOLDS['max_mito_pct'], color='red', linestyle='--', label='Threshold')
    axes[0, 2].set_xlabel('Mitochondrial %')
    axes[0, 2].set_ylabel('Number of Spots')
    axes[0, 2].set_title('Mitochondrial % Distribution')
    axes[0, 2].legend()
    
    # Plot 4: Spatial - Total Counts
    try:
        sc.pl.spatial(adata, color='total_counts', size=1.5, ax=axes[1, 0], show=False, 
                      title='Total Counts (Spatial)', cmap='viridis')
    except Exception as e:
        axes[1, 0].text(0.5, 0.5, f'Spatial plot failed:\n{e}', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Total Counts (Spatial)')
    
    # Plot 5: Spatial - Genes Detected
    try:
        sc.pl.spatial(adata, color='n_genes_by_counts', size=1.5, ax=axes[1, 1], show=False,
                      title='Genes Detected (Spatial)', cmap='viridis')
    except Exception as e:
        axes[1, 1].text(0.5, 0.5, f'Spatial plot failed:\n{e}', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Genes Detected (Spatial)')
    
    # Plot 6: Spatial - Mitochondrial %
    try:
        sc.pl.spatial(adata, color='pct_counts_mt', size=1.5, ax=axes[1, 2], show=False,
                      title='Mitochondrial % (Spatial)', cmap='RdYlBu_r')
    except Exception as e:
        axes[1, 2].text(0.5, 0.5, f'Spatial plot failed:\n{e}', ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Mitochondrial % (Spatial)')
    
    plt.tight_layout()
    plt.savefig(sample_qc_dir / 'qc_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'QC Scatter Plots: {dataset_name} / {sample_name}', fontsize=14, fontweight='bold')
    
    # Counts vs Genes
    axes[0].scatter(adata.obs['total_counts'], adata.obs['n_genes_by_counts'], 
                   alpha=0.3, s=5, c='steelblue')
    axes[0].set_xlabel('Total Counts')
    axes[0].set_ylabel('Genes Detected')
    axes[0].set_title('Counts vs Genes')
    
    # Counts vs Mito%
    axes[1].scatter(adata.obs['total_counts'], adata.obs['pct_counts_mt'],
                   alpha=0.3, s=5, c='coral')
    axes[1].set_xlabel('Total Counts')
    axes[1].set_ylabel('Mitochondrial %')
    axes[1].set_title('Counts vs Mitochondrial %')
    
    plt.tight_layout()
    plt.savefig(sample_qc_dir / 'qc_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 Plots saved: {sample_qc_dir}")


def run_qc_dataset(dataset_id):
    """Run QC on all samples in a dataset"""
    print(f"\n{'='*70}")
    print(f"📊 Running QC: {dataset_id}")
    print(f"{'='*70}")
    
    dataset_path = PROCESSED_DATA_DIR / dataset_id
    
    if not dataset_path.exists():
        print(f"   ❌ Dataset not found: {dataset_path}")
        print(f"   💡 Run: python organize_visium_data.py {dataset_id}")
        return []
    
    # Find all h5ad files
    samples = find_h5ad_samples(dataset_path)
    
    if not samples:
        print(f"   ⚠️  No .h5ad files found")
        print(f"   💡 Run: python organize_visium_data.py {dataset_id}")
        return []
    
    print(f"   Found {len(samples)} sample(s)")
    
    # Run QC on each sample
    all_metrics = []
    for sample in samples:
        metrics = run_qc_single_sample(sample, dataset_id)
        if metrics:
            all_metrics.append(metrics)
    
    # Save dataset-level summary
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        
        # Create directory with forward slashes replaced
        output_subdir = QC_OUTPUT_DIR / dataset_id
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_subdir / 'qc_summary.csv'
        df.to_csv(output_file, index=False)
        print(f"\n   📄 Summary saved: {output_file}")
    
    return all_metrics


def run_qc_all():
    """Run QC on all datasets and create master report"""
    all_results = []
    
    for dataset_id in KNOWN_DATASETS:
        dataset_path = PROCESSED_DATA_DIR / dataset_id
        if dataset_path.exists():
            results = run_qc_dataset(dataset_id)
            all_results.extend(results)
    
    # Create master report
    if all_results:
        create_master_report(all_results)
    else:
        print("\n⚠️  No QC results to report")
    
    return all_results


def create_master_report(all_results):
    """Create a master QC report across all datasets"""
    
    df = pd.DataFrame(all_results)
    
    # Save master CSV
    QC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    master_file = QC_OUTPUT_DIR / 'master_qc_report.csv'
    df.to_csv(master_file, index=False)
    
    # Create summary figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('HGSOC Spatial Atlas - Master QC Report', fontsize=16, fontweight='bold')
    
    # Plot 1: Mean genes per spot by dataset
    df.boxplot(column='mean_genes_per_spot', by='dataset', ax=axes[0, 0])
    axes[0, 0].axhline(QC_THRESHOLDS['min_genes_per_spot'], color='red', linestyle='--')
    axes[0, 0].set_xlabel('Dataset')
    axes[0, 0].set_ylabel('Mean Genes per Spot')
    axes[0, 0].set_title('Mean Genes per Spot')
    plt.sca(axes[0, 0])
    plt.xticks(rotation=45, ha='right')
    
    # Plot 2: Mean counts per spot by dataset
    df.boxplot(column='mean_counts_per_spot', by='dataset', ax=axes[0, 1])
    axes[0, 1].axhline(QC_THRESHOLDS['min_counts_per_spot'], color='red', linestyle='--')
    axes[0, 1].set_xlabel('Dataset')
    axes[0, 1].set_ylabel('Mean Counts per Spot')
    axes[0, 1].set_title('Mean Counts per Spot')
    plt.sca(axes[0, 1])
    plt.xticks(rotation=45, ha='right')
    
    # Plot 3: Mitochondrial % by dataset
    df.boxplot(column='mean_mito_pct', by='dataset', ax=axes[1, 0])
    axes[1, 0].axhline(QC_THRESHOLDS['max_mito_pct'], color='red', linestyle='--')
    axes[1, 0].set_xlabel('Dataset')
    axes[1, 0].set_ylabel('Mean Mitochondrial %')
    axes[1, 0].set_title('Mean Mitochondrial %')
    plt.sca(axes[1, 0])
    plt.xticks(rotation=45, ha='right')
    
    # Plot 4: QC status counts
    qc_counts = df['qc_status'].value_counts()
    colors = ['green' if x == 'PASS' else 'orange' for x in qc_counts.index]
    axes[1, 1].bar(qc_counts.index, qc_counts.values, color=colors)
    axes[1, 1].set_xlabel('QC Status')
    axes[1, 1].set_ylabel('Number of Samples')
    axes[1, 1].set_title('QC Status Summary')
    
    plt.tight_layout()
    plt.savefig(QC_OUTPUT_DIR / 'master_qc_report.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print(f"\n{'='*70}")
    print("📊 MASTER QC SUMMARY")
    print(f"{'='*70}")
    print(f"Total samples: {len(df)}")
    print(f"PASS: {(df['qc_status'] == 'PASS').sum()}")
    print(f"CHECK: {(df['qc_status'] == 'CHECK').sum()}")
    print(f"\n✅ Master report: {master_file}")


def update_master_report():
    """Update master report by combining all dataset summaries"""
    all_results = []
    
    for csv_file in QC_OUTPUT_DIR.rglob("qc_summary.csv"):
        if csv_file.name != 'master_qc_report.csv':
            df = pd.read_csv(csv_file)
            all_results.extend(df.to_dict('records'))
    
    if all_results:
        create_master_report(all_results)
    else:
        print("⚠️  No dataset summaries found to combine")


def list_status():
    """List QC status for all datasets"""
    print(f"\n{'Dataset':<30} {'Processed':<12} {'QC Done':<10} {'Samples':<8}")
    print("-" * 65)
    
    for dataset_id in KNOWN_DATASETS:
        processed_path = PROCESSED_DATA_DIR / dataset_id
        qc_summary = QC_OUTPUT_DIR / dataset_id / 'qc_summary.csv'
        
        # Check processed data
        if processed_path.exists():
            h5ad_files = list(processed_path.glob("*.h5ad"))
            processed_status = "✅" if h5ad_files else "❌"
            n_processed = len(h5ad_files)
        else:
            processed_status = "❌"
            n_processed = 0
        
        # Check QC status
        if qc_summary.exists():
            qc_status = "✅"
        else:
            qc_status = "❌" if n_processed > 0 else "-"
        
        print(f"{dataset_id:<30} {processed_status:<12} {qc_status:<10} {n_processed:<8}")


def main():
    """Main entry point"""
    if not PACKAGES_AVAILABLE:
        sys.exit(1)
    
    QC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if len(sys.argv) < 2:
        print_header()
        print("\nUsage:")
        print("  python run_qc_pipeline.py <platform/dataset_id>")
        print("  python run_qc_pipeline.py --all")
        print("  python run_qc_pipeline.py --list")
        print("  python run_qc_pipeline.py --update")
        print("\nExamples:")
        print("  python run_qc_pipeline.py visium/yamamoto_2025")
        print("  python run_qc_pipeline.py --all")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == '--all' or command == '-a':
        print_header()
        run_qc_all()
    
    elif command == '--list' or command == '-l':
        print_header()
        list_status()
    
    elif command == '--update' or command == '-u':
        print_header()
        update_master_report()
    
    elif command == '--help' or command == '-h':
        print_header()
        print("\nUsage:")
        print("  python run_qc_pipeline.py <platform/dataset_id>")
        print("  python run_qc_pipeline.py --all")
        print("  python run_qc_pipeline.py --list")
        print("  python run_qc_pipeline.py --update")
    
    else:
        dataset_id = command
        print_header(dataset_id)
        results = run_qc_dataset(dataset_id)
        
        if results:
            print("\n💡 Updating master report...")
            update_master_report()
    
    print(f"\n{'='*70}")
    print("✅ QC PIPELINE COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()