# organize_visium.py
"""
Convert organized 10X folder structure to h5ad format.

This script is simple because organize_for_spacet.py does the heavy lifting.

Usage:
    python organize_visium.py visium/yamamoto_2025
    python organize_visium.py --all
    python organize_visium.py --list

Input:  01_raw_data/visium/<dataset>/organized/<sample>/
Output: 02_processed_data/visium/<dataset>/<sample>.h5ad
"""

import sys
from pathlib import Path

# Check for required packages
try:
    import scanpy as sc
    SCANPY_AVAILABLE = True
except ImportError:
    print("❌ scanpy not installed. Run: pip install scanpy")
    SCANPY_AVAILABLE = False

# Configuration
BASE_DIR = Path(r"D:\HGSOC_Spatial_Atlas")
RAW_DIR = BASE_DIR / "01_raw_data"
PROCESSED_DIR = BASE_DIR / "02_processed_data"

# Known datasets
KNOWN_DATASETS = [
    "visium/yamamoto_2025",
    "visium/ju_2024",
    "visium/denisenko_2022",
    "visium/10X_ov_standard",
    "visium/10X_ov_11mm",
]


def print_header(dataset_id=None):
    """Print script header."""
    print("\n" + "=" * 70)
    print("🧬 Visium → h5ad Converter")
    print("=" * 70)
    if dataset_id:
        print(f"📂 Dataset: {dataset_id}")
    print(f"📁 Input: {RAW_DIR}/<dataset>/organized/")
    print(f"📁 Output: {PROCESSED_DIR}")
    print("=" * 70)


def convert_sample(sample_dir, output_path):
    """Convert a single sample to h5ad using MTX files + manual spatial setup."""
    import json
    import numpy as np
    import pandas as pd
    from PIL import Image
    
    sample_name = sample_dir.name
    print(f"\n📬 Converting: {sample_name}")
    
    # Check required directories
    matrix_dir = sample_dir / "filtered_feature_bc_matrix"
    spatial_dir = sample_dir / "spatial"
    
    if not matrix_dir.exists():
        print(f"   ❌ Missing: filtered_feature_bc_matrix/")
        return False
    
    if not spatial_dir.exists():
        print(f"   ❌ Missing: spatial/")
        return False
    
    try:
        # Step 1: Read expression matrix (MTX format)
        matrix_file = matrix_dir / "matrix.mtx.gz"
        with open(matrix_file, 'rb') as f:
            header = f.read(4)
            if header[:3] == b'\x89HD':
                # It's an HDF5 file
                adata = sc.read_10x_h5(matrix_file)
            else:
                print("   📥 Reading MTX format")
                adata = sc.read_10x_mtx(matrix_dir)
                
        adata.var_names_make_unique()
        
        # Step 2: Load tissue positions
        positions_file = spatial_dir / "tissue_positions_list.csv"
        if not positions_file.exists():
            positions_file = spatial_dir / "tissue_positions.csv"
        
        if not positions_file.exists():
            print(f"   ❌ Missing: tissue_positions file")
            return False
        
        # Try reading with/without header
        try:
            positions = pd.read_csv(positions_file, header=None)
            positions.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row', 'pxl_col']
        except:
            positions = pd.read_csv(positions_file, index_col=0)
            positions = positions.reset_index()
            positions.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row', 'pxl_col']
        
        positions = positions.set_index('barcode')
        
        # Step 3: Match barcodes
        common_barcodes = adata.obs_names.intersection(positions.index)
        n_matched = len(common_barcodes)
        
        if n_matched == 0:
            print(f"   ❌ No barcode matches!")
            return False
        
        adata = adata[common_barcodes].copy()
        positions = positions.loc[common_barcodes]
        
        # Step 4: Add spatial coordinates to adata
        adata.obs['in_tissue'] = positions['in_tissue'].astype(int).values
        adata.obs['array_row'] = positions['array_row'].astype(int).values
        adata.obs['array_col'] = positions['array_col'].astype(int).values
        adata.obsm['spatial'] = positions[['pxl_col', 'pxl_row']].values.astype(float)
        
        # Step 5: Load scale factors
        scalefactors_file = spatial_dir / "scalefactors_json.json"
        with open(scalefactors_file, 'r') as f:
            scalefactors = json.load(f)
        
        # Step 6: Load images
        images = {}
        for img_name, img_key in [('tissue_hires_image.png', 'hires'), 
                                   ('tissue_lowres_image.png', 'lowres')]:
            img_path = spatial_dir / img_name
            if img_path.exists():
                images[img_key] = np.array(Image.open(img_path))
        
        # Step 7: Set up spatial structure (matches sc.read_visium format)
        adata.uns['spatial'] = {
            sample_name: {
                'images': images,
                'scalefactors': scalefactors,
                'metadata': {}
            }
        }
        
        # Filter to in-tissue spots only
        adata = adata[adata.obs['in_tissue'] == 1].copy()
        
        print(f"   📊 Spots: {adata.n_obs}")
        print(f"   🧬 Genes: {adata.n_vars}")
        
        # Save
        output_file = output_path / f"{sample_name}.h5ad"
        adata.write_h5ad(output_file)
        
        print(f"   ✅ Saved: {output_file.name}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def convert_dataset(dataset_id):
    """Convert all samples in a dataset to h5ad."""
    print(f"\n{'='*70}")
    print(f"📊 Converting: {dataset_id}")
    print(f"{'='*70}")
    
    organized_path = RAW_DIR / dataset_id / "organized"
    output_path = PROCESSED_DIR / dataset_id
    
    if not organized_path.exists():
        print(f"❌ Organized data not found: {organized_path}")
        print(f"   Run first: python organize_for_spacet.py {dataset_id}")
        return False
    
    # Find samples
    samples = [d for d in organized_path.iterdir() if d.is_dir() and not d.name.startswith('_')]
    
    if not samples:
        print(f"❌ No samples found in {organized_path}")
        return False
    
    print(f"🔍 Found {len(samples)} sample(s)")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert each sample
    successful = 0
    for sample_dir in sorted(samples):
        if convert_sample(sample_dir, output_path):
            successful += 1
    
    # Summary
    print(f"\n{'='*70}")
    print(f"✅ CONVERSION COMPLETE")
    print(f"{'='*70}")
    print(f"📊 Converted: {successful}/{len(samples)} samples")
    print(f"📁 Output: {output_path}")
    
    return successful > 0


def convert_all():
    """Convert all known datasets."""
    results = []
    
    for dataset_id in KNOWN_DATASETS:
        organized_path = RAW_DIR / dataset_id / "organized"
        if organized_path.exists():
            success = convert_dataset(dataset_id)
            results.append({'dataset': dataset_id, 'success': success})
        else:
            print(f"\n⚠️  Skipping {dataset_id} (not organized)")
            results.append({'dataset': dataset_id, 'success': False})
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*70}")
    
    successful = sum(1 for r in results if r['success'])
    print(f"Successful: {successful}/{len(results)}")
    
    for r in results:
        status = "✅" if r['success'] else "❌"
        print(f"   {status} {r['dataset']}")


def list_status():
    """List conversion status."""
    print(f"\n{'Dataset':<30} {'Organized':<12} {'h5ad':<10} {'Samples':<8}")
    print("-" * 65)
    
    for dataset_id in KNOWN_DATASETS:
        organized_path = RAW_DIR / dataset_id / "organized"
        processed_path = PROCESSED_DIR / dataset_id
        
        # Check organized
        if organized_path.exists():
            org_samples = [d for d in organized_path.iterdir() if d.is_dir()]
            org_status = "✅" if org_samples else "❌"
        else:
            org_status = "❌"
            org_samples = []
        
        # Check h5ad
        if processed_path.exists():
            h5ad_files = list(processed_path.glob("*.h5ad"))
            h5ad_status = "✅" if h5ad_files else "❌"
            n_h5ad = len(h5ad_files)
        else:
            h5ad_status = "❌"
            n_h5ad = 0
        
        print(f"{dataset_id:<30} {org_status:<12} {h5ad_status:<10} {n_h5ad:<8}")


def main():
    """Main entry point."""
    if not SCANPY_AVAILABLE:
        sys.exit(1)
    
    if len(sys.argv) < 2:
        print_header()
        print("\nUsage:")
        print("  python organize_visium.py <platform/dataset_id>")
        print("  python organize_visium.py visium/yamamoto_2025")
        print("  python organize_visium.py --all")
        print("  python organize_visium.py --list")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == '--list' or command == '-l':
        print_header()
        list_status()
    
    elif command == '--all' or command == '-a':
        print_header()
        convert_all()
    
    elif command == '--help' or command == '-h':
        print_header()
        print(__doc__)
    
    else:
        dataset_id = command
        print_header(dataset_id)
        convert_dataset(dataset_id)


if __name__ == "__main__":
    main()



"""
## Summary of changes

### organize_visium.py (simplified)

| Change | What it does |
|--------|--------------|
| **Much simpler** | Only ~100 lines instead of ~200 |
| **No config needed** | Just calls `sc.read_visium()` on organized folders |
| **Standard structure assumed** | Expects consistent 10X folder structure from organize_for_spacet.py |

---

## Workflow after these changes
```
Step 1: python organize_for_spacet.py visium/yamamoto_2025
        Raw TAR → organized 10X folders

Step 2: python organize_visium.py visium/yamamoto_2025
        Organized folders → h5ad files

Step 3: python run_qc_pipeline.py visium/yamamoto_2025
        h5ad files → QC reports
"""

