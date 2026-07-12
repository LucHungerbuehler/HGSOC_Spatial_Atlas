"""
analysis_utils.py
=================
Shared utility functions for SNAI1-ac spatial transcriptomics analysis.

This module contains statistical functions, spatial analysis helpers, and I/O utilities
used by both spatial_signature_analysis.py and tumor_subset_analysis.py.

Author: [Your name]
Date: 2024
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, skew, kurtosis, rankdata
from pathlib import Path


# =============================================================================
# STATISTICAL FUNCTIONS
# =============================================================================

def cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size between two groups.
    
    Cohen's d = (mean1 - mean2) / pooled_std
    
    Interpretation:
        |d| < 0.2: negligible
        |d| 0.2-0.5: small
        |d| 0.5-0.8: medium
        |d| > 0.8: large
    
    Parameters
    ----------
    group1 : array-like
        First group of values
    group2 : array-like
        Second group of values
        
    Returns
    -------
    float
        Cohen's d effect size (positive if group1 > group2)
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)
    
    n1, n2 = len(group1), len(group2)
    
    if n1 < 2 or n2 < 2:
        return np.nan
    
    var1 = group1.var(ddof=1)
    var2 = group2.var(ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (group1.mean() - group2.mean()) / pooled_std


def bimodality_coefficient(data):
    """
    Calculate the bimodality coefficient.
    
    BC = (skewness^2 + 1) / (kurtosis + 3)
    
    A BC > 0.555 suggests bimodality (Pfister et al., 2013).
    The value 0.555 is the BC of a uniform distribution.
    
    Parameters
    ----------
    data : array-like
        Data to assess for bimodality
        
    Returns
    -------
    tuple
        (bimodality_coefficient, is_bimodal)
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    
    if len(data) < 4:
        return np.nan, False
    
    s = skew(data)
    k = kurtosis(data, fisher=True)  # Fisher's definition (normal = 0)
    
    # Bimodality coefficient formula
    # Note: scipy's kurtosis with fisher=True gives excess kurtosis
    # We need kurtosis + 3 for the formula (which equals the raw kurtosis)
    bc = (s**2 + 1) / (k + 3)
    
    is_bimodal = bc > 0.555
    
    return bc, is_bimodal


def partial_spearman(x, y, covar):
    """
    Calculate partial Spearman correlation between x and y, controlling for covariate.
    
    Uses residualization approach with rank-transformed variables.
    
    Parameters
    ----------
    x : array-like
        First variable
    y : array-like
        Second variable
    covar : array-like
        Covariate to control for
        
    Returns
    -------
    tuple
        (correlation coefficient, p-value)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    covar = np.asarray(covar)
    
    # Remove any rows with NaN
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(covar))
    x = x[mask]
    y = y[mask]
    covar = covar[mask]
    
    if len(x) < 4:
        return np.nan, np.nan
    
    # Rank transform all variables
    x_rank = rankdata(x)
    y_rank = rankdata(y)
    covar_rank = rankdata(covar)
    
    # Residualize x and y on covariate using linear regression
    x_resid = x_rank - np.polyval(np.polyfit(covar_rank, x_rank, 1), covar_rank)
    y_resid = y_rank - np.polyval(np.polyfit(covar_rank, y_rank, 1), covar_rank)
    
    # Correlate residuals
    r, p = spearmanr(x_resid, y_resid)
    
    return r, p


# =============================================================================
# SPATIAL ANALYSIS FUNCTIONS
# =============================================================================

def get_neighbor_composition(spot_indices, conn_matrix, fractions):
    """
    Get mean cell type fractions of neighbors for a set of spots.
    
    Parameters
    ----------
    spot_indices : array-like
        Indices of spots to analyze
    conn_matrix : sparse matrix
        Spatial connectivity matrix (from scanpy neighbors)
    fractions : ndarray
        Cell type fractions array (n_spots x n_celltypes)
        
    Returns
    -------
    ndarray
        Mean cell type fractions across all neighbors of all spots
    """
    neighbor_fractions = []
    
    for idx in spot_indices:
        # Get neighbors of this spot
        neighbors = conn_matrix[idx].nonzero()[1]
        if len(neighbors) > 0:
            neighbor_fractions.append(fractions[neighbors].mean(axis=0))
    
    if len(neighbor_fractions) == 0:
        return np.full(fractions.shape[1], np.nan)
    
    return np.array(neighbor_fractions).mean(axis=0)


def get_neighbor_composition_excluding_group(spot_indices, group_indices_to_exclude, 
                                              conn_matrix, fractions):
    """
    Get mean cell type fractions of neighbors, excluding neighbors in a specified group.
    
    Useful for asking "what's around HIGH spots, excluding other HIGH spots?"
    
    Parameters
    ----------
    spot_indices : array-like
        Indices of spots to analyze
    group_indices_to_exclude : array-like
        Indices of neighbors to exclude
    conn_matrix : sparse matrix
        Spatial connectivity matrix
    fractions : ndarray
        Cell type fractions array (n_spots x n_celltypes)
        
    Returns
    -------
    ndarray
        Mean cell type fractions of non-excluded neighbors
    """
    exclude_set = set(group_indices_to_exclude)
    neighbor_fractions = []
    
    for idx in spot_indices:
        neighbors = conn_matrix[idx].nonzero()[1]
        # Filter out neighbors in the exclusion group
        filtered_neighbors = [n for n in neighbors if n not in exclude_set]
        
        if len(filtered_neighbors) > 0:
            neighbor_fractions.append(fractions[filtered_neighbors].mean(axis=0))
    
    if len(neighbor_fractions) == 0:
        return np.full(fractions.shape[1], np.nan)
    
    return np.array(neighbor_fractions).mean(axis=0)


def get_rings(spot_idx, conn_matrix, n_rings=3):
    """
    Get spot indices for each ring around a central spot.
    
    Ring 1 = immediate neighbors
    Ring 2 = neighbors of ring 1, excluding spot and ring 1
    Ring 3 = neighbors of ring 2, excluding all previous
    
    Parameters
    ----------
    spot_idx : int
        Index of central spot
    conn_matrix : sparse matrix
        Spatial connectivity matrix
    n_rings : int
        Number of rings to compute
        
    Returns
    -------
    dict
        {ring_number: set of spot indices}
    """
    rings = {}
    seen = {spot_idx}
    
    # Ring 1: immediate neighbors
    ring1 = set(conn_matrix[spot_idx].nonzero()[1])
    rings[1] = ring1
    seen.update(ring1)
    
    # Subsequent rings
    current_ring = ring1
    for r in range(2, n_rings + 1):
        next_ring = set()
        for idx in current_ring:
            neighbors = set(conn_matrix[idx].nonzero()[1])
            next_ring.update(neighbors - seen)
        rings[r] = next_ring
        seen.update(next_ring)
        current_ring = next_ring
    
    return rings


def get_ring_composition(spot_indices, conn_matrix, fractions, n_rings=3):
    """
    Get mean cell type fractions for each ring around a set of spots.
    
    Parameters
    ----------
    spot_indices : array-like
        Indices of central spots
    conn_matrix : sparse matrix
        Spatial connectivity matrix
    fractions : ndarray
        Cell type fractions array (n_spots x n_celltypes)
    n_rings : int
        Number of rings to compute
        
    Returns
    -------
    dict
        {ring_number: mean_fractions_array}
    """
    ring_compositions = {r: [] for r in range(1, n_rings + 1)}
    
    for idx in spot_indices:
        rings = get_rings(idx, conn_matrix, n_rings)
        for r in range(1, n_rings + 1):
            if len(rings[r]) > 0:
                ring_fractions = fractions[list(rings[r])].mean(axis=0)
                ring_compositions[r].append(ring_fractions)
    
    # Average across all spots
    result = {}
    for r in range(1, n_rings + 1):
        if len(ring_compositions[r]) > 0:
            result[r] = np.array(ring_compositions[r]).mean(axis=0)
        else:
            result[r] = np.full(fractions.shape[1], np.nan)
    
    return result


def get_ring_composition_rescaled(spot_indices, conn_matrix, fractions, celltype_names,
                                   n_rings=3, min_nonmal=0.05):
    """
    Get rescaled (non-malignant) cell type fractions for each ring.
    
    Rescales each spot's non-malignant fractions to sum to 1, then averages.
    Spots with non-malignant fraction < min_nonmal are excluded (too noisy).
    
    Parameters
    ----------
    spot_indices : array-like
        Indices of central spots
    conn_matrix : sparse matrix
        Spatial connectivity matrix
    fractions : ndarray
        Cell type fractions array (n_spots x n_celltypes)
    celltype_names : list
        Names of cell types (must include 'Malignant')
    n_rings : int
        Number of rings to compute
    min_nonmal : float
        Minimum non-malignant fraction to include spot
        
    Returns
    -------
    tuple
        (dict of ring compositions, dict of spot counts per ring)
    """
    mal_idx = celltype_names.index('Malignant')
    
    ring_compositions = {r: [] for r in range(1, n_rings + 1)}
    ring_counts = {r: 0 for r in range(1, n_rings + 1)}
    
    for idx in spot_indices:
        rings = get_rings(idx, conn_matrix, n_rings)
        
        for r in range(1, n_rings + 1):
            ring_spots = list(rings[r])
            if len(ring_spots) == 0:
                continue
            
            # Get fractions for ring spots
            ring_fracs = fractions[ring_spots].copy()
            
            # Calculate non-malignant total for each spot
            non_mal_total = 1 - ring_fracs[:, mal_idx]
            
            # Filter spots with sufficient non-malignant fraction
            valid_mask = non_mal_total >= min_nonmal
            if valid_mask.sum() == 0:
                continue
            
            valid_fracs = ring_fracs[valid_mask]
            valid_non_mal = non_mal_total[valid_mask]
            
            # Rescale: divide each non-malignant fraction by total non-malignant
            rescaled = valid_fracs.copy()
            for i in range(len(celltype_names)):
                if i != mal_idx:
                    rescaled[:, i] = valid_fracs[:, i] / valid_non_mal
            
            # Average across valid spots in this ring
            ring_compositions[r].append(rescaled.mean(axis=0))
            ring_counts[r] += valid_mask.sum()
    
    # Average across all central spots
    result = {}
    for r in range(1, n_rings + 1):
        if len(ring_compositions[r]) > 0:
            result[r] = np.array(ring_compositions[r]).mean(axis=0)
        else:
            result[r] = np.full(len(celltype_names), np.nan)
    
    return result, ring_counts


def classify_by_neighbor_threshold(spot_idx, conn_matrix, mask, min_neighbors=5):
    """
    Check if a spot and at least min_neighbors of its neighbors meet a mask criteria.
    
    Used for threshold-based hotspot detection.
    
    Parameters
    ----------
    spot_idx : int
        Index of spot to check
    conn_matrix : sparse matrix
        Spatial connectivity matrix
    mask : array-like of bool
        Boolean mask indicating which spots meet criteria
    min_neighbors : int
        Minimum number of neighbors that must also meet criteria
        
    Returns
    -------
    bool
        True if spot qualifies as hotspot
    """
    # Spot itself must meet criteria
    if not mask[spot_idx]:
        return False
    
    # Check neighbors
    neighbors = conn_matrix[spot_idx].nonzero()[1]
    n_neighbors_meeting_criteria = mask[neighbors].sum()
    
    return n_neighbors_meeting_criteria >= min_neighbors


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def jaccard(set1, set2):
    """
    Calculate Jaccard index: intersection / union.
    
    Parameters
    ----------
    set1 : set
        First set
    set2 : set
        Second set
        
    Returns
    -------
    float
        Jaccard index (0 to 1)
    """
    if len(set1) == 0 and len(set2) == 0:
        return np.nan
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def convert_keys_to_str(d):
    """
    Recursively convert dictionary keys to strings for h5ad compatibility.
    
    h5ad format requires string keys in uns dictionaries.
    
    Parameters
    ----------
    d : dict or any
        Dictionary to convert (or any other value)
        
    Returns
    -------
    dict or any
        Dictionary with string keys (or unchanged value if not dict)
    """
    if isinstance(d, dict):
        return {str(k): convert_keys_to_str(v) for k, v in d.items()}
    return d


# =============================================================================
# I/O HELPERS
# =============================================================================

def ensure_dir(path):
    """
    Create directory if it doesn't exist.
    
    Parameters
    ----------
    path : str or Path
        Directory path to create
        
    Returns
    -------
    Path
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig, filepath, dpi=150):
    """
    Save figure and close to free memory.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    filepath : str or Path
        Output path
    dpi : int
        Resolution
    """
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    import matplotlib.pyplot as plt
    plt.close(fig)


def format_pvalue(p):
    """
    Format p-value for display.
    
    Parameters
    ----------
    p : float
        P-value
        
    Returns
    -------
    str
        Formatted string
    """
    if np.isnan(p):
        return "NA"
    elif p < 0.001:
        return f"{p:.2e}"
    else:
        return f"{p:.4f}"