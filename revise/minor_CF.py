import numpy as np
import scipy.sparse as sp

from scipy.spatial import KDTree
from tqdm import tqdm


def replace_effect_spots(adata, celltype_col="Level1", no_effect_col="no_effect", n_search_neighbors=10):
    """
    Replace gene expression of cells where no_effect_col is False with the mean expression
    of n_search_neighbors nearest cells of the same cell type.
    
    Parameters:
    - adata: AnnData object containing spatial data and gene expression
    - celltype_col: Column name in adata.obs for cell type annotations (default: "Level1")
    - no_effect_col: Column name in adata.obs indicating cells to replace (default: "no_effect")
    - n_search_neighbors: Number of nearest neighbors to consider (default: 10)
    
    Returns:
    - adata: Modified AnnData object with updated .X for specified cells
    """
    # Ensure adata.X is a dense array for modification
    if not isinstance(adata.X, np.ndarray):
        expr = adata.X.toarray()
    else:
        expr = adata.X

    # Extract spatial coordinates and relevant columns
    spot_coords = adata.obsm['spatial']
    cell_types = adata.obs[celltype_col].values
    no_effect = adata.obs[no_effect_col].values
    
    # Build KDTree for spatial coordinates
    kdtree = KDTree(spot_coords)
    
    # Get unique cell types
    unique_cell_types = np.unique(cell_types)
    
    # Identify cells to replace (where no_effect is False)
    cells_to_replace = np.where(~no_effect)[0]
    
    for cell_idx in tqdm(cells_to_replace, desc="Replacing cell expressions"):
        cell_type = cell_types[cell_idx]
        
        # Get indices of cells with the same cell type
        same_type_mask = cell_types == cell_type
        same_type_indices = np.where(same_type_mask)[0]
        
        if len(same_type_indices) <= 1:  # Only the cell itself, no other neighbors
            continue
        
        # Get coordinates of the current cell
        cell_coord = spot_coords[cell_idx].reshape(1, -1)
        
        # Query nearest neighbors
        _, neighbor_indices = kdtree.query(cell_coord, k=n_search_neighbors + 1)  # +1 to include self
        
        # Filter neighbors to those of the same cell type, excluding self
        valid_indices = [idx for idx in neighbor_indices[0] if idx in same_type_indices and idx != cell_idx]
        
        if len(valid_indices) == 0:  # No valid neighbors found
            continue
        
        # Limit to requested number of neighbors
        valid_indices = valid_indices[:n_search_neighbors]
        
        # Get expression data for selected neighbors
        neighbor_X = expr[valid_indices]
        
        # Compute mean expression
        mean_expr = np.mean(neighbor_X, axis=0)
        
        # Replace the cell's expression
        adata.X[cell_idx] = np.round(mean_expr)
    # Update AnnData.X with the modified expression matrix
    adata.X = sp.csr_matrix(adata.X)
    
    return adata