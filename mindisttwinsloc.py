import numpy as np

def mindisttwinsloc(dataloc1, dataloc2, DM):
    # Ensure inputs are numpy arrays
    dataloc1 = np.array(dataloc1)
    dataloc2 = np.array(dataloc2)
    
    # Extract the submatrix
    dists = DM[np.ix_(dataloc1, dataloc2)]
    
    # Handle sparse matrices
    from scipy import sparse
    if sparse.issparse(dists):
        # Find the minimum value
        min_val = dists.min()
        
        # Find coordinates where value equals minimum
        rows, cols, vals = sparse.find(dists)
        min_indices = [(i, j) for i, j, v in zip(rows, cols, vals) if v == min_val]
        
        if min_indices:
            a, b = min_indices[0]  # Take the first occurrence
        else:
            raise ValueError("No minimum value found in the distance matrix")
    else:
        # For dense arrays
        min_val = np.min(dists)
        indices = np.where(dists == min_val)
        a, b = indices[0][0], indices[1][0]  # Take the first occurrence
    
    # Map back to original indices
    linkloc1 = dataloc1[a]
    linkloc2 = dataloc2[b]
    
    return linkloc1, linkloc2