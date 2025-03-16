import numpy as np
from scipy import sparse

def uniqueZ(Z, old_ljmat):
    """
    Generate a unique version of Z (newZ) based on the first two columns and update 
    the connectivity (or linkage) matrix old_ljmat based on columns 3 and 4 of Z.
    Now with support for sparse matrices.
    
    Detailed Explanation:
    - If Z is empty, newZ is set to an empty array.
    - Otherwise, the function:
        1. Sorts the values in each row of the first two columns of Z.
        2. Finds the unique rows (based on the sorted values) and uses their indices (order)
           to select the corresponding rows from the original Z, forming newZ.
        3. Then, for columns 3 and 4 of Z, each row is sorted and the subset corresponding to 
           the unique rows (using the same indices 'order') is selected (Uni_sortrow_Y).
        4. For each row in the full sorted version of columns 3 and 4 (sortrow_Y), it checks 
           whether the row is a member of Uni_sortrow_Y.
        5. For any rows that are NOT members (rmv), the function sets the corresponding entries 
           in old_ljmat to 0 (in both symmetric positions).
    
    Parameters
    ----------
    Z : numpy.ndarray
        A 2D array where:
            - Columns 0 and 1 (MATLAB columns 1 and 2) are used to determine uniqueness.
            - Columns 2 and 3 (MATLAB columns 3 and 4) are used to update old_ljmat.
    old_ljmat : numpy.ndarray or scipy.sparse matrix
        A connectivity matrix that will be updated. For each row in Z (columns 3 and 4) that 
        is not among the unique ones, the corresponding entries in old_ljmat are set to 0.
        (Assumes that the indices in columns 3 and 4 are already 0-based. If they are 1-based,
        subtract 1 when using them as indices.)
    
    Returns
    -------
    newZ : numpy.ndarray
        A matrix containing the unique rows of Z (based on the first two columns).
    ljmat : numpy.ndarray or scipy.sparse matrix
        The updated connectivity matrix after processing.
    """
    # If Z is empty, set newZ to an empty array and return old_ljmat unchanged.
    if Z.size == 0:
        newZ = np.array([])
        ljmat = old_ljmat
        return newZ, ljmat

    # Check if old_ljmat is a sparse matrix
    is_sparse = sparse.issparse(old_ljmat)
    
    # Create a copy of old_ljmat to modify
    if is_sparse:
        # For sparse matrices, we'll work with the same format
        ljmat = old_ljmat.copy()
    else:
        ljmat = old_ljmat.copy()

    # -------------------------------------------------------------------------
    # Step 1: Process columns 1-2 (Python indices 0-1)
    # Sort the first two columns of Z row-wise.
    sortrow_Z = np.sort(Z[:, [0, 1]], axis=1)
    
    # Find unique rows in sortrow_Z.
    _, order = np.unique(sortrow_Z, axis=0, return_index=True)
    
    # Create newZ by selecting rows of Z corresponding to the unique indices.
    newZ = Z[order, :].copy()
    
    # -------------------------------------------------------------------------
    # Step 2: Process columns 3-4 (Python indices 2-3)
    # Sort the third and fourth columns of Z row-wise.
    sortrow_Y = np.sort(Z[:, [2, 3]], axis=1)
    
    # Select the rows corresponding to the unique set from step 1.
    Uni_sortrow_Y = sortrow_Y[order, :]
    
    # Check, for each row in sortrow_Y, whether it is a member of Uni_sortrow_Y.
    def ismember_rows(A, B):
        # Create a view of each row as a single element (of type void) for comparison.
        A_view = np.ascontiguousarray(A).view(np.dtype((np.void, A.dtype.itemsize * A.shape[1])))
        B_view = np.ascontiguousarray(B).view(np.dtype((np.void, B.dtype.itemsize * B.shape[1])))
        return np.in1d(A_view, B_view)
    
    test = ismember_rows(sortrow_Y, Uni_sortrow_Y)
    
    # Identify the rows in sortrow_Y that are NOT in Uni_sortrow_Y.
    rmv = sortrow_Y[~test, :]
    
    # -------------------------------------------------------------------------
    # Step 3: Update ljmat based on the non-unique rows.
    if rmv.size > 0:
        rmv_num = rmv.shape[0]
        
        if is_sparse:
            # For sparse matrices, we'll collect all indices to modify and do it efficiently
            i1_indices = [int(rmv[j, 0]) for j in range(rmv_num)]
            i2_indices = [int(rmv[j, 1]) for j in range(rmv_num)]
            
            # Convert to format that allows item assignment if needed
            if not hasattr(ljmat, 'tolil'):
                ljmat = ljmat.tocsr()  # Default to CSR if unknown format
                
            # Convert to LIL format for efficient item assignment
            ljmat_lil = ljmat.tolil()
            
            # Set values to zero
            for i1, i2 in zip(i1_indices, i2_indices):
                ljmat_lil[i1, i2] = 0
                ljmat_lil[i2, i1] = 0  # ensure symmetry
                
            # Convert back to the original format or to CSR for efficiency
            if hasattr(old_ljmat, 'format'):
                ljmat = ljmat_lil.asformat(old_ljmat.format)
            else:
                ljmat = ljmat_lil.tocsr()
        else:
            # For dense matrices, use the original approach
            for j in range(rmv_num):
                i1 = int(rmv[j, 0])
                i2 = int(rmv[j, 1])
                ljmat[i1, i2] = 0
                ljmat[i2, i1] = 0  # ensure symmetry

    return newZ, ljmat