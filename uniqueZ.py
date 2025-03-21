from typing import Tuple, Union
import numpy as np
import numpy.typing as npt
from scipy import sparse

def uniqueZ(
    Z: npt.NDArray[np.float_],
    old_ljmat: Union[npt.NDArray[np.float_], sparse.spmatrix]
) -> Tuple[npt.NDArray[np.float_], Union[npt.NDArray[np.float_], sparse.spmatrix]]:
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
    Z : npt.NDArray[np.float_]
        A 2D array where:
            - Columns 0 and 1 (MATLAB columns 1 and 2) are used to determine uniqueness.
            - Columns 2 and 3 (MATLAB columns 3 and 4) are used to update old_ljmat.
    old_ljmat : Union[npt.NDArray[np.float_], sparse.spmatrix]
        A connectivity matrix that will be updated. For each row in Z (columns 3 and 4) that 
        is not unique, the corresponding entries in old_ljmat will be set to 0.
        
    Returns
    -------
    Tuple[npt.NDArray[np.float_], Union[npt.NDArray[np.float_], sparse.spmatrix]]
        A tuple containing:
            - newZ: The unique version of Z
            - old_ljmat: The updated connectivity matrix
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
    def ismember_rows(
        A: npt.NDArray[np.float_],
        B: npt.NDArray[np.float_]
    ) -> Tuple[npt.NDArray[np.bool_], npt.NDArray[np.int_]]:
        """
        Find rows in A that are members of B and return their indices.
        
        Parameters
        ----------
        A : npt.NDArray[np.float_]
            First array for comparison
        B : npt.NDArray[np.float_]
            Second array to check membership against
            
        Returns
        -------
        Tuple[npt.NDArray[np.bool_], npt.NDArray[np.int_]]
            A tuple containing:
                - Boolean array indicating membership of each row in A
                - Array of indices where each row in A is found in B (-1 if not found)
        """
        # Create a view of each row as a single element (of type void) for comparison
        A_view = np.ascontiguousarray(A).view(np.dtype((np.void, A.dtype.itemsize * A.shape[1])))
        B_view = np.ascontiguousarray(B).view(np.dtype((np.void, B.dtype.itemsize * B.shape[1])))
        
        # Find which rows in A are present in B
        membership = np.in1d(A_view, B_view)
        
        # Initialize indices array with -1 (indicating "not found")
        indices = np.full(len(A), -1, dtype=np.int_)
        
        # For each row in A that has a match in B, find its index in B
        for i in np.where(membership)[0]:
            # Find the first matching row in B
            matches = np.where((B == A[i]).all(axis=1))[0]
            if len(matches) > 0:
                indices[i] = matches[0]
        
        return membership, indices
    
    test, indices = ismember_rows(sortrow_Y, Uni_sortrow_Y)
    
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