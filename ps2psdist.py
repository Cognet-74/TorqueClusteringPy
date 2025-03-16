from typing import Union, List, Sequence
import numpy as np
import numpy.typing as npt

def ps2psdist(
    Loc_dataset1: Union[List[int], npt.NDArray[np.int_]],
    Loc_dataset2: Union[List[int], npt.NDArray[np.int_]],
    DM: npt.NDArray[np.float_]
) -> float:
    """
    Compute a distance measure between two sets of points using a precomputed distance matrix.
    
    This function is a direct translation of the MATLAB ps2psdist function and maintains
    identical behavior. It extracts the submatrix of distances from DM corresponding to the
    indices in Loc_dataset1 and Loc_dataset2, then computes the minimum distance between
    any point in the first group and any point in the second group.
    
    Note: Assumes indices are 0-based (Python style) rather than 1-based (MATLAB style).
    If MATLAB indices are provided, subtract 1 from each index.
    
    Parameters:
        Loc_dataset1 (Union[List[int], npt.NDArray[np.int_]]): 
            Indices of points in the first group.
        Loc_dataset2 (Union[List[int], npt.NDArray[np.int_]]): 
            Indices of points in the second group.
        DM (npt.NDArray[np.float_]): 
            Precomputed distance matrix where DM[i, j] is the distance between point i and point j.
    
    Returns:
        float: The computed distance measure between the two groups.
    """
    
    # Convert lists to arrays if needed for consistent handling
    Loc_dataset1 = np.asarray(Loc_dataset1, dtype=int)
    Loc_dataset2 = np.asarray(Loc_dataset2, dtype=int)
    
    # Handle empty arrays case consistently with MATLAB
    if Loc_dataset1.size == 0 or Loc_dataset2.size == 0:
        return float('inf')  # MATLAB convention for no valid distances
    
    # Check single-point case (commented out for exact alignment with current MATLAB code)
    # if Loc_dataset1.size == 1 and Loc_dataset2.size == 1:
    #     return float(DM[Loc_dataset1[0], Loc_dataset2[0]])
    
    # Extract submatrix exactly as MATLAB does
    # Replicate the direct matrix indexing from MATLAB
    # This creates a matrix where each row i corresponds to Loc_dataset1[i] 
    # and each column j corresponds to Loc_dataset2[j]
    dists = DM[np.ix_(Loc_dataset1, Loc_dataset2)]
    
    # Apply min operations sequentially to match MATLAB's min(min(dists))
    # First get min along rows (axis=1), then get min of those values
    # This is equivalent to MATLAB's min(min(dists))
    min_per_row = np.min(dists, axis=1)
    Cdist = np.min(min_per_row)
    
    # Handle NaN values consistently with MATLAB
    if np.isnan(Cdist):
        # In MATLAB, if all values are NaN, min returns NaN
        return float(Cdist)
    
    return float(Cdist)