"""
Torque Clustering - Python Implementation with Exact MATLAB Compatibility
Copyright (C) Jie Yang

Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International (CC BY-NC-SA 4.0)

This code is intended for academic and research purposes only.
Commercial use is strictly prohibited. Please contact the author for licensing inquiries.

Author: Jie Yang (jie.yang.uts@gmail.com)
Python adaptation with MATLAB-matching behavior
"""

from typing import Tuple
import numpy as np
import numpy.typing as npt


def qac(sort_p: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """
    Python implementation of MATLAB's Qac function to calculate quality ratios.
    Matches MATLAB's exact behavior from the provided Qac.m file.
    
    Args:
        sort_p: Sorted torque values
        
    Returns:
        Quality ratio array with same length as input
    """
    p_num = len(sort_p)
    ind_num = p_num - 1
    # Initialize array of zeros with shape (1, p_num) to match MATLAB's row vector
    ind = np.zeros(p_num, dtype=np.float64)
    
    # Calculate ratios exactly as in MATLAB
    for i in range(ind_num):
        # MATLAB's division handles zeros automatically (returns inf)
        # i+1 in Python = i+1 in MATLAB (since MATLAB is 1-indexed)
        ind[i] = sort_p[i] / sort_p[i+1]
    
    # Set last element to NaN (p_num is already correct index since Python is 0-indexed)
    ind[p_num-1] = np.nan
    
    return ind


def matlab_setdiff(a, b):
    """
    Replicate MATLAB's setdiff function behavior exactly.
    
    MATLAB setdiff(A,B) returns the values in A that are not in B, with:
    - Result sorted in ascending order
    - No duplicate values in the result
    - NaN values in A included if not in B
    
    Args:
        a: First array
        b: Second array
        
    Returns:
        Array of values in a but not in b, sorted ascending
    """
    # Ensure inputs are numpy arrays
    a = np.asarray(a)
    b = np.asarray(b)
    
    # Handle NaN values specially (MATLAB includes NaN if in A but not in B)
    a_has_nan = np.any(np.isnan(a))
    b_has_nan = np.any(np.isnan(b))
    
    # Get non-NaN elements for normal processing
    a_no_nan = a[~np.isnan(a)]
    b_no_nan = b[~np.isnan(b)]
    
    # Use numpy's setdiff1d for the main calculation
    result = np.setdiff1d(a_no_nan, b_no_nan)
    
    # Add NaN if it was in a but not in b
    if a_has_nan and not b_has_nan:
        result = np.append(result, np.nan)
    
    return result


def matlab_intersect(a, b):
    """
    Replicate MATLAB's intersect function behavior exactly.
    
    MATLAB intersect(A,B) returns the values common to both A and B, with:
    - Result sorted in ascending order
    - No duplicate values in the result
    - Proper handling of NaN values (NaN in both arrays is considered a match in MATLAB)
    
    Args:
        a: First array
        b: Second array
        
    Returns:
        Array of values common to both a and b, sorted ascending
    """
    # Ensure inputs are numpy arrays
    a = np.asarray(a)
    b = np.asarray(b)
    
    # Handle NaN values specially
    a_nan = np.isnan(a)
    b_nan = np.isnan(b)
    a_has_nan = np.any(a_nan)
    b_has_nan = np.any(b_nan)
    
    # Get non-NaN elements for normal processing
    a_no_nan = a[~a_nan]
    b_no_nan = b[~b_nan]
    
    # Find intersection of non-NaN values
    common = np.intersect1d(a_no_nan, b_no_nan)
    
    # Add NaN if it was in both arrays
    if a_has_nan and b_has_nan:
        common = np.append(common, np.nan)
    
    return common


def matlab_logical_and(a, b, c):
    """
    Replicate MATLAB's element-wise logical AND operator for arrays.
    Handles NaN values and type conversion exactly as MATLAB does.
    
    Args:
        a, b, c: Arrays or scalars to combine with logical AND
        
    Returns:
        Boolean array with the result of element-wise a & b & c
    """
    # Convert inputs to numpy arrays
    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)
    
    # Handle NaN values (MATLAB treats NaN as false in logical operations)
    a_logical = np.zeros_like(a, dtype=bool)
    b_logical = np.zeros_like(b, dtype=bool)
    c_logical = np.zeros_like(c, dtype=bool)
    
    # Convert non-zero and non-NaN values to True
    a_non_nan = ~np.isnan(a)
    b_non_nan = ~np.isnan(b)
    c_non_nan = ~np.isnan(c)
    
    a_logical[a_non_nan] = a[a_non_nan] != 0
    b_logical[b_non_nan] = b[b_non_nan] != 0
    c_logical[c_non_nan] = c[c_non_nan] != 0
    
    # Perform element-wise AND (no short-circuiting)
    result = a_logical & b_logical & c_logical
    
    return result


def matlab_find_max(arr):
    """
    Replicate MATLAB's find(arr==max(arr)) behavior exactly.
    
    Key behaviors to match:
    1. MATLAB's max ignores NaN values
    2. MATLAB's equality comparison with a scalar is element-wise
    3. MATLAB's find returns indices where condition is true
    4. If all elements are NaN, max returns NaN and no indices match
    
    Args:
        arr: Input array to find maximum values
        
    Returns:
        Array of indices where arr equals its maximum value
    """
    arr = np.asarray(arr)
    
    # If all values are NaN, MATLAB returns an empty array
    if np.all(np.isnan(arr)):
        return np.array([], dtype=int)
    
    # MATLAB's max ignores NaN values
    max_val = np.nanmax(arr)
    
    # Find indices where arr equals max_val
    # Use a small tolerance for floating-point comparisons
    tol = 1e-14
    indices = np.where(np.abs(arr - max_val) < tol)[0]
    
    # Return as a 1D array
    return indices


def matlab_sort(arr, direction='descend'):
    """
    Replicate MATLAB's sort function with stable sorting.
    
    Args:
        arr: Array to sort
        direction: 'ascend' or 'descend'
        
    Returns:
        Tuple of (sorted_array, indices)
    """
    arr = np.asarray(arr)
    
    # Handle NaN values (MATLAB puts NaNs at the end)
    nan_mask = np.isnan(arr)
    non_nan_mask = ~nan_mask
    
    # Get non-NaN values and their indices
    non_nan_values = arr[non_nan_mask]
    non_nan_indices = np.where(non_nan_mask)[0]
    
    # Sort non-NaN values
    if direction == 'descend':
        sorted_indices = np.argsort(-non_nan_values, kind='mergesort')
    else:
        sorted_indices = np.argsort(non_nan_values, kind='mergesort')
    
    # Get sorted non-NaN values and indices
    sorted_non_nan_values = non_nan_values[sorted_indices]
    sorted_non_nan_indices = non_nan_indices[sorted_indices]
    
    # Get NaN values and their indices
    nan_indices = np.where(nan_mask)[0]
    
    # Combine sorted non-NaN values/indices with NaN values/indices
    sorted_values = np.zeros_like(arr)
    sorted_values[non_nan_mask] = sorted_non_nan_values
    sorted_values[nan_mask] = np.nan
    
    if nan_indices.size > 0:
        sorted_indices = np.concatenate((sorted_non_nan_indices, nan_indices))
    else:
        sorted_indices = sorted_non_nan_indices
        
    return sorted_values, sorted_indices


def matlab_mean(arr, axis=None):
    """
    Replicate MATLAB's mean behavior with NaN values.
    MATLAB ignores NaN values when computing the mean.
    
    Args:
        arr: Input array
        axis: Axis along which to compute mean
        
    Returns:
        Mean of arr, ignoring NaNs
    """
    return np.nanmean(arr, axis=axis)


def Nab_dec(
    p: npt.NDArray[np.float_],
    mass: npt.NDArray[np.float_],
    R: npt.NDArray[np.float_],
    florderloc: npt.NDArray[np.int_],
    use_std_adjustment: bool = True,
    adjustment_factor: float = 0.5
) -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """
    Determine the number of abnormal merges to cut based on torque analysis.
    
    Args:
        p: Array of torque values
        mass: Array of mass values
        R: Array of distance values
        florderloc: Array of first layer location indices
        use_std_adjustment: Whether to use standard deviation for threshold adjustment (default: True)
        adjustment_factor: Factor to multiply standard deviation for threshold adjustment (default: 0.5)
    
    Returns:
        Tuple of (NAB, resolution) arrays
    """
    # Convert inputs to float64 for consistency with MATLAB
    p = np.float64(p)
    mass = np.float64(mass)
    R = np.float64(R)
    
    # Sort values in descending order
    sort_p_1, ind1 = matlab_sort(p)
    sort_mass_1, _ = matlab_sort(mass)
    sort_R_1, _ = matlab_sort(R)
    
    # Calculate means
    p_mean = matlab_mean(sort_p_1)
    mass_mean = matlab_mean(sort_mass_1)
    R_mean = matlab_mean(sort_R_1)
    
    # Calculate standard deviations if using std adjustment
    if use_std_adjustment:
        p_std = np.nanstd(sort_p_1)
        mass_std = np.nanstd(sort_mass_1)
        R_std = np.nanstd(sort_R_1)
        
        # Adjust thresholds using standard deviation
        R_threshold = R_mean - adjustment_factor * R_std
        mass_threshold = mass_mean - adjustment_factor * mass_std
        p_threshold = p_mean - adjustment_factor * p_std
    else:
        # Use means directly as thresholds
        R_threshold = R_mean
        mass_threshold = mass_mean
        p_threshold = p_mean
    
    # Identify points that meet criteria with adjusted thresholds
    a = (sort_R_1 >= R_threshold)
    b = (sort_mass_1 >= mass_threshold)
    c = (sort_p_1 >= p_threshold)
    
    # Combine criteria with improved noise handling
    d = matlab_logical_and(a, b, c)
    
    # Find indices that satisfy all criteria
    resolution = np.where(d)[0]
    
    # Calculate combined index for cluster determination
    if len(resolution) > 0:
        combined_index = ind1[resolution]
        max_index = np.nanmax(combined_index)
        NAB = resolution[combined_index == max_index]
    else:
        NAB = np.array([], dtype=np.int_)
    
    return NAB, resolution


def validate_with_test_case(test_p, test_mass, test_R, test_florderloc):
    """
    Validate the Python implementation with a test case.
    Prints detailed outputs from each step for debugging.
    
    Args:
        test_p: Test torque values
        test_mass: Test mass values
        test_R: Test R values
        test_florderloc: Test excluded indices
    """
    print("Input arrays:")
    print(f"p: {test_p}")
    print(f"mass: {test_mass}")
    print(f"R: {test_R}")
    print(f"florderloc: {test_florderloc}")
    
    # Run the algorithm
    NAB, resolution = Nab_dec(test_p, test_mass, test_R, test_florderloc)
    
    print("\nResults:")
    print(f"NAB: {NAB}")
    print(f"resolution: {resolution}")
    
    # Additional step-by-step outputs for validation
    sort_p, order = matlab_sort(test_p, 'descend')
    print("\nIntermediate values:")
    print(f"sort_p: {sort_p}")
    print(f"order: {order}")
    
    sort_R = test_R[order]
    sort_mass = test_mass[order]
    print(f"sort_R: {sort_R}")
    print(f"sort_mass: {sort_mass}")
    
    ind1 = qac(sort_p)
    print(f"ind1: {ind1}")
    
    # Return results for comparison
    return NAB, resolution

