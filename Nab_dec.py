import numpy as np

def qac(sort_p):
    """
    Torque Clustering - Python Implementation matching MATLAB behavior

    Calculates the ratio of adjacent elements in a sorted array.
    Matches MATLAB's handling of division by zero which returns Inf.

    Args:
        sort_p: A 1D numpy array representing the sorted input.

    Returns:
        A 1D numpy array containing the ratios. The last element is NaN.
    """
    # Ensure input is a numpy array
    sort_p = np.asarray(sort_p)
    
    p_num = len(sort_p)
    
    # Preallocate full result array (more efficient than appending later)
    ind = np.zeros(p_num)
    
    # Replace zeros with a very small value to mimic MATLAB's behavior
    # MATLAB automatically converts division by zero to Inf
    sort_p_safe = np.copy(sort_p)
    sort_p_safe[sort_p_safe == 0] = np.finfo(float).eps  # smallest positive float
    
    # Calculate ratios (p_num-1 elements)
    for i in range(p_num-1):
        ind[i] = sort_p[i] / sort_p_safe[i+1]
    
    # Set last element to NaN
    ind[p_num-1] = np.nan
    
    return ind


def Nab_dec(p, mass, R, florderloc):
    """
    Torque Clustering - Python Implementation matching MATLAB behavior
    
    This code is intended for academic and research purposes only.
    Commercial use is strictly prohibited. Please contact the author for licensing inquiries.
    
    Author: Jie Yang (jie.yang.uts@gmail.com)
    Python adaptation with MATLAB-matching behavior
    
    Args:
        p: torque of each connection
        mass: mass values
        R: R values
        florderloc: indices to exclude
        
    Returns:
        NAB: indices where the combined index equals the maximum value
        resolution: indices that satisfy conditions a, b, and c
    """
    
    sort_p, order = np.sort(p)[::-1], np.argsort(p)[::-1]  # Sort in descending order
    sort_R = R[order]
    sort_mass = mass[order]
    
    # Copy arrays for modification
    sort_p_1 = sort_p.copy()
    sort_p_1[florderloc] = np.nan
    sort_R_1 = sort_R.copy()
    sort_R_1[florderloc] = np.nan
    sort_mass_1 = sort_mass.copy()
    sort_mass_1[florderloc] = np.nan
    
    # Calculate quality ratio
    ind1 = qac(sort_p)
    ind1[florderloc] = np.nan
    
    num_p = len(p)
    loc = np.arange(num_p)  # Create an array of indices
    
    # Find non-excluded indices
    non_florderloc = np.setdiff1d(loc, florderloc)
    
    # Calculate means of non-excluded elements
    R_mean = np.nanmean(sort_R[non_florderloc])
    mass_mean = np.nanmean(sort_mass[non_florderloc])
    p_mean = np.nanmean(sort_p[non_florderloc])
    
    # Identify points that meet all three criteria
    a = (sort_R_1 >= R_mean)
    b = (sort_mass_1 >= mass_mean)
    c = (sort_p_1 >= p_mean)
    
    # Combined criteria - mimicking MATLAB's logical operations
    combined = np.logical_and.reduce([a, b, c])
    
    # Get indices that meet criteria
    resolution = loc[combined]
    
    # Calculate the final index based on resolution
    if resolution.size > 0:  # Check if resolution is not empty
        ind2 = np.zeros(num_p)
        for i in range(num_p):
            gd = np.arange(i + 1)  # Python range is exclusive of the end
            common_cn = np.intersect1d(gd, resolution)
            ind2[i] = len(common_cn) / len(resolution)
        
        # Combine indicators
        ind = ind1 * ind2
    else:
        # If no points meet criteria, use only ind1
        ind = ind1
    
    # Handle case when all values in ind might be NaN
    # Find maximum value and indices where it occurs
    if np.all(np.isnan(ind)):
        NAB = np.array([], dtype=int)
    else:
        max_ind_val = np.nanmax(ind)
        NAB = np.where(ind == max_ind_val)[0]
    
    return NAB, resolution