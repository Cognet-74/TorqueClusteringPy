import numpy as np

def ps2psdist(Loc_dataset1, Loc_dataset2, DM):
    """
    Compute a distance measure between two sets of points using a precomputed distance matrix.
    
    This function extracts the submatrix of distances from DM corresponding to the indices
    in Loc_dataset1 (first group) and Loc_dataset2 (second group), and then computes a single 
    scalar distance measure between these two groups.
    
    The active implementation returns the minimum distance between any point in the first group
    and any point in the second group.
    
    Alternative approaches (commented out) include:
      - Directly returning the distance if each group contains exactly one point.
      - Computing the average distance between the groups.
      - Computing the maximum distance between the groups.
      - Computing the norm of the difference between the means of the two groups.
    
    Parameters:
        Loc_dataset1 : list or array-like
            Indices of points in the first group.
        Loc_dataset2 : list or array-like
            Indices of points in the second group.
        DM : 2D numpy array
            Precomputed distance matrix where DM[i, j] is the distance between point i and point j.
    
    Returns:
        Cdist : float
            The computed distance measure between the two groups.
    """
    
    # If both groups contain exactly one point, one could simply return the distance between these two points.
    # Uncomment the following if you want to use this approach:
    # if len(Loc_dataset1) == 1 and len(Loc_dataset2) == 1:
    #     return DM[Loc_dataset1[0], Loc_dataset2[0]]
    
    # Alternative approach: Exclude rows and columns not in the two groups.
    # For example, if you wanted to remove the other locations, you could do:
    # datanum = DM.shape[0]
    # all_loc = np.arange(datanum)
    # diff_Loc1 = np.setdiff1d(all_loc, Loc_dataset1)
    # diff_Loc2 = np.setdiff1d(all_loc, Loc_dataset2)
    # dists = DM.copy()
    # dists = np.delete(dists, diff_Loc1, axis=0)
    # dists = np.delete(dists, diff_Loc2, axis=1)
    #
    # The above code removes rows not in Loc_dataset1 and columns not in Loc_dataset2.
    
    # Extract the submatrix of DM corresponding to the two sets of indices.
    dists = DM[np.ix_(Loc_dataset1, Loc_dataset2)]
    
    # Alternative 1: Compute the average distance between the two groups.
    # (Translation of "类间平均距离": "average distance between the classes")
    # Uncomment the following line to use this approach:
    # Cdist = np.mean(dists)
    
    # Alternative 2: Compute the maximum distance between the two groups.
    # Uncomment the following line to use this alternative:
    # Cdist = np.max(dists)
    
    # Alternative 3: Compute the norm of the difference between the means of the two groups.
    # (This would use the means of the datasets to compute a distance.)
    # Uncomment the following lines to use this alternative:
    # dataset1_mean = np.mean(DM[Loc_dataset1, :], axis=0)
    # dataset2_mean = np.mean(DM[Loc_dataset2, :], axis=0)
    # Cdist = np.linalg.norm(dataset1_mean - dataset2_mean)
    
    # Active approach: Compute the minimum distance between any two points in the two groups.
    # (Translation of "两类间最短距离": "the shortest distance between the two classes")
    Cdist = np.min(dists)
    
    return Cdist
