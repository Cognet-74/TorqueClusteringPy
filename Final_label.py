import numpy as np

def Final_label(labels1, labels2):
    """
    FINAL_LABEL: Compute the final cluster labels by filtering out noise.
    
    For each unique label in labels1, this function finds the "main" subset of indices 
    from labels2 that is fully contained in the corresponding group of labels1 and has 
    the largest size. Any indices in that group (from labels1) that are not part of the 
    main subset are marked as noise (set to 0).
    
    Parameters:
        labels1 : array-like of integers
                  Cluster labels from method 1.
        labels2 : array-like of integers
                  Cluster labels from method 2.
                  
    Returns:
        Idx : numpy array of integers
              Final cluster labels, where noise points are assigned the label 0.
    """
    labels1 = np.array(labels1)
    labels2 = np.array(labels2)
    
    # Start with labels1 as the initial label assignment.
    Idx = labels1.copy()
    
    # Get unique labels from both inputs.
    uni_labels1 = np.unique(labels1)
    uni_labels2 = np.unique(labels2)
    
    # Process each unique label in labels1.
    for ul in uni_labels1:
        # Find indices in labels1 that equal this unique label.
        class_loc = np.where(labels1 == ul)[0]
        mainloc = np.array([], dtype=int)  # Initialize main location as empty.
        
        # For each unique label in labels2.
        for u2 in uni_labels2:
            # Find indices in labels2 that equal this unique label.
            zj_loc = np.where(labels2 == u2)[0]
            # If all indices in zj_loc are in class_loc and the size is larger than current mainloc:
            if zj_loc.size > 0 and np.all(np.isin(zj_loc, class_loc)) and (zj_loc.size > mainloc.size):
                mainloc = zj_loc
        
        # Compute the indices that are in class_loc but not in mainloc (i.e., noise).
        class_noise_loc = np.setdiff1d(class_loc, mainloc)
        # Set the corresponding entries in Idx to 0.
        Idx[class_noise_loc] = 0
        
    return Idx

# Example usage:
if __name__ == '__main__':
    # Example input membership vectors:
    labels1 = [1, 1, 2, 2, 3, 3, 1, 2, 3]
    labels2 = [1, 1, 2, 2, 3, 3, 2, 2, 3]
    
    final_labels = Final_label(labels1, labels2)
    print("Final labels:", final_labels)
