from typing import Union, List
import numpy as np
import numpy.typing as npt

def Final_label(
    labels1: Union[List[int], npt.NDArray[np.int_]],
    labels2: Union[List[int], npt.NDArray[np.int_]]
) -> npt.NDArray[np.int_]:
    """
    FINAL_LABEL: Compute the final cluster labels by filtering out noise.
    
    This function processes each unique label in labels1 and finds the largest
    subset of points from labels2 that is fully contained within each cluster
    of labels1. Points that are not part of this main subset are marked as noise.
    
    Parameters:
        labels1 (Union[List[int], npt.NDArray[np.int_]]): 
            Cluster labels from method 1.
        labels2 (Union[List[int], npt.NDArray[np.int_]]): 
            Cluster labels from method 2.
                  
    Returns:
        npt.NDArray[np.int_]: Final cluster labels, where noise points are assigned the label 0.
    
    Note:
        This implementation precisely matches the MATLAB version's behavior.
    """
    # Convert to numpy arrays while preserving input data types
    labels1 = np.asarray(labels1)
    labels2 = np.asarray(labels2)
    
    # Start with labels1 as the initial label assignment
    Idx = labels1.copy()
    
    # Get unique labels from both inputs
    uni_labels1 = np.unique(labels1)
    uni_labels2 = np.unique(labels2)
    
    # Process each unique label in labels1
    for i in range(len(uni_labels1)):
        # Find indices in labels1 that equal this unique label
        class_loc = np.where(labels1 == uni_labels1[i])[0]
        mainloc = np.array([])  # Initialize main location as empty
        
        # For each unique label in labels2
        for j in range(len(uni_labels2)):
            # Find indices in labels2 that equal this unique label
            zj_loc = np.where(labels2 == uni_labels2[j])[0]
            
            # Match MATLAB's behavior: all(ismember()) && numel() > numel()
            # Note: all([]) is True in MATLAB
            ismember_result = np.isin(zj_loc, class_loc)
            if len(ismember_result) > 0:
                all_ismember = np.all(ismember_result)
            else:
                all_ismember = True  # Match MATLAB's all([]) behavior
                
            if all_ismember and len(zj_loc) > len(mainloc):
                mainloc = zj_loc
        
        # Compute the indices that are in class_loc but not in mainloc (i.e., noise)
        class_noise_loc = np.setdiff1d(class_loc, mainloc)
        
        # Set the corresponding entries in Idx to 0
        Idx[class_noise_loc] = 0
        
    return Idx

# Example usage:
if __name__ == '__main__':
    # Example input membership vectors:
    labels1 = [1, 1, 2, 2, 3, 3, 1, 2, 3]
    labels2 = [1, 1, 2, 2, 3, 3, 2, 2, 3]
    
    final_labels = Final_label(labels1, labels2)
    print("Final labels:", final_labels)
    
    # Test with edge cases
    print("\nEdge cases:")
    # Empty arrays
    print("Empty arrays:", Final_label([], []))
    # Single cluster
    print("Single cluster:", Final_label([1, 1, 1], [1, 2, 3]))
    # All noise
    print("All noise:", Final_label([1, 2, 3], [4, 5, 6]))