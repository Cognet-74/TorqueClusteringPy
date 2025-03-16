import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from hungarian import hungarian  # Assume this is an equivalent implementation

def spconvert_from_triplets(triplets, shape=None):
    """
    Construct a sparse matrix from a list of triplets.
    
    Parameters:
      triplets: a 2D numpy array with three columns [row, col, value],
                where row and col are 1-indexed (MATLAB-style).
      shape   : tuple (optional) specifying the shape of the output matrix.
    
    Returns:
      A scipy.sparse.csr_matrix with rows and columns converted to 0-indexing.
    """
    triplets = np.array(triplets)
    
    if triplets.size == 0:
        if shape is None:
            shape = (0, 0)
        return csr_matrix(shape)
        
    rows = triplets[:, 0] - 1  # convert from 1-indexed to 0-indexed
    cols = triplets[:, 1] - 1
    data = triplets[:, 2]
    
    if shape is None:
        shape = (int(rows.max()) + 1, int(cols.max()) + 1)
        
    # Return as CSR for more efficient matrix multiplication
    return csr_matrix((data, (rows, cols)), shape=shape)

def accuracy(true_labels, cluster_labels):
    """
    Compute clustering accuracy using the true and cluster labels.
    
    Inputs:
      true_labels    : 1D array-like of true labels (length n).
      cluster_labels : 1D array-like of cluster labels (length n).
    
    Output:
      score : Clustering accuracy in percent.
    
    The function computes a confusion matrix and uses the Hungarian algorithm
    to find the optimal matching between clusters and true labels.
    """
    true_labels = np.asarray(true_labels).flatten()
    cluster_labels = np.asarray(cluster_labels).flatten()
    
    # Check if input arrays have the same length
    if len(true_labels) != len(cluster_labels):
        raise ValueError(f"Length mismatch: true_labels has length {len(true_labels)}, "
                         f"but cluster_labels has length {len(cluster_labels)}.")
    
    n = len(true_labels)
    
    if n == 0:
        return 0.0
        
    # Convert to consecutive integers starting from 1
    # This ensures we don't have unnecessary sparsity from large label values
    unique_true = np.unique(true_labels)
    unique_cluster = np.unique(cluster_labels)
    
    true_map = {label: i+1 for i, label in enumerate(unique_true)}
    cluster_map = {label: i+1 for i, label in enumerate(unique_cluster)}
    
    mapped_true = np.array([true_map[label] for label in true_labels])
    mapped_cluster = np.array([cluster_map[label] for label in cluster_labels])
    
    # Get dimensions for confusion matrix
    n_true_clusters = len(unique_true)
    n_pred_clusters = len(unique_cluster)
    
    # Construct the "cat" matrix for true labels
    triplets_cat = np.column_stack((np.arange(1, n+1), mapped_true, np.ones(n)))
    cat = spconvert_from_triplets(triplets_cat, shape=(n, n_true_clusters))
    
    # Construct the "cls" matrix for cluster labels
    triplets_cls = np.column_stack((np.arange(1, n+1), mapped_cluster, np.ones(n)))
    cls = spconvert_from_triplets(triplets_cls, shape=(n, n_pred_clusters))
    
    # Transpose to make rows correspond to cluster labels (already in CSR format)
    cls = cls.transpose()
    
    # Compute the confusion matrix using sparse matrix multiplication
    # This avoids converting to dense arrays until necessary
    cmat = cls.dot(cat)
    
    # Only convert to array for the Hungarian algorithm when needed
    # For very large matrices, you might want to implement a sparse version of Hungarian
    cmat_array = cmat.toarray()
    
    # Call the Hungarian algorithm on the negative confusion matrix
    Matching, cost = hungarian(-cmat_array)
    
    # Compute accuracy as 100 * (-cost / n)
    score = 100 * (-cost / n)
    return score

def accuracy_with_mapping(true_labels, cluster_labels):
    """
    Compute clustering accuracy and return the optimal label mapping.
    
    Inputs:
      true_labels    : 1D array-like of true labels.
      cluster_labels : 1D array-like of cluster labels.
    
    Outputs:
      score : Clustering accuracy in percent.
      mapping : Dictionary mapping cluster labels to true labels.
    """
    true_labels = np.asarray(true_labels).flatten()
    cluster_labels = np.asarray(cluster_labels).flatten()
    
    # Check if input arrays have the same length
    if len(true_labels) != len(cluster_labels):
        raise ValueError(f"Length mismatch: true_labels has length {len(true_labels)}, "
                         f"but cluster_labels has length {len(cluster_labels)}.")
    
    n = len(true_labels)
    
    if n == 0:
        return 0.0, {}
        
    # Convert to consecutive integers starting from 1
    unique_true = np.unique(true_labels)
    unique_cluster = np.unique(cluster_labels)
    
    true_map = {label: i+1 for i, label in enumerate(unique_true)}
    cluster_map = {label: i+1 for i, label in enumerate(unique_cluster)}
    
    # Create reverse maps to convert back after matching
    true_reverse_map = {i+1: label for i, label in enumerate(unique_true)}
    cluster_reverse_map = {i+1: label for i, label in enumerate(unique_cluster)}
    
    mapped_true = np.array([true_map[label] for label in true_labels])
    mapped_cluster = np.array([cluster_map[label] for label in cluster_labels])
    
    # Get dimensions for confusion matrix
    n_true_clusters = len(unique_true)
    n_pred_clusters = len(unique_cluster)
    
    # Construct sparse matrices
    triplets_cat = np.column_stack((np.arange(1, n+1), mapped_true, np.ones(n)))
    cat = spconvert_from_triplets(triplets_cat, shape=(n, n_true_clusters))
    
    triplets_cls = np.column_stack((np.arange(1, n+1), mapped_cluster, np.ones(n)))
    cls = spconvert_from_triplets(triplets_cls, shape=(n, n_pred_clusters))
    cls = cls.transpose()
    
    # Compute confusion matrix
    cmat = cls.dot(cat)
    cmat_array = cmat.toarray()
    
    # Call Hungarian algorithm
    Matching, cost = hungarian(-cmat_array)
    
    # Create label mapping from cluster labels to true labels
    label_mapping = {}
    for cluster_idx, true_idx in enumerate(Matching):
        # Check if true_idx is a scalar
        if isinstance(true_idx, (int, float)):
            if true_idx < n_true_clusters:  # Ensure the index is valid
                original_cluster = cluster_reverse_map[cluster_idx + 1]
                original_true = true_reverse_map[true_idx + 1]
                label_mapping[original_cluster] = original_true
        else:
            # Handle case where true_idx is an array
            # In this case, we want the first valid index
            for idx in true_idx:
                if idx < n_true_clusters:
                    original_cluster = cluster_reverse_map[cluster_idx + 1]
                    original_true = true_reverse_map[idx + 1]
                    label_mapping[original_cluster] = original_true
                    break
    
    # Compute accuracy
    score = 100 * (-cost / n)
    return score, label_mapping

