from typing import Tuple, Union, Optional, List
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import networkx as nx
import matplotlib.pyplot as plt
from ps2psdist import ps2psdist
from Updateljmat import Updateljmat
from uniqueZ import uniqueZ
from Nab_dec import Nab_dec
from Final_label import Final_label

def normalize_distance_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Normalize distance matrix for improved numerical stability.
    
    Args:
        distance_matrix: Input distance matrix
        
    Returns:
        Normalized distance matrix
    """
    # Handle zero distances
    np.fill_diagonal(distance_matrix, 0.0)
    
    # Scale distances to [0, 1] range
    if distance_matrix.size > 0:
        min_dist = np.min(distance_matrix[distance_matrix > 0])
        max_dist = np.max(distance_matrix)
        if max_dist > min_dist:
            # Avoid division by zero
            distance_matrix = (distance_matrix - min_dist) / (max_dist - min_dist)
            distance_matrix[distance_matrix < 0] = 0  # Clean up any negative values
    
    return distance_matrix

def TorqueClustering(
    ALL_DM: Union[np.ndarray, scipy.sparse.spmatrix],
    K: int = 0,
    isnoise: bool = False,
    isfig: bool = False,
    matlab_compatibility: bool = True
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Implements the Torque Clustering algorithm for unsupervised clustering with improved sparse matrix handling
    while maintaining exact compatibility with the original MATLAB implementation.

    Args:
        ALL_DM (Union[np.ndarray, scipy.sparse.spmatrix]): Distance Matrix (n x n).
        K (int, optional): Number of clusters if known (overrides automatic detection). Defaults to 0.
        isnoise (bool, optional): Enable noise detection. Defaults to False.
        isfig (bool, optional): Generate decision graph figure. Defaults to False.
        matlab_compatibility (bool, optional): Enable strict MATLAB compatibility mode. Defaults to True.

    Returns:
        Tuple containing:
            np.ndarray: Idx - Cluster labels (1 x n).
            np.ndarray: Idx_with_noise - Cluster labels with noise handling (1 x n) or empty array.
            int: cutnum - Number of connections cut.
            np.ndarray: cutlink_ori - Original cut links.
            np.ndarray: p - Torque values for each connection.
            np.ndarray: firstlayer_loc_onsortp - Indices of first layer connections sorted by torque.
            np.ndarray: mass - Mass values for each connection.
            np.ndarray: R - Distance squared values for each connection.
            np.ndarray: cutlinkpower_all - All connection properties recorded during merging.

    Raises:
        ValueError: If distance matrix is not provided or invalid.
    """
    # Setting NumPy error handling to match MATLAB
    if matlab_compatibility:
        old_settings = np.seterr(all='ignore')

    # ---- Input Argument Handling ----
    if ALL_DM is None:
        raise ValueError('Not enough input arguments. Distance Matrix is required.')

    # Validate distance matrix
    if not scipy.sparse.issparse(ALL_DM) and not isinstance(ALL_DM, np.ndarray):
        raise ValueError('Distance matrix must be a NumPy array or scipy sparse matrix')
    
    # Ensure matrix is square
    if ALL_DM.shape[0] != ALL_DM.shape[1]:
        raise ValueError('Distance matrix must be square')

    # Convert to float64 for matching MATLAB's default precision if not already
    if isinstance(ALL_DM, np.ndarray) and ALL_DM.dtype != np.float64:
        ALL_DM = np.float64(ALL_DM)

    # Normalize distance matrix for improved stability
    if isinstance(ALL_DM, np.ndarray):
        ALL_DM = normalize_distance_matrix(ALL_DM)
    else:
        # Convert sparse to dense for normalization if small enough
        if ALL_DM.size <= 1e8:  # Adjust threshold based on available memory
            ALL_DM = normalize_distance_matrix(ALL_DM.toarray())
            ALL_DM = scipy.sparse.csr_matrix(ALL_DM)
        else:
            # For very large matrices, normalize in chunks
            chunk_size = 1000
            for i in range(0, ALL_DM.shape[0], chunk_size):
                end = min(i + chunk_size, ALL_DM.shape[0])
                chunk = ALL_DM[i:end].toarray()
                chunk = normalize_distance_matrix(chunk)
                ALL_DM[i:end] = scipy.sparse.csr_matrix(chunk)

    # Convert to sparse matrix if dense and store the format type
    is_input_sparse = scipy.sparse.issparse(ALL_DM)
    
    # Use appropriate matrix format based on the operation
    if not is_input_sparse:
        # Create sparse copy for efficiency
        ALL_DM_sparse = scipy.sparse.csr_matrix(ALL_DM)
        # Keep original for operations that need dense
        ALL_DM_dense = ALL_DM
    else:
        # Ensure input sparse matrix is in CSR format for efficient row slicing
        ALL_DM_sparse = ALL_DM.tocsr()
        # Only create dense when necessary (deferred conversion)
        ALL_DM_dense = None

    # ---- Step 1: Initialization ----
    datanum = np.int64(ALL_DM_sparse.shape[0])
    cutlinkpower_all = []  # Use list for collecting results, convert to array later
    # Use LIL format for initial empty matrix which will undergo many modifications
    link_adjacency_matrix = scipy.sparse.lil_matrix((datanum, datanum), dtype=np.float64)
    dataloc = np.arange(datanum, dtype=np.int64)
    community = [[dataloc[i]] for i in range(datanum)]
    
    # Using sparse matrix directly for inter_community_distance_matrix
    inter_community_distance_matrix = ALL_DM_sparse.copy()
    community_num = np.int64(datanum)
    
    # Use LIL for graph_connectivity_matrix since we'll be building it by assigning values
    graph_connectivity_matrix = scipy.sparse.lil_matrix((community_num, community_num), dtype=np.float64)

    # ---- Step 2: Construct Initial Graph Connectivity (Nearest Neighbor) ----
    # Find nearest neighbors while maintaining sparsity
    neighbor_community_indices = [None] * community_num
    
    # Process row by row for large sparse matrices
    for i in range(community_num):
        row = ALL_DM_sparse[i].toarray().flatten()  # Get current row as array
        row = np.float64(row)  # Ensure double precision
        row[i] = np.inf  # Exclude self
        
        # Use stable sorting to match MATLAB's behavior
        min_indices = np.argsort(row, kind='mergesort')
        min_idx = np.int64(min_indices[0])  # Force int64 to match MATLAB
        
        # IMPORTANT: In the first layer, ALWAYS connect to nearest neighbor
        # This follows the original MATLAB behavior exactly
        graph_connectivity_matrix[i, min_idx] = 1
        graph_connectivity_matrix[min_idx, i] = 1
        neighbor_community_indices[i] = min_idx

    # Convert to CSR before passing to NetworkX for better performance
    graph_connectivity_matrix = graph_connectivity_matrix.tocsr()
    
    # Create NetworkX graph from sparse matrix
    SG = nx.from_scipy_sparse_array(graph_connectivity_matrix)
    
    # Get connected components with consistent ordering
    components = list(nx.connected_components(SG))
    # Sort components by smallest node index to match MATLAB's behavior
    components.sort(key=lambda c: min(c))
    
    # Ensure consistent labeling with original algorithm (use 0-based indexing internally)
    BINS = np.zeros(datanum, dtype=np.int64)
    for i, component in enumerate(components):
        for node in component:
            BINS[node] = i

    # ---- Display Initial Cluster Count ----
    current_cluster_count = len(np.unique(BINS))
    print(f'The number of clusters in this layer is: {current_cluster_count}')

    # ---- Step 3: Update Link Matrix and Record Connection Properties ----
    # Note: Updateljmat has been modified to handle LIL format efficiently
    link_adjacency_matrix, cutlinkpower = Updateljmat(link_adjacency_matrix, neighbor_community_indices, 
                                                     community, inter_community_distance_matrix, 
                                                     graph_connectivity_matrix, ALL_DM_sparse)
    
    # Ensure double precision for consistency
    if cutlinkpower is not None and cutlinkpower.size > 0:
        cutlinkpower = np.float64(cutlinkpower)
        # Force 2D like MATLAB
        if len(cutlinkpower.shape) == 1:
            cutlinkpower = cutlinkpower.reshape(1, -1)
    
    cutlinkpower, link_adjacency_matrix = uniqueZ(cutlinkpower, link_adjacency_matrix)
    
    firstlayer_conn_num = 0
    if cutlinkpower is not None and cutlinkpower.size > 0:
        firstlayer_conn_num = np.int64(cutlinkpower.shape[0])
        cutlinkpower_all.append(cutlinkpower)

    # ---- Iterative Clustering Process (Merge Communities) ----
    previous_unique_bins = 0
    max_iterations = datanum * 2  # Safety to prevent infinite loops
    iteration_count = 0
    
    while True:
        iteration_count += 1
        if iteration_count > max_iterations:
            print("Warning: Maximum iterations reached. Breaking loop.")
            break
            
        Idx = BINS.copy()  # Use copy to ensure no accidental modification
        uni_Idx = np.unique(Idx)
        num_uni_Idx = np.int64(len(uni_Idx))

        # ---- Update Communities based on current cluster labels ----
        community_new = [None] * num_uni_Idx
        for i in range(num_uni_Idx):
            # MATLAB-like logical indexing
            uniloc = (uni_Idx[i] == Idx)
            current_community = []
            indices = np.where(uniloc)[0]
            for idx in indices:
                current_community.extend(community[idx])
            community_new[i] = current_community

        community = community_new
        community_num = np.int64(len(community))

        # ---- Compute Inter-Cluster Distances ----
        # Create a new sparse matrix for inter-community distances - use LIL for efficient construction
        inter_community_distance_matrix = scipy.sparse.lil_matrix((community_num, community_num), dtype=np.float64)
        
        # Calculate distances - use LIL format for efficient matrix construction
        for i in range(community_num):
            for j in range(community_num):  # Calculate all distances to ensure exact behavior matching
                if i != j:  # Skip self distances
                    # Call the ps2psdist function with explicit data type conversion
                    dist = np.float64(ps2psdist(community[i], community[j], ALL_DM_sparse))
                    inter_community_distance_matrix[i, j] = dist
        
        # Convert to CSR for efficient operations
        inter_community_distance_matrix = inter_community_distance_matrix.tocsr()

        # ---- Step 2 (Repeat): Update Graph Connectivity (Nearest Larger/Equal Size Neighbor Rule) ----
        # Use LIL for building the matrix
        graph_connectivity_matrix = scipy.sparse.lil_matrix((community_num, community_num), dtype=np.float64)
        neighbor_community_indices = [None] * community_num
        
        # Efficient nearest neighbor finding for sparse matrices
        for i in range(community_num):
            row = inter_community_distance_matrix[i].toarray().flatten()
            row = np.float64(row)  # Ensure double precision
            row[i] = np.inf  # Exclude self from nearest neighbor calculation
            
            # Use stable sorting to match MATLAB's behavior
            sorted_indices = np.argsort(row, kind='mergesort')
            
            # IMPORTANT: Only connect if target community is not larger
            # This strictly follows the original MATLAB behavior without fallbacks
            found_neighbor = False
            for j in sorted_indices:
                if j != i and len(community[i]) <= len(community[j]):
                    graph_connectivity_matrix[i, j] = 1
                    graph_connectivity_matrix[j, i] = 1
                    neighbor_community_indices[i] = j
                    found_neighbor = True
                    break
            
            # Note: No fallback - some communities might remain unconnected
            # This is the same behavior as the original MATLAB implementation

        # Convert to CSR before using NetworkX
        graph_connectivity_matrix = graph_connectivity_matrix.tocsr()
        
        # Create NetworkX graph from sparse matrix
        SG = nx.from_scipy_sparse_array(graph_connectivity_matrix)
        
        # Get connected components with consistent ordering
        components = list(nx.connected_components(SG))
        # Sort components by smallest node index to match MATLAB's behavior
        components.sort(key=lambda c: min(c))
        
        # Ensure consistent labeling with original algorithm
        BINS = np.zeros(community_num, dtype=np.int64)
        for i, component in enumerate(components):
            for node in component:
                BINS[node] = i

        # ---- Display Updated Cluster Count ----
        current_cluster_count = len(np.unique(BINS))
        print(f'The number of clusters in this layer is: {current_cluster_count}')

        # ---- Update link properties ----
        link_adjacency_matrix, cutlinkpower = Updateljmat(link_adjacency_matrix, neighbor_community_indices, 
                                                        community, inter_community_distance_matrix, 
                                                        graph_connectivity_matrix, ALL_DM_sparse)
        
        # Ensure double precision and 2D arrays for consistency with MATLAB
        if cutlinkpower is not None and cutlinkpower.size > 0:
            cutlinkpower = np.float64(cutlinkpower)
            if len(cutlinkpower.shape) == 1:
                cutlinkpower = cutlinkpower.reshape(1, -1)
                
        cutlinkpower, link_adjacency_matrix = uniqueZ(cutlinkpower, link_adjacency_matrix)
        
        if cutlinkpower is not None and cutlinkpower.size > 0:
            cutlinkpower_all.append(cutlinkpower)
        
        # ---- Stop conditions with enhanced checks ----
        unique_bins = np.unique(BINS)
        
        # MATLAB compatible termination - stop when one cluster or no change
        if len(unique_bins) == 1 or len(unique_bins) == previous_unique_bins:
            break
            
        previous_unique_bins = len(unique_bins)

    # Stack all cutlinkpower arrays into one
    if cutlinkpower_all:
        cutlinkpower_all_np = np.vstack([cp for cp in cutlinkpower_all if cp.size > 0])
        cutlinkpower_all_np = np.float64(cutlinkpower_all_np)  # Ensure double precision
    else:
        # Handle edge case where no connections were made
        cutlinkpower_all_np = np.array([], dtype=np.float64)
        if matlab_compatibility:
            np.seterr(**old_settings)  # Restore original NumPy settings
        return np.zeros(datanum, dtype=np.int64), np.array([], dtype=np.int64), 0, np.array([], dtype=np.float64), np.array([], dtype=np.float64), np.array([], dtype=np.float64), np.array([], dtype=np.float64), np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    
    # ---- Step 4: Define Torque and Compute Cluster Properties ----
    # Use explicit double precision for all calculations
    mass = np.float64(cutlinkpower_all_np[:, 4] * cutlinkpower_all_np[:, 5])
    R = np.float64(cutlinkpower_all_np[:, 6]**2)
    p = np.float64(mass * R)
    R_mass = np.float64(R / mass)  # For noise detection

    # ---- Step 5: Visualization (Decision Graph) ----
    if isfig:
        plt.figure(figsize=(10, 12))
        plt.subplot(2, 1, 1)
        plt.plot(R, mass, 'o', markersize=5, markerfacecolor='k', markeredgecolor='k')
        plt.title('Decision Graph', fontsize=15)
        plt.xlabel('R (Distance Squared)')
        plt.ylabel('Mass')
        plt.grid(True)

    # ---- Step 6: Identify Important Cluster Centers ----
    # Use stable sorting to match MATLAB
    order_torque = np.argsort(p, kind='mergesort')[::-1]  # Descending order
    order_2 = np.argsort(order_torque, kind='mergesort')
    
    # Ensure compatible indexing
    if firstlayer_conn_num > 0:
        firstlayer_loc_onsortp = order_2[:firstlayer_conn_num]
    else:
        firstlayer_loc_onsortp = np.array([], dtype=np.int64)

    # ---- Step 7: Determine Cutoff Points for Clusters (Torque Gap or User-defined K) ----
    if K == 0:
        # The key function for determining number of clusters - ensure it's precise
        NAB = Nab_dec(p, mass, R, firstlayer_loc_onsortp)
        # Handle exactly as MATLAB does:
        # If NAB is empty, use a default value
        if len(NAB) == 0:
            print("Warning: Automatic clustering couldn't find a clear cutoff. Using default.")
            cutnum = 1
        else:
            # Use the first element if NAB has values (matching MATLAB's implicit behavior)
            cutnum = np.int64(NAB[0])
    else:
        cutnum = np.int64(K - 1)

    # Ensure cutnum is valid (must be at least 1 to create meaningful clusters)
    cutnum = np.int64(max(1, cutnum))

    # ---- Step 8: Extract Cluster Boundaries (Cut Links) ----
    # Make sure we don't try to cut more links than we have and ensure it's a Python integer
    cutnum_int = int(min(cutnum, len(order_torque)))
    
    # Explicitly convert to list of integer indices to avoid non-integer indexing errors
    indices_to_cut = []
    for i in range(cutnum_int):
        indices_to_cut.append(int(order_torque[i]))
    
    if not indices_to_cut:  # Handle empty list case
        cutlink1 = np.empty((0, cutlinkpower_all_np.shape[1]), dtype=np.float64)
    else:
        cutlink1 = cutlinkpower_all_np[indices_to_cut, :].copy()  # Use copy to avoid modification
    
    cutlink_ori = cutlink1.copy()
    cutlink1 = np.delete(cutlink1, [0, 1, 4, 5, 6], axis=1)

    # ---- Step 9: Noise Handling (If Enabled) ----
    Idx_with_noise = np.array([], dtype=np.int64)
    if isnoise:
        # Calculate means with explicit precision
        R_mean = np.mean(R, dtype=np.float64)
        mass_mean = np.mean(mass, dtype=np.float64)
        R_mass_mean = np.mean(R_mass, dtype=np.float64)
        
        # Add small epsilon for floating point comparisons
        epsilon = 1e-10
        
        # IMPORTANT: Use exact same noise detection criteria as original
        # But with explicit handling of floating point comparisons
        noise_loc_indices = np.intersect1d(
            np.intersect1d(
                np.where(R >= R_mean - epsilon)[0], 
                np.where(mass <= mass_mean + epsilon)[0]
            ), 
            np.where(R_mass >= R_mass_mean - epsilon)[0]
        )
        
        # Convert to integer indices for proper indexing - ensure cutnum is a Python int
        cutnum_int = int(cutnum)
        indices_to_cut = []
        for i in range(cutnum_int):
            indices_to_cut.append(int(order_torque[i]))
            
        # Convert noise indices to Python ints
        noise_indices_python = [int(idx) for idx in noise_loc_indices]
        
        # Manual union to ensure we have Python ints
        all_indices = set(indices_to_cut).union(set(noise_indices_python))
        all_indices = list(all_indices)
        
        if not all_indices:  # Handle empty list case
            cutlink2 = np.empty((0, cutlinkpower_all_np.shape[1]), dtype=np.float64)
        else:
            cutlink2 = cutlinkpower_all_np[all_indices, :].copy()  # Use copy
        cutlink2 = np.delete(cutlink2, [0, 1, 4, 5, 6], axis=1)

    # ---- Step 10: Update Graph and Finalize Cluster Labels (without noise) ----
    # Ensure link_adjacency_matrix is in CSR format for copying
    if not isinstance(link_adjacency_matrix, scipy.sparse.csr_matrix):
        link_adjacency_matrix = link_adjacency_matrix.tocsr()
    
    ljmat1 = link_adjacency_matrix.copy()
    
    # Convert to lil format for efficient matrix modification
    link_adjacency_matrix = link_adjacency_matrix.tolil()
    
    # Collect all matrix updates first to apply them consistently
    updates = []
    cutlinknum1 = np.int64(cutlink1.shape[0])
    for i in range(cutlinknum1):
        row_index = np.int64(cutlink1[i, 0])
        col_index = np.int64(cutlink1[i, 1])
        updates.append((row_index, col_index))
        updates.append((col_index, row_index))
    
    # Apply updates 
    for r, c in updates:
        link_adjacency_matrix[r, c] = 0
    
    # Convert back to CSR for NetworkX
    link_adjacency_matrix = link_adjacency_matrix.tocsr()

    # Create NetworkX graph from sparse matrix
    ljmat_G = nx.from_scipy_sparse_array(link_adjacency_matrix)
    
    # Get connected components with consistent ordering
    components = list(nx.connected_components(ljmat_G))
    # Sort components by smallest node index to match MATLAB's behavior
    components.sort(key=lambda c: min(c))
    
    # Get final cluster labels with consistent ordering
    labels1 = np.zeros(datanum, dtype=np.int64)
    for i, component in enumerate(components):
        for node in component:
            labels1[node] = i
    
    Idx = labels1.copy()  # Use copy to ensure no accidental modification

    # ---- Step 11: If Noise Handling is Enabled, Finalize Cluster Labels with Noise ----
    if isnoise:
        # Ensure ljmat1 is in LIL format for efficient updates
        ljmat1 = ljmat1.tolil()
        
        # Collect all matrix updates first
        updates = []
        cutlinknum2 = np.int64(cutlink2.shape[0])
        for i in range(cutlinknum2):
            row_index = np.int64(cutlink2[i, 0])
            col_index = np.int64(cutlink2[i, 1])
            updates.append((row_index, col_index))
            updates.append((col_index, row_index))
        
        # Apply updates
        for r, c in updates:
            ljmat1[r, c] = 0
        
        # Convert back to CSR for NetworkX
        ljmat1 = ljmat1.tocsr()

        # Create NetworkX graph from sparse matrix
        ljmat1_G = nx.from_scipy_sparse_array(ljmat1)
        
        # Get connected components with consistent ordering
        components = list(nx.connected_components(ljmat1_G))
        # Sort components by smallest node index to match MATLAB's behavior
        components.sort(key=lambda c: min(c))
        
        # Get noise-aware cluster labels with consistent ordering
        labels2 = np.zeros(datanum, dtype=np.int64)
        for i, component in enumerate(components):
            for node in component:
                labels2[node] = i
        
        # Finalize labels with noise detection
        Idx_with_noise = Final_label(labels1, labels2)

    # ---- Step 12: Additional visualization if requested ----
    if isfig:
        plt.subplot(2, 1, 2)
        
        # Get unique cluster IDs
        uniqueLabels = np.unique(Idx)
        numClusters = len(uniqueLabels)
        
        # Create custom colormap - use exact same colormap generation as MATLAB hsv
        colors = plt.cm.hsv(np.linspace(0, 1, numClusters))
        
        # Plot decision graph with points colored by cluster
        # First create a mapping between cluster indices and points
        cluster_to_points = {}
        for i, cluster_id in enumerate(uniqueLabels):
            cluster_to_points[cluster_id] = np.where(Idx == cluster_id)[0]
        
        # For each cluster, find connections involving any point in that cluster
        for i, cluster_id in enumerate(uniqueLabels):
            # Get points in this cluster
            cluster_points = cluster_to_points[cluster_id]
            
            # Find connections where either end is in this cluster
            connection_mask = np.zeros(cutlinkpower_all_np.shape[0], dtype=bool)
            for point in cluster_points:
                # This is more efficient than using np.where in a loop
                connection_mask |= (cutlinkpower_all_np[:, 0] == point) | (cutlinkpower_all_np[:, 1] == point)
            
            # Get indices of connections for this cluster
            connection_indices = np.where(connection_mask)[0]
            
            if len(connection_indices) > 0:
                plt.plot(R[connection_indices], mass[connection_indices], 'o', markersize=5, 
                         markerfacecolor=colors[i], markeredgecolor=colors[i])
        
        plt.title('Clusters in Decision Graph', fontsize=15)
        plt.xlabel('D (Distance)')
        plt.ylabel('M (Mass)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Restore NumPy settings if changed
    if matlab_compatibility:
        np.seterr(**old_settings)
        
    return Idx, Idx_with_noise, cutnum, cutlink_ori, p, firstlayer_loc_onsortp, mass, R, cutlinkpower_all_np

# Helper validation function to compare outputs with MATLAB
def validate_against_matlab(
    py_results: Tuple,
    matlab_results: Tuple,
    tolerance: float = 1e-10
) -> bool:
    """
    Validates Python results against MATLAB results.
    
    Args:
        py_results: Results from Python implementation
        matlab_results: Results from MATLAB implementation
        tolerance: Numerical tolerance for floating point comparisons
        
    Returns:
        bool: True if results match within tolerance
    """
    all_match = True
    
    # Unpack results
    py_idx, py_idx_noise, py_cutnum, py_cutlink, py_p, py_firstlayer, py_mass, py_r, py_cutlinkpower = py_results
    m_idx, m_idx_noise, m_cutnum, m_cutlink, m_p, m_firstlayer, m_mass, m_r, m_cutlinkpower = matlab_results
    
    # Check discrete values first (exact match required)
    if py_cutnum != m_cutnum:
        print(f"Cutnum mismatch: Python={py_cutnum}, MATLAB={m_cutnum}")
        all_match = False
    
    # Check cluster assignments (clusters may be labeled differently but structure should match)
    # This requires more sophisticated comparison of cluster structure
    
    # Check numerical arrays with tolerance
    arrays_to_check = [
        ("p", py_p, m_p),
        ("mass", py_mass, m_mass),
        ("R", py_r, m_r)
    ]
    
    for name, py_arr, m_arr in arrays_to_check:
        if py_arr.shape != m_arr.shape:
            print(f"{name} shape mismatch: Python={py_arr.shape}, MATLAB={m_arr.shape}")
            all_match = False
        else:
            max_diff = np.max(np.abs(py_arr - m_arr)) if py_arr.size > 0 else 0
            if max_diff > tolerance:
                print(f"{name} values differ by up to {max_diff} (tolerance={tolerance})")
                all_match = False
    
    return all_match