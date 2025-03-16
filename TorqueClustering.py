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

def TorqueClustering(ALL_DM, K=0, isnoise=False, isfig=False):
    """
    Implements the Torque Clustering algorithm for unsupervised clustering with improved sparse matrix handling
    while maintaining exact compatibility with the original MATLAB implementation.

    Args:
        ALL_DM (numpy.ndarray or scipy.sparse matrix): Distance Matrix (n x n).
        K (int, optional): Number of clusters if known (overrides automatic detection). Defaults to 0.
        isnoise (bool, optional): Enable noise detection. Defaults to False.
        isfig (bool, optional): Generate decision graph figure. Defaults to False.

    Returns:
        Idx (numpy.ndarray): Cluster labels (1 x n).
        Idx_with_noise (numpy.ndarray): Cluster labels with noise handling (1 x n) or empty array.
        cutnum (int): Number of connections cut.
        cutlink_ori (numpy.ndarray): Original cut links.
        p (numpy.ndarray): Torque values for each connection.
        firstlayer_loc_onsortp (numpy.ndarray): Indices of first layer connections sorted by torque.
        mass (numpy.ndarray): Mass values for each connection.
        R (numpy.ndarray): Distance squared values for each connection.
        cutlinkpower_all (numpy.ndarray): All connection properties recorded during merging.
    """

    # ---- Input Argument Handling ----
    if ALL_DM is None:
        raise ValueError('Not enough input arguments. Distance Matrix is required.')

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
    datanum = ALL_DM_sparse.shape[0]
    cutlinkpower_all = []
    # Use LIL format for initial empty matrix which will undergo many modifications
    link_adjacency_matrix = scipy.sparse.lil_matrix((datanum, datanum))
    dataloc = np.arange(datanum)
    community = [[dataloc[i]] for i in range(datanum)]
    
    # Using sparse matrix directly for inter_community_distance_matrix
    inter_community_distance_matrix = ALL_DM_sparse.copy()
    community_num = datanum
    
    # Use LIL for graph_connectivity_matrix since we'll be building it by assigning values
    graph_connectivity_matrix = scipy.sparse.lil_matrix((community_num, community_num))

    # ---- Step 2: Construct Initial Graph Connectivity (Nearest Neighbor) ----
    # Find nearest neighbors while maintaining sparsity
    neighbor_community_indices = [None] * community_num
    
    # Process row by row for large sparse matrices
    for i in range(community_num):
        row = ALL_DM_sparse[i].toarray().flatten()  # Get current row as array
        row[i] = np.inf  # Exclude self
        min_idx = np.argmin(row)
        # IMPORTANT: In the first layer, ALWAYS connect to nearest neighbor
        # This follows the original MATLAB behavior exactly
        graph_connectivity_matrix[i, min_idx] = 1
        graph_connectivity_matrix[min_idx, i] = 1
        neighbor_community_indices[i] = min_idx

    # Convert to CSR before passing to NetworkX for better performance
    graph_connectivity_matrix = graph_connectivity_matrix.tocsr()
    
    # Create NetworkX graph from sparse matrix
    SG = nx.from_scipy_sparse_array(graph_connectivity_matrix)
    
    # Ensure consistent labeling with original algorithm (use 0-based indexing internally)
    BINS = np.zeros(datanum, dtype=int)
    for i, component in enumerate(nx.connected_components(SG)):
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
    cutlinkpower, link_adjacency_matrix = uniqueZ(cutlinkpower, link_adjacency_matrix)
    firstlayer_conn_num = cutlinkpower.shape[0] if cutlinkpower.size > 0 else 0
    
    if cutlinkpower.size > 0:
        cutlinkpower_all.append(cutlinkpower)

    # ---- Iterative Clustering Process (Merge Communities) ----
    while True:
        Idx = BINS
        uni_Idx = np.unique(Idx)
        num_uni_Idx = len(uni_Idx)

        # ---- Update Communities based on current cluster labels ----
        community_new = [None] * num_uni_Idx
        for i in range(num_uni_Idx):
            uniloc = (uni_Idx[i] == Idx)
            current_community = []
            indices = np.where(uniloc)[0]
            for idx in indices:
                current_community.extend(community[idx])
            community_new[i] = current_community

        community = community_new
        community_num = len(community)

        # ---- Compute Inter-Cluster Distances ----
        # Create a new sparse matrix for inter-community distances - use LIL for efficient construction
        inter_community_distance_matrix = scipy.sparse.lil_matrix((community_num, community_num))
        
        # Calculate distances - use LIL format for efficient matrix construction
        for i in range(community_num):
            for j in range(community_num):  # Calculate all distances to ensure exact behavior matching
                if i != j:  # Skip self distances
                    dist = ps2psdist(community[i], community[j], ALL_DM_sparse)
                    inter_community_distance_matrix[i, j] = dist
        
        # Convert to CSR for efficient operations
        inter_community_distance_matrix = inter_community_distance_matrix.tocsr()

        # ---- Step 2 (Repeat): Update Graph Connectivity (Nearest Larger/Equal Size Neighbor Rule) ----
        # Use LIL for building the matrix
        graph_connectivity_matrix = scipy.sparse.lil_matrix((community_num, community_num))
        neighbor_community_indices = [None] * community_num
        
        # Efficient nearest neighbor finding for sparse matrices
        for i in range(community_num):
            row = inter_community_distance_matrix[i].toarray().flatten()
            row[i] = np.inf  # Exclude self from nearest neighbor calculation
            sorted_indices = np.argsort(row)
            
            # IMPORTANT: Only connect if target community is not larger
            # This strictly follows the original MATLAB behavior without fallbacks
            for j in sorted_indices:
                if j != i and len(community[i]) <= len(community[j]):
                    graph_connectivity_matrix[i, j] = 1
                    graph_connectivity_matrix[j, i] = 1
                    neighbor_community_indices[i] = j
                    break
            
            # Note: No fallback - some communities might remain unconnected
            # This is the same behavior as the original MATLAB implementation

        # Convert to CSR before using NetworkX
        graph_connectivity_matrix = graph_connectivity_matrix.tocsr()
        
        # Create NetworkX graph from sparse matrix
        SG = nx.from_scipy_sparse_array(graph_connectivity_matrix)
        
        # Ensure consistent labeling with original algorithm
        BINS = np.zeros(community_num, dtype=int)
        for i, component in enumerate(nx.connected_components(SG)):
            for node in component:
                BINS[node] = i

        # ---- Display Updated Cluster Count ----
        current_cluster_count = len(np.unique(BINS))
        print(f'The number of clusters in this layer is: {current_cluster_count}')

        # ---- Update link properties ----
        link_adjacency_matrix, cutlinkpower = Updateljmat(link_adjacency_matrix, neighbor_community_indices, 
                                                        community, inter_community_distance_matrix, 
                                                        graph_connectivity_matrix, ALL_DM_sparse)
        cutlinkpower, link_adjacency_matrix = uniqueZ(cutlinkpower, link_adjacency_matrix)
        
        if cutlinkpower.size > 0:
            cutlinkpower_all.append(cutlinkpower)
        
        # ---- Stop if only one cluster remains ----
        if len(np.unique(BINS)) == 1:
            break

    # Stack all cutlinkpower arrays into one
    if cutlinkpower_all:
        cutlinkpower_all_np = np.vstack(cutlinkpower_all)
    else:
        # Handle edge case where no connections were made
        cutlinkpower_all_np = np.array([])
        return np.zeros(datanum), np.array([]), 0, np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    
    # ---- Step 4: Define Torque and Compute Cluster Properties ----
    mass = cutlinkpower_all_np[:, 4] * cutlinkpower_all_np[:, 5]
    R = cutlinkpower_all_np[:, 6]**2
    p = mass * R
    R_mass = R / mass  # For noise detection

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
    order_torque = np.argsort(p)[::-1]  # Descending order
    order_2 = np.argsort(order_torque)
    firstlayer_loc_onsortp = order_2[:firstlayer_conn_num]


    # ---- Step 7: Determine Cutoff Points for Clusters (Torque Gap or User-defined K) ----
    if K == 0:
        NAB = Nab_dec(p, mass, R, firstlayer_loc_onsortp)
        # Handle exactly as MATLAB does:
        # If NAB is empty, use a default value
        if len(NAB) == 0:
            print("Warning: Automatic clustering couldn't find a clear cutoff. Using default.")
            cutnum = 1
        else:
            # Use the first element if NAB has values (matching MATLAB's implicit behavior)
            cutnum = int(NAB[0])
    else:
        cutnum = K - 1

    # Ensure cutnum is valid (must be at least 1 to create meaningful clusters)
    cutnum = max(1, cutnum)

    # ---- Step 8: Extract Cluster Boundaries (Cut Links) ----
    # Make sure we don't try to cut more links than we have
    cutnum = min(cutnum, len(order_torque))
    cutlink1 = cutlinkpower_all_np[order_torque[:cutnum], :]
    cutlink_ori = cutlink1.copy()
    cutlink1 = np.delete(cutlink1, [0, 1, 4, 5, 6], axis=1)

    # ---- Step 9: Noise Handling (If Enabled) ----
    Idx_with_noise = np.array([])
    if isnoise:
        # IMPORTANT: Use exact same noise detection criteria as original
        noise_loc_indices = np.intersect1d(
            np.intersect1d(
                np.where(R >= np.mean(R))[0], 
                np.where(mass <= np.mean(mass))[0]
            ), 
            np.where(R_mass >= np.mean(R_mass))[0]
        )
        cutlink2 = cutlinkpower_all_np[np.union1d(order_torque[:cutnum], noise_loc_indices), :]
        cutlink2 = np.delete(cutlink2, [0, 1, 4, 5, 6], axis=1)

    # ---- Step 10: Update Graph and Finalize Cluster Labels (without noise) ----
    # Ensure link_adjacency_matrix is in CSR format for copying
    if not isinstance(link_adjacency_matrix, scipy.sparse.csr_matrix):
        link_adjacency_matrix = link_adjacency_matrix.tocsr()
    
    ljmat1 = link_adjacency_matrix.copy()
    
    # Convert to lil format for efficient matrix modification
    link_adjacency_matrix = link_adjacency_matrix.tolil()
    
    # Cut the links in the original matrix
    cutlinknum1 = cutlink1.shape[0]
    for i in range(cutlinknum1):
        row_index = int(cutlink1[i, 0])
        col_index = int(cutlink1[i, 1])
        link_adjacency_matrix[row_index, col_index] = 0
        link_adjacency_matrix[col_index, row_index] = 0
    
    # Convert back to CSR for NetworkX
    link_adjacency_matrix = link_adjacency_matrix.tocsr()

    # Create NetworkX graph from sparse matrix
    ljmat_G = nx.from_scipy_sparse_array(link_adjacency_matrix)
    
    # Get final cluster labels
    labels1 = np.zeros(datanum, dtype=int)
    for i, component in enumerate(nx.connected_components(ljmat_G)):
        for node in component:
            labels1[node] = i
    
    Idx = labels1

    # ---- Step 11: If Noise Handling is Enabled, Finalize Cluster Labels with Noise ----
    if isnoise:
        # Ensure ljmat1 is in LIL format for efficient updates
        ljmat1 = ljmat1.tolil()
        cutlinknum2 = cutlink2.shape[0]
        for i in range(cutlinknum2):
            row_index = int(cutlink2[i, 0])
            col_index = int(cutlink2[i, 1])
            ljmat1[row_index, col_index] = 0
            ljmat1[col_index, row_index] = 0
        
        # Convert back to CSR for NetworkX
        ljmat1 = ljmat1.tocsr()

        # Create NetworkX graph from sparse matrix
        ljmat1_G = nx.from_scipy_sparse_array(ljmat1)
        
        # Get noise-aware cluster labels
        labels2 = np.zeros(datanum, dtype=int)
        for i, component in enumerate(nx.connected_components(ljmat1_G)):
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
        
        # Create custom colormap
        colors = plt.cm.hsv(np.linspace(0, 1, numClusters))
        
        # Plot decision graph with points colored by cluster - WITH FIX
        for i in range(numClusters):
            clusterIdx = (Idx == uniqueLabels[i])
            
            # Here's the key fix: Map between Idx and R/mass arrays
            # In MATLAB, this mapping happens implicitly during indexing
            # In Python, we need to handle it explicitly
            
            # First, find all points in this cluster
            cluster_points = np.where(clusterIdx)[0]
            
            # Then, find all connections where either end is in this cluster
            # This matches how MATLAB would implicitly map between arrays
            connection_indices = []
            for point in cluster_points:
                # Find connections involving this point
                point_connections = np.where(
                    (cutlinkpower_all_np[:, 0] == point) | 
                    (cutlinkpower_all_np[:, 1] == point)
                )[0]
                connection_indices.extend(point_connections)
            
            # Remove duplicates
            connection_indices = np.unique(connection_indices)
            
            # Only plot if we found connections
            if len(connection_indices) > 0:
                plt.plot(R[connection_indices], mass[connection_indices], 'o', markersize=5, 
                        markerfacecolor=colors[i], markeredgecolor=colors[i])
        
        plt.title('Clusters in Decision Graph', fontsize=15)
        plt.xlabel('D (Distance)')
        plt.ylabel('M (Mass)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return Idx, Idx_with_noise, cutnum, cutlink_ori, p, firstlayer_loc_onsortp, mass, R, cutlinkpower_all_np