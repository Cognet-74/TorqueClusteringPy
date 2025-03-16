import numpy as np
import logging
import scipy.sparse as sp
from mindisttwinsloc import mindisttwinsloc

# Set up logger
logger = logging.getLogger(__name__)

def Updateljmat(old_ljmat, neiborloc, community, commu_DM, G, ALL_DM):
    """
    Update the connectivity matrix (ljmat) and record cut link power information.
    
    Parameters
    ----------
    old_ljmat : numpy.ndarray or scipy.sparse matrix
        The original connectivity (linkage) matrix. Can be either dense or sparse.
    neiborloc : list
        A list where each element is either a neighbor community index or 
        None (or empty list) indicating no neighbor.
    community : list of lists
        A list where each element is a list of indices representing a community.
    commu_DM : numpy.ndarray
        A matrix representing distances or related metrics between communities.
    G : numpy.ndarray or scipy.sparse matrix
        A matrix used to update the connectivity matrix when communities have only one element.
        Can be either dense or sparse.
    ALL_DM : numpy.ndarray or scipy.sparse matrix
        A full distance matrix used by mindisttwinsloc to find minimum distances.
        Can be either dense or sparse.
    
    Returns
    -------
    new_ljmat : numpy.ndarray or scipy.sparse matrix
        The updated connectivity matrix. Will be the same type (dense or sparse) as the input.
    cutlinkpower : numpy.ndarray
        A matrix recording information about the cut links.
        Columns: [community_min, neighbor_min, link_point1, link_point2, 
                 community_size, neighbor_size, community_distance]
    
    Raises
    ------
    TypeError
        If inputs are not of the expected types
    ValueError
        If dimensions or values are invalid
    IndexError
        If indices are out of bounds
    RuntimeError
        If there are errors during processing
    """
    
    # Input validation - Type checking
    if not (isinstance(old_ljmat, np.ndarray) or sp.issparse(old_ljmat)):
        raise TypeError("old_ljmat must be a numpy array or scipy sparse matrix")
    
    if not (isinstance(commu_DM, np.ndarray) or sp.issparse(commu_DM)):
        raise TypeError("commu_DM must be a numpy array or scipy sparse matrix")
        
    if not (isinstance(G, np.ndarray) or sp.issparse(G)):
        raise TypeError("G must be a numpy array or scipy sparse matrix")
        
    if not (isinstance(ALL_DM, np.ndarray) or sp.issparse(ALL_DM)):
        raise TypeError("ALL_DM must be a numpy array or scipy sparse matrix")
    
    if not isinstance(community, list) or not isinstance(neiborloc, list):
        raise TypeError("community and neiborloc must be lists")
    
    # Input validation - Empty checks
    if not community:
        raise ValueError("community list cannot be empty")
    
    # Input validation - Dimension checking
    if old_ljmat.shape[0] != old_ljmat.shape[1]:
        raise ValueError("old_ljmat must be a square matrix")
        
    # Check if input is sparse and convert to appropriate format for efficient modification
    old_ljmat_is_sparse = sp.issparse(old_ljmat)
    G_is_sparse = sp.issparse(G)
    
    # Convert to LIL format if sparse for efficient modification
    if old_ljmat_is_sparse:
        logger.debug("Converting old_ljmat to LIL format for efficient modification")
        old_ljmat = old_ljmat.tolil()
    
    if G_is_sparse and not isinstance(G, sp.csr_matrix):
        logger.debug("Converting G to CSR format for efficient indexing")
        G = G.tocsr()
    
    community_num = len(community)
    if commu_DM.shape[0] != community_num or commu_DM.shape[1] != community_num:
        raise ValueError(f"commu_DM dimensions {commu_DM.shape} must match the number of communities ({community_num})")
    
    if len(neiborloc) != community_num:
        raise ValueError(f"Length of neiborloc ({len(neiborloc)}) must match number of communities ({community_num})")
    
    # Validate community structures
    for i, comm in enumerate(community):
        if not isinstance(comm, list):
            raise TypeError(f"Each element in community must be a list, found {type(comm)} at index {i}")
        if len(comm) == 0:
            raise ValueError(f"Empty community found at index {i}")
    
    # Helper function to check if a neighbor is valid
    def is_valid_neighbor(n, idx):
        """Determine if n is a valid neighbor index"""
        if n is None or (isinstance(n, list) and len(n) == 0):
            return False
        
        # If n is a list but should be a scalar, warn but use first element
        if isinstance(n, list) and len(n) > 0:
            logger.warning(f"Expected scalar at neiborloc[{idx}], found list {n}. Using first element.")
            n = n[0]
            
        # Check bounds
        if not isinstance(n, (int, np.integer)) or n < 0 or n >= community_num:
            raise ValueError(f"Invalid neighbor index {n} at position {idx}. Must be between 0 and {community_num-1}")
        
        return True
    
    # Determine the number of elements in the first community
    pd = len(community[0])
    logger.debug(f"Processing {community_num} communities with {pd} elements in first community")
    
    if pd > 1:
        # Count non-empty neighbor entries
        valid_neighbors = [i for i, n in enumerate(neiborloc) if is_valid_neighbor(n, i)]
        cutlinknum = len(valid_neighbors)
        
        logger.debug(f"Found {cutlinknum} valid neighbor relationships")
        
        # If no valid links exist, return with warning
        if cutlinknum == 0:
            logger.warning("No valid neighbor relationships found. Returning unchanged matrix.")
            # Convert back to CSR if it was originally sparse
            if old_ljmat_is_sparse:
                return old_ljmat.tocsr(), np.zeros((0, 7))
            return old_ljmat.copy(), np.zeros((0, 7))
        
        # Initialize cutlinkpower matrix
        cutlinkpower = np.zeros((cutlinknum, 7))
        
        th = 0  # Using 0-indexing
        
        for i in valid_neighbors:
            # Get the neighbor index safely
            if isinstance(neiborloc[i], list):
                neighbor_idx = neiborloc[i][0]
            else:
                neighbor_idx = neiborloc[i]
            
            try:
                # Find the pair of points with minimum distance between communities
                linkloc1, linkloc2 = mindisttwinsloc(community[i], community[neighbor_idx], ALL_DM)
            except Exception as e:
                raise RuntimeError(f"Error finding minimum distance between communities {i} and {neighbor_idx}: {str(e)}")
            
            # Check if indices are within bounds before updating matrices
            if (linkloc1 >= old_ljmat.shape[0] or linkloc2 >= old_ljmat.shape[1] or 
                linkloc1 < 0 or linkloc2 < 0):
                raise IndexError(f"Link locations ({linkloc1}, {linkloc2}) out of bounds for ljmat with shape {old_ljmat.shape}")
            
            # Get minimum values from each community as representatives
            try:
                xx = min(community[i])
                yy = min(community[neighbor_idx])
            except Exception as e:
                raise ValueError(f"Error finding minimum values in communities: {str(e)}")
            
            # Update connectivity matrix (for both sparse and dense)
            old_ljmat[linkloc1, linkloc2] = 1
            old_ljmat[linkloc2, linkloc1] = 1
            
            # Record cut link information
            cutlinkpower[th, 0] = xx
            cutlinkpower[th, 1] = yy
            cutlinkpower[th, 2] = linkloc1
            cutlinkpower[th, 3] = linkloc2
            cutlinkpower[th, 4] = len(community[i])
            cutlinkpower[th, 5] = len(community[neighbor_idx])
            cutlinkpower[th, 6] = commu_DM[i, neighbor_idx]
            
            th += 1
    
    elif pd == 1:
        logger.debug("Processing single-element communities")
        
        # Validate all neighbors for single-element communities
        for i, n in enumerate(neiborloc):
            is_valid_neighbor(n, i)
        
        cutlinkpower = np.zeros((community_num, 7))
        
        th = 0
        for i in range(community_num):
            # Convert to int if it's a list
            if isinstance(neiborloc[i], list):
                neighbor_idx = neiborloc[i][0]
            else:
                neighbor_idx = neiborloc[i]
            
            # Check if communities have the expected single element
            if len(community[i]) != 1 or len(community[neighbor_idx]) != 1:
                raise ValueError(f"Expected single-element communities, found {len(community[i])} and {len(community[neighbor_idx])} elements")
            
            # For single-element communities, use the only element
            linkloc1 = community[i][0]
            linkloc2 = community[neighbor_idx][0]
            
            # Check bounds
            if (linkloc1 >= G.shape[0] or linkloc2 >= G.shape[1] or 
                linkloc1 < 0 or linkloc2 < 0):
                raise IndexError(f"Link locations ({linkloc1}, {linkloc2}) out of bounds for G with shape {G.shape}")
            
            # Record cut link information
            cutlinkpower[th, 0] = linkloc1
            cutlinkpower[th, 1] = linkloc2
            cutlinkpower[th, 2] = linkloc1
            cutlinkpower[th, 3] = linkloc2
            cutlinkpower[th, 4] = 1  # Size is 1
            cutlinkpower[th, 5] = 1  # Size is 1
            cutlinkpower[th, 6] = commu_DM[i, neighbor_idx]
            
            th += 1
        
        # Replace connectivity matrix with G for single-element communities
        if G_is_sparse:
            # Convert to LIL for consistency if original was sparse
            if old_ljmat_is_sparse:
                old_ljmat = G.tolil()
            else:
                old_ljmat = G.toarray()
        else:
            old_ljmat = G.copy()
    
    else:
        raise ValueError(f"Invalid community size: {pd}. Must be at least 1.")
    
    new_ljmat = old_ljmat
    logger.debug(f"Updated ljmat with {cutlinkpower.shape[0]} links")
    
    # Convert back to CSR if original was sparse
    if old_ljmat_is_sparse:
        new_ljmat = new_ljmat.tocsr()
    
    return new_ljmat, cutlinkpower