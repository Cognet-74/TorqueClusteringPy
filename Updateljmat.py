import numpy as np
import logging
import scipy.sparse as sp
from mindisttwinsloc import mindisttwinsloc

# Set up logger
logger = logging.getLogger(__name__)

def Updateljmat(old_ljmat, neiborloc, community, commu_DM, G, ALL_DM):
    """
    Update the connectivity matrix (ljmat) and record cut link power information.
    
    [Existing docstring is maintained]
    """
    
    # Matrix type detection (maintained improvement)
    old_ljmat_is_sparse = sp.issparse(old_ljmat)
    G_is_sparse = sp.issparse(G)
    
    # Input validation as warnings instead of errors
    if not isinstance(community, list):
        logger.warning("community should be a list - unexpected behavior may occur")
    if not isinstance(neiborloc, list):
        logger.warning("neiborloc should be a list - unexpected behavior may occur")
    
    community_num = len(community)
    logger.debug(f"Processing {community_num} communities")
    
    # MATLAB compatible empty check
    def is_matlab_empty(n):
        """Match MATLAB's emptiness check"""
        return n is None or (isinstance(n, list) and len(n) == 0)
    
    # Determine the number of elements in the first community
    pd = len(community[0])
    
    if pd > 1:
        # Count non-empty neighbor entries (MATLAB compatible)
        cutlinknum = sum(1 for n in neiborloc if not is_matlab_empty(n))
        
        # Initialize cutlinkpower matrix
        cutlinkpower = np.zeros((cutlinknum, 7))
        
        th = 0  # Using 0-indexing
        
        for i in range(community_num):
            if not is_matlab_empty(neiborloc[i]):
                # MATLAB compatible access
                neighbor_idx = neiborloc[i]
                
                # Find the pair of points with minimum distance
                linkloc1, linkloc2 = mindisttwinsloc(community[i], community[neighbor_idx], ALL_DM)
                
                # Get minimum values from each community (exactly as MATLAB)
                xx = min(community[i])
                yy = min(community[neighbor_idx])
                
                # Update connectivity matrix (MATLAB operation)
                old_ljmat[linkloc1, linkloc2] = 1
                old_ljmat[linkloc2, linkloc1] = 1
                
                # Record cut link information (identical to MATLAB)
                cutlinkpower[th, 0] = xx
                cutlinkpower[th, 1] = yy
                cutlinkpower[th, 2] = linkloc1
                cutlinkpower[th, 3] = linkloc2
                cutlinkpower[th, 4] = len(community[i])
                cutlinkpower[th, 5] = len(community[neighbor_idx])
                cutlinkpower[th, 6] = commu_DM[i, neighbor_idx]
                
                th += 1
    
    elif pd == 1:
        # Direct MATLAB implementation for single-element communities
        cutlinkpower = np.zeros((community_num, 7))
        
        th = 0
        for i in range(community_num):
            # Direct access to match MATLAB behavior
            linkloc1 = community[i][0]
            linkloc2 = community[neiborloc[i]][0]
            
            # Record cut link information
            cutlinkpower[th, 0] = linkloc1
            cutlinkpower[th, 1] = linkloc2
            cutlinkpower[th, 2] = linkloc1
            cutlinkpower[th, 3] = linkloc2
            cutlinkpower[th, 4] = 1
            cutlinkpower[th, 5] = 1
            cutlinkpower[th, 6] = commu_DM[i, neiborloc[i]]
            
            th += 1
        
        # Direct assignment to match MATLAB
        old_ljmat = G
    
    new_ljmat = old_ljmat
    
    # Convert back to original format if needed
    if old_ljmat_is_sparse and not sp.issparse(new_ljmat):
        new_ljmat = sp.csr_matrix(new_ljmat)
    elif not old_ljmat_is_sparse and sp.issparse(new_ljmat):
        new_ljmat = new_ljmat.toarray()
    
    return new_ljmat, cutlinkpower