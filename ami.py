import numpy as np
from scipy.special import comb

def contingency(mem1, mem2):
    """
    Build a contingency table from two membership arrays, following MATLAB implementation.
    
    Parameters:
        mem1: array-like of cluster labels for clustering 1 (positive integers)
        mem2: array-like of cluster labels for clustering 2 (positive integers)
        
    Returns:
        cont: a 2D numpy array where cont[i, j] is the number of data points
              assigned to cluster i+1 in mem1 and cluster j+1 in mem2.
    """
    mem1 = np.asarray(mem1, dtype=int)
    mem2 = np.asarray(mem2, dtype=int)
    
    # Get the maximum cluster labels
    n_clusters1 = np.max(mem1)
    n_clusters2 = np.max(mem2)
    
    # Initialize contingency matrix
    cont = np.zeros((n_clusters1, n_clusters2), dtype=int)
    
    # Fill contingency matrix
    for i in range(len(mem1)):
        # Subtract 1 because Python indices start at 0 while MATLAB indices start at 1
        cont[mem1[i] - 1, mem2[i] - 1] += 1
    
    return cont

def nchoosek(n, k):
    """
    MATLAB-compatible implementation of nchoosek
    """
    # Use scipy's comb function with exact=True for integer results matching MATLAB
    return comb(n, k, exact=True)

def ami(true_mem, mem=None):
    """
    Calculate the Adjusted Mutual Information (AMI) between two clusterings.
    MATLAB-compatible implementation based on Nguyen Xuan Vinh's code.
    
    Parameters:
        true_mem: Either a contingency table (if mem is None) or an array-like of 
                  cluster labels for the first clustering.
        mem:      (Optional) An array-like of cluster labels for the second clustering.
                  If omitted, true_mem is assumed to be a precomputed contingency table.
    
    Returns:
        AMI_val: Adjusted Mutual Information (AMI_max)
    """
    eps = np.finfo(float).eps  # Machine epsilon
    
    # Parse inputs exactly as MATLAB does
    if mem is None:
        # Contingency table pre-supplied
        T = np.asarray(true_mem, dtype=float)
    else:
        # Build the contingency table from membership arrays
        true_mem = np.asarray(true_mem, dtype=int)
        mem = np.asarray(mem, dtype=int)
        
        R = np.max(true_mem)
        C = np.max(mem)
        n = len(mem)
        N = n
        
        # Identify & remove the missing labels
        list_t = np.zeros(R, dtype=bool)
        list_m = np.zeros(C, dtype=bool)
        
        # MATLAB's ismember behavior
        for i in range(1, R + 1):
            if i in true_mem:
                list_t[i-1] = True
                
        for i in range(1, C + 1):
            if i in mem:
                list_m[i-1] = True
        
        # Build contingency table
        T_full = contingency(true_mem, mem)
        # Select only rows and columns containing actual clusters
        T = T_full[list_t, :][:, list_m].astype(float)
    
    # Calculate Rand index and other metrics (following MATLAB exactly)
    n = int(np.sum(T))
    N = n
    C = T.copy()
    
    # Row and column sums
    a = np.sum(C, axis=1)
    b = np.sum(C, axis=0)
    
    # Sum of squares of sums of rows & columns
    nis = np.sum(a**2)
    njs = np.sum(b**2)
    
    # Total number of pairs of entities
    t1 = nchoosek(n, 2)
    
    # Sum over rows & columns of nij^2
    t2 = np.sum(C**2)
    
    t3 = 0.5 * (nis + njs)
    
    # Expected index (for adjustment)
    nc = (n*(n**2+1) - (n+1)*nis - (n+1)*njs + 2*(nis*njs)/n) / (2*(n-1))
    
    A = t1 + t2 - t3  # Number of agreements
    D = -t2 + t3      # Number of disagreements
    
    # Adjusted Rand Index
    if t1 == nc:
        AR = 0  # Avoid division by zero
    else:
        AR = (A - nc) / (t1 - nc)
    
    # Other indices
    RI = A / t1                # Rand Index
    MIRKIN = D / t1            # Mirkin
    HI = (A - D) / t1          # Hubert
    Dri = 1 - RI               # Distance version of the RI
    Dari = 1 - AR              # Distance version of the ARI
    
    # Get updated dimensions
    R, C = T.shape
    
    # Extract row and column sums
    if C > 1:
        a = np.sum(T, axis=1)
    else:
        a = T.flatten()
    
    if R > 1:
        b = np.sum(T, axis=0)
    else:
        b = T.flatten()
    
    # Calculate Entropies (using safe division and log)
    Ha = -np.sum((a/n) * np.log(a/n + eps))
    Hb = -np.sum((b/n) * np.log(b/n + eps))
    
    # Calculate MI (unadjusted)
    MI = 0
    for i in range(R):
        for j in range(C):
            if T[i, j] > 0:
                MI += T[i, j] * np.log(T[i, j] * n / (a[i] * b[j]) + eps)
    MI = MI / n
    
    # Correcting for agreement by chance
    AB = np.outer(a, b)
    E3 = (AB / n**2) * np.log(AB / n**2 + eps)
    E3[~np.isfinite(E3)] = 0  # Handle any non-finite values
    
    EPLNP = np.zeros((R, C))
    bound = np.zeros((R, C))
    
    # Pre-compute log values exactly as MATLAB does
    max_val = int(min(np.max(a), np.max(b)))
    LogNij = np.log(np.arange(1, max_val + 1) / N + eps)
    
    for i in range(R):
        for j in range(C):
            # Initial nij calculation
            nij = max(1, int(a[i] + b[j] - N))
            X = sorted([nij, int(N - a[i] - b[j] + nij)])
            x1, x2 = X[0], X[1]
            
            # Calculate numerator and denominator arrays for p0
            if N - b[j] > x2:
                nom1 = np.arange(a[i] - nij + 1, a[i] + 1)
                nom2 = np.arange(b[j] - nij + 1, b[j] + 1)
                nom3 = np.arange(x2 + 1, N - b[j] + 1)
                nom = np.concatenate((nom1, nom2, nom3)).astype(float)
                
                dem1 = np.arange(N - a[i] + 1, N + 1)
                dem2 = np.arange(1, x1 + 1)
                dem = np.concatenate((dem1, dem2)).astype(float)
            else:
                nom1 = np.arange(a[i] - nij + 1, a[i] + 1)
                nom2 = np.arange(b[j] - nij + 1, b[j] + 1)
                nom = np.concatenate((nom1, nom2)).astype(float)
                
                dem1 = np.arange(N - a[i] + 1, N + 1)
                dem2 = np.arange(N - b[j] + 1, x2 + 1)
                dem3 = np.arange(1, x1 + 1)
                dem = np.concatenate((dem1, dem2, dem3)).astype(float)
            
            # Calculate p0 using log-domain calculations
            log_p0 = np.sum(np.log(nom)) - np.sum(np.log(dem)) - np.log(N)
            p0 = np.exp(log_p0)
            
            # Initial probability sum
            sumPnij = p0
            
            # Initial EPLNP calculation
            # Note: Use nij-1 for Python 0-based indexing into LogNij
            try:
                EPLNP[i, j] = nij * LogNij[nij-1] * p0
            except IndexError:
                # Handle edge case where nij > max_val
                EPLNP[i, j] = 0
            
            # Calculate p1 for next iteration
            if nij < min(a[i], b[j]):
                # Initial p1 calculation
                p1 = p0 * (a[i] - nij) * (b[j] - nij) / (nij + 1) / (N - a[i] - b[j] + nij + 1)
                
                # Loop through remaining possible nij values
                for nij_val in range(nij + 1, int(min(a[i], b[j])) + 1):
                    if not np.isfinite(p1) or p1 < np.finfo(float).tiny:
                        break  # Avoid numerical underflow
                    
                    # Accumulate probabilities
                    sumPnij += p1
                    
                    # Update EPLNP
                    try:
                        EPLNP[i, j] += nij_val * LogNij[nij_val-1] * p1
                    except IndexError:
                        # Handle edge case
                        pass
                    
                    # Update p1 for next iteration
                    if nij_val < min(a[i], b[j]):
                        p1 = p1 * (a[i] - nij_val) * (b[j] - nij_val) / (nij_val + 1) / (N - a[i] - b[j] + nij_val + 1)
            
            # Calculate bound
            CC = N * (a[i] - 1) * (b[j] - 1) / (a[i] * b[j] * (N - 1) + eps) + N / (a[i] * b[j] + eps)
            bound[i, j] = a[i] * b[j] / (N**2) * np.log(CC + eps)
    
    # Calculate EMI and bounds
    EMI_bound = np.sum(bound)
    EMI_bound_2 = np.log(R * C / N + (N - R) * (N - C) / (N * (N - 1)) + eps)
    EMI = np.sum(EPLNP - E3)
    
    # Calculate AMI and NMI
    NMI = MI / np.sqrt(Ha * Hb + eps)
    
    # Handle edge cases to match MATLAB behavior
    if max(Ha, Hb) - EMI == 0:
        AMI_val = 1.0  # Perfect agreement when denominator is zero
    else:
        AMI_val = (MI - EMI) / (max(Ha, Hb) - EMI)
    
    # If expected MI is problematic, use NMI
    if not np.isfinite(EMI) or abs(EMI) > EMI_bound:
        print(f"The EMI is small: EMI < {EMI_bound}, setting AMI=NMI")
        AMI_val = NMI
    
    # Ensure finite result
    if not np.isfinite(AMI_val):
        print("Warning: AMI calculation resulted in a non-finite value, using NMI instead")
        AMI_val = NMI
    
    return AMI_val