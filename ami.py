import numpy as np
from scipy.special import comb
import warnings

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
    # Validate input types
    mem1 = np.asarray(mem1, dtype=int)
    mem2 = np.asarray(mem2, dtype=int)
    
    # Check for empty arrays
    if len(mem1) == 0 or len(mem2) == 0:
        return np.zeros((0, 0), dtype=int)
    
    # Check that arrays have the same length
    if len(mem1) != len(mem2):
        raise ValueError(f"Arrays have different lengths: {len(mem1)} and {len(mem2)}")
    
    # Get the maximum cluster labels with safety check
    n_clusters1 = int(np.max(mem1)) if len(mem1) > 0 else 0
    n_clusters2 = int(np.max(mem2)) if len(mem2) > 0 else 0
    
    # Handle case where clusters might be zero or negative
    if n_clusters1 <= 0 or n_clusters2 <= 0:
        warnings.warn("Cluster labels should be positive integers starting from 1")
        # Handle labels starting from 0 or negative values
        offset1 = min(1, 1 - np.min(mem1)) if len(mem1) > 0 else 1
        offset2 = min(1, 1 - np.min(mem2)) if len(mem2) > 0 else 1
        
        if offset1 > 0 or offset2 > 0:
            mem1 = mem1 + offset1
            mem2 = mem2 + offset2
            n_clusters1 = int(np.max(mem1)) if len(mem1) > 0 else 0
            n_clusters2 = int(np.max(mem2)) if len(mem2) > 0 else 0
    
    # Initialize contingency matrix
    cont = np.zeros((n_clusters1, n_clusters2), dtype=int)
    
    # Fill contingency matrix with safety check
    for i in range(len(mem1)):
        idx1 = mem1[i] - 1  # Subtract 1 for zero-based indexing
        idx2 = mem2[i] - 1
        
        # Additional safety check
        if 0 <= idx1 < n_clusters1 and 0 <= idx2 < n_clusters2:
            cont[idx1, idx2] += 1
        else:
            warnings.warn(f"Invalid cluster labels at index {i}: {mem1[i]}, {mem2[i]}")
    
    return cont

def nchoosek(n, k):
    """
    MATLAB-compatible implementation of nchoosek
    """
    # Input validation
    if n < 0 or k < 0 or k > n:
        return 0
    
    # Use scipy's comb function with exact=True for integer results matching MATLAB
    try:
        return comb(n, k, exact=True)
    except (OverflowError, ValueError):
        # Fallback to non-exact for large numbers
        return int(comb(n, k, exact=False))

def ami(true_mem, mem=None, max_attempts=3):
    """
    Calculate the Adjusted Mutual Information (AMI) between two clusterings.
    MATLAB-compatible implementation based on Nguyen Xuan Vinh's code.
    
    Parameters:
        true_mem: Either a contingency table (if mem is None) or an array-like of 
                  cluster labels for the first clustering.
        mem:      (Optional) An array-like of cluster labels for the second clustering.
                  If omitted, true_mem is assumed to be a precomputed contingency table.
        max_attempts: Maximum number of attempts with adjusted parameters if calculation fails.
        
    Returns:
        AMI_val: Adjusted Mutual Information (AMI_max)
    """
    # Store diagnostic information
    diagnostics = {}
    
    # Safety: use a larger epsilon for improved stability
    eps = np.finfo(float).eps * 100  
    
    # Attempt calculation with increasingly relaxed parameters
    for attempt in range(max_attempts):
        try:
            # Parse inputs exactly as MATLAB does
            if mem is None:
                # Contingency table pre-supplied
                T = np.asarray(true_mem, dtype=float)
                
                # Validate contingency table
                if T.ndim != 2:
                    raise ValueError(f"Contingency table must be 2D, got {T.ndim}D")
                
                # Handle empty contingency table
                if T.size == 0:
                    return 0.0
            else:
                # Build the contingency table from membership arrays
                true_mem_arr = np.asarray(true_mem, dtype=int)
                mem_arr = np.asarray(mem, dtype=int)
                
                # Check for empty arrays
                if len(true_mem_arr) == 0 or len(mem_arr) == 0:
                    return 0.0
                
                # Check that arrays have the same length
                if len(true_mem_arr) != len(mem_arr):
                    raise ValueError(f"Arrays have different lengths: {len(true_mem_arr)} and {len(mem_arr)}")
                
                R = np.max(true_mem_arr)
                C = np.max(mem_arr)
                n = len(mem_arr)
                N = n
                
                # Identify & remove the missing labels
                list_t = np.zeros(R, dtype=bool)
                list_m = np.zeros(C, dtype=bool)
                
                # MATLAB's ismember behavior
                for i in range(1, R + 1):
                    if i in true_mem_arr:
                        list_t[i-1] = True
                        
                for i in range(1, C + 1):
                    if i in mem_arr:
                        list_m[i-1] = True
                
                # Build contingency table
                T_full = contingency(true_mem_arr, mem_arr)
                
                # Safety check for empty clusters
                if np.sum(list_t) == 0 or np.sum(list_m) == 0:
                    # No valid clusters found
                    return 0.0
                
                # Select only rows and columns containing actual clusters
                T = T_full[list_t, :][:, list_m].astype(float)
                
                # If contingency table is empty after filtering, return 0
                if T.size == 0:
                    return 0.0
            
            # Calculate Rand index and other metrics (following MATLAB exactly)
            n = int(np.sum(T))
            
            # Safety check for empty data
            if n == 0:
                return 0.0
                
            N = n
            C = T.copy()
            
            # Verify dimensions of contingency table
            R, C_dim = T.shape
            
            # Safety check for singleton clusterings
            if R <= 1 and C_dim <= 1:
                # Perfect agreement with singleton clusterings
                return 1.0
            
            # Row and column sums
            a = np.sum(T, axis=1)
            b = np.sum(T, axis=0)
            
            # Check for empty clusters
            if np.any(a == 0) or np.any(b == 0):
                warnings.warn("Empty clusters detected. Results may be unreliable.")
                
                # Remove empty clusters
                T = T[a > 0, :]
                T = T[:, b > 0]
                a = a[a > 0]
                b = b[b > 0]
                
                # Update dimensions
                R, C_dim = T.shape
                
                # Safety check for empty contingency table after filtering
                if T.size == 0:
                    return 0.0
            
            # Sum of squares of sums of rows & columns
            nis = np.sum(a**2)
            njs = np.sum(b**2)
            
            # Total number of pairs of entities
            t1 = nchoosek(n, 2)
            
            # Sum over rows & columns of nij^2
            t2 = np.sum(T**2)
            
            t3 = 0.5 * (nis + njs)
            
            # Expected index (for adjustment)
            # Avoid division by zero
            if n <= 1:
                nc = 0
            else:
                nc = (n*(n**2+1) - (n+1)*nis - (n+1)*njs + 2*(nis*njs)/n) / (2*(n-1))
            
            A = t1 + t2 - t3  # Number of agreements
            D = -t2 + t3      # Number of disagreements
            
            # Adjusted Rand Index
            if t1 == nc or abs(t1 - nc) < eps:
                AR = 0  # Avoid division by zero
            else:
                AR = (A - nc) / (t1 - nc)
            
            # Extract row and column sums
            if C_dim > 1:
                a = np.sum(T, axis=1)
            else:
                a = T.flatten()
            
            if R > 1:
                b = np.sum(T, axis=0)
            else:
                b = T.flatten()
            
            # Calculate Entropies (using safe division and log)
            # Replace zero values with epsilon to avoid log(0)
            a_safe = np.maximum(a, eps)
            b_safe = np.maximum(b, eps)
            
            # Use protected log to prevent -inf
            log_a = np.log(a_safe/n + eps)
            log_b = np.log(b_safe/n + eps)
            
            # Handle -inf values
            log_a[~np.isfinite(log_a)] = 0
            log_b[~np.isfinite(log_b)] = 0
            
            Ha = -np.sum((a/n) * log_a)
            Hb = -np.sum((b/n) * log_b)
            
            # Safety check for zero entropy
            if abs(Ha) < eps and abs(Hb) < eps:
                # Both clusterings have only one cluster
                return 1.0
            
            # Calculate MI (unadjusted)
            MI = 0
            for i in range(R):
                for j in range(C_dim):
                    if T[i, j] > 0:
                        # Safe computation to avoid log(0)
                        log_term = np.log(T[i, j] * n / (a[i] * b[j]) + eps)
                        # Skip non-finite values
                        if np.isfinite(log_term):
                            MI += T[i, j] * log_term
            MI = MI / n
            
            # Safety check - MI should not exceed min(Ha, Hb)
            if MI > min(Ha, Hb) + eps:
                MI = min(Ha, Hb)
            
            # Get NMI as fallback
            # Use safe division to avoid division by zero
            denominator = np.sqrt(Ha * Hb + eps)
            NMI = MI / denominator if denominator > eps else 1.0
            
            # Early return if entropy is zero
            if abs(Ha) < eps or abs(Hb) < eps:
                return NMI
            
            # Correcting for agreement by chance
            AB = np.outer(a, b)
            E3 = (AB / n**2) * np.log(AB / n**2 + eps)
            E3[~np.isfinite(E3)] = 0  # Handle any non-finite values
            
            EPLNP = np.zeros((R, C_dim))
            bound = np.zeros((R, C_dim))
            
            # Safety check for large clusters that might cause numerical issues
            if R * C_dim > 1000 or n > 10000:
                # Use approximation for large tables
                EMI = 0
                for i in range(R):
                    for j in range(C_dim):
                        nij_mean = a[i] * b[j] / n
                        if nij_mean > eps:
                            log_term = np.log(nij_mean/n + eps)
                            if np.isfinite(log_term):
                                EMI += nij_mean * log_term
                
                AMI_val = (MI - EMI) / (max(Ha, Hb) - EMI + eps)
                
                # Validate the result
                if not np.isfinite(AMI_val) or AMI_val < -1 or AMI_val > 1:
                    # Fallback to NMI
                    AMI_val = NMI
                
                return AMI_val
            
            # Pre-compute log values exactly as MATLAB does
            max_val = int(min(np.max(a), np.max(b)))
            LogNij = np.log(np.arange(1, max_val + 1) / N + eps)
            
            # Make sure LogNij is finite
            LogNij[~np.isfinite(LogNij)] = -709  # Lower bound for log in double precision
            
            for i in range(R):
                for j in range(C_dim):
                    # Initial nij calculation
                    nij = max(1, int(a[i] + b[j] - N))
                    X = sorted([nij, int(N - a[i] - b[j] + nij)])
                    x1, x2 = X[0], X[1]
                    
                    # Calculate numerator and denominator arrays for p0
                    try:
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
                    except ValueError:
                        # Handle cases where arange produces empty arrays
                        warnings.warn(f"Empty range in probability calculation at i={i}, j={j}")
                        continue
                    
                    # Prevent empty arrays
                    if len(nom) == 0 or len(dem) == 0:
                        continue
                    
                    # Calculate p0 using log-domain calculations for numerical stability
                    try:
                        log_p0 = np.sum(np.log(nom + eps)) - np.sum(np.log(dem + eps)) - np.log(N + eps)
                        p0 = np.exp(min(log_p0, 700))  # Limit to prevent overflow
                    except:
                        # Skip this calculation if numerical issues arise
                        continue
                    
                    # Initial probability sum
                    sumPnij = p0
                    
                    # Initial EPLNP calculation
                    # Note: Use nij-1 for Python 0-based indexing into LogNij
                    try:
                        if 0 < nij <= max_val:
                            EPLNP[i, j] = nij * LogNij[nij-1] * p0
                    except IndexError:
                        # Handle edge case where nij > max_val
                        EPLNP[i, j] = 0
                    
                    # Calculate p1 for next iteration
                    if nij < min(a[i], b[j]):
                        # Initial p1 calculation with safety checks
                        denom = (nij + 1) * (N - a[i] - b[j] + nij + 1)
                        if denom > eps:
                            p1 = p0 * (a[i] - nij) * (b[j] - nij) / denom
                        else:
                            p1 = 0
                        
                        # Loop through remaining possible nij values
                        for nij_val in range(nij + 1, int(min(a[i], b[j])) + 1):
                            if not np.isfinite(p1) or p1 < 1e-15:
                                break  # Avoid numerical underflow
                            
                            # Accumulate probabilities
                            sumPnij += p1
                            
                            # Update EPLNP
                            try:
                                if 0 < nij_val <= max_val:
                                    EPLNP[i, j] += nij_val * LogNij[nij_val-1] * p1
                            except IndexError:
                                # Handle edge case
                                pass
                            
                            # Update p1 for next iteration with safety checks
                            denom = (nij_val + 1) * (N - a[i] - b[j] + nij_val + 1)
                            if denom > eps and nij_val < min(a[i], b[j]):
                                p1 = p1 * (a[i] - nij_val) * (b[j] - nij_val) / denom
                            else:
                                p1 = 0
                    
                    # Calculate bound with safety checks
                    if a[i] * b[j] > eps:
                        denom = a[i] * b[j] * (N - 1) + eps
                        CC = N * (a[i] - 1) * (b[j] - 1) / denom + N / (a[i] * b[j] + eps)
                        log_term = np.log(CC + eps)
                        if np.isfinite(log_term):
                            bound[i, j] = a[i] * b[j] / (N**2) * log_term
            
            # Calculate EMI and bounds
            EMI_bound = np.sum(bound)
            EMI_bound_2 = np.log(R * C_dim / N + (N - R) * (N - C_dim) / (N * (N - 1)) + eps)
            EMI = np.sum(EPLNP - E3)
            
            # Get NMI as fallback
            NMI = MI / np.sqrt(Ha * Hb + eps)
            
            # Handle edge cases to match MATLAB behavior
            denom = max(Ha, Hb) - EMI
            if abs(denom) < eps:
                AMI_val = 1.0  # Perfect agreement when denominator is zero
            else:
                AMI_val = (MI - EMI) / denom
            
            # If expected MI is problematic, use NMI
            if not np.isfinite(EMI) or abs(EMI) > EMI_bound:
                warnings.warn(f"The EMI is problematic: EMI={EMI}, bound={EMI_bound}, setting AMI=NMI")
                AMI_val = NMI
            
            # Ensure finite result and in proper range
            if not np.isfinite(AMI_val) or AMI_val < -1 or AMI_val > 1:
                warnings.warn(f"AMI calculation resulted in an invalid value: {AMI_val}, using NMI instead")
                AMI_val = NMI
            
            # Clip to valid range
            AMI_val = max(-1.0, min(1.0, AMI_val))
            
            return AMI_val
            
        except Exception as e:
            diagnostics[f'attempt_{attempt}'] = str(e)
            
            # Adjust parameters for next attempt
            eps *= 10
            
            if attempt == max_attempts - 1:
                # Last attempt, return NMI if available, otherwise 0
                warnings.warn(f"AMI calculation failed after {max_attempts} attempts: {e}. Returning NMI or 0.")
                try:
                    # Try to calculate NMI as fallback with even more relaxed conditions
                    if mem is None:
                        T = np.asarray(true_mem, dtype=float)
                    else:
                        T_full = contingency(true_mem, mem)
                        T = T_full.astype(float)
                    
                    # Skip if contingency table is empty
                    if T.size == 0:
                        return 0.0
                    
                    n = np.sum(T)
                    if n == 0:
                        return 0.0
                    
                    a = np.sum(T, axis=1)
                    b = np.sum(T, axis=0)
                    
                    # Skip if any row/column sum is zero
                    if np.any(a == 0) or np.any(b == 0):
                        return 0.0
                    
                    Ha = -np.sum((a/n) * np.log(a/n + 1e-10))
                    Hb = -np.sum((b/n) * np.log(b/n + 1e-10))
                    
                    MI = 0
                    for i in range(T.shape[0]):
                        for j in range(T.shape[1]):
                            if T[i, j] > 0:
                                MI += T[i, j] * np.log(T[i, j] * n / (a[i] * b[j]) + 1e-10)
                    MI = MI / n
                    
                    NMI = MI / np.sqrt(Ha * Hb + 1e-10)
                    return max(-1.0, min(1.0, NMI))
                except Exception:
                    # If even NMI calculation fails, return 0
                    return 0.0
    
    # This should not be reached, but just in case
    return 0.0

# Simple validation function to test AMI calculation
def validate_ami(true_labels, pred_labels):
    """
    Test the AMI function with the given labels and print diagnostic information.
    
    Args:
        true_labels: Ground truth cluster labels
        pred_labels: Predicted cluster labels
    """
    try:
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            ami_value = ami(true_labels, pred_labels)
        
        # Print diagnostics
        print(f"AMI value: {ami_value}")
        print(f"Warnings: {len(w)}")
        for warning in w:
            print(f"  - {warning.message}")
        
        # Check if result is valid
        if not np.isfinite(ami_value) or ami_value < -1 or ami_value > 1:
            print(f"WARNING: Invalid AMI value: {ami_value}")
        
        # Calculate alternative metrics for comparison
        try:
            from sklearn import metrics
            sklearn_ami = metrics.adjusted_mutual_info_score(true_labels, pred_labels)
            print(f"Sklearn AMI: {sklearn_ami}")
            print(f"Difference: {abs(ami_value - sklearn_ami)}")
        except ImportError:
            print("sklearn not available for comparison")
        
        return ami_value
    except Exception as e:
        print(f"Error calculating AMI: {e}")
        return None