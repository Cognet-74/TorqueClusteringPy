"""
TorqueClustering_Full.py

A complete implementation of the Torque Clustering algorithm
as described in the paper. This version follows these steps:
  1. Each sample is initialized as its own cluster (mass = 1).
  2. Iterative merging: For each cluster, find its 1-nearest neighbor 
     with mass greater than or equal to its own, merge them, and record the merge.
  3. Compute the torque for each merge:
         Tau = (mass_A * mass_B) * (distance^2)
  4. Sort all merges in descending order by torque.
  5. Compute the Torque Gap (TGap) for each merge with a weighting factor:
         TGap[i] = omega_i * (tau_i / tau_(i+1))
     where omega_i is the fraction of merges among the top-i that exceed the mean
     values of mass, distance, and torque.
  6. Determine the number of abnormal merges (L) to cut (L = argmax(TGap) + 1).
  7. Optionally, remove halo (noise) merges based on the condition:
         M <= mean(M),  D >= mean(D), and D/M >= mean(D/M)
  8. Reconstruct the final clustering partition by replaying the merges in ascending
     order of torque, skipping the abnormal (and halo, if enabled) merges.
  
Note: This reference implementation is not optimized for large datasets.
It uses a naive O(n^2) distance computation.
"""

import numpy as np

def torque_clustering_paper(X, is_noise=False):
    """
    Fully implements the paper's Torque Clustering algorithm, step by step.
    
    Args:
        X (np.ndarray): Data array of shape (n_samples, n_features).
        is_noise (bool): If True, apply halo-removal rules to detect noise.
    
    Returns:
        labels (np.ndarray): Final cluster labels for each point (0-based).
        final_k (int): Number of clusters found (L+1, possibly more if noise is removed).
        merges_sorted (np.ndarray): Structured array of all merges sorted by descending torque.
                                    Each row contains fields:
                                      'cA', 'cB', 'M', 'D', 'Tau', 'pointsA', 'pointsB'
    """
    n = X.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=int), n, np.empty((0, 5))
    
    #----------------------------------------------------------------------
    # Step 1: Initialize each point as its own cluster.
    #----------------------------------------------------------------------
    cluster_dict = {}
    for i in range(n):
        cluster_dict[i] = {'points': [i], 'mass': 1}
    
    active_clusters = set(cluster_dict.keys())
    merges_chrono = []  # List to store merges in chronological order.
    
    def cluster_distance_sq(cA, cB):
        """Compute minimum squared distance between any point in cluster cA and any point in cluster cB."""
        ptsA = cluster_dict[cA]['points']
        ptsB = cluster_dict[cB]['points']
        min_d2 = np.inf
        for a in ptsA:
            diffs = X[ptsB] - X[a]  # shape: (len(ptsB), n_features)
            dist2 = np.sum(diffs**2, axis=1)
            cand = dist2.min()
            if cand < min_d2:
                min_d2 = cand
        return min_d2
    
    #----------------------------------------------------------------------
    # Step 2: Iteratively merge clusters until only one remains.
    # For each cluster, merge it with its nearest cluster whose mass is >= its own.
    #----------------------------------------------------------------------
    while len(active_clusters) > 1:
        merged_this_round = set()
        sorted_by_mass = sorted(active_clusters, key=lambda c: cluster_dict[c]['mass'])
        
        for c in sorted_by_mass:
            if c in merged_this_round:
                continue
            mass_c = cluster_dict[c]['mass']
            best_dist2 = np.inf
            best_nn = None
            for other in active_clusters:
                # Ensure that 'other' still exists in cluster_dict.
                if other not in cluster_dict:
                    continue
                if other == c:
                    continue
                if cluster_dict[other]['mass'] >= mass_c:
                    d2 = cluster_distance_sq(c, other)
                    if d2 < best_dist2:
                        best_dist2 = d2
                        best_nn = other
            if best_nn is not None:
                mass_nn = cluster_dict[best_nn]['mass']
                M_ij = mass_c * mass_nn
                D_ij = best_dist2
                Tau_ij = M_ij * D_ij
                
                merges_chrono.append({
                    'cA': c,
                    'cB': best_nn,
                    'pointsA': list(cluster_dict[c]['points']),
                    'pointsB': list(cluster_dict[best_nn]['points']),
                    'M': M_ij,
                    'D': D_ij,
                    'Tau': Tau_ij
                })
                
                # Merge cluster c into best_nn.
                cluster_dict[best_nn]['points'].extend(cluster_dict[c]['points'])
                cluster_dict[best_nn]['mass'] = mass_nn + mass_c
                del cluster_dict[c]
                merged_this_round.add(c)
                merged_this_round.add(best_nn)
        active_clusters = set(cluster_dict.keys())
        if len(active_clusters) == 1 or len(merged_this_round) == 0:
            break
    
    if len(merges_chrono) == 0:
        labels = np.arange(n)
        return labels, n, np.empty((0, 5))
    
    #----------------------------------------------------------------------
    # Step 3: Convert chronological merges into a structured array and sort by Tau descending.
    #----------------------------------------------------------------------
    dtype_ = [
        ('cA', 'i4'), ('cB', 'i4'),
        ('M', 'f8'), ('D', 'f8'), ('Tau', 'f8'),
        ('pointsA', 'O'), ('pointsB', 'O')
    ]
    merges_arr = np.zeros(len(merges_chrono), dtype=dtype_)
    for i, mdict in enumerate(merges_chrono):
        merges_arr[i]['cA'] = mdict['cA']
        merges_arr[i]['cB'] = mdict['cB']
        merges_arr[i]['M']   = mdict['M']
        merges_arr[i]['D']   = mdict['D']
        merges_arr[i]['Tau'] = mdict['Tau']
        merges_arr[i]['pointsA'] = mdict['pointsA']
        merges_arr[i]['pointsB'] = mdict['pointsB']
    
    sort_idx = np.argsort(merges_arr['Tau'])[::-1]
    merges_sorted = merges_arr[sort_idx]
    
    #----------------------------------------------------------------------
    # Step 4: Compute Torque Gap (TGap) with weighting factor omega_i.
    #----------------------------------------------------------------------
    M_all = merges_sorted['M']
    D_all = merges_sorted['D']
    Tau_all = merges_sorted['Tau']
    
    mean_M = M_all.mean()
    mean_D = D_all.mean()
    mean_Tau = Tau_all.mean()
    
    # Define "LargeC": merges with M >= mean_M, D >= mean_D, and Tau >= mean_Tau.
    is_large = ((M_all >= mean_M) &
                (D_all >= mean_D) &
                (Tau_all >= mean_Tau))
    large_indices = np.where(is_large)[0]
    n_large = len(large_indices)
    
    TGap_vals = []
    for i in range(len(merges_sorted) - 1):
        top_i_indices = np.arange(i + 1)
        large_in_top_i = np.intersect1d(top_i_indices, large_indices).size
        w_i = (large_in_top_i / n_large) if n_large > 0 else 0.0
        tau_i = merges_sorted['Tau'][i]
        tau_i1 = merges_sorted['Tau'][i + 1]
        ratio = tau_i / tau_i1 if tau_i1 > 1e-15 else 1e9
        TGap_vals.append(w_i * ratio)
    TGap_vals = np.array(TGap_vals, dtype=float)
    if len(TGap_vals) == 0:
        labels = np.zeros(n, dtype=int)
        return labels, 1, merges_sorted
    L = np.argmax(TGap_vals) + 1  # Remove top L merges.
    
    #----------------------------------------------------------------------
    # Step 5: Optional Halo (Noise) Removal.
    # Halo condition: M <= mean_M, D >= mean_D, and D/M >= mean(D/M)
    #----------------------------------------------------------------------
    if is_noise:
        ratio_DM = merges_sorted['D'] / merges_sorted['M']
        mean_ratio = ratio_DM.mean()
        halo_mask = ((merges_sorted['M'] <= mean_M) &
                     (merges_sorted['D'] >= mean_D) &
                     (ratio_DM >= mean_ratio))
    else:
        halo_mask = np.zeros(len(merges_sorted), dtype=bool)
    
    # Determine indices to skip: top L merges (abnormal) and halo merges.
    # We work with indices in the sorted array; convert these to indices in the original merges array.
    skip_indices = set(sort_idx[:L])
    if is_noise:
        halo_indices = set(sort_idx[np.where(halo_mask)[0]])
        skip_indices = skip_indices.union(halo_indices)
    
    #----------------------------------------------------------------------
    # Step 6: Reconstruct Final Clusters.
    # Re-play the merges in ascending order of Tau, skipping merges in skip_indices.
    # We'll use a Union-Find (Disjoint Set) structure on the original n points.
    #----------------------------------------------------------------------
    asc_idx = np.argsort(merges_arr['Tau'])  # ascending order indices
    parent = np.arange(n, dtype=int)
    rank = np.zeros(n, dtype=int)
    
    def find(u):
        while u != parent[u]:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    
    def union(u, v):
        ru, rv = find(u), find(v)
        if ru == rv:
            return
        if rank[ru] < rank[rv]:
            parent[ru] = rv
        elif rank[ru] > rank[rv]:
            parent[rv] = ru
        else:
            parent[rv] = ru
            rank[ru] += 1
    
    for idx in asc_idx:
        if idx in skip_indices:
            continue
        ptsA = merges_arr[idx]['pointsA']
        ptsB = merges_arr[idx]['pointsB']
        if len(ptsA) > 0 and len(ptsB) > 0:
            union(ptsA[0], ptsB[0])
    
    labels = np.zeros(n, dtype=int)
    for i in range(n):
        labels[i] = find(i)
    
    unique_roots = np.unique(labels)
    root_map = {r: i for i, r in enumerate(unique_roots)}
    for i in range(n):
        labels[i] = root_map[labels[i]]
    
    final_k = len(unique_roots)
    return labels, final_k, merges_sorted

if __name__ == '__main__':
    # Simple test dataset for debugging
    X_test = np.array([
        [0.0, 0.0],
        [0.9, 0.1],
        [1.8, 0.2],
        [5.0, 5.0],
        [5.1, 5.1],
        [5.2, 4.9]
    ])
    
    labels, k, merges_sorted = torque_clustering_paper(X_test, is_noise=True)
    print("Final labels:", labels)
    print("Number of clusters:", k)
    print("Sorted merges (first few rows):")
    print(merges_sorted[:min(5, len(merges_sorted))])
