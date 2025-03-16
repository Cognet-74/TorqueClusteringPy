import numpy as np

def torque_clustering_paper(X, is_noise=False):
    """
    Fully implements the paper's Torque Clustering algorithm, step by step:
    
    1) Each sample is its own cluster (mass=1).
    2) Iteratively merge each cluster with its 1-nearest cluster
       whose mass >= its own, recording merges in chronological order.
    3) Compute torque for each merge: Tau = (mass product)*(distance^2).
    4) Sort merges by Tau descending and use the "Torque Gap" (TGap) with
       the paper's weighting factor w_i to decide how many top merges to cut.
    5) (Optional) Remove "halo" merges if is_noise=True.
    6) Reconstruct final clusters by replaying merges in ascending Tau,
       skipping the merges that were cut or are halo merges.

    Args:
        X (np.ndarray): shape (n_samples, n_features).
        is_noise (bool): If True, apply the paper's halo-removal rules.

    Returns:
        labels (np.ndarray): Final cluster labels for each point (0-based).
        final_k (int): Number of clusters found = L+1 (possibly more if noise).
        merges_sorted (np.ndarray): Array of merges [cA, cB, M, D, Tau] sorted desc by Tau
    """
    n = X.shape[0]
    if n <= 1:
        # trivial
        return np.zeros(n, dtype=int), n, np.empty((0,5))

    #----------------------------------------------------------------------
    # 1) Initialize each point as its own cluster
    #----------------------------------------------------------------------
    # cluster_dict[c_id] = {
    #    'points': list of point indices,
    #    'mass': integer (size of cluster)
    # }
    cluster_dict = {}
    for i in range(n):
        cluster_dict[i] = {'points': [i], 'mass': 1}

    active_clusters = set(cluster_dict.keys())

    # We'll store merges in a chronological list of dicts:
    # each item: {
    #   'cA': int,
    #   'cB': int,
    #   'pointsA': [...],
    #   'pointsB': [...],
    #   'M': float,
    #   'D': float,
    #   'Tau': float
    # }
    merges_chrono = []

    def cluster_distance_sq(cA, cB):
        """Compute minimal squared distance between any point in cA and any point in cB."""
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
    # 2) Iteratively merge clusters until only 1 remains
    #    In each pass, we sort clusters by mass ascending and attempt
    #    to merge each with its nearest-larger-or-equal cluster.
    #----------------------------------------------------------------------
    while len(active_clusters) > 1:
        merged_this_round = set()

        # sort clusters by ascending mass
        sorted_by_mass = sorted(active_clusters, key=lambda c: cluster_dict[c]['mass'])

        for c in sorted_by_mass:
            if c in merged_this_round:
                continue
            mass_c = cluster_dict[c]['mass']

            # find nearest cluster among active_clusters with mass >= mass_c
            best_dist2 = np.inf
            best_nn = None
            for other in active_clusters:
                if other == c:
                    continue
                if cluster_dict[other]['mass'] >= mass_c:
                    d2 = cluster_distance_sq(c, other)
                    if d2 < best_dist2:
                        best_dist2 = d2
                        best_nn = other

            if best_nn is not None:
                # record the merge
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

                # unify c -> best_nn
                cluster_dict[best_nn]['points'].extend(cluster_dict[c]['points'])
                cluster_dict[best_nn]['mass'] = mass_nn + mass_c
                # remove c
                del cluster_dict[c]
                merged_this_round.add(c)
                merged_this_round.add(best_nn)

        # update active_clusters
        # any cluster merged is removed, except the "absorber" stays.
        # to simplify, just re-collect what's left in cluster_dict
        active_clusters = set(cluster_dict.keys())
        if len(active_clusters) == 1:
            break  # done

        # if no merges happened, we're stuck
        if len(merged_this_round) == 0:
            break

    # If merges_chrono is empty => no merges => each point is separate
    if len(merges_chrono) == 0:
        labels = np.arange(n)
        return labels, n, np.empty((0,5))

    #----------------------------------------------------------------------
    # 3) We have a chronological list of merges. Sort them by Tau descending
    #    to apply the TGap procedure.
    #----------------------------------------------------------------------
    # Convert merges_chrono to a structured array for convenience
    # We'll keep only cA, cB, M, D, Tau for TGap, but also keep pointsA, pointsB for final partition
    dtype_ = [
        ('cA','i4'), ('cB','i4'),
        ('M','f8'), ('D','f8'), ('Tau','f8'),
        ('pointsA','O'), ('pointsB','O')  # store Python object for lists
    ]
    merges_arr = np.zeros(len(merges_chrono), dtype=dtype_)
    for i,mdict in enumerate(merges_chrono):
        merges_arr[i]['cA'] = mdict['cA']
        merges_arr[i]['cB'] = mdict['cB']
        merges_arr[i]['M']   = mdict['M']
        merges_arr[i]['D']   = mdict['D']
        merges_arr[i]['Tau'] = mdict['Tau']
        merges_arr[i]['pointsA'] = mdict['pointsA']
        merges_arr[i]['pointsB'] = mdict['pointsB']

    # sort descending by Tau
    sort_idx = np.argsort(merges_arr['Tau'])[::-1]
    merges_sorted = merges_arr[sort_idx]

    #----------------------------------------------------------------------
    # 4) Paper's TGap procedure with weighting factor omega_i
    #----------------------------------------------------------------------
    M_all = merges_sorted['M']
    D_all = merges_sorted['D']
    Tau_all = merges_sorted['Tau']

    mean_M = M_all.mean()
    mean_D = D_all.mean()
    mean_Tau = Tau_all.mean()

    # define the set of "large" merges
    is_large = (
        (M_all >= mean_M) &
        (D_all >= mean_D) &
        (Tau_all >= mean_Tau)
    )
    large_indices = np.where(is_large)[0]
    n_large = len(large_indices)

    # TGap[i] = omega_i * (tau[i]/tau[i+1])
    # with i up to len(merges_sorted)-2 or -1
    TGap_vals = []
    for i in range(len(merges_sorted)-1):
        top_i_indices = np.arange(i+1)  # the merges from 0..i
        # how many among top i are large
        large_in_top_i = np.intersect1d(top_i_indices, large_indices).size
        if n_large == 0:
            w_i = 0.0
        else:
            # fraction of large merges among the entire LargeC set
            w_i = large_in_top_i / n_large
        tau_i   = merges_sorted['Tau'][i]
        tau_i1  = merges_sorted['Tau'][i+1]
        ratio   = tau_i / tau_i1 if tau_i1>1e-15 else 1e9
        TGap_i  = w_i * ratio
        TGap_vals.append(TGap_i)

    TGap_vals = np.array(TGap_vals, dtype=float)
    if len(TGap_vals) == 0:
        # edge case: if we had only 1 merge?
        # then there's no gap to cut => final clusters=1
        # But if there's only 1 merge, that merges 2 clusters => final = n-1 clusters for the top layer?
        # The simplest is to say everything is in one cluster => for the sake of completeness.
        labels = np.zeros(n, dtype=int)
        return labels, 1, merges_sorted

    L = np.argmax(TGap_vals) + 1  # if TGap max at i => cut i+1 merges

    #----------------------------------------------------------------------
    # 5) Optional Halo Removal
    #    "Halo_C = { merges with M<=meanM, D>=meanD, D/M>=mean(D/M) }"
    #----------------------------------------------------------------------
    if is_noise:
        ratio_DM = merges_sorted['D'] / merges_sorted['M']
        mean_ratio = ratio_DM.mean()
        # halo merges
        halo_mask = (
            (merges_sorted['M'] <= mean_M) &
            (merges_sorted['D'] >= mean_D) &
            (ratio_DM >= mean_ratio)
        )
    else:
        halo_mask = np.zeros(len(merges_sorted), dtype=bool)

    #----------------------------------------------------------------------
    # 6) Reconstruct the final clusters by replaying merges in
    #    *ascending* Tau order, skipping:
    #    - The top L merges in descending Tau (i.e. merges_sorted[:L])
    #    - Any halo merges if is_noise=True
    #----------------------------------------------------------------------
    # We'll do a standard Union-Find on the n original points.
    # merges_sorted is descending by Tau, so the "top L merges" are merges_sorted[:L].
    # We'll skip them. Then to replay merges in ascending order, let's invert merges_sorted.
    skip_indices = set(sort_idx[:L])  # these are the merges to remove (top L in desc)
    if is_noise:
        # also skip halo merges
        halo_indices = np.where(halo_mask)[0]
        # but halo_indices are indices in merges_sorted => need global merges arr index
        halo_global = set(sort_idx[halo_indices]) 
        skip_indices = skip_indices.union(halo_global)

    # Build a map from sorted index => original global index
    # but we already have that: sort_idx. merges_sorted[i] = merges_arr[ sort_idx[i] ]
    # We'll union merges in ascending tau => merges_arr sorted ascending by Tau
    ascending_idx = np.argsort(merges_arr['Tau'])  # ascending

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

    # Re-apply merges in ascending Tau
    for idx in ascending_idx:
        if idx in skip_indices:
            # skip
            continue
        # merges_arr[idx] is a chronological merge of cA with cB, but we have pointsA, pointsB
        ptsA = merges_arr[idx]['pointsA']
        ptsB = merges_arr[idx]['pointsB']
        # unify their points in union-find
        # pick the first point of each set as a "representative"
        # (we could do them all, but it's enough to unify just the first point in each cluster
        # because eventually that merges entire sets.)
        if len(ptsA) > 0 and len(ptsB) > 0:
            union(ptsA[0], ptsB[0])

    # Now read off the final cluster labels
    labels = np.zeros(n, dtype=int)
    # find() each point, then re-label
    # gather unique roots
    for i in range(n):
        labels[i] = find(i)
    # compress those root labels into 0..(k-1)
    unique_roots = np.unique(labels)
    root_map = {}
    for i, r in enumerate(unique_roots):
        root_map[r] = i
    for i in range(n):
        labels[i] = root_map[labels[i]]

    final_k = len(unique_roots)

    return labels, final_k, merges_sorted
