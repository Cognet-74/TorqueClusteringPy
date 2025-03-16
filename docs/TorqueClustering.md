# TorqueClustering Algorithm Documentation

## Overview

The TorqueClustering algorithm is a novel approach to clustering that uses physical analogies to identify natural clusters in data. It works by analyzing the "torque" between points in the feature space, which helps identify cluster boundaries and automatically determine the optimal number of clusters.

## Algorithm Details

### Core Concepts

1. **Torque Calculation**
   - Uses distance and mass to compute torque between points
   - Higher torque indicates stronger cluster boundaries
   - Automatically identifies cluster centers and boundaries

2. **Mass Computation**
   - Each point's mass is determined by its local density
   - Helps identify dense regions that form cluster cores

3. **Distance Analysis**
   - Uses pairwise distances between points
   - Supports multiple distance metrics (Euclidean, Cosine)
   - Can handle both dense and sparse distance matrices

### Main Function

```python
def TorqueClustering(
    ALL_DM: Union[np.ndarray, scipy.sparse.spmatrix],
    K: int = 0,
    isnoise: bool = False,
    isfig: bool = False
) -> Tuple[np.ndarray, ...]:
```

#### Parameters

- `ALL_DM`: Distance matrix (n x n)
  - Can be dense numpy array or sparse matrix
  - Must be symmetric
  - Diagonal should be zeros

- `K`: Number of clusters
  - 0: Automatic detection (recommended)
  - >0: Force specific number of clusters

- `isnoise`: Noise detection flag
  - True: Enable noise detection
  - False: Treat all points as part of clusters

- `isfig`: Visualization flag
  - True: Generate decision graph
  - False: No visualization

#### Returns

The function returns a tuple containing:

1. `Idx`: Cluster labels (1-based indexing)
2. `Idx_with_noise`: Labels with noise points marked as 0
3. `cutnum`: Number of connections cut
4. `cutlink_ori`: Original cut links
5. `p`: Torque values for each connection
6. `firstlayer_loc_onsortp`: First layer connection indices
7. `mass`: Mass values for each connection
8. `R`: Distance squared values
9. `cutlinkpower_all`: Cut properties record

### Algorithm Steps

1. **Initialization**
   ```python
   # Convert input to appropriate format
   if not scipy.sparse.issparse(ALL_DM):
       ALL_DM_sparse = scipy.sparse.csr_matrix(ALL_DM)
   ```

2. **Mass Calculation**
   ```python
   # Compute mass for each point based on local density
   mass = compute_mass(ALL_DM_sparse)
   ```

3. **Torque Analysis**
   ```python
   # Calculate torque between points
   p = compute_torque(ALL_DM_sparse, mass)
   ```

4. **Cluster Identification**
   ```python
   # Find cluster boundaries using torque values
   cutlinks = find_cluster_boundaries(p, mass)
   ```

5. **Label Assignment**
   ```python
   # Assign cluster labels to points
   Idx = assign_labels(cutlinks, ALL_DM_sparse.shape[0])
   ```

### Noise Detection

When `isnoise=True`, the algorithm:
1. Identifies sparse regions in the data
2. Marks points in these regions as noise (label 0)
3. Returns both clean and noisy labelings

### Visualization

When `isfig=True`, generates:
1. Decision graph showing torque vs. distance
2. Cluster boundary visualization
3. Noise point identification (if enabled)

## Usage Examples

### Basic Usage

```python
import numpy as np
from scipy.spatial.distance import cdist
from TorqueClustering import TorqueClustering

# Prepare data
data = np.random.rand(100, 2)
DM = cdist(data, data, metric='euclidean')

# Run clustering
idx = TorqueClustering(DM)[0]
```

### With Noise Detection

```python
# Enable noise detection
idx_clean, idx_with_noise = TorqueClustering(DM, isnoise=True)[:2]
```

### Forced Number of Clusters

```python
# Force 5 clusters
idx = TorqueClustering(DM, K=5)[0]
```

### Sparse Matrix Input

```python
import scipy.sparse

# Convert to sparse matrix
DM_sparse = scipy.sparse.csr_matrix(DM)

# Run clustering
idx = TorqueClustering(DM_sparse)[0]
```

## Performance Considerations

1. **Memory Usage**
   - For large datasets, use sparse matrices
   - Memory scales with O(n²) for dense matrices
   - Sparse matrices can significantly reduce memory usage

2. **Computation Time**
   - Main bottleneck is distance matrix computation
   - Sparse matrices can speed up computation
   - Scales approximately O(n²) with dataset size

3. **Optimization Tips**
   - Pre-compute and cache distance matrix
   - Use appropriate distance metric for data type
   - Consider dimensionality reduction for high-dim data

## Error Handling

The algorithm includes various error checks:

1. **Input Validation**
   ```python
   if ALL_DM is None:
       raise ValueError('Distance Matrix is required')
   ```

2. **Matrix Properties**
   ```python
   if not is_symmetric(ALL_DM):
       raise ValueError('Distance Matrix must be symmetric')
   ```

3. **Parameter Validation**
   ```python
   if K < 0:
       raise ValueError('K must be non-negative')
   ```

## References

1. Original Paper: [Link to paper]
2. MATLAB Implementation: [Link to MATLAB code]
3. Related Algorithms:
   - DBSCAN
   - Spectral Clustering
   - Hierarchical Clustering 