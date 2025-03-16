# Diagnostic Output in TorqueClustering

## Overview

The TorqueClustering algorithm now provides detailed diagnostic information about the clustering process, threshold calculations, and final results. This information helps understand how the algorithm makes decisions and how different parameters affect the clustering outcome.

## Diagnostic Structure

The diagnostic output is a dictionary containing several nested sections:

### 1. Parameters

```python
diagnostics['parameters'] = {
    'K': int,                      # Number of clusters (0 for auto-detection)
    'isnoise': bool,               # Whether noise detection was enabled
    'use_std_adjustment': bool,    # Whether std deviation adjustment was used
    'adjustment_factor': float,    # The adjustment factor used
    'matlab_compatibility': bool   # Whether MATLAB compatibility mode was enabled
}
```

### 2. Input Matrix Information

```python
diagnostics['input_matrix'] = {
    'shape': tuple,                # Shape of the distance matrix
    'is_sparse': bool,             # Whether input was sparse
    'dtype': str                   # Data type of the matrix
}
```

### 3. Threshold Analysis

```python
diagnostics['threshold_analysis'] = {
    'input_stats': {
        'p_min': float,            # Minimum torque value
        'p_max': float,            # Maximum torque value
        'mass_min': float,         # Minimum mass value
        'mass_max': float,         # Maximum mass value
        'R_min': float,            # Minimum distance value
        'R_max': float            # Maximum distance value
    },
    'means': {
        'p_mean': float,           # Mean torque
        'mass_mean': float,        # Mean mass
        'R_mean': float           # Mean distance
    },
    'standard_deviations': {
        'p_std': float,            # Std dev of torque
        'mass_std': float,         # Std dev of mass
        'R_std': float            # Std dev of distance
    },
    'thresholds': {
        'p_threshold': float,      # Final torque threshold
        'mass_threshold': float,   # Final mass threshold
        'R_threshold': float      # Final distance threshold
    },
    'criteria_counts': {
        'points_above_R_threshold': int,    # Points meeting R criterion
        'points_above_mass_threshold': int, # Points meeting mass criterion
        'points_above_p_threshold': int    # Points meeting torque criterion
    }
}
```

### 4. Clustering Results

```python
diagnostics['clustering_results'] = {
    'cutnum': int,                 # Number of cuts made
    'total_clusters': int,         # Total number of clusters found
    'cluster_sizes': dict         # Size of each cluster
}
```

### 5. Noise Detection (if enabled)

```python
diagnostics['noise_detection'] = {
    'total_clusters_with_noise': int,      # Clusters after noise detection
    'noise_points_count': int,             # Number of noise points
    'cluster_sizes_with_noise': dict      # Cluster sizes including noise
}
```

## Using Diagnostic Information

### 1. Threshold Tuning

Monitor the effect of `adjustment_factor` on thresholds:
```python
# Example: Check if thresholds are too strict
thresholds = diagnostics['threshold_analysis']['thresholds']
criteria_counts = diagnostics['threshold_analysis']['criteria_counts']

if criteria_counts['points_above_p_threshold'] < 10:
    print("Warning: Very few points above torque threshold")
```

### 2. Cluster Quality Assessment

Analyze cluster size distribution:
```python
cluster_sizes = diagnostics['clustering_results']['cluster_sizes']
avg_size = sum(cluster_sizes.values()) / len(cluster_sizes)
print(f"Average cluster size: {avg_size}")
```

### 3. Noise Detection Evaluation

Check noise detection effectiveness:
```python
if 'noise_detection' in diagnostics:
    noise_ratio = (diagnostics['noise_detection']['noise_points_count'] / 
                  sum(diagnostics['clustering_results']['cluster_sizes'].values()))
    print(f"Noise ratio: {noise_ratio:.2%}")
```

## Best Practices

1. **Parameter Tuning**:
   - Use `input_stats` to understand data scale
   - Compare `thresholds` with data distribution
   - Adjust parameters based on `criteria_counts`

2. **Quality Control**:
   - Monitor `cluster_sizes` for balance
   - Check `noise_points_count` for outliers
   - Verify threshold effectiveness

3. **Troubleshooting**:
   - Compare `means` and `thresholds`
   - Check `criteria_counts` for bottlenecks
   - Review `standard_deviations` for data spread

## Example Usage

```python
from TorqueClustering import TorqueClustering
import numpy as np

# Run clustering with diagnostics
labels, *rest, diagnostics = TorqueClustering(
    distance_matrix,
    use_std_adjustment=True,
    adjustment_factor=0.5
)

# Analyze thresholds
thresholds = diagnostics['threshold_analysis']['thresholds']
print(f"Torque threshold: {thresholds['p_threshold']:.2f}")

# Check cluster distribution
sizes = diagnostics['clustering_results']['cluster_sizes']
print(f"Number of clusters: {len(sizes)}")
print(f"Cluster sizes: {sizes}")

# Evaluate noise detection
if diagnostics.get('noise_detection'):
    print(f"Noise points: {diagnostics['noise_detection']['noise_points_count']}")
``` 