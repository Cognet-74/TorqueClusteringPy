# TorqueClusteringPy

A Python implementation of the Torque Clustering algorithm, designed for efficient and accurate clustering of various datasets. This implementation maintains exact compatibility with the original MATLAB version while leveraging Python's scientific computing capabilities.

## Overview

TorqueClustering is a novel clustering algorithm that uses physical analogies to identify clusters in data. It works by:
1. Computing pairwise distances between data points
2. Analyzing the "torque" between points to identify natural cluster boundaries
3. Automatically determining the optimal number of clusters
4. Handling noise and outliers effectively

## Features

- Automatic determination of cluster numbers
- Robust handling of various data shapes and distributions
- Support for both dense and sparse distance matrices
- Noise detection capabilities
- Multiple distance metric support (Euclidean, Cosine)
- Comprehensive evaluation metrics (NMI, AMI, AC)

## Installation

```bash
git clone https://github.com/Cognet-74/TorqueClusteringPy.git
cd TorqueClusteringPy
pip install -r requirements.txt
```

## Dependencies

- Python >= 3.6
- NumPy
- SciPy
- Matplotlib
- h5py (for specific dataset formats)

## Quick Start

```python
import numpy as np
from scipy.spatial.distance import cdist
from TorqueClustering import TorqueClustering

# Prepare your data
data = np.array(...)  # Your data points
DM = cdist(data, data, metric='euclidean')  # Distance matrix

# Run clustering
idx = TorqueClustering(DM, K=0, isnoise=False, isfig=True)[0]

# idx contains the cluster assignments for each point
```

## Module Structure

The codebase consists of several key modules:

1. `TorqueClustering.py`: Core clustering algorithm implementation
2. `TorqueClustering_Run.py`: Example usage and benchmark datasets
3. `uniqueZ.py`: Unique cluster identification utilities
4. `Nab_dec.py`: Decision graph analysis
5. `ps2psdist.py`: Point set distance calculations
6. `Final_label.py`: Final cluster label assignment
7. `evaluatecluster.py`: Clustering evaluation metrics
8. `ami.py`: Adjusted Mutual Information calculation
9. `nmi.py`: Normalized Mutual Information calculation
10. `accuracy.py`: Clustering accuracy metrics

## Detailed Documentation

### Core Algorithm (TorqueClustering.py)

The main clustering function accepts the following parameters:

```python
def TorqueClustering(
    ALL_DM: Union[np.ndarray, scipy.sparse.spmatrix],
    K: int = 0,
    isnoise: bool = False,
    isfig: bool = False
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Main clustering function.
    
    Args:
        ALL_DM: Distance Matrix (n x n)
        K: Number of clusters (0 for automatic detection)
        isnoise: Enable noise detection
        isfig: Generate decision graph figure
    
    Returns:
        Tuple containing:
        - Idx: Cluster labels
        - Idx_with_noise: Labels with noise handling
        - cutnum: Number of connections cut
        - cutlink_ori: Original cut links
        - p: Torque values
        - firstlayer_loc_onsortp: First layer indices
        - mass: Mass values
        - R: Distance values
        - cutlinkpower_all: Cut properties
    """
```

### Data Processing (process_dataset function)

```python
def process_dataset(
    data_dict: Union[Dict, np.ndarray],
    label_column: int = 2,
    add_to_labels: int = 1
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Standardized dataset processing.
    
    Args:
        data_dict: Input data (dictionary or array)
        label_column: Label column index
        add_to_labels: Label offset value
    
    Returns:
        Tuple of (data array, labels array)
    """
```

## Usage Examples

### Basic Clustering

```python
from TorqueClustering import TorqueClustering
import numpy as np
from scipy.spatial.distance import cdist

# Generate sample data
np.random.seed(42)
n_samples = 300
data = np.concatenate([
    np.random.normal(0, 1, (100, 2)),
    np.random.normal(4, 1, (100, 2)),
    np.random.normal(8, 1, (100, 2))
])

# Compute distance matrix
DM = cdist(data, data, metric='euclidean')

# Perform clustering
idx = TorqueClustering(DM, K=0, isnoise=False, isfig=True)[0]

# Evaluate results (if ground truth available)
from evaluatecluster import evaluatecluster
true_labels = np.repeat([0, 1, 2], 100)
NMI, AC = evaluatecluster(idx, true_labels)
print(f"NMI: {NMI:.4f}, AC: {AC:.4f}")
```

### Handling Different Data Types

```python
# Load and process various data formats
from TorqueClustering_Run import process_dataset, load_data

# MATLAB data
data, labels = process_dataset(load_data('data1.mat'))

# Text data
data, labels = process_dataset(load_data('data.txt'))

# Custom data
data = np.random.rand(100, 3)
DM = cdist(data, data, metric='euclidean')
idx = TorqueClustering(DM)[0]
```

## Evaluation Metrics

The package includes several evaluation metrics:

1. **NMI (Normalized Mutual Information)**
   - Range: [0, 1]
   - Higher values indicate better clustering
   - Implementation in `nmi.py`

2. **AMI (Adjusted Mutual Information)**
   - Adjusts for chance
   - Implementation in `ami.py`

3. **AC (Accuracy)**
   - Direct cluster assignment accuracy
   - Implementation in `accuracy.py`

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

This project is licensed under the terms of the LICENSE.txt file.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{torqueclusteringpy,
    title = {TorqueClusteringPy: Python Implementation of Torque Clustering},
    author = {Your Name},
    year = {2024},
    url = {https://github.com/Cognet-74/TorqueClusteringPy}
}
```

## Contact

For questions and support, please open an issue on the GitHub repository.
