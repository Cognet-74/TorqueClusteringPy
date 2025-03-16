# API Reference

## Core Module: TorqueClustering

### Main Functions

#### TorqueClustering

```python
def TorqueClustering(
    ALL_DM: np.ndarray,
    K: Optional[int] = None,
    isnoise: bool = False,
    isfig: bool = False
) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """
    Main clustering function implementing the Torque Clustering algorithm.

    Parameters
    ----------
    ALL_DM : np.ndarray
        Distance matrix of shape (n_samples, n_samples)
    K : Optional[int]
        Number of clusters. If None, automatically determined
    isnoise : bool
        Whether to detect and label noise points
    isfig : bool
        Whether to generate visualization figures

    Returns
    -------
    labels : np.ndarray
        Cluster labels for each point
    torque_threshold : float
        Final torque threshold used for clustering
    quality_score : float
        Clustering quality score
    torque_values : np.ndarray
        Computed torque values for each point
    """
```

#### process_dataset

```python
def process_dataset(
    data: Union[np.ndarray, str, Path],
    metric: str = 'euclidean',
    precomputed: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process input dataset and compute distance matrix.

    Parameters
    ----------
    data : Union[np.ndarray, str, Path]
        Input data as array or path to data file
    metric : str
        Distance metric to use
    precomputed : bool
        Whether input is a precomputed distance matrix

    Returns
    -------
    distance_matrix : np.ndarray
        Computed distance matrix
    raw_data : np.ndarray
        Original data points
    """
```

### Utility Functions

#### compute_torque

```python
def compute_torque(
    distance_matrix: np.ndarray,
    mass_values: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute torque values for each point.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Pairwise distance matrix
    mass_values : Optional[np.ndarray]
        Pre-computed mass values for each point

    Returns
    -------
    torque_values : np.ndarray
        Computed torque for each point
    """
```

#### compute_mass

```python
def compute_mass(
    distance_matrix: np.ndarray,
    sigma: Optional[float] = None
) -> np.ndarray:
    """
    Compute mass values for each point.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Pairwise distance matrix
    sigma : Optional[float]
        Bandwidth parameter for mass computation

    Returns
    -------
    mass_values : np.ndarray
        Computed mass for each point
    """
```

#### detect_noise

```python
def detect_noise(
    distance_matrix: np.ndarray,
    torque_values: np.ndarray,
    threshold: float
) -> np.ndarray:
    """
    Detect noise points in the dataset.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Pairwise distance matrix
    torque_values : np.ndarray
        Computed torque values
    threshold : float
        Threshold for noise detection

    Returns
    -------
    noise_mask : np.ndarray
        Boolean mask indicating noise points
    """
```

## Data Processing Module

### Input/Output Functions

#### load_data

```python
def load_data(
    file_path: Union[str, Path],
    format: str = 'auto'
) -> np.ndarray:
    """
    Load data from file.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to data file
    format : str
        File format ('auto', 'mat', 'txt', 'h5')

    Returns
    -------
    data : np.ndarray
        Loaded data array
    """
```

#### save_results

```python
def save_results(
    labels: np.ndarray,
    output_path: Union[str, Path],
    metadata: Optional[Dict] = None
) -> None:
    """
    Save clustering results to file.

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels
    output_path : Union[str, Path]
        Path to save results
    metadata : Optional[Dict]
        Additional metadata to save
    """
```

## Visualization Module

### Plotting Functions

#### plot_clusters

```python
def plot_clusters(
    data: np.ndarray,
    labels: np.ndarray,
    title: str = "Clustering Results",
    show_noise: bool = True,
    marker_size: int = 30
) -> None:
    """
    Plot clustering results.

    Parameters
    ----------
    data : np.ndarray
        Input data points
    labels : np.ndarray
        Cluster labels
    title : str
        Plot title
    show_noise : bool
        Whether to show noise points
    marker_size : int
        Size of scatter plot markers
    """
```

#### plot_torque

```python
def plot_torque(
    torque_values: np.ndarray,
    threshold: float,
    title: str = "Torque Analysis"
) -> None:
    """
    Plot torque analysis results.

    Parameters
    ----------
    torque_values : np.ndarray
        Computed torque values
    threshold : float
        Torque threshold
    title : str
        Plot title
    """
```

## Evaluation Module

### Metrics

#### nmi

```python
def nmi(
    labels_true: np.ndarray,
    labels_pred: np.ndarray
) -> float:
    """
    Calculate Normalized Mutual Information.

    Parameters
    ----------
    labels_true : np.ndarray
        Ground truth labels
    labels_pred : np.ndarray
        Predicted cluster labels

    Returns
    -------
    score : float
        NMI score in range [0, 1]
    """
```

#### accuracy

```python
def accuracy(
    labels_true: np.ndarray,
    labels_pred: np.ndarray
) -> float:
    """
    Calculate clustering accuracy.

    Parameters
    ----------
    labels_true : np.ndarray
        Ground truth labels
    labels_pred : np.ndarray
        Predicted cluster labels

    Returns
    -------
    score : float
        Accuracy score in range [0, 1]
    """
```

## Types and Constants

### Custom Types

```python
# Type aliases
DistanceMatrix = np.ndarray  # Shape: (n_samples, n_samples)
Labels = np.ndarray  # Shape: (n_samples,)
TorqueValues = np.ndarray  # Shape: (n_samples,)
MassValues = np.ndarray  # Shape: (n_samples,)

# Configuration types
class ClusteringConfig(TypedDict):
    K: Optional[int]
    isnoise: bool
    isfig: bool
    metric: str
```

### Constants

```python
# Default values
DEFAULT_K = None  # Auto-determine number of clusters
DEFAULT_METRIC = 'euclidean'
DEFAULT_NOISE_THRESHOLD = 0.1

# Supported distance metrics
DISTANCE_METRICS = [
    'euclidean',
    'manhattan',
    'cosine',
    'correlation'
]

# File formats
SUPPORTED_FORMATS = [
    'mat',  # MATLAB files
    'txt',  # Text files
    'h5'    # HDF5 files
]
```

## Error Handling

### Custom Exceptions

```python
class TorqueClusteringError(Exception):
    """Base exception for TorqueClustering errors."""
    pass

class InvalidDataError(TorqueClusteringError):
    """Raised when input data is invalid."""
    pass

class ConvergenceError(TorqueClusteringError):
    """Raised when algorithm fails to converge."""
    pass
```

### Error Messages

```python
ERROR_MESSAGES = {
    'invalid_distance_matrix': 'Distance matrix must be square and symmetric',
    'invalid_k': 'K must be positive integer or None',
    'convergence_failed': 'Algorithm failed to converge',
    'file_not_found': 'Data file not found: {}',
    'unsupported_format': 'Unsupported file format: {}'
}
```

## Configuration

### Environment Variables

```python
# Performance settings
MAX_THREADS = int(os.getenv('MAX_THREADS', '4'))
USE_GPU = bool(os.getenv('USE_GPU', 'false').lower() == 'true')

# Debug settings
DEBUG_MODE = bool(os.getenv('DEBUG_MODE', 'false').lower() == 'true')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
```

### Logging Configuration

```python
import logging

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('TorqueClustering')
```

## Examples

### Basic Usage

```python
from TorqueClustering import TorqueClustering, process_dataset

# Load and process data
data = load_data('data.txt')
distance_matrix = process_dataset(data, metric='euclidean')

# Run clustering
labels, threshold, quality, torque = TorqueClustering(
    distance_matrix,
    K=None,
    isnoise=True,
    isfig=True
)
```

### Advanced Usage

```python
# Custom configuration
config = ClusteringConfig(
    K=5,
    isnoise=True,
    isfig=True,
    metric='correlation'
)

# Process with custom settings
try:
    distance_matrix = process_dataset(
        data,
        metric=config['metric'],
        precomputed=False
    )
    
    labels, *metrics = TorqueClustering(
        distance_matrix,
        K=config['K'],
        isnoise=config['isnoise'],
        isfig=config['isfig']
    )
    
except TorqueClusteringError as e:
    logger.error(f"Clustering failed: {e}")
    raise
```

## Performance Tips

### Memory Optimization

```python
# Use sparse matrices for large datasets
from scipy.sparse import csr_matrix

def optimize_memory(distance_matrix):
    if distance_matrix.size > 1e6:
        return csr_matrix(distance_matrix)
    return distance_matrix
```

### Parallel Processing

```python
# Enable parallel processing
from concurrent.futures import ThreadPoolExecutor

def parallel_compute(func, data, max_workers=MAX_THREADS):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(func, data))
```

## Contributing Guidelines

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines on:
- Code style
- Testing requirements
- Documentation standards
- Pull request process

## Version History

- v1.0.0: Initial release
- v1.1.0: Added GPU support
- v1.2.0: Improved noise detection
- v1.3.0: Added new distance metrics 