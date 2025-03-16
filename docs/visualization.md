# Visualization Utilities Documentation

## Overview

TorqueClusteringPy provides comprehensive visualization tools to help understand clustering results, analyze algorithm performance, and debug clustering issues. These utilities support both 2D and 3D data visualization, with customizable plotting options.

## Core Visualization Functions

### 1. Plot Clusters

```python
def plot_clusters(
    data: np.ndarray,
    labels: np.ndarray,
    title: str = "Clustering Results",
    show_noise: bool = True,
    marker_size: int = 30
) -> None:
    """
    Plot clustering results with different colors for each cluster.
    
    Parameters:
        data: Input data points (n_samples, n_features)
        labels: Cluster labels (n_samples,)
        title: Plot title
        show_noise: Whether to show noise points (labeled as -1)
        marker_size: Size of scatter plot markers
    """
```

#### Usage Example
```python
from visualization import plot_clusters

# Basic usage
plot_clusters(X, labels)

# With custom settings
plot_clusters(X, labels, 
             title="My Clustering Results",
             show_noise=True,
             marker_size=50)
```

### 2. Plot Torque Analysis

```python
def plot_torque(
    torque_values: np.ndarray,
    threshold: float,
    title: str = "Torque Analysis"
) -> None:
    """
    Visualize torque values and threshold for cluster determination.
    
    Parameters:
        torque_values: Computed torque values
        threshold: Torque threshold for clustering
        title: Plot title
    """
```

#### Usage Example
```python
from visualization import plot_torque

plot_torque(torque_vals, threshold=0.5)
```

### 3. Interactive 3D Plot

```python
def plot_3d_clusters(
    data: np.ndarray,
    labels: np.ndarray,
    azimuth: float = -60,
    elevation: float = 30
) -> None:
    """
    Create interactive 3D plot of clustering results.
    
    Parameters:
        data: 3D input data (n_samples, 3)
        labels: Cluster labels
        azimuth: Initial azimuth angle
        elevation: Initial elevation angle
    """
```

#### Usage Example
```python
from visualization import plot_3d_clusters

plot_3d_clusters(X_3d, labels)
```

## Advanced Visualization Features

### 1. Cluster Evolution Plot

```python
def plot_cluster_evolution(
    data: np.ndarray,
    labels_history: List[np.ndarray],
    titles: List[str]
) -> None:
    """
    Show how clusters evolve through algorithm iterations.
    
    Parameters:
        data: Input data points
        labels_history: List of labels at each iteration
        titles: Subplot titles for each iteration
    """
```

#### Usage Example
```python
from visualization import plot_cluster_evolution

plot_cluster_evolution(X, 
                      labels_history=[labels_iter1, labels_iter2],
                      titles=["Initial", "Final"])
```

### 2. Distance Matrix Visualization

```python
def plot_distance_matrix(
    distance_matrix: np.ndarray,
    log_scale: bool = False
) -> None:
    """
    Visualize pairwise distance matrix.
    
    Parameters:
        distance_matrix: Pairwise distances
        log_scale: Use logarithmic color scale
    """
```

#### Usage Example
```python
from visualization import plot_distance_matrix

plot_distance_matrix(dist_matrix, log_scale=True)
```

## Customization Options

### 1. Color Schemes

```python
def set_color_scheme(
    scheme: str = "default",
    custom_colors: List[str] = None
) -> None:
    """
    Set color scheme for cluster visualization.
    
    Parameters:
        scheme: Predefined scheme ('default', 'bright', 'pastel')
        custom_colors: List of custom color hex codes
    """
```

### 2. Plot Styling

```python
def set_plot_style(
    style_dict: Dict[str, Any]
) -> None:
    """
    Configure global plot styling.
    
    Parameters:
        style_dict: Dictionary of matplotlib style parameters
    """
```

## Best Practices

### 1. Memory Management

```python
# For large datasets
def plot_clusters_subset(
    data: np.ndarray,
    labels: np.ndarray,
    max_points: int = 10000
):
    # Randomly sample points if dataset is too large
    if len(data) > max_points:
        idx = np.random.choice(len(data), max_points, replace=False)
        data = data[idx]
        labels = labels[idx]
    plot_clusters(data, labels)
```

### 2. Interactive Features

```python
def add_interactive_features(ax):
    """Add zoom, pan, and hover functionality."""
    ax.set_picker(True)
    ax.figure.canvas.mpl_connect('pick_event', on_pick)
```

### 3. Saving Plots

```python
def save_visualization(
    fig,
    filename: str,
    dpi: int = 300,
    format: str = 'png'
):
    """Save plot with high quality settings."""
    fig.savefig(filename, dpi=dpi, format=format, bbox_inches='tight')
```

## Common Issues and Solutions

### 1. Overlapping Points

```python
# Solution: Use transparency
plt.scatter(X[:, 0], X[:, 1], c=labels, alpha=0.5)
```

### 2. Memory Issues with Large Datasets

```python
# Solution: Use downsampling
def downsample_for_plot(data, labels, target_size=10000):
    if len(data) > target_size:
        idx = np.random.choice(len(data), target_size, replace=False)
        return data[idx], labels[idx]
    return data, labels
```

### 3. 3D Plot Performance

```python
# Solution: Optimize 3D rendering
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1,1,1])  # Equal aspect ratio
```

## Examples

### 1. Basic Clustering Visualization

```python
# Example of basic cluster visualization
data = np.random.randn(1000, 2)
labels = cluster_algorithm(data)
plot_clusters(data, labels)
```

### 2. Advanced Multi-plot Analysis

```python
def create_analysis_dashboard(data, labels):
    """Create comprehensive visualization dashboard."""
    fig = plt.figure(figsize=(15, 10))
    
    # Cluster plot
    ax1 = fig.add_subplot(221)
    plot_clusters(data, labels, ax=ax1)
    
    # Distance matrix
    ax2 = fig.add_subplot(222)
    plot_distance_matrix(compute_distances(data), ax=ax2)
    
    # Torque analysis
    ax3 = fig.add_subplot(223)
    plot_torque(compute_torque(data), ax=ax3)
    
    # Statistics
    ax4 = fig.add_subplot(224)
    plot_cluster_stats(labels, ax=ax4)
    
    plt.tight_layout()
    return fig
```

## Contributing

To add new visualization features:

1. Create visualization function:
```python
def new_visualization(data, **kwargs):
    """
    New visualization type.
    
    Args:
        data: Input data
        **kwargs: Additional parameters
    """
    # Implementation
    pass
```

2. Add to visualization module:
```python
# In visualization/__init__.py
from .new_viz import new_visualization
```

3. Update documentation and add examples

## References

1. Matplotlib Documentation
2. Seaborn for Statistical Visualizations
3. Plotly for Interactive Plots 