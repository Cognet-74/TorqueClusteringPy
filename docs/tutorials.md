# Tutorials and Examples

## Basic Examples

### 1. Simple Clustering

```python
import numpy as np
from TorqueClustering import TorqueClustering, process_dataset

# Generate sample data
np.random.seed(42)
n_samples = 300
data = np.concatenate([
    np.random.normal(0, 1, (100, 2)),   # Cluster 1
    np.random.normal(4, 1, (100, 2)),   # Cluster 2
    np.random.normal([2, 4], 1, (100, 2))  # Cluster 3
])

# Process data and run clustering
distance_matrix = process_dataset(data, metric='euclidean')
labels, threshold, quality, torque = TorqueClustering(
    distance_matrix,
    K=None,  # Auto-determine number of clusters
    isnoise=False,
    isfig=True  # Show visualization
)

print(f"Number of clusters found: {len(np.unique(labels))}")
print(f"Clustering quality score: {quality:.3f}")
```

### 2. Handling Noise

```python
# Generate data with noise
clean_data = np.random.normal(0, 1, (200, 2))
noise = np.random.uniform(-5, 5, (50, 2))
data = np.vstack([clean_data, noise])

# Run clustering with noise detection
distance_matrix = process_dataset(data)
labels, threshold, quality, torque = TorqueClustering(
    distance_matrix,
    isnoise=True,  # Enable noise detection
    isfig=True
)

# Analyze results
noise_points = np.sum(labels == -1)
print(f"Detected {noise_points} noise points")
```

## Advanced Tutorials

### 1. Custom Distance Metrics

```python
from scipy.spatial.distance import cdist

def custom_distance(X, Y):
    """Custom distance metric combining Euclidean and cosine distances."""
    euclidean = cdist(X, Y, metric='euclidean')
    cosine = cdist(X, Y, metric='cosine')
    return 0.7 * euclidean + 0.3 * cosine

# Use custom metric
data = np.random.randn(200, 10)
distance_matrix = cdist(data, data, metric=custom_distance)

# Run clustering
labels = TorqueClustering(distance_matrix, precomputed=True)[0]
```

### 2. Large Dataset Handling

```python
from scipy.sparse import csr_matrix
import h5py

def process_large_dataset(file_path, chunk_size=1000):
    """Process large dataset in chunks."""
    with h5py.File(file_path, 'r') as f:
        data = f['data']
        n_samples = data.shape[0]
        
        # Initialize sparse distance matrix
        distance_matrix = csr_matrix((n_samples, n_samples))
        
        # Process in chunks
        for i in range(0, n_samples, chunk_size):
            chunk = data[i:i+chunk_size]
            distances = process_dataset(chunk)
            distance_matrix[i:i+chunk_size] = distances
            
    return distance_matrix

# Use with large dataset
distance_matrix = process_large_dataset('large_data.h5')
labels = TorqueClustering(distance_matrix, K=10)[0]
```

### 3. Real-time Clustering

```python
class RealTimeClustering:
    def __init__(self, buffer_size=1000):
        self.buffer = np.zeros((buffer_size, 2))
        self.buffer_idx = 0
        self.labels = None
        
    def update(self, new_point):
        """Update clustering with new data point."""
        # Add to buffer
        self.buffer[self.buffer_idx] = new_point
        self.buffer_idx = (self.buffer_idx + 1) % len(self.buffer)
        
        # Recluster when buffer is full
        if self.buffer_idx == 0:
            distance_matrix = process_dataset(self.buffer)
            self.labels = TorqueClustering(distance_matrix)[0]
            
        return self.labels

# Usage
clusterer = RealTimeClustering()
for _ in range(2000):
    new_point = np.random.randn(2)
    labels = clusterer.update(new_point)
```

## Practical Applications

### 1. Image Segmentation

```python
from PIL import Image
import numpy as np

def segment_image(image_path):
    """Segment image using TorqueClustering."""
    # Load and preprocess image
    img = Image.open(image_path)
    pixels = np.array(img).reshape(-1, 3)  # RGB values
    
    # Cluster pixels
    distance_matrix = process_dataset(pixels, metric='euclidean')
    labels = TorqueClustering(distance_matrix, K=5)[0]
    
    # Reconstruct segmented image
    segmented = labels.reshape(img.size[::-1])
    return segmented

# Example usage
segmented_image = segment_image('sample.jpg')
```

### 2. Time Series Clustering

```python
def cluster_time_series(series_data, window_size=10):
    """Cluster time series data using sliding windows."""
    # Create windows
    n_series = len(series_data)
    windows = np.zeros((n_series, window_size))
    
    for i in range(n_series - window_size + 1):
        windows[i] = series_data[i:i+window_size]
    
    # Compute DTW distance matrix
    from dtaidistance import dtw
    distance_matrix = dtw.distance_matrix(windows)
    
    # Cluster
    labels = TorqueClustering(distance_matrix, isnoise=True)[0]
    return labels

# Example with stock data
import yfinance as yf
stock_data = yf.download('AAPL', start='2020-01-01')['Close']
clusters = cluster_time_series(stock_data.values)
```

### 3. Text Document Clustering

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import pdist, squareform

def cluster_documents(documents):
    """Cluster text documents using TF-IDF and TorqueClustering."""
    # Convert to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Compute cosine distance matrix
    distances = pdist(tfidf_matrix.toarray(), metric='cosine')
    distance_matrix = squareform(distances)
    
    # Cluster
    labels = TorqueClustering(distance_matrix)[0]
    return labels

# Example usage
documents = [
    "Machine learning is fascinating",
    "Deep learning revolutionizes AI",
    "Python is a great language",
    "Programming in Python is fun"
]
doc_clusters = cluster_documents(documents)
```

## Performance Optimization

### 1. GPU Acceleration

```python
import cupy as cp

def gpu_clustering(data):
    """Run clustering on GPU."""
    # Transfer data to GPU
    gpu_data = cp.array(data)
    
    # Compute distance matrix on GPU
    gpu_distances = cp.zeros((len(data), len(data)))
    for i in range(len(data)):
        diff = gpu_data - gpu_data[i]
        gpu_distances[i] = cp.sqrt(cp.sum(diff**2, axis=1))
    
    # Transfer back to CPU for clustering
    cpu_distances = gpu_distances.get()
    return TorqueClustering(cpu_distances)[0]
```

### 2. Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def parallel_distance_computation(data, n_jobs=4):
    """Compute distance matrix in parallel."""
    chunk_size = len(data) // n_jobs
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        distances = list(executor.map(
            lambda x: process_dataset(x, metric='euclidean'),
            chunks
        ))
    
    return np.vstack(distances)
```

## Visualization Examples

### 1. Advanced Plotting

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_clusters(data, labels):
    """Create interactive 3D plot of clusters."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each cluster
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        ax.scatter(
            data[mask, 0],
            data[mask, 1],
            data[mask, 2],
            c=[color],
            label=f'Cluster {label}'
        )
    
    ax.legend()
    plt.show()
```

### 2. Cluster Evolution Animation

```python
import matplotlib.animation as animation

def animate_clustering(data, n_frames=50):
    """Create animation of clustering process."""
    fig, ax = plt.subplots()
    
    def update(frame):
        ax.clear()
        # Run clustering with varying parameters
        k = int(np.ceil(frame / 10)) + 2
        labels = TorqueClustering(process_dataset(data), K=k)[0]
        
        # Plot current state
        scatter = ax.scatter(data[:, 0], data[:, 1], c=labels)
        ax.set_title(f'Frame {frame}, K={k}')
        return scatter,
    
    ani = animation.FuncAnimation(
        fig, update, frames=n_frames,
        interval=200, blit=True
    )
    return ani
```

## Testing and Validation

### 1. Clustering Validation

```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score

def validate_clustering(data, labels):
    """Compute various clustering validation metrics."""
    metrics = {
        'silhouette': silhouette_score(data, labels),
        'calinski_harabasz': calinski_harabasz_score(data, labels)
    }
    
    # Custom stability score
    stability = compute_stability(data, labels)
    metrics['stability'] = stability
    
    return metrics

def compute_stability(data, labels, n_samples=10):
    """Compute clustering stability score."""
    scores = []
    n_points = len(data)
    
    for _ in range(n_samples):
        # Subsample data
        idx = np.random.choice(n_points, size=int(0.8*n_points))
        sub_data = data[idx]
        
        # Recluster
        sub_labels = TorqueClustering(process_dataset(sub_data))[0]
        scores.append(compare_clusterings(labels[idx], sub_labels))
    
    return np.mean(scores)
```

### 2. Benchmark Suite

```python
import time
from sklearn.datasets import make_blobs

def run_benchmarks():
    """Run performance benchmarks."""
    results = {}
    
    # Test different dataset sizes
    for n_samples in [100, 1000, 10000]:
        data, _ = make_blobs(n_samples=n_samples, centers=5)
        
        start_time = time.time()
        distance_matrix = process_dataset(data)
        labels = TorqueClustering(distance_matrix)[0]
        elapsed = time.time() - start_time
        
        results[n_samples] = {
            'time': elapsed,
            'memory': get_memory_usage()
        }
    
    return results

def get_memory_usage():
    """Get current memory usage."""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB
```

## Additional Resources

- See [examples/](../examples/) directory for more examples
- Check [notebooks/](../notebooks/) for interactive tutorials
- Visit [documentation](https://torqueclustering.readthedocs.io) for complete API reference 