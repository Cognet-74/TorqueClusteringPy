# Clustering Evaluation Metrics Documentation

## Overview

TorqueClusteringPy includes a comprehensive set of evaluation metrics to assess clustering quality. These metrics help compare clustering results with ground truth labels and evaluate clustering performance without ground truth.

## Available Metrics

### 1. Normalized Mutual Information (NMI)

```python
def nmi(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Calculate Normalized Mutual Information between two clusterings.
    """
```

#### Properties
- Range: [0, 1]
- 0: No mutual information
- 1: Perfect correlation
- Symmetric: NMI(a,b) = NMI(b,a)
- Normalized for cluster size variations

#### Usage
```python
from nmi import nmi

# Calculate NMI
nmi_score = nmi(true_labels, predicted_labels)
```

### 2. Adjusted Mutual Information (AMI)

```python
def ami(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Calculate Adjusted Mutual Information between two clusterings.
    """
```

#### Properties
- Adjusts for chance
- Range: [-1, 1]
- 0: Random labeling
- 1: Perfect match
- Handles varying numbers of clusters

#### Usage
```python
from ami import ami

# Calculate AMI
ami_score = ami(true_labels, predicted_labels)
```

### 3. Clustering Accuracy (AC)

```python
def accuracy(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Calculate clustering accuracy using optimal label matching.
    """
```

#### Properties
- Direct measure of correct assignments
- Uses Hungarian algorithm for optimal matching
- Range: [0, 1]
- Accounts for label permutations

#### Usage
```python
from accuracy import accuracy

# Calculate accuracy
ac_score = accuracy(true_labels, predicted_labels)
```

## Comprehensive Evaluation

### evaluatecluster Function

```python
def evaluatecluster(
    labels_pred: np.ndarray,
    labels_true: np.ndarray
) -> Tuple[float, float]:
    """
    Compute multiple evaluation metrics at once.
    """
```

#### Returns
- NMI score
- Accuracy score

#### Usage
```python
from evaluatecluster import evaluatecluster

# Get multiple metrics
nmi_score, ac_score = evaluatecluster(predicted_labels, true_labels)
```

## Implementation Details

### 1. NMI Calculation

```python
def nmi(labels_true, labels_pred):
    # Convert labels to contingency matrix
    contingency = contingency_matrix(labels_true, labels_pred)
    
    # Calculate entropies
    h_true = entropy(labels_true)
    h_pred = entropy(labels_pred)
    
    # Calculate mutual information
    mi = mutual_info(contingency)
    
    # Normalize
    nmi = 2.0 * mi / (h_true + h_pred)
    return nmi
```

### 2. AMI Calculation

```python
def ami(labels_true, labels_pred):
    # Calculate observed mutual information
    mi_observed = mutual_info(contingency_matrix(labels_true, labels_pred))
    
    # Calculate expected mutual information
    mi_expected = expected_mutual_info(labels_true, labels_pred)
    
    # Calculate max possible MI
    h_true = entropy(labels_true)
    h_pred = entropy(labels_pred)
    mi_max = max(h_true, h_pred)
    
    # Adjust for chance
    ami = (mi_observed - mi_expected) / (mi_max - mi_expected)
    return ami
```

### 3. Accuracy Calculation

```python
def accuracy(labels_true, labels_pred):
    # Create cost matrix
    cost_matrix = create_cost_matrix(labels_true, labels_pred)
    
    # Use Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Calculate accuracy
    accuracy = np.sum(labels_true == col_ind[labels_pred]) / len(labels_true)
    return accuracy
```

## Best Practices

### 1. Metric Selection

Choose metrics based on your needs:
- NMI: General clustering quality
- AMI: When chance-adjusted measure is needed
- AC: When exact matches are important

### 2. Label Handling

```python
# Ensure labels are 0-based
labels = np.asarray(labels)
if labels.min() > 0:
    labels = labels - labels.min()

# Handle noise points (label -1 or 0)
mask = labels >= 0
labels_clean = labels[mask]
```

### 3. Multiple Evaluations

```python
def evaluate_all_metrics(labels_true, labels_pred):
    results = {
        'nmi': nmi(labels_true, labels_pred),
        'ami': ami(labels_true, labels_pred),
        'accuracy': accuracy(labels_true, labels_pred)
    }
    return results
```

## Common Issues and Solutions

### 1. Label Mismatch

```python
# Problem: Different label ranges
labels1 = [1, 1, 2, 2]
labels2 = [0, 0, 1, 1]

# Solution: Use label-agnostic metrics
nmi_score = nmi(labels1, labels2)  # Works correctly
```

### 2. Memory Issues

```python
# Problem: Large contingency matrices
n_samples = 1000000

# Solution: Sparse matrices
from scipy.sparse import csr_matrix
contingency = csr_matrix((n_samples, n_samples))
```

### 3. Speed Optimization

```python
# Problem: Slow computation for large datasets
# Solution: Vectorized operations
def fast_contingency(labels_true, labels_pred):
    return np.histogram2d(labels_true, labels_pred)[0]
```

## Visualization

### 1. Metric Comparison

```python
def plot_metrics(labels_true, labels_pred):
    metrics = evaluate_all_metrics(labels_true, labels_pred)
    
    plt.figure(figsize=(10, 5))
    plt.bar(metrics.keys(), metrics.values())
    plt.title('Clustering Evaluation Metrics')
    plt.ylim(0, 1)
    plt.show()
```

### 2. Confusion Matrix

```python
def plot_confusion_matrix(labels_true, labels_pred):
    contingency = contingency_matrix(labels_true, labels_pred)
    plt.imshow(contingency, cmap='viridis')
    plt.colorbar()
    plt.title('Clustering Confusion Matrix')
    plt.show()
```

## References

1. NMI: [Strehl & Ghosh (2002)](link)
2. AMI: [Vinh et al. (2010)](link)
3. Hungarian Algorithm: [Kuhn (1955)](link)

## Contributing

To add new metrics:

1. Create metric function:
```python
def new_metric(labels_true, labels_pred):
    # Implementation
    return score
```

2. Add to evaluatecluster:
```python
def evaluatecluster(labels_true, labels_pred):
    # Existing metrics
    new_score = new_metric(labels_true, labels_pred)
    return nmi_score, ac_score, new_score
```

3. Update documentation and tests 