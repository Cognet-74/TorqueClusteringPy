# Threshold Adjustment in TorqueClustering

## Overview

The TorqueClustering algorithm uses thresholds to determine cluster boundaries and identify abnormal merges. These thresholds can now be configured using two parameters:

1. `use_std_adjustment`: Controls whether to use standard deviation-based adjustment
2. `adjustment_factor`: Controls the magnitude of the adjustment

## Parameters

### use_std_adjustment (bool)

- **Default**: `True`
- **Purpose**: Controls whether thresholds are adjusted using standard deviation
- **Behavior**:
  - When `True`: Thresholds are calculated as `mean - (adjustment_factor * std)`
  - When `False`: Thresholds are set directly to the mean values

### adjustment_factor (float)

- **Default**: `0.5`
- **Purpose**: Controls how much the standard deviation affects the threshold
- **Range**: Typically between 0.0 and 1.0
  - Lower values (e.g., 0.3) = More conservative clustering
  - Higher values (e.g., 0.7) = More aggressive clustering

## Usage Examples

```python
from TorqueClustering import TorqueClustering
import numpy as np
from scipy.spatial.distance import cdist

# Example 1: Default behavior (with std adjustment)
labels = TorqueClustering(distance_matrix)

# Example 2: Disable std adjustment
labels = TorqueClustering(
    distance_matrix,
    use_std_adjustment=False
)

# Example 3: Custom adjustment factor
labels = TorqueClustering(
    distance_matrix,
    use_std_adjustment=True,
    adjustment_factor=0.3  # More conservative
)
```

## Impact on Clustering

### Standard Deviation Adjustment

Using standard deviation adjustment (`use_std_adjustment=True`) helps in:
- Adapting to the natural variability in your data
- Handling datasets with different scales
- Providing more robust cluster boundaries

### Adjustment Factor

The `adjustment_factor` parameter allows fine-tuning:
- **Lower values** (e.g., 0.3):
  - More conservative clustering
  - Fewer clusters detected
  - Better for noisy data
- **Higher values** (e.g., 0.7):
  - More aggressive clustering
  - More clusters detected
  - Better for well-separated data

## Best Practices

1. **Start with Defaults**:
   - Begin with default values (`use_std_adjustment=True`, `adjustment_factor=0.5`)
   - These work well for most datasets

2. **Adjusting Parameters**:
   - If getting too many clusters: decrease `adjustment_factor`
   - If getting too few clusters: increase `adjustment_factor`
   - If results are unstable: try `use_std_adjustment=False`

3. **Dataset-Specific Tuning**:
   - Dense, well-separated clusters: Higher `adjustment_factor` (0.6-0.8)
   - Noisy or overlapping clusters: Lower `adjustment_factor` (0.3-0.4)
   - Very uniform data: Consider `use_std_adjustment=False`

## Technical Details

### Threshold Calculation

When `use_std_adjustment=True`:
```python
threshold = mean - (adjustment_factor * std)
```

When `use_std_adjustment=False`:
```python
threshold = mean
```

### Affected Metrics

The adjustment affects three key metrics:
1. Mass threshold
2. Distance (R) threshold
3. Torque (p) threshold

Each metric uses the same adjustment parameters but is calculated independently. 