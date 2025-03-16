# Dataset Configuration System

The TorqueClustering algorithm now includes a sophisticated configuration system that automatically adapts parameters based on dataset characteristics. This system helps optimize clustering performance for different types of datasets.

## Overview

The configuration system provides:
1. Automatic dataset analysis and characterization
2. Preset configurations for common dataset types
3. Dynamic parameter adjustment based on dataset statistics
4. Detailed diagnostic information about configuration decisions

## Dataset Types

The system recognizes several dataset types, each with optimized parameters:

### 1. Noisy Datasets
- **Characteristics**: High coefficient of variation, high kurtosis
- **Configuration**:
  - `adjustment_factor`: 0.3 (conservative)
  - `use_std_adjustment`: True
  - `isnoise`: True
- **Best for**: Datasets with significant noise or outliers

### 2. Subtle Structure Datasets
- **Characteristics**: Low coefficient of variation, low skewness
- **Configuration**:
  - `adjustment_factor`: 0.7 (aggressive)
  - `use_std_adjustment`: True
  - `isnoise`: False
- **Best for**: Datasets with fine cluster boundaries

### 3. Well-Separated Datasets
- **Characteristics**: Moderate statistics, clear structure
- **Configuration**:
  - `adjustment_factor`: 0.5 (standard)
  - `use_std_adjustment`: True
  - `isnoise`: False
- **Best for**: Datasets with clear cluster boundaries

### 4. High-Dimensional Datasets
- **Characteristics**: Dimensionality > 100
- **Configuration**:
  - `adjustment_factor`: 0.4 (conservative)
  - `use_std_adjustment`: True
  - `isnoise`: True
- **Best for**: High-dimensional data affected by curse of dimensionality

### 5. Sparse Datasets
- **Characteristics**: High sparsity (>80% zeros)
- **Configuration**:
  - `adjustment_factor`: 0.6 (aggressive)
  - `use_std_adjustment`: True
  - `isnoise`: True
- **Best for**: Sparse distance matrices

## Usage

### Basic Usage with Auto-Configuration

```python
from TorqueClustering import TorqueClustering
import numpy as np

# Create or load your distance matrix
distance_matrix = ...

# Run clustering with automatic configuration
labels, *rest, diagnostics = TorqueClustering(
    distance_matrix,
    auto_config=True  # Enable automatic configuration
)
```

### Forcing a Specific Configuration

```python
# Force a specific dataset type
labels, *rest, diagnostics = TorqueClustering(
    distance_matrix,
    auto_config=True,
    dataset_type='noisy'  # Force noisy dataset configuration
)
```

### Manual Configuration

```python
# Disable auto-configuration and set parameters manually
labels, *rest, diagnostics = TorqueClustering(
    distance_matrix,
    auto_config=False,
    use_std_adjustment=True,
    adjustment_factor=0.5,
    isnoise=True
)
```

## Dataset Analysis

The system analyzes several key characteristics:

1. **Statistical Properties**:
   - Mean and standard deviation of distances
   - Skewness and kurtosis
   - Coefficient of variation

2. **Structural Properties**:
   - Dimensionality
   - Sparsity
   - Distance range

3. **Quality Metrics**:
   - Distance distribution
   - Matrix characteristics

## Configuration Process

1. **Analysis Phase**:
   ```python
   from dataset_config import analyze_distance_matrix
   
   stats = analyze_distance_matrix(distance_matrix)
   print(f"Coefficient of Variation: {stats['coefficient_variation']:.2f}")
   print(f"Dimensionality: {stats['dimensionality']}")
   ```

2. **Type Detection**:
   ```python
   from dataset_config import get_dataset_type
   
   dataset_type = get_dataset_type(stats)
   print(f"Detected Dataset Type: {dataset_type}")
   ```

3. **Configuration Application**:
   ```python
   from dataset_config import get_recommended_config
   
   config = get_recommended_config(distance_matrix)
   print(f"Recommended adjustment factor: {config['adjustment_factor']}")
   ```

## Diagnostic Information

The configuration system provides detailed diagnostics:

```python
# Print configuration summary
from dataset_config import print_config_summary

print_config_summary(config)
```

Example output:
```
Dataset Configuration Summary:
------------------------------
Detected Dataset Type: high_dimensional
Description: Configuration for high-dimensional datasets

Clustering Parameters:
- Standard Deviation Adjustment: Enabled
- Adjustment Factor: 0.40
- Noise Detection: Enabled

Dataset Statistics:
- Dimensionality: 1000
- Coefficient of Variation: 1.50
- Sparsity: 5.00%
- Skewness: 0.75
- Kurtosis: 3.20
------------------------------
```

## Best Practices

1. **Start with Auto-Configuration**:
   - Let the system analyze your dataset first
   - Review the configuration summary
   - Adjust only if necessary

2. **Dataset-Specific Tuning**:
   - For noisy datasets, verify noise detection results
   - For high-dimensional data, monitor clustering stability
   - For sparse matrices, check connectivity preservation

3. **Configuration Overrides**:
   - Use dataset_type override when you know your data characteristics
   - Adjust individual parameters only when auto-config results need fine-tuning
   - Document any manual parameter adjustments

4. **Monitoring and Validation**:
   - Review diagnostic information
   - Validate clustering results
   - Adjust configuration if needed

## Advanced Usage

### Custom Dataset Types

You can extend the configuration system with custom dataset types:

```python
from dataset_config import PRESET_CONFIGS

# Add custom configuration
PRESET_CONFIGS['custom_type'] = {
    'use_std_adjustment': True,
    'adjustment_factor': 0.45,
    'isnoise': True,
    'description': 'Custom configuration for specific use case'
}
```

### Fine-Tuning Analysis

Adjust analysis parameters for large datasets:

```python
# Increase sample size for more accurate analysis
stats = analyze_distance_matrix(
    distance_matrix,
    sample_size=50000  # Default is 10000
)
```

### Combining Configurations

Mix different configurations based on complex criteria:

```python
config1 = get_recommended_config(distance_matrix, override_type='noisy')
config2 = get_recommended_config(distance_matrix, override_type='high_dimensional')

# Create hybrid configuration
hybrid_config = config1.copy()
hybrid_config['adjustment_factor'] = (
    config1['adjustment_factor'] + config2['adjustment_factor']
) / 2
``` 