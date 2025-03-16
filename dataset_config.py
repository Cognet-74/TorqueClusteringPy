"""
Dataset Configuration System for TorqueClustering

This module provides dataset-specific configurations and automatic parameter tuning
for different types of datasets based on their characteristics.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from scipy.sparse import issparse, spmatrix
from scipy.stats import skew, kurtosis

# Preset configurations for different dataset types
PRESET_CONFIGS = {
    'noisy': {
        'use_std_adjustment': True,
        'adjustment_factor': 0.3,  # More conservative for noisy data
        'isnoise': True,          # Enable noise detection
        'description': 'Configuration for datasets with significant noise'
    },
    'subtle': {
        'use_std_adjustment': True,
        'adjustment_factor': 0.7,  # More aggressive to detect subtle structure
        'isnoise': False,         # Disable noise detection
        'description': 'Configuration for datasets with subtle cluster boundaries'
    },
    'well_separated': {
        'use_std_adjustment': True,
        'adjustment_factor': 0.5,  # Standard adjustment
        'isnoise': False,         # No need for noise detection
        'description': 'Configuration for datasets with clear cluster boundaries'
    },
    'high_dimensional': {
        'use_std_adjustment': True,
        'adjustment_factor': 0.4,  # More conservative due to curse of dimensionality
        'isnoise': True,          # Enable noise detection
        'description': 'Configuration for high-dimensional datasets'
    },
    'sparse': {
        'use_std_adjustment': True,
        'adjustment_factor': 0.6,  # More aggressive for sparse connections
        'isnoise': True,          # Enable noise detection
        'description': 'Configuration for sparse distance matrices'
    }
}

def analyze_distance_matrix(
    distance_matrix: np.ndarray,
    sample_size: int = 10000
) -> Dict[str, float]:
    """
    Analyze distance matrix characteristics to determine dataset properties.
    
    Args:
        distance_matrix: Input distance matrix
        sample_size: Maximum number of distances to sample for large matrices
    
    Returns:
        Dictionary containing dataset characteristics
    """
    # Handle sparse matrices
    is_sparse = issparse(distance_matrix)
    if is_sparse:
        # Sample random elements from sparse matrix
        total_elements = distance_matrix.shape[0] * distance_matrix.shape[1]
        if total_elements > sample_size:
            row_indices = np.random.randint(0, distance_matrix.shape[0], sample_size)
            col_indices = np.random.randint(0, distance_matrix.shape[1], sample_size)
            distances = np.array([distance_matrix[i, j] for i, j in zip(row_indices, col_indices)])
        else:
            distances = distance_matrix.data
    else:
        # For dense matrices, flatten and sample if necessary
        distances = distance_matrix.flatten()
        if len(distances) > sample_size:
            distances = np.random.choice(distances, sample_size, replace=False)
    
    # Calculate statistical properties
    stats = {
        'mean_distance': float(np.mean(distances)),
        'std_distance': float(np.std(distances)),
        'skewness': float(skew(distances)),
        'kurtosis': float(kurtosis(distances)),
        'sparsity': float(np.sum(distances == 0) / len(distances)),
        'dimensionality': distance_matrix.shape[0],
        'distance_range': float(np.max(distances) - np.min(distances)),
        'coefficient_variation': float(np.std(distances) / np.mean(distances) if np.mean(distances) != 0 else np.inf)
    }
    
    return stats

def get_dataset_type(stats: Dict[str, float]) -> str:
    """
    Determine dataset type based on statistical properties.
    
    Args:
        stats: Dictionary of dataset statistics
    
    Returns:
        String indicating the dataset type
    """
    # Define thresholds for classification
    if stats['sparsity'] > 0.8:
        return 'sparse'
    elif stats['dimensionality'] > 100:
        return 'high_dimensional'
    elif stats['coefficient_variation'] > 2.0 or stats['kurtosis'] > 5.0:
        return 'noisy'
    elif stats['coefficient_variation'] < 0.5 and abs(stats['skewness']) < 0.5:
        return 'subtle'
    else:
        return 'well_separated'

def get_recommended_config(
    distance_matrix: np.ndarray,
    override_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get recommended configuration based on dataset characteristics.
    
    Args:
        distance_matrix: Input distance matrix
        override_type: Optional string to force a specific preset configuration
    
    Returns:
        Dictionary containing recommended configuration parameters
    """
    if override_type is not None:
        if override_type not in PRESET_CONFIGS:
            raise ValueError(f"Unknown dataset type: {override_type}")
        return PRESET_CONFIGS[override_type].copy()
    
    # Analyze dataset characteristics
    stats = analyze_distance_matrix(distance_matrix)
    dataset_type = get_dataset_type(stats)
    
    # Get base configuration
    config = PRESET_CONFIGS[dataset_type].copy()
    
    # Fine-tune parameters based on specific statistics
    if stats['coefficient_variation'] > 3.0:
        config['adjustment_factor'] *= 0.8  # More conservative for highly variable data
    if stats['dimensionality'] > 1000:
        config['adjustment_factor'] *= 0.9  # Even more conservative for very high dimensions
    
    # Add analysis results to config
    config['dataset_analysis'] = stats
    config['detected_type'] = dataset_type
    
    return config

def apply_config(
    distance_matrix: np.ndarray,
    config: Dict[str, Any]
) -> Tuple[bool, float, bool]:
    """
    Extract clustering parameters from configuration.
    
    Args:
        distance_matrix: Input distance matrix (for validation)
        config: Configuration dictionary
    
    Returns:
        Tuple of (use_std_adjustment, adjustment_factor, isnoise)
    """
    # Validate configuration
    required_keys = {'use_std_adjustment', 'adjustment_factor', 'isnoise'}
    if not all(key in config for key in required_keys):
        raise ValueError("Configuration missing required parameters")
    
    # Extract parameters
    use_std_adjustment = bool(config['use_std_adjustment'])
    adjustment_factor = float(config['adjustment_factor'])
    isnoise = bool(config['isnoise'])
    
    # Validate parameters
    if not 0.0 <= adjustment_factor <= 1.0:
        raise ValueError("adjustment_factor must be between 0 and 1")
    
    return use_std_adjustment, adjustment_factor, isnoise

def print_config_summary(config: Dict[str, Any]) -> None:
    """
    Print a human-readable summary of the configuration.
    
    Args:
        config: Configuration dictionary
    """
    print("\nDataset Configuration Summary:")
    print("-" * 30)
    
    if 'detected_type' in config:
        print(f"Detected Dataset Type: {config['detected_type']}")
        print(f"Description: {PRESET_CONFIGS[config['detected_type']]['description']}")
    
    print("\nClustering Parameters:")
    print(f"- Standard Deviation Adjustment: {'Enabled' if config['use_std_adjustment'] else 'Disabled'}")
    print(f"- Adjustment Factor: {config['adjustment_factor']:.2f}")
    print(f"- Noise Detection: {'Enabled' if config['isnoise'] else 'Disabled'}")
    
    if 'dataset_analysis' in config:
        stats = config['dataset_analysis']
        print("\nDataset Statistics:")
        print(f"- Dimensionality: {stats['dimensionality']}")
        print(f"- Coefficient of Variation: {stats['coefficient_variation']:.2f}")
        print(f"- Sparsity: {stats['sparsity']:.2%}")
        print(f"- Skewness: {stats['skewness']:.2f}")
        print(f"- Kurtosis: {stats['kurtosis']:.2f}")
    
    print("-" * 30) 