# Data Processing Utilities Documentation

## Overview

The data processing utilities in TorqueClusteringPy provide standardized methods for loading, preprocessing, and handling various data formats. These utilities ensure consistent data handling across different input types and formats.

## Core Functions

### process_dataset

```python
def process_dataset(
    data_dict: Union[Dict, np.ndarray],
    label_column: int = 2,
    add_to_labels: int = 1
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Standardized dataset processing function.
    """
```

#### Parameters

- `data_dict`: Input data
  - Can be dictionary with 'data' and 'datalabels' keys
  - Can be numpy array with optional label column
  - Supports both dense and sparse formats

- `label_column`: Index of label column
  - Default: 2 (third column)
  - Set to -1 for last column
  - Ignored if labels are in separate field

- `add_to_labels`: Value to add to labels
  - Default: 1 (for 0/1-based indexing conversion)
  - Used to align with different indexing conventions

#### Returns

- Tuple containing:
  1. Processed data array (features only)
  2. Labels array (if available, None otherwise)

### load_data

```python
def load_data(filename: str) -> Union[Dict[str, np.ndarray], np.ndarray]:
    """
    Load data from various file formats.
    """
```

#### Supported Formats

1. **MATLAB Files (.mat)**
   - Both v5 and v7.3 formats
   - Handles structured arrays
   - Supports multiple variable storage

2. **Text Files (.txt, .dat)**
   - Space/tab-delimited formats
   - CSV files
   - Custom delimiters

3. **HDF5 Files (.h5)**
   - High-dimensional datasets
   - Compressed storage
   - Complex data structures

## Usage Examples

### Loading MATLAB Data

```python
# Load data from MATLAB file
data_dict = load_data('data1.mat')
data, labels = process_dataset(data_dict)

# Specify different label column
data, labels = process_dataset(data_dict, label_column=-1)
```

### Loading Text Data

```python
# Load from text file
data = load_data('data.txt')
data, labels = process_dataset(data, label_column=2)
```

### Custom Data Processing

```python
# Process numpy array with labels in last column
data = np.random.rand(100, 4)  # 3 features + 1 label column
processed_data, labels = process_dataset(data, label_column=-1)
```

## Data Format Requirements

### MATLAB Files

```matlab
% Example MATLAB structure
data.X = rand(100, 3);        % Features
data.labels = randi(3, 100);  % Labels
save('data.mat', 'data');
```

### Text Files

```text
# Format: feature1 feature2 feature3 label
1.2 3.4 5.6 1
7.8 9.0 1.2 2
...
```

### HDF5 Files

```python
# HDF5 structure
/data           # Features dataset
/labels         # Labels dataset
/metadata       # Optional metadata
```

## Error Handling

The utilities include robust error handling:

1. **File Loading**
   ```python
   try:
       data = load_data(filename)
   except FileNotFoundError:
       print(f"File {filename} not found")
   except ValueError as e:
       print(f"Invalid file format: {e}")
   ```

2. **Data Validation**
   ```python
   # Check data dimensions
   if data.ndim == 1:
       data = data.reshape(-1, 1)
   
   # Validate label column
   if label_column >= data.shape[1]:
       raise ValueError("Label column index out of bounds")
   ```

3. **Type Conversion**
   ```python
   # Ensure numeric data
   data = np.asarray(data, dtype=np.float64)
   
   # Handle label conversion
   if labels is not None:
       labels = np.asarray(labels, dtype=np.int_)
   ```

## Best Practices

1. **Data Preparation**
   - Normalize/scale features if needed
   - Handle missing values before processing
   - Ensure consistent data types

2. **Label Handling**
   - Use consistent indexing convention
   - Verify label ranges
   - Handle multi-class labels appropriately

3. **Memory Management**
   - Use sparse formats for large datasets
   - Process data in batches if needed
   - Clean up temporary files

## Common Issues and Solutions

1. **Data Loading Issues**
   ```python
   # Problem: Mixed data types
   data = np.genfromtxt('data.txt', dtype=None)
   
   # Solution: Specify dtype
   data = np.genfromtxt('data.txt', dtype=float)
   ```

2. **Label Processing**
   ```python
   # Problem: Inconsistent label format
   labels = data[:, -1]  # May include strings/mixed types
   
   # Solution: Convert and validate
   labels = np.asarray(labels, dtype=int)
   assert labels.min() >= 0, "Negative labels found"
   ```

3. **Memory Errors**
   ```python
   # Problem: Large dense matrices
   DM = np.zeros((n, n))  # May cause memory error
   
   # Solution: Use sparse matrix
   DM = scipy.sparse.lil_matrix((n, n))
   ```

## Performance Optimization

1. **File Loading**
   - Use memory mapping for large files
   - Implement lazy loading for big datasets
   - Cache frequently used data

2. **Data Processing**
   - Vectorize operations when possible
   - Use efficient array operations
   - Minimize data copying

3. **Memory Usage**
   - Use appropriate data types
   - Release memory when possible
   - Monitor memory consumption

## Contributing

To add support for new data formats:

1. Update `load_data`:
   ```python
   def load_data(filename):
       if filename.endswith('.new_format'):
           return load_new_format(filename)
       # ... existing formats ...
   ```

2. Implement format-specific loader:
   ```python
   def load_new_format(filename):
       # Implementation
       return data
   ```

3. Update documentation and tests 