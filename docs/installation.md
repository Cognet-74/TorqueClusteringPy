# Installation and Setup Guide

## System Requirements

- Python 3.8 or higher
- pip package manager
- Git (for version control)
- 4GB RAM minimum (8GB recommended for large datasets)
- Operating Systems:
  - Linux (Ubuntu 18.04+, CentOS 7+)
  - macOS (10.14+)
  - Windows 10/11

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Cognet-74/TorqueClusteringPy.git
cd TorqueClusteringPy
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv

# Activate virtual environment
# On Windows
venv\Scripts\activate
# On Linux/macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Detailed Installation Steps

### 1. Python Installation

#### Windows
1. Download Python from [python.org](https://python.org)
2. Run installer with "Add Python to PATH" checked
3. Verify installation:
   ```bash
   python --version
   pip --version
   ```

#### Linux
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip

# CentOS/RHEL
sudo yum install python3 python3-pip
```

#### macOS
```bash
# Using Homebrew
brew install python
```

### 2. Dependencies

The project requires the following main dependencies:

```python
# requirements.txt
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
pandas>=1.3.0
```

Additional dependencies for development:

```python
# dev-requirements.txt
pytest>=6.2.0
black>=21.5b2
flake8>=3.9.0
mypy>=0.910
```

### 3. GPU Support (Optional)

For GPU acceleration:

1. Install CUDA Toolkit (if using NVIDIA GPU)
2. Install cupy:
   ```bash
   pip install cupy-cuda11x  # Replace x with your CUDA version
   ```

## Configuration

### 1. Environment Variables

Create a `.env` file in the project root:

```bash
# .env
PYTHONPATH=.
MAX_THREADS=4
USE_GPU=false
```

### 2. Project Structure

```
TorqueClusteringPy/
├── src/
│   ├── __init__.py
│   ├── TorqueClustering.py
│   └── utils/
├── tests/
├── docs/
├── examples/
├── requirements.txt
└── README.md
```

## Development Setup

### 1. Install Development Dependencies

```bash
pip install -r dev-requirements.txt
```

### 2. Configure Git Hooks

```bash
# Install pre-commit
pip install pre-commit
pre-commit install
```

### 3. IDE Configuration

#### VSCode Settings

```json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.analysis.typeCheckingMode": "basic"
}
```

#### PyCharm Settings

1. Enable Python Integrated Tools
2. Set Black as formatter
3. Enable type checking

## Common Installation Issues

### 1. Package Conflicts

```bash
# Solution: Use virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Import Errors

```bash
# Solution: Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/TorqueClusteringPy"
```

### 3. GPU Issues

```bash
# Check CUDA installation
nvidia-smi

# Verify cupy installation
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
```

## Testing Installation

### 1. Run Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_clustering.py

# Run with coverage
pytest --cov=src tests/
```

### 2. Verify Installation

```python
# test_installation.py
from TorqueClustering import cluster_data
import numpy as np

# Generate sample data
X = np.random.randn(100, 2)

# Run clustering
labels = cluster_data(X)
print("Clustering successful!")
```

## Upgrading

### 1. Update Repository

```bash
git pull origin main
```

### 2. Update Dependencies

```bash
pip install --upgrade -r requirements.txt
```

## Contributing

### 1. Fork Repository

1. Fork on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/TorqueClusteringPy.git
   ```

### 2. Setup Development Environment

```bash
# Create branch
git checkout -b feature-name

# Install dev dependencies
pip install -r dev-requirements.txt
```

### 3. Submit Changes

1. Commit changes
2. Push to your fork
3. Create Pull Request

## Support

### Getting Help

1. Check documentation
2. Search issues on GitHub
3. Create new issue if needed

### Reporting Issues

Include:
1. Python version
2. OS details
3. Error message
4. Minimal reproducible example

## References

1. [Python Installation Guide](https://python.org/downloads)
2. [pip Documentation](https://pip.pypa.io)
3. [Virtual Environments](https://docs.python.org/3/tutorial/venv.html)
4. [Git Documentation](https://git-scm.com/doc) 