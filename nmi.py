import numpy as np
from scipy.sparse import coo_matrix, issparse

def nmi(x, y):
    """
    Compute normalized mutual information I(x,y)/sqrt(H(x)*H(y)) for two discrete variables x and y.
    
    Inputs:
        x, y : array-like or sparse matrix
            Two integer vectors (or arrays) of the same length, or sparse matrices with the same dimensions.
    
    Output:
        z : float
            Normalized mutual information defined as z = I(x,y)/sqrt(H(x)*H(y)).
    
    Originally written by Mo Chen (sth4nth@gmail.com) [translated from MATLAB].
    Updated to handle sparse matrices.
    
    Note:
        If x and y are not sparse matrices, they are assumed to be one-dimensional arrays.
        If x and y are sparse matrices, they should have the same shape.
    """
    # Check if inputs are sparse matrices
    x_sparse = issparse(x)
    y_sparse = issparse(y)
    
    if x_sparse and y_sparse:
        # Both inputs are sparse matrices
        assert x.shape == y.shape, "Input sparse matrices x and y must have the same shape."
        n = x.shape[0]
        
        # For sparse matrices, we'll use them directly
        Mx = x.tocsr()
        My = y.tocsr()
    else:
        # Convert inputs to numpy arrays and flatten them to 1D
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        
        # Ensure that x and y have the same length
        assert x.size == y.size, "Input vectors x and y must have the same number of elements."
        n = x.size
        
        # Offset the values so that the smallest value becomes 0
        l = min(np.min(x), np.min(y))
        x = x - l
        y = y - l
        
        # Determine the number of discrete states
        k = int(max(np.max(x), np.max(y)) + 1)
        
        # Create an array of row indices
        idx = np.arange(n)
        
        # Build the sparse indicator matrices for x and y
        data = np.ones(n)
        Mx = coo_matrix((data, (idx, x)), shape=(n, k)).tocsr()
        My = coo_matrix((data, (idx, y)), shape=(n, k)).tocsr()
    
    # Compute the joint distribution matrix Pxy
    Pxy_matrix = (Mx.T @ My) / n
    # Extract the nonzero entries (the joint probabilities)
    Pxy = Pxy_matrix.data
    
    # Filter out zero probabilities (can happen due to numerical issues)
    Pxy = Pxy[Pxy > 0]
    
    # Joint entropy Hxy = -sum(Pxy * log2(Pxy))
    Hxy = -np.dot(Pxy, np.log2(Pxy))
    
    # Compute marginal probabilities
    Px = np.array(Mx.sum(axis=0)).flatten() / n
    Py = np.array(My.sum(axis=0)).flatten() / n
    
    # Remove zero entries to avoid issues with log2(0)
    Px = Px[Px > 0]
    Py = Py[Py > 0]
    
    # Compute entropies for x and y
    Hx = -np.dot(Px, np.log2(Px))
    Hy = -np.dot(Py, np.log2(Py))
    
    # Mutual information: MI = Hx + Hy - Hxy
    MI = Hx + Hy - Hxy
    
    # Normalized mutual information: sqrt((MI/Hx)*(MI/Hy))
    z = np.sqrt((MI / Hx) * (MI / Hy))
    # Ensure non-negativity
    z = max(0, z)
    
    return z