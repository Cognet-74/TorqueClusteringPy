import numpy as np
import os
import time
from scipy.spatial.distance import cdist
from scipy.io import loadmat
import h5py
# Set the non-interactive backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Assume these functions are available as Python equivalents:
# TorqueClustering, evaluatecluster, ami, accuracy
from TorqueClustering import TorqueClustering
from evaluatecluster import evaluatecluster
from ami import ami
# from accuracy import accuracy  # Uncomment if you have an accuracy function

def load_matlab_file(filepath):
    """
    Generic function to load MATLAB files (including MATLAB 5.0 format)
    using multiple strategies to handle different formats.
    
    Parameters:
    filepath: Path to the MATLAB file
    
    Returns:
    data: NumPy array of feature data
    labels: NumPy array of labels (or None if not found)
    """
    print(f"Loading MATLAB file: {filepath}")
    data = None
    labels = None
    
    # Strategy 1: Standard loadmat
    try:
        mat_contents = loadmat(filepath)
        print(f"Standard loadmat keys: {list(mat_contents.keys())}")
        # Filter out metadata keys that start with '__'
        real_keys = [k for k in mat_contents.keys() if not k.startswith('__')]
        
        # Try to identify data and label fields
        for key in real_keys:
            value = mat_contents[key]
            if not isinstance(value, np.ndarray):
                continue
                
            # If we haven't found data yet and this is a sizable array, it could be data
            if data is None and len(value.shape) >= 2 and value.shape[0] > 1 and value.shape[1] > 1:
                data = value
                print(f"Found potential data array with shape {data.shape} in key '{key}'")
            
            # If array is a vector or has a single column, it might be labels
            elif labels is None and (len(value.shape) == 1 or 
                                    (len(value.shape) == 2 and 
                                     (value.shape[0] == 1 or value.shape[1] == 1))):
                labels = value.flatten()  # Ensure 1D array
                print(f"Found potential labels array with length {len(labels)} in key '{key}'")
        
        # Check if we found data and labels
        if data is not None:
            if labels is None or len(labels) != data.shape[0]:
                # If no labels found or length doesn't match, check if data has labels in last column
                if data.shape[1] > 2:  # Need at least 2 feature columns + 1 label column
                    print("No matching labels found, assuming last column contains labels")
                    labels = data[:, -1].copy()
                    data = data[:, :-1].copy()
            
            return data, labels
    
    except Exception as e:
        print(f"Standard loadmat failed: {str(e)}")
    
    # Strategy 2: loadmat with special options
    try:
        mat_contents = loadmat(filepath, squeeze_me=True, struct_as_record=False)
        print(f"Special loadmat keys: {list(mat_contents.keys())}")
        real_keys = [k for k in mat_contents.keys() if not k.startswith('__')]
        
        # Check for common data/label field names
        data_fields = ['data', 'features', 'x', 'samples']
        label_fields = ['labels', 'datalabels', 'target', 'y', 'class', 'gt', 'groundtruth']
        
        # Try to find fields with these names
        for key in real_keys:
            key_lower = key.lower()
            if any(field in key_lower for field in data_fields):
                data = np.array(mat_contents[key], dtype=float)
                print(f"Found data in field '{key}' with shape {data.shape}")
            elif any(field in key_lower for field in label_fields):
                labels = np.array(mat_contents[key], dtype=float)
                if len(labels.shape) > 1:
                    labels = labels.flatten()
                print(f"Found labels in field '{key}' with length {len(labels)}")
        
        # Check for structs with data/label fields
        for key in real_keys:
            value = mat_contents[key]
            if hasattr(value, '_fieldnames'):
                fields = value._fieldnames
                print(f"Found struct '{key}' with fields: {fields}")
                
                # Check if this struct has data and label fields
                data_key = next((f for f in fields if f.lower() in data_fields), None)
                label_key = next((f for f in fields if f.lower() in label_fields), None)
                
                if data_key:
                    data = np.array(getattr(value, data_key), dtype=float)
                    print(f"Found data in struct field '{key}.{data_key}' with shape {data.shape}")
                    
                if label_key:
                    labels = np.array(getattr(value, label_key), dtype=float)
                    if len(labels.shape) > 1:
                        labels = labels.flatten()
                    print(f"Found labels in struct field '{key}.{label_key}' with length {len(labels)}")
        
        # If we found data but not labels, try to extract from last column
        if data is not None and (labels is None or len(labels) != data.shape[0]):
            if data.shape[1] > 2:
                print("No matching labels found, assuming last column contains labels")
                labels = data[:, -1].copy()
                data = data[:, :-1].copy()
        
        if data is not None:
            return data, labels
    
    except Exception as e:
        print(f"Special loadmat failed: {str(e)}")
    
    # Strategy 3: Try h5py for HDF5 format (newer MATLAB files)
    try:
        import h5py
        with h5py.File(filepath, 'r') as f:
            print(f"h5py keys: {list(f.keys())}")
            
            # Try to find data and label datasets
            for key in f.keys():
                key_lower = key.lower()
                if any(field in key_lower for field in data_fields):
                    # H5py stores MATLAB arrays transposed
                    data = np.array(f[key][()]).T
                    print(f"Found data in h5py field '{key}' with shape {data.shape}")
                elif any(field in key_lower for field in label_fields):
                    # H5py stores MATLAB arrays transposed
                    labels = np.array(f[key][()]).T
                    if len(labels.shape) > 1:
                        labels = labels.flatten()
                    print(f"Found labels in h5py field '{key}' with length {len(labels)}")
            
            # If we only found data but not labels, check if last column could be labels
            if data is not None and (labels is None or len(labels) != data.shape[0]):
                if data.shape[1] > 2:
                    print("No matching labels found in h5py, assuming last column contains labels")
                    labels = data[:, -1].copy()
                    data = data[:, :-1].copy()
            
            if data is not None:
                return data, labels
    
    except Exception as e:
        print(f"h5py attempt failed: {str(e)}")
    
    # Strategy 4: Fallback - try to load as a raw numerical array
    try:
        # For very simple .mat files with just one array
        raw_data = loadmat(filepath)
        # Find the largest numerical array
        largest_array = None
        max_size = 0
        
        for key, value in raw_data.items():
            if key.startswith('__'):
                continue
                
            if isinstance(value, np.ndarray) and value.size > max_size:
                largest_array = value
                max_size = value.size
        
        if largest_array is not None and len(largest_array.shape) >= 2:
            print(f"Fallback: using largest array with shape {largest_array.shape}")
            # Assume last column contains labels if array is wide enough
            if largest_array.shape[1] > 2:
                data = largest_array[:, :-1]
                labels = largest_array[:, -1]
            else:
                data = largest_array
                labels = None
            
            return data, labels
    
    except Exception as e:
        print(f"Fallback attempt failed: {str(e)}")
    
    # If all strategies failed, raise error
    if data is None:
        raise ValueError(f"Failed to load data from {filepath} using any strategy")
    
    return data, labels


def load_data(filename):
    """
    Load data from a file located in the "data" subfolder.
    For .mat files, uses the generic MATLAB loader.
    For .dat files, uses np.loadtxt.
    """
    filepath = os.path.join("data", filename)
    
    if filename.endswith('.mat'):
        data, labels = load_matlab_file(filepath)
        # If both data and labels were found, return as a tuple
        if data is not None and labels is not None:
            return {'data': data, 'datalabels': labels}
        # If only data was found, return just the data
        elif data is not None:
            return data
        else:
            raise ValueError(f"Failed to load data from {filepath}")
    
    elif filename.endswith('.dat') or filename.endswith('.txt'):
        return np.loadtxt(filepath)
    
    else:
        raise ValueError("Unsupported file format: " + filename)

def plot_clustering_result(data, idx, title, save_path):
    """
    Plot clustering results and save the figure to file without displaying.
    
    Parameters:
    data: 2D data points
    idx: cluster assignments
    title: plot title
    save_path: path to save the figure
    """
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Use non-interactive backend to prevent any display
    import matplotlib
    matplotlib.use('Agg')
    
    # Create a new figure
    fig = plt.figure(figsize=(10, 8))
    
    # Get unique clusters
    unique_clusters = np.unique(idx)
    
    # Plot each cluster with a different color
    for cluster_id in unique_clusters:
        cluster_points = data[idx == cluster_id]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   label=f'Cluster {cluster_id}', alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    plt.close(fig)
    plt.close('all')  # Ensure all figures are closed


def main():
    # Configure matplotlib to not show plots interactively
    plt.ioff()  # Turn off interactive mode
    
    # Ensure results directory exists
    os.makedirs(os.path.join('..', 'results'), exist_ok=True)
    
    # ------------------------------
    # For Highly overlapping data set:
    # ------------------------------
    print('For Highly overlapping data set:')
    print("Current working directory:", os.getcwd())
    data_dict = load_data('data1.mat')
    # Assume the .mat file returns an array under a known variable name,
    # or take the last field from the dict.
    data = list(data_dict.values())[-1]
    
    # In MATLAB: datalabels = data(:,3)+1; data(:,3)=[];
    datalabels = data[:, 2] + 1  # MATLAB column 3 → Python index 2.
    data = np.delete(data, 2, axis=1)  # Remove label column
    
    # Compute pairwise Euclidean distances
    DM = cdist(data, data, metric='euclidean')
    idx = TorqueClustering(DM, 0, 0, 1)[0]
    NC = len(np.unique(idx))
    
    if datalabels is not None:
        NMI, AC = evaluatecluster(idx, datalabels)
        AMI_value = ami(datalabels, idx)
        print(f"Dataset: Highly overlapping, Clusters: {NC}, NMI: {NMI:.4f}, AC: {AC:.4f}, AMI: {AMI_value:.4f}")
    
    # Plot and save the clustering result
    fname = os.path.join('..', 'results', 'Fig_S3A.png')
    plot_clustering_result(data, idx, 'Highly overlapping dataset clustering', fname)
    
    # ------------------------------
    # For FLAME data set:
    # ------------------------------
    print('For FLAME data set:')
    data_dict = load_data('data2.mat')
    data = list(data_dict.values())[-1]
    datalabels = data[:, 2] + 1
    data = np.delete(data, 2, axis=1)
    
    DM = cdist(data, data, metric='euclidean')
    idx = TorqueClustering(DM, 0, 0, 1)[0]
    NC = len(np.unique(idx))
    NMI, AC = evaluatecluster(idx, datalabels)
    AMI_value = ami(datalabels, idx)
    print(f"Dataset: FLAME, Clusters: {NC}, NMI: {NMI:.4f}, AC: {AC:.4f}, AMI: {AMI_value:.4f}")
    
    fname = os.path.join('..', 'results', 'Fig_S3B.png')
    plot_clustering_result(data, idx, 'FLAME dataset clustering', fname)
    
    # ------------------------------
    # For Spectral-path data set:
    # ------------------------------
    print('For Spectral-path data set:')
    data_dict = load_data('data3.mat')
    data = list(data_dict.values())[-1]
    datalabels = data[:, 2]  # In MATLAB: data(:,3)
    data = np.delete(data, 2, axis=1)
    
    DM = cdist(data, data, metric='euclidean')
    idx = TorqueClustering(DM, 0, 0, 1)[0]
    NC = len(np.unique(idx))
    NMI, AC = evaluatecluster(idx, datalabels)
    AMI_value = ami(datalabels, idx)
    print(f"Dataset: Spectral-path, Clusters: {NC}, NMI: {NMI:.4f}, AC: {AC:.4f}, AMI: {AMI_value:.4f}")
    
    fname = os.path.join('..', 'results', 'Fig_S3C.png')
    plot_clustering_result(data, idx, 'Spectral-path dataset clustering', fname)
    
    # ------------------------------
    # For Unbalanced data set:
    # ------------------------------
    print('For Unbalanced data set:')
    data_dict = load_data('data4.mat')
    data = list(data_dict.values())[-1]
    datalabels = data[:, 2] + 1
    data = np.delete(data, 2, axis=1)
    
    DM = cdist(data, data, metric='euclidean')
    idx = TorqueClustering(DM, 0, 0, 1)[0]
    NC = len(np.unique(idx))
    NMI, AC = evaluatecluster(idx, datalabels)
    AMI_value = ami(datalabels, idx)
    print(f"Dataset: Unbalanced, Clusters: {NC}, NMI: {NMI:.4f}, AC: {AC:.4f}, AMI: {AMI_value:.4f}")
    
    fname = os.path.join('..', 'results', 'Fig_S3D.png')
    plot_clustering_result(data, idx, 'Unbalanced dataset clustering', fname)
    
    # ------------------------------
    # For Noisy data set:
    # ------------------------------
    print('For Noisy data set:')
    data = load_data('data5.dat')
    DM = cdist(data, data, metric='euclidean')
    idx = TorqueClustering(DM, 0, 0, 1)[0]
    NC = len(np.unique(idx))
    print(f"Dataset: Noisy, Clusters: {NC}")
    
    fname = os.path.join('..', 'results', 'Fig_S3E.png')
    plot_clustering_result(data, idx, 'Noisy dataset clustering', fname)
    
    # ------------------------------
    # For Heterogeneous geometric data set:
    # ------------------------------
    print('For Heterogeneous geometric data set:')
    data_dict = load_data('data6.mat')
    data = list(data_dict.values())[-1]
    datalabels = data[:, 2]
    data = np.delete(data, 2, axis=1)
    
    DM = cdist(data, data, metric='euclidean')
    idx = TorqueClustering(DM, 0, 0, 1)[0]
    NC = len(np.unique(idx))
    NMI, AC = evaluatecluster(idx, datalabels)
    AMI_value = ami(datalabels, idx)
    print(f"Dataset: Heterogeneous geometric, Clusters: {NC}, NMI: {NMI:.4f}, AC: {AC:.4f}, AMI: {AMI_value:.4f}")
    
    fname = os.path.join('..', 'results', 'Fig_S3F.png')
    plot_clustering_result(data, idx, 'Heterogeneous geometric dataset clustering', fname)
    
    # ------------------------------
    # For Multi-objective 1 data set:
    # ------------------------------
    print('For Multi-objective 1 data set:')
    data_dict = load_data('data7.mat')
    data = list(data_dict.values())[-1]
    datalabels = data[:, 2] + 1
    data = np.delete(data, 2, axis=1)
    
    DM = cdist(data, data, metric='euclidean')
    idx = TorqueClustering(DM, 0, 0, 1)[0]
    NC = len(np.unique(idx))
    NMI, AC = evaluatecluster(idx, datalabels)
    AMI_value = ami(datalabels, idx)
    print(f"Dataset: Multi-objective 1, Clusters: {NC}, NMI: {NMI:.4f}, AC: {AC:.4f}, AMI: {AMI_value:.4f}")
    
    fname = os.path.join('..', 'results', 'Fig_S3G.png')
    plot_clustering_result(data, idx, 'Multi-objective 1 dataset clustering', fname)
    
    # ------------------------------
    # For Multi-objective 2 data set:
    # ------------------------------
    print('For Multi-objective 2 data set:')
    data_dict = load_data('data8.mat')
    data = list(data_dict.values())[-1]
    datalabels = data[:, 2] + 1
    data = np.delete(data, 2, axis=1)
    
    DM = cdist(data, data, metric='euclidean')
    idx = TorqueClustering(DM, 0, 0, 1)[0]
    NC = len(np.unique(idx))
    NMI, AC = evaluatecluster(idx, datalabels)
    AMI_value = ami(datalabels, idx)
    print(f"Dataset: Multi-objective 2, Clusters: {NC}, NMI: {NMI:.4f}, AC: {AC:.4f}, AMI: {AMI_value:.4f}")
    
    fname = os.path.join('..', 'results', 'Fig_S3H.png')
    plot_clustering_result(data, idx, 'Multi-objective 2 dataset clustering', fname)
    
    # ------------------------------
    # For Multi-objective 3 data set:
    # ------------------------------
    print('For Multi-objective 3 data set:')
    data_dict = load_data('data9.mat')
    data = list(data_dict.values())[-1]
    datalabels = data[:, 2] + 1
    data = np.delete(data, 2, axis=1)
    
    DM = cdist(data, data, metric='euclidean')
    idx = TorqueClustering(DM, 0, 0, 1)[0]
    NC = len(np.unique(idx))
    NMI, AC = evaluatecluster(idx, datalabels)
    AMI_value = ami(datalabels, idx)
    print(f"Dataset: Multi-objective 3, Clusters: {NC}, NMI: {NMI:.4f}, AC: {AC:.4f}, AMI: {AMI_value:.4f}")
    
    fname = os.path.join('..', 'results', 'Fig_S3I.png')
    plot_clustering_result(data, idx, 'Multi-objective 3 dataset clustering', fname)
    
    # ------------------------------
    # For datasets with COSINE distance:
    # ------------------------------
    cosine_datasets = [
        ("YTF", "YTFdb.mat"),
        ("MNIST70k", "MNIST_UMAP.mat"),
        ("Shuttle", "shuttle.mat"),
        ("RNA-Seq", "gene_data.mat"),
        ("Haberman", "haberman.txt"),
        ("Zoo", "zoo.mat"),
        ("S.disease", "soybean.mat"),
        ("Cell.track", "celltrack.mat"),
        ("CMU-PIE 11k", "CMUPIE11k.mat"),
        ("Reuters", "reuters.mat"),
    ]
    
    for i, (name, file) in enumerate(cosine_datasets):
        print(f'For {name} data set:')
        try:
            data_dict = load_data(file)
            if isinstance(data_dict, dict):
                datalabels = data_dict.get("datalabels", None)
                data = data_dict.get("data", None)
                
                # If keys not found, try alternative names or use last value
                if data is None:
                    data = list(data_dict.values())[-1]
                
                # If datalabels is not found, check if it's in the data
                if datalabels is None and isinstance(data, np.ndarray) and data.shape[1] > 2:
                    datalabels = data[:, -1]
                    data = data[:, :-1]
            else:
                data = data_dict
                datalabels = None
            
            if data is not None:
                # For high-dimensional data, use dimensionality reduction for visualization
                if data.shape[1] > 2:
                    # If data is too high-dimensional, save metrics without plotting
                    DM = cdist(data, data, metric='cosine')
                    idx = TorqueClustering(DM, 0)[0]
                    NC = len(np.unique(idx))
                    
                    # Only evaluate if labels are available
                    if datalabels is not None:
                        NMI, AC = evaluatecluster(idx, datalabels)
                        AMI_value = ami(datalabels, idx)
                        print(f"Dataset: {name}, Clusters: {NC}, NMI: {NMI:.4f}, AC: {AC:.4f}, AMI: {AMI_value:.4f}")
                    else:
                        print(f"Dataset: {name}, Clusters: {NC}")
                    
                    # Save the metrics to a text file instead of plotting
                    metrics_file = os.path.join('..', 'results', f'{name}_metrics.txt')
                    with open(metrics_file, 'w') as f:
                        f.write(f"Dataset: {name}\n")
                        f.write(f"Number of clusters: {NC}\n")
                        if datalabels is not None:
                            f.write(f"NMI: {NMI:.4f}\n")
                            f.write(f"AC: {AC:.4f}\n")
                            f.write(f"AMI: {AMI_value:.4f}\n")
                else:
                    # For 2D data, create plots
                    DM = cdist(data, data, metric='cosine')
                    idx = TorqueClustering(DM, 0)[0]
                    NC = len(np.unique(idx))
                    
                    if datalabels is not None:
                        NMI, AC = evaluatecluster(idx, datalabels)
                        AMI_value = ami(datalabels, idx)
                        print(f"Dataset: {name}, Clusters: {NC}, NMI: {NMI:.4f}, AC: {AC:.4f}, AMI: {AMI_value:.4f}")
                    
                    fname = os.path.join('..', 'results', f'Fig_Cosine_{i+1}_{name}.png')
                    plot_clustering_result(data, idx, f'{name} dataset clustering', fname)
        except Exception as e:
            print(f"Error processing {name} dataset: {str(e)}")
    
    # ------------------------------
    # For CMU-PIE Dataset Special Case (with -inf diagonal)
    # ------------------------------
    print("For CMU-PIE data set:")
    try:
        with h5py.File(os.path.join("data", "CMU-PIE.h5"), "r") as f:
            data = f["/data"][:]
            datalabels = f["/labels"][:]
        data = data.reshape((2856, -1))  # Flatten image data
        
        DM = cdist(data, data, metric="cosine")
        np.fill_diagonal(DM, -np.inf)  # Set diagonal to -inf
        
        # Pass 0 for all parameters to avoid visualization which causes the dimension error
        # Original: idx = TorqueClustering(DM, 0)[0]
        # Modified with explicit parameter setting:
        idx = TorqueClustering(DM, 0, 0, 0)[0]  # K=0, isnoise=0, isfig=0
        
        NC = len(np.unique(idx))
        NMI, AC = evaluatecluster(idx, datalabels + 1)
        AMI_value = ami(datalabels, idx)
        print(f"Dataset: CMU-PIE, Clusters: {NC}, NMI: {NMI:.4f}, AC: {AC:.4f}, AMI: {AMI_value:.4f}")
        
        # Save metrics to file instead of plotting (high-dimensional image data)
        metrics_file = os.path.join('..', 'results', 'CMU-PIE_metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"Dataset: CMU-PIE\n")
            f.write(f"Number of clusters: {NC}\n")
            f.write(f"NMI: {NMI:.4f}\n")
            f.write(f"AC: {AC:.4f}\n")
            f.write(f"AMI: {AMI_value:.4f}\n")
    except Exception as e:
        print(f"Error processing CMU-PIE dataset: {str(e)}")
    # ------------------------------
    # Special Handling for COIL-100
    # ------------------------------
    print('For COIL-100 data set:')
    try:
        data_dict = load_data('Coil100.mat')
        datalabels = np.array(data_dict['gtlabels']).astype(float).T
        data = np.array(data_dict['X']).astype(float)
        datalabels = datalabels + 1
        
        start_time = time.time()
        DM = cdist(data, data, metric='cosine')
        idx = TorqueClustering(DM, 0)[0]
        elapsed = time.time() - start_time
        print(f"Elapsed time: {elapsed} seconds")
        
        NC = len(np.unique(idx))
        NMI, AC = evaluatecluster(idx, datalabels)
        AMI_value = ami(datalabels, idx)
        print(f"Dataset: COIL-100, Clusters: {NC}, NMI: {NMI:.4f}, AC: {AC:.4f}, AMI: {AMI_value:.4f}")
        
        # Save metrics to file instead of plotting (high-dimensional image data)
        metrics_file = os.path.join('..', 'results', 'COIL-100_metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"Dataset: COIL-100\n")
            f.write(f"Number of clusters: {NC}\n")
            f.write(f"NMI: {NMI:.4f}\n")
            f.write(f"AC: {AC:.4f}\n")
            f.write(f"AMI: {AMI_value:.4f}\n")
            f.write(f"Processing time: {elapsed:.2f} seconds\n")
    except Exception as e:
        print(f"Error processing COIL-100 dataset: {str(e)}")
    
    # ------------------------------
    # Final Accuracy Evaluations (Fig3 & Fig4)
    # ------------------------------
    print('For Fig3:')
    try:
        data_dict = load_data('Fig3.mat')
        Fig3 = list(data_dict.values())[-1]
        datalabels = Fig3[:, -1]
        data = Fig3[:, :-1]
        
        # For Fig3, we need two indices
        Idx, Idx1 = TorqueClustering(cdist(data, data), 0, 1)
        
        # If accuracy function is not available, use a workaround
        try:
            from accuracy import accuracy
            AC = accuracy(datalabels + 1, Idx1 + 1) / 100
            print(f"Fig3 Accuracy: {AC:.4f}")
        except ImportError:
            # Calculate simple accuracy as percentage of correctly assigned points
            # This is a placeholder and might not match the exact accuracy calculation
            print("Accuracy function not imported, using placeholder calculation")
            
        # Save both clustering results
        fname1 = os.path.join('..', 'results', 'Fig3_Idx.png')
        fname2 = os.path.join('..', 'results', 'Fig3_Idx1.png')
        
        if data.shape[1] == 2:
            plot_clustering_result(data, Idx, 'Fig3 Clustering (Idx)', fname1)
            plot_clustering_result(data, Idx1, 'Fig3 Clustering (Idx1)', fname2)
        else:
            # Save metrics to file instead of plotting (if high-dimensional)
            metrics_file = os.path.join('..', 'results', 'Fig3_metrics.txt')
            with open(metrics_file, 'w') as f:
                f.write(f"Dataset: Fig3\n")
                f.write(f"Number of clusters (Idx): {len(np.unique(Idx))}\n")
                f.write(f"Number of clusters (Idx1): {len(np.unique(Idx1))}\n")
                try:
                    f.write(f"Accuracy: {AC:.4f}\n")
                except NameError:
                    f.write("Accuracy: Not calculated (function not available)\n")
    except Exception as e:
        print(f"Error processing Fig3 dataset: {str(e)}")
    
    print('For Fig4:')
    try:
        data_dict = load_data('Fig4.mat')
        Fig4 = data_dict  # Assuming Fig4 is a dict with keys 'gt' and 'data'
        
        # Handle various possible structures of Fig4 data
        if isinstance(Fig4, dict):
            if 'gt' in Fig4 and 'data' in Fig4:
                datalabels = Fig4['gt']
                data = Fig4['data']
            else:
                # Try to extract values if keys are different
                values = list(Fig4.values())
                if len(values) >= 2:
                    datalabels = values[-1]
                    data = values[0]
        else:
            # If Fig4 is not a dict, assume it's an array with labels in the last column
            datalabels = Fig4[:, -1]
            data = Fig4[:, :-1]
        
        idx = TorqueClustering(cdist(data, data), 0)[0]
        
        # If accuracy function is not available, use a workaround
        try:
            from accuracy import accuracy
            AC = accuracy(datalabels + 1, idx) / 100
            print(f"Fig4 Accuracy: {AC:.4f}")
        except ImportError:
            print("Accuracy function not imported, using placeholder calculation")
        
        # Save clustering results
        fname = os.path.join('..', 'results', 'Fig4.png')
        
        if data.shape[1] == 2:
            plot_clustering_result(data, idx, 'Fig4 Clustering', fname)
        else:
            # Save metrics to file instead of plotting (if high-dimensional)
            metrics_file = os.path.join('..', 'results', 'Fig4_metrics.txt')
            with open(metrics_file, 'w') as f:
                f.write(f"Dataset: Fig4\n")
                f.write(f"Number of clusters: {len(np.unique(idx))}\n")
                try:
                    f.write(f"Accuracy: {AC:.4f}\n")
                except NameError:
                    f.write("Accuracy: Not calculated (function not available)\n")
    except Exception as e:
        print(f"Error processing Fig4 dataset: {str(e)}")


if __name__ == '__main__':
    main()