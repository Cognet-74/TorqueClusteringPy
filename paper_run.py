import numpy as np
import os
import time
from scipy.spatial.distance import cdist
from scipy.io import loadmat
import h5py
from typing import Dict, Union, Tuple, Optional
import numpy.typing as npt

# Set the non-interactive backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Updated import: use the new full Torque Clustering implementation.
from TorqueClustering_Full import torque_clustering_paper
from evaluatecluster import evaluatecluster
from ami import ami
from accuracy import accuracy  # Uncomment if you have an accuracy function

def load_matlab_file(filepath: str) -> Tuple[Optional[npt.NDArray], Optional[npt.NDArray]]:
    """
    Generic function to load MATLAB files (including MATLAB 5.0 format)
    using multiple strategies to handle different formats.
    
    Parameters:
        filepath (str): Path to the MATLAB file
    
    Returns:
        Tuple[Optional[npt.NDArray], Optional[npt.NDArray]]: 
            - data: NumPy array of feature data
            - labels: NumPy array of labels (or None if not found)
    """
    print(f"Loading MATLAB file: {filepath}")
    
    try:
        # First try loading with scipy.io.loadmat (works for MATLAB 5.0 format)
        mat_contents = loadmat(filepath)
        
        # Try to find the data array - look for common variable names
        data_vars = ['data', 'X', 'features']
        data = None
        for var in data_vars:
            if var in mat_contents:
                data = mat_contents[var]
                break
        
        # If no known data variable found, take the last non-special variable
        if data is None:
            # Filter out special MATLAB variables that start with '__'
            regular_vars = [k for k in mat_contents.keys() if not k.startswith('__')]
            if regular_vars:
                data = mat_contents[regular_vars[-1]]
        
        # Try to find labels - look for common variable names
        label_vars = ['labels', 'y', 'datalabels', 'truth', 'groundtruth']
        labels = None
        for var in label_vars:
            if var in mat_contents:
                labels = mat_contents[var]
                break
        
        # Ensure data is 2D
        if data is not None and data.ndim == 1:
            data = data.reshape(-1, 1)
        
        return data, labels
        
    except NotImplementedError:
        # If scipy.io.loadmat fails, try h5py (for newer MATLAB formats)
        try:
            with h5py.File(filepath, 'r') as f:
                # Try to find the data array
                data_vars = ['data', 'X', 'features']
                data = None
                for var in data_vars:
                    if var in f:
                        data = np.array(f[var])
                        break
                
                # If no known data variable found, take the first dataset
                if data is None and len(f.keys()) > 0:
                    first_key = list(f.keys())[0]
                    data = np.array(f[first_key])
                
                # Try to find labels
                label_vars = ['labels', 'y', 'datalabels', 'truth', 'groundtruth']
                labels = None
                for var in label_vars:
                    if var in f:
                        labels = np.array(f[var])
                        break
                
                # Ensure data is 2D
                if data is not None and data.ndim == 1:
                    data = data.reshape(-1, 1)
                
                return data, labels
                
        except Exception as e:
            print(f"Error loading with h5py: {e}")
            return None, None
            
    except Exception as e:
        print(f"Error loading MATLAB file: {e}")
        return None, None


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
    plt.close('all')

def process_dataset(
    data_dict: Union[Dict, np.ndarray],
    label_column: int = 2,
    add_to_labels: int = 1
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Standardized function to process dataset loading results.
    
    Args:
        data_dict: Either a dictionary containing data and labels or a numpy array
        label_column: Which column contains the labels (if labels are in the data)
        add_to_labels: Value to add to labels (for 0/1-based indexing conversion)
    
    Returns:
        Tuple of (data array, labels array or None)
    """
    # Handle both dictionary and direct array returns
    if isinstance(data_dict, dict):
        if 'data' in data_dict and 'datalabels' in data_dict:
            data = data_dict['data']
            datalabels = data_dict['datalabels']
        else:
            valid_keys = [k for k in data_dict.keys() if not k.startswith('__')]
            if valid_keys:
                data = data_dict[valid_keys[-1]]
            else:
                raise ValueError("No valid data found in dictionary")
            datalabels = None
    else:
        data = data_dict
        datalabels = None
    
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    if datalabels is None and data.shape[1] > label_column:
        datalabels = data[:, label_column] + add_to_labels
        data = np.delete(data, label_column, axis=1)
    elif datalabels is not None:
        datalabels = np.asarray(datalabels)
        if datalabels.ndim > 1:
            datalabels = datalabels.ravel()
    
    return data, datalabels

def main():
    plt.ioff()  # Turn off interactive mode
    os.makedirs(os.path.join('..', 'results'), exist_ok=True)
    
    # ------------------------------
    # For Highly overlapping data set:
    # ------------------------------
    print('For Highly overlapping data set:')
    data, datalabels = process_dataset(load_data('data1.mat'))
    # Run full Torque Clustering on raw data
    labels, final_k, merges_sorted = torque_clustering_paper(data, is_noise=True)
    NC = final_k
    
    if datalabels is not None:
        NMI, AC = evaluatecluster(labels, datalabels)
        AMI_value = ami(datalabels, labels)
        print(f"Dataset: Highly overlapping, Clusters: {NC}, NMI: {NMI:.4f}, AC: {AC:.4f}, AMI: {AMI_value:.4f}")
    
    fname = os.path.join('..', 'results', 'Fig_S3A.png')
    plot_clustering_result(data, labels, 'Highly overlapping dataset clustering', fname)
    
    # ------------------------------
    # For FLAME data set:
    # ------------------------------
    print('For FLAME data set:')
    data, datalabels = process_dataset(load_data('data2.mat'))
    labels, final_k, _ = torque_clustering_paper(data, is_noise=True)
    NC = final_k
    
    if datalabels is not None:
        NMI, AC = evaluatecluster(labels, datalabels)
        AMI_value = ami(datalabels, labels)
        print(f"Dataset: FLAME, Clusters: {NC}, NMI: {NMI:.4f}, AC: {AC:.4f}, AMI: {AMI_value:.4f}")
    
    fname = os.path.join('..', 'results', 'Fig_S3B.png')
    plot_clustering_result(data, labels, 'FLAME dataset clustering', fname)
    
    # ------------------------------
    # For Spectral-path data set:
    # ------------------------------
    print('For Spectral-path data set:')
    data, datalabels = process_dataset(load_data('data3.mat'), label_column=2, add_to_labels=0)
    labels, final_k, _ = torque_clustering_paper(data, is_noise=True)
    NC = final_k
    NMI, AC = evaluatecluster(labels, datalabels)
    AMI_value = ami(datalabels, labels)
    print(f"Dataset: Spectral-path, Clusters: {NC}, NMI: {NMI:.4f}, AC: {AC:.4f}, AMI: {AMI_value:.4f}")
    
    fname = os.path.join('..', 'results', 'Fig_S3C.png')
    plot_clustering_result(data, labels, 'Spectral-path dataset clustering', fname)
    
    # ------------------------------
    # For Unbalanced data set:
    # ------------------------------
    print('For Unbalanced data set:')
    data, datalabels = process_dataset(load_data('data4.mat'), label_column=2, add_to_labels=1)
    labels, final_k, _ = torque_clustering_paper(data, is_noise=True)
    NC = final_k
    NMI, AC = evaluatecluster(labels, datalabels)
    AMI_value = ami(datalabels, labels)
    print(f"Dataset: Unbalanced, Clusters: {NC}, NMI: {NMI:.4f}, AC: {AC:.4f}, AMI: {AMI_value:.4f}")
    
    fname = os.path.join('..', 'results', 'Fig_S3D.png')
    plot_clustering_result(data, labels, 'Unbalanced dataset clustering', fname)
    
    # ------------------------------
    # For Noisy data set:
    # ------------------------------
    print('For Noisy data set:')
    data, datalabels = process_dataset(load_data('data5.dat'))
    labels, final_k, _ = torque_clustering_paper(data, is_noise=True)
    NC = final_k
    print(f"Dataset: Noisy, Clusters: {NC}")
    
    fname = os.path.join('..', 'results', 'Fig_S3E.png')
    plot_clustering_result(data, labels, 'Noisy dataset clustering', fname)
    
    # ------------------------------
    # For Heterogeneous geometric data set:
    # ------------------------------
    print('For Heterogeneous geometric data set:')
    data, datalabels = process_dataset(load_data('data6.mat'), label_column=2, add_to_labels=0)
    labels, final_k, _ = torque_clustering_paper(data, is_noise=True)
    NC = final_k
    NMI, AC = evaluatecluster(labels, datalabels)
    AMI_value = ami(datalabels, labels)
    print(f"Dataset: Heterogeneous geometric, Clusters: {NC}, NMI: {NMI:.4f}, AC: {AC:.4f}, AMI: {AMI_value:.4f}")
    
    fname = os.path.join('..', 'results', 'Fig_S3F.png')
    plot_clustering_result(data, labels, 'Heterogeneous geometric dataset clustering', fname)
    
    # ------------------------------
    # For Multi-objective 1 data set:
    # ------------------------------
    print('For Multi-objective 1 data set:')
    data, datalabels = process_dataset(load_data('data7.mat'), label_column=2, add_to_labels=1)
    labels, final_k, _ = torque_clustering_paper(data, is_noise=True)
    NC = final_k
    NMI, AC = evaluatecluster(labels, datalabels)
    AMI_value = ami(datalabels, labels)
    print(f"Dataset: Multi-objective 1, Clusters: {NC}, NMI: {NMI:.4f}, AC: {AC:.4f}, AMI: {AMI_value:.4f}")
    
    fname = os.path.join('..', 'results', 'Fig_S3G.png')
    plot_clustering_result(data, labels, 'Multi-objective 1 dataset clustering', fname)
    
    # ------------------------------
    # For Multi-objective 2 data set:
    # ------------------------------
    print('For Multi-objective 2 data set:')
    data, datalabels = process_dataset(load_data('data8.mat'), label_column=2, add_to_labels=1)
    labels, final_k, _ = torque_clustering_paper(data, is_noise=True)
    NC = final_k
    NMI, AC = evaluatecluster(labels, datalabels)
    AMI_value = ami(datalabels, labels)
    print(f"Dataset: Multi-objective 2, Clusters: {NC}, NMI: {NMI:.4f}, AC: {AC:.4f}, AMI: {AMI_value:.4f}")
    
    fname = os.path.join('..', 'results', 'Fig_S3H.png')
    plot_clustering_result(data, labels, 'Multi-objective 2 dataset clustering', fname)
    
    # ------------------------------
    # For Multi-objective 3 data set:
    # ------------------------------
    print('For Multi-objective 3 data set:')
    data, datalabels = process_dataset(load_data('data9.mat'), label_column=2, add_to_labels=1)
    labels, final_k, _ = torque_clustering_paper(data, is_noise=True)
    NC = final_k
    NMI, AC = evaluatecluster(labels, datalabels)
    AMI_value = ami(datalabels, labels)
    print(f"Dataset: Multi-objective 3, Clusters: {NC}, NMI: {NMI:.4f}, AC: {AC:.4f}, AMI: {AMI_value:.4f}")
    
    fname = os.path.join('..', 'results', 'Fig_S3I.png')
    plot_clustering_result(data, labels, 'Multi-objective 3 dataset clustering', fname)
    
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
            data, datalabels = process_dataset(load_data(file))
            
            if data is not None:
                # For high-dimensional data, if necessary use cosine distance.
                DM = cdist(data, data, metric='cosine')  # for evaluation/visualization purposes
                # Here, run the clustering on raw data:
                labels, final_k, _ = torque_clustering_paper(data, is_noise=True)
                NC = final_k
                
                if datalabels is not None:
                    NMI, AC = evaluatecluster(labels, datalabels)
                    AMI_value = ami(datalabels, labels)
                    print(f"Dataset: {name}, Clusters: {NC}, NMI: {NMI:.4f}, AC: {AC:.4f}, AMI: {AMI_value:.4f}")
                else:
                    print(f"Dataset: {name}, Clusters: {NC}")
                
                fname = os.path.join('..', 'results', f'Fig_Cosine_{i+1}_{name}.png')
                plot_clustering_result(data, labels, f'{name} dataset clustering', fname)
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
        
        labels, final_k, _ = torque_clustering_paper(data, is_noise=False)
        NC = final_k
        NMI, AC = evaluatecluster(labels, datalabels + 1)
        AMI_value = ami(datalabels, labels)
        print(f"Dataset: CMU-PIE, Clusters: {NC}, NMI: {NMI:.4f}, AC: {AC:.4f}, AMI: {AMI_value:.4f}")
        
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
        data, datalabels = process_dataset(data_dict)
        
        start_time = time.time()
        # Use cosine distance for high-dimensional data
        DM = cdist(data, data, metric='cosine')
        labels, final_k, _ = torque_clustering_paper(data, is_noise=True)
        elapsed = time.time() - start_time
        print(f"Elapsed time: {elapsed} seconds")
        
        NC = final_k
        NMI, AC = evaluatecluster(labels, datalabels)
        AMI_value = ami(datalabels, labels)
        print(f"Dataset: COIL-100, Clusters: {NC}, NMI: {NMI:.4f}, AC: {AC:.4f}, AMI: {AMI_value:.4f}")
        
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
        data, datalabels = process_dataset(load_data('Fig3.mat'), label_column=-1, add_to_labels=0)
        # Run clustering on raw data
        labels, final_k, _ = torque_clustering_paper(data, is_noise=True)
        try:
            from accuracy import accuracy
            AC = accuracy(datalabels + 1, labels + 1) / 100
            print(f"Fig3 Accuracy: {AC:.4f}")
        except ImportError:
            print("Accuracy function not imported, using placeholder calculation")
            
        fname1 = os.path.join('..', 'results', 'Fig3_Idx.png')
        fname2 = os.path.join('..', 'results', 'Fig3_Idx1.png')
        
        if data.shape[1] == 2:
            plot_clustering_result(data, labels, 'Fig3 Clustering (Idx)', fname1)
            # If a second index array is needed, it could be computed similarly.
            plot_clustering_result(data, labels, 'Fig3 Clustering (Idx1)', fname2)
        else:
            metrics_file = os.path.join('..', 'results', 'Fig3_metrics.txt')
            with open(metrics_file, 'w') as f:
                f.write(f"Dataset: Fig3\n")
                f.write(f"Number of clusters: {len(np.unique(labels))}\n")
                try:
                    f.write(f"Accuracy: {AC:.4f}\n")
                except NameError:
                    f.write("Accuracy: Not calculated\n")
    except Exception as e:
        print(f"Error processing Fig3 dataset: {str(e)}")
    
    print('For Fig4:')
    try:
        data, datalabels = process_dataset(load_data('Fig4.mat'), label_column=-1, add_to_labels=0)
        labels, final_k, _ = torque_clustering_paper(data, is_noise=True)
        try:
            from accuracy import accuracy
            AC = accuracy(datalabels + 1, labels + 1) / 100
            print(f"Fig4 Accuracy: {AC:.4f}")
        except ImportError:
            print("Accuracy function not imported, using placeholder calculation")
        
        fname = os.path.join('..', 'results', 'Fig4.png')
        if data.shape[1] == 2:
            plot_clustering_result(data, labels, 'Fig4 Clustering', fname)
        else:
            metrics_file = os.path.join('..', 'results', 'Fig4_metrics.txt')
            with open(metrics_file, 'w') as f:
                f.write(f"Dataset: Fig4\n")
                f.write(f"Number of clusters: {len(np.unique(labels))}\n")
                try:
                    f.write(f"Accuracy: {AC:.4f}\n")
                except NameError:
                    f.write("Accuracy: Not calculated\n")
    except Exception as e:
        print(f"Error processing Fig4 dataset: {str(e)}")

if __name__ == '__main__':
    main()
