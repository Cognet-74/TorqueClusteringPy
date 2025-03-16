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

# Import necessary functions
from TorqueClustering import TorqueClustering
from evaluatecluster import evaluatecluster
from ami import ami
# from accuracy import accuracy  # Uncomment if you have an accuracy function

def load_txt_data(filepath):
    """
    Load data from a text file.
    
    Parameters:
    filepath: Path to the text/dat file
    
    Returns:
    data: NumPy array of data
    """
    print(f"Loading text file: {filepath}")
    try:
        data = np.loadtxt(filepath)
        print(f"Loaded data with shape {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading text file: {str(e)}")
        raise

def load_csv_data(filepath, delimiter=',', header=None):
    """
    Load data from a CSV file.
    
    Parameters:
    filepath: Path to the CSV file
    delimiter: Character used to separate values (default: ',')
    header: Row number to use as column names (default: None)
    
    Returns:
    data: NumPy array of data
    column_names: List of column names if header is not None
    """
    print(f"Loading CSV file: {filepath}")
    try:
        import pandas as pd
        
        # Read CSV with pandas for better handling of headers, missing values, etc.
        df = pd.read_csv(filepath, delimiter=delimiter, header=header)
        print(f"Loaded CSV with shape {df.shape}")
        
        # Check for missing values
        if df.isna().any().any():
            print("Warning: CSV contains missing values. They will be handled by pandas defaults.")
        
        # If header was specified, return both data and column names
        if header is not None:
            return df.values, df.columns.tolist()
        else:
            return df.values
            
    except ImportError:
        print("Pandas not available, falling back to numpy.loadtxt")
        try:
            # Fallback to numpy if pandas is not available
            data = np.loadtxt(filepath, delimiter=delimiter, skiprows=1 if header is not None else 0)
            print(f"Loaded data with shape {data.shape}")
            return data
        except Exception as e:
            print(f"Error loading CSV file with numpy: {str(e)}")
            raise
    except Exception as e:
        print(f"Error loading CSV file: {str(e)}")
        raise

def load_data(filename, delimiter=',', header=None):
    """
    Load data from a file located in the "data" subfolder.
    Supports multiple file formats:
    - .txt/.dat files: uses np.loadtxt
    - .csv files: uses load_csv_data (pandas or numpy fallback)
    
    Parameters:
    filename: Name of the file in the data folder
    delimiter: Character used to separate values in CSV files (default: ',')
    header: Row number to use as column names in CSV files (default: None)
    
    Returns:
    data: NumPy array of data
    column_names: List of column names if header is not None and format is CSV
    """
    filepath = os.path.join("data", filename)
    
    if filename.endswith('.dat') or filename.endswith('.txt'):
        return load_txt_data(filepath)
    elif filename.endswith('.csv'):
        return load_csv_data(filepath, delimiter, header)
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

def plot_high_dim_clustering(data, idx, title, save_path):
    """
    Plot clustering results for high-dimensional data using PCA or t-SNE.
    
    Parameters:
    data: high-dimensional data points
    idx: cluster assignments
    title: plot title
    save_path: path to save the figure
    """
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Use non-interactive backend
    import matplotlib
    matplotlib.use('Agg')
    
    # Create a figure
    fig = plt.figure(figsize=(15, 6))
    
    # Get unique clusters
    unique_clusters = np.unique(idx)
    
    # Generate colors for clusters
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
    
    # Try to import sklearn for dimensionality reduction
    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        # Create subplots for PCA and t-SNE
        plt.subplot(1, 2, 1)
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data)
        
        # Plot each cluster with a different color
        for i, cluster_id in enumerate(unique_clusters):
            cluster_points = data_pca[idx == cluster_id]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                      color=colors[i], label=f'Cluster {cluster_id}', alpha=0.7)
        
        plt.title(f"{title} (PCA)")
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Apply t-SNE for dimensionality reduction
        plt.subplot(1, 2, 2)
        
        # t-SNE can be slow for larger datasets
        if data.shape[0] <= 5000:  # Only apply t-SNE for reasonably sized datasets
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, data.shape[0]//5))
            data_tsne = tsne.fit_transform(data)
            
            # Plot each cluster with a different color
            for i, cluster_id in enumerate(unique_clusters):
                cluster_points = data_tsne[idx == cluster_id]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                          color=colors[i], label=f'Cluster {cluster_id}', alpha=0.7)
            
            plt.title(f"{title} (t-SNE)")
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.legend(loc='best')
            plt.grid(True, linestyle='--', alpha=0.7)
        else:
            plt.text(0.5, 0.5, "Dataset too large for t-SNE", 
                   horizontalalignment='center', verticalalignment='center')
        
    except ImportError:
        # If sklearn is not available, use a simple 2D projection
        plt.text(0.5, 0.5, "Install scikit-learn for PCA/t-SNE visualization", 
               horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
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
    # For Haberman dataset only:
    # ------------------------------
    print('Processing Haberman dataset:')
    try:
        # Load the Haberman dataset
        data = load_data('haberman.csv')
        
        # Check if the last column contains class labels
        if data.shape[1] > 2:
            datalabels = data[:, -1]  # Assume last column contains labels
            data_features = data[:, :-1]  # Features are all columns except the last
            print(f"Features shape: {data_features.shape}, Labels shape: {datalabels.shape}")
        else:
            # If no labels column, just use all data for clustering
            data_features = data
            datalabels = None
            print(f"Features shape: {data_features.shape}, No labels detected")
        
        # Compute pairwise distances using cosine distance
        DM = cdist(data_features, data_features, metric='cosine')
        
        # Run TorqueClustering algorithm
        start_time = time.time()
        idx = TorqueClustering(DM, 0)[0]
        elapsed = time.time() - start_time
        print(f"Clustering completed in {elapsed:.2f} seconds")
        
        # Get number of clusters detected
        NC = len(np.unique(idx))
        print(f"Number of clusters detected: {NC}")
        
        # Variables to store evaluation metrics
        NMI = None
        AC = None
        AMI_value = None
        
        # If we have ground truth labels, evaluate clustering performance
        if datalabels is not None:
            try:
                # Convert labels to integers if they're not already
                int_labels = datalabels.astype(int)
                
                # Ensure idx is also integer type
                idx_int = idx.astype(int)
                
                # Debug information
                print(f"Original labels range: {np.min(int_labels)} to {np.max(int_labels)}")
                print(f"Cluster indices range: {np.min(idx_int)} to {np.max(idx_int)}")
                
                # Ensure both arrays are 1-dimensional vectors
                int_labels = int_labels.flatten()
                idx_int = idx_int.flatten()
                
                # Validate that the arrays are the same length
                if len(int_labels) != len(idx_int):
                    raise ValueError(f"Label length ({len(int_labels)}) doesn't match cluster assignment length ({len(idx_int)})")
                
                # Check indexing and convert if needed
                # If the evaluatecluster function expects 1-indexed clusters (MATLAB style)
                if np.min(idx_int) == 0 and np.min(int_labels) == 1:
                    # Convert from 0-indexed to 1-indexed to match labels
                    idx_int = idx_int + 1
                elif np.min(int_labels) == 0 and np.min(idx_int) == 1:
                    # Convert labels from 0-indexed to 1-indexed to match clusters
                    int_labels = int_labels + 1
                
                # Evaluate clustering
                NMI, AC = evaluatecluster(idx_int, int_labels)
                AMI_value = ami(int_labels, idx_int)
                
                print(f"Clustering evaluation metrics:")
                print(f"NMI: {NMI:.4f}, AC: {AC:.4f}, AMI: {AMI_value:.4f}")
            except Exception as e:
                print(f"Error during evaluation: {str(e)}")
                import traceback
                traceback.print_exc()  # More detailed error information
                print("Continuing without evaluation metrics...")
        
        # Save metrics to a text file
        metrics_file = os.path.join('..', 'results', 'Haberman_metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"Dataset: Haberman\n")
            f.write(f"Number of clusters: {NC}\n")
            f.write(f"Processing time: {elapsed:.2f} seconds\n")
            
            # Only write metrics if they were successfully calculated
            if NMI is not None and AC is not None and AMI_value is not None:
                f.write(f"NMI: {NMI:.4f}\n")
                f.write(f"AC: {AC:.4f}\n") 
                f.write(f"AMI: {AMI_value:.4f}\n")
            else:
                f.write("Evaluation metrics: Not available due to errors\n")
        
        # If data is 2D, create and save standard visualization
        if data_features.shape[1] == 2:
            fname = os.path.join('..', 'results', 'Haberman_clustering.png')
            plot_clustering_result(data_features, idx, 'Haberman Dataset Clustering', fname)
        else:
            print("Data has more than 2 dimensions, using dimensionality reduction...")
            fname = os.path.join('..', 'results', 'Haberman_clustering_high_dim.png')
            try:
                plot_high_dim_clustering(data_features, idx, 'Haberman Dataset Clustering', fname)
            except Exception as e:
                print(f"Error during high-dimensional visualization: {str(e)}")
                print("Skipping visualization")
            
    except Exception as e:
        print(f"Error processing Haberman dataset: {str(e)}")
        import traceback
        traceback.print_exc()  # Print detailed stack trace for debugging

if __name__ == '__main__':
    main()