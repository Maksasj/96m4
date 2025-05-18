import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# If you want to use UMAP, uncomment the next line
# import umap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import argparse
import pandas as pd

def load_data(filepath):
    """Loads N-dimensional data from a comma-separated file."""
    try:
        data = pd.read_csv(filepath, header=None, comment='#').values
        # Check if all data is numeric, try to convert if not, and raise error if issues persist
        if not np.issubdtype(data.dtype, np.number):
            try:
                data = data.astype(float)
            except ValueError as e:
                print(f"Error: Could not convert all data to numeric values in {filepath}.")
                print(f"Pandas error: {e}")
                print("Please ensure your file contains only comma-separated numbers.")
                return None
        print(f"Successfully loaded data with shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"Error: Input file '{filepath}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: Input file '{filepath}' is empty.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading the data: {e}")
        return None

def reduce_dimensions(data, method='pca', n_components=2, perplexity=30.0, random_state=42):
    """
    Reduces data dimensionality to n_components using the specified method.
    'pca', 'tsne', or 'umap'.
    """
    print(f"Reducing dimensions using {method.upper()} to {n_components} components...")
    if method == 'pca':
        # It's often good practice to scale data before PCA
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        reducer = PCA(n_components=n_components, random_state=random_state)
        reduced_data = reducer.fit_transform(scaled_data)
    elif method == 'tsne':
        # t-SNE can be sensitive to scaling, but often applied directly or after PCA for pre-reduction
        # For very high dimensions, it's common to pre-reduce with PCA first
        if data.shape[1] > 50: # Heuristic: if more than 50 features, pre-reduce with PCA
            print("Input data has >50 features, applying PCA pre-reduction for t-SNE.")
            pca_pre = PCA(n_components=min(50, data.shape[1]), random_state=random_state)
            data_for_tsne = pca_pre.fit_transform(data)
        else:
            data_for_tsne = data
        
        # Adjust perplexity if number of samples is too small
        effective_perplexity = min(perplexity, data_for_tsne.shape[0] - 1)
        if effective_perplexity <= 0 :
             effective_perplexity = min(5.0, data_for_tsne.shape[0] -1) # A small default if too few samples
             if effective_perplexity <=0: # Still not enough samples for t-SNE
                 print("Error: Not enough samples for t-SNE after potential PCA pre-reduction.")
                 return None
             print(f"Warning: Perplexity adjusted to {effective_perplexity} due to small sample size ({data_for_tsne.shape[0]} samples).")


        reducer = TSNE(n_components=n_components, perplexity=effective_perplexity, random_state=random_state, n_iter=300) # n_iter can be increased
        reduced_data = reducer.fit_transform(data_for_tsne)
    # elif method == 'umap': # Uncomment if you have umap-learn installed
    #     reducer = umap.UMAP(n_components=n_components, random_state=random_state, n_neighbors=15, min_dist=0.1)
    #     reduced_data = reducer.fit_transform(data)
    else:
        raise ValueError("Unsupported dimensionality reduction method. Choose 'pca' or 'tsne'.") # Add 'umap' if using
    print("Dimensionality reduction complete.")
    return reduced_data

def perform_clustering(data_2d, n_clusters, random_state=42):
    """Performs K-Means clustering on the 2D data."""
    print(f"Performing K-Means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    cluster_labels = kmeans.fit_predict(data_2d)
    print("Clustering complete.")
    return cluster_labels

def save_output(data_2d, cluster_labels, output_filepath):
    """Saves the 2D data and cluster labels to the output file."""
    print(f"Saving processed data to '{output_filepath}'...")
    with open(output_filepath, 'w') as f:
        f.write("x y groupName\n") # Header for PGFPlots table
        for i in range(data_2d.shape[0]):
            f.write(f"{data_2d[i, 0]:.4f} {data_2d[i, 1]:.4f} cluster_{cluster_labels[i]}\n")
    print("Output file saved successfully.")

def main():
    parser = argparse.ArgumentParser(description="Reduce N-dimensional vector data to 2D, cluster it, and save for PGFPlots.")
    parser.add_argument("input_file", help="Path to the input text file (comma-separated N-dimensional vectors, one per line).")
    parser.add_argument("output_file", help="Path to save the processed 2D data with cluster labels.")
    parser.add_argument("-n", "--n_clusters", type=int, default=3, help="Number of clusters for K-Means (default: 3).")
    parser.add_argument("-m", "--method", type=str, default="pca", choices=["pca", "tsne"], help="Dimensionality reduction method: 'pca' or 'tsne' (default: pca).") # Add "umap" to choices if using
    parser.add_argument("--perplexity", type=float, default=30.0, help="Perplexity for t-SNE (default: 30.0). Ignored for PCA.")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducible results (default: 42).")

    args = parser.parse_args()

    # 1. Load data
    raw_data = load_data(args.input_file)
    if raw_data is None:
        return

    # 2. Reduce dimensions
    # Ensure there are enough samples for the chosen perplexity if using t-SNE
    if args.method == "tsne" and raw_data.shape[0] <= args.perplexity:
        print(f"Warning: Number of samples ({raw_data.shape[0]}) is less than or equal to perplexity ({args.perplexity}).")
        print("t-SNE may not perform well or might error out. Consider reducing perplexity or using PCA.")
        # The reduce_dimensions function also has a check, but this is an earlier warning.

    reduced_data_2d = reduce_dimensions(raw_data, method=args.method, perplexity=args.perplexity, random_state=args.random_state)
    if reduced_data_2d is None:
        print("Failed to reduce dimensions. Exiting.")
        return

    # 3. Perform clustering
    if reduced_data_2d.shape[0] < args.n_clusters:
        print(f"Error: Number of data points ({reduced_data_2d.shape[0]}) is less than the number of clusters ({args.n_clusters}).")
        print("Please reduce the number of clusters or provide more data.")
        return
    cluster_labels = perform_clustering(reduced_data_2d, args.n_clusters, random_state=args.random_state)

    # 4. Save output
    save_output(reduced_data_2d, cluster_labels, args.output_file)

if __name__ == "__main__":
    main()