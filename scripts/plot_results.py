import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Check for sklearn (required for t-SNE on high-dim data)
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    has_sklearn = True
except ImportError:
    has_sklearn = False

def load_labels_robust(filepath):
    """
    Robustly loads labels handling headers, indices, and NaNs.
    """
    try:
        # Read as string to avoid parsing errors
        df = pd.read_csv(filepath, header=None, dtype=str)
        
        # Take the last column (usually: index, label)
        y_raw = df.iloc[:, -1].values
        
        # Convert to numeric, coercing errors (like headers "Label") to NaN
        y_numeric = pd.to_numeric(y_raw, errors='coerce')
        
        # Remove NaNs
        mask = ~np.isnan(y_numeric)
        y = y_numeric[mask].astype(int)
        
        return y
    except Exception as e:
        print(f"Error reading labels: {e}")
        sys.exit(1)

def main():
    print("--- Spectral Clustering Visualization Tool ---")
    
    # 1. ARGUMENTS PARSING
    if len(sys.argv) < 3:
        print("Usage: python plot_results.py [data_file.csv] [label_file.csv]")
        print("Example: python plot_results.py data/mixed_dataset.csv data/mixed_dataset_labels_mpi.csv")
        sys.exit(1)
    
    data_file = sys.argv[1]
    label_file = sys.argv[2]

    print(f"Data:   {data_file}")
    print(f"Labels: {label_file}")

    if not os.path.exists(data_file) or not os.path.exists(label_file):
        print("Error: Files not found.")
        sys.exit(1)

    # 2. LOAD DATA
    try:
        # Data usually has no header (x, y) or (d1, d2... dn)
        X = pd.read_csv(data_file, header=None).values
        # If header was present (string detected), reload skipping row 0
        if X.dtype == object:
             X = pd.read_csv(data_file, header=0).values
    except Exception as e:
        print(f"Error reading data: {e}")
        sys.exit(1)

    # 3. LOAD LABELS
    y = load_labels_robust(label_file)

    # 4. ALIGN LENGTHS (Fix eventual off-by-one errors)
    min_len = min(len(X), len(y))
    if len(X) != len(y):
        print(f"Warning: Length mismatch: Data={len(X)}, Labels={len(y)}")
        print(f"Trimming to {min_len} points.")
        X = X[:min_len]
        y = y[:min_len]
    
    # 5. PLOT LOGIC (2D vs High-Dim)
    dims = X.shape[1]
    print(f"Dataset Dimensions: {min_len} rows x {dims} columns")

    plt.figure(figsize=(10, 8))
    title_str = ""
    
    if dims == 2:
        print(">> Mode: 2D SCATTER (Synthetic Data)")
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', s=5, alpha=0.8)
        plt.xlabel("X")
        plt.ylabel("Y")
        title_str = "Spectral Clustering Result (Synthetic 2D)"

    else:
        print(">> Mode: HIGH-DIMENSIONAL (Biological Data / t-SNE)")
        if not has_sklearn:
            print("Error: scikit-learn required for High-Dim visualization.")
            sys.exit(1)
            
        print("   1. Running PCA (reducing to 50 dims)...")
        X_pca = PCA(n_components=min(50, dims)).fit_transform(X)
        
        print("   2. Running t-SNE (projection to 2D)...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
        X_embedded = tsne.fit_transform(X_pca)
        
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='tab10', s=1, alpha=0.6)
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        title_str = "Spectral Clustering Result (t-SNE Projection)"

    # 6. FINALIZE PLOT
    n_clusters = len(np.unique(y))
    plt.colorbar(label=f'Cluster ID (Total: {n_clusters})')
    plt.title(f"{title_str}\nN={min_len} points", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    out_name = "clustering_result.png"
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    print(f"Success! Plot saved as: {out_name}")

if __name__ == "__main__":
    main()