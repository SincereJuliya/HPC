import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def main(data_csv, labels_csv, output_path):
    # Ensure the output directory exists
    os.makedirs("plots", exist_ok=True)

    # UNIQUE NAMING LOGIC:
    # If the output path is the default one, we rename it based on labels_csv
    # e.g., if labels are 'mixed_dataset_1_labels.csv', plot becomes 'mixed_dataset_1_labels.png'
    if "cluster_plot.png" in output_path:
        base_name = os.path.splitext(os.path.basename(labels_csv))[0]
        output_path = os.path.join("plots", f"{base_name}.png")

    # Load data
    try:
        data = pd.read_csv(data_csv)
        labels = pd.read_csv(labels_csv)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Visualization
    plt.figure(figsize=(8, 8))
    # Using iloc to get first two columns regardless of their names (x/y or 0/1)
    scatter = plt.scatter(data.iloc[:, 0], data.iloc[:, 1], 
                        c=labels['label'], cmap='tab10', s=10, alpha=0.6)
    
    plt.title(f"Spectral Clustering Result\nSource: {os.path.basename(labels_csv)}")
    plt.colorbar(scatter, label="Cluster ID")
    plt.axis("equal")
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=200)
    print(f"✅ Success! Plot saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path to input coordinates
    parser.add_argument("--data", type=str, default="data/mixed_dataset_1.csv")
    # Path to cluster labels
    parser.add_argument("--labels", type=str, default="data/mixed_dataset_1_labels.csv")
    # Output image path (automatically adjusted if left as default)
    parser.add_argument("--output", type=str, default="plots/cluster_plot.png")
    
    args = parser.parse_args()
    main(args.data, args.labels, args.output)