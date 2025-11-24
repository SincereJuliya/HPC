import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def main(data_csv, labels_csv, output="plots/cluster_plot.png"):
    os.makedirs(os.path.dirname(output), exist_ok=True)

    data = pd.read_csv(data_csv)
    labels = pd.read_csv(labels_csv)

    if len(data) != len(labels):
        raise ValueError("Number of points and labels must match")

    plt.figure(figsize=(6,6))
    unique_labels = labels['label'].unique()
    for lab in unique_labels:
        pts = data[labels['label']==lab]
        plt.scatter(pts['x'], pts['y'], s=15, label=f"Cluster {lab}")

    plt.title("Spectral Clustering Result")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    print(f"Plot saved to {output}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../scripts/data/mixed_dataset.csv",
                        help="CSV with points (x,y)")
    parser.add_argument("--labels", type=str, default="../data/mixed_dataset_labels.csv",
                        help="CSV with cluster labels")
    parser.add_argument("--output", type=str, default="../plots/cluster_plot.png",
                        help="Output image file")
    args = parser.parse_args()

    main(args.data, args.labels, args.output)
