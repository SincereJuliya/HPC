# High Performance Spectral Clustering

![C++](https://img.shields.io/badge/C++-17-blue.svg)
![MPI](https://img.shields.io/badge/MPI-OpenMP-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A highly optimized, hybrid parallel implementation of the **Spectral Clustering** algorithm.
This project combines **MPI** (Distributed Memory) and **OpenMP** (Shared Memory) to efficiently cluster large datasets using the Normalized Symmetric Laplacian and K-Means.

## ��� Features

* **Hybrid Parallelism:** Uses MPI for inter-node communication and OpenMP for intra-node acceleration.
* **Self-Tuning Similarity:** Implements Zelnik-Manor & Perona (2004) local scaling for adaptive sigma.
* **Sparse/Dense Optimization:** Efficient matrix operations using the Eigen library.
* **Robust K-Means:** Distributed K-Means implementation for the final clustering step.
* **Visualization:** Python scripts included for 2D scatter plots and high-dimensional t-SNE projections.

## ��� Project Structure

```text
Spectral_Clustering_HPC/
├── src/                        # C++ Source Code
│   ├── main.cpp                # Sequential Entry Point
│   ├── mainMPI.cpp             # Parallel Entry Point
│   ├── SpectralClustering.cpp  # Sequential Implementation
│   ├── SpectralClustering.hpp  # Header sequential Implementation
│   ├── SpectralClusteringMPI.hpp # Header parallel Implementation
│   └── SpectralClusteringMPI.cpp # Parallel Implementation
│
├── scripts/                    # Helper Scripts
│   ├── run_seq.pbs             # PBS Job Script (Sequential)
│   ├── run_mpi.pbs             # PBS Job Script (Parallel)
│   ├── generate_data.py        # Dataset Generator
│   └── plot_results.py         # Visualization Tool
│
├── data/                       # Datasets & Results
│   ├── mixed_dataset.csv             # Input: Synthetic Benchmark (Spirals/Circles) for validation
│   ├── mixed_dataset_labels_mpi.csv  # Output: Cluster labels for the synthetic dataset
│   ├── dataset_mca_20k.csv           # Input: Mouse Cell Atlas (20k cells) for scalability testing
│   └── dataset_mca_20k_labels.csv    # Output: Cluster labels for the biological dataset
│
├── eigen_local/                # (Optional) Local copy of Eigen 3 library
├── Makefile                    # Build automation
└── README.md                   # Documentation
```

## ���️ Prerequisites

To build and run this project, you need:

* **C++ Compiler** with C++17 support (e.g., GCC 9+).
* **MPI Implementation** (MPICH or OpenMPI).
* **Eigen 3 Library:**
  * If not installed globally (`/usr/include/eigen3`), place the library files in a folder named `eigen_local` in the project root.
* **Python 3** (Optional, for visualization) with pandas, matplotlib, and scikit-learn.

## ��� Compilation

A Makefile is provided to compile both Sequential and Parallel versions automatically.

```bash
# Clean previous builds
make clean

# Compile everything
make
```

This will generate two executables:

* `spectral_seq`: Sequential implementation.
* `spectral_mpi`: Hybrid Parallel implementation.

Note: The Makefile uses `mpicxx` for both targets to ensure C++17 support on older HPC environments.

## ��� Usage

### 1. Running Locally (Interactive)

**Sequential Run:**

```bash
# Syntax: ./spectral_seq <file> <k> <knn> <sigma> <runs>
./spectral_seq data/mixed_dataset.csv 6 10 0.05 5
```

**Parallel Run (MPI):**

```bash
# Syntax: mpirun -np <ranks> ./spectral_mpi <file> <k> <knn> <sigma> <runs>
# Example: Run on 4 processes
mpirun -np 4 ./spectral_mpi data/mixed_dataset.csv 6 10 0.05 10
```

**Parameters:**

* `<file>`: Path to CSV dataset (x, y coordinates).
* `<k>`: Number of clusters to find.
* `<knn>`: Number of nearest neighbors for similarity graph.
* `<sigma>`: Gaussian kernel width (use -1 for auto-tuning).
* `<runs>`: Number of K-Means restarts (for stability).

### 2. Running on HPC Cluster (PBS/Torque)

Use the provided scripts in the `scripts/` folder to submit jobs to the queue.

**Submit Parallel Job:**

```bash
qsub scripts/run_mpi.pbs
```

**Submit Sequential Job:**

```bash
qsub scripts/run_seq.pbs
```

You can customize the run configuration (nodes, cores, dataset) by editing the variables inside the `.pbs` files.

## ��� Visualization

After running the simulation, result labels are saved in the `data/` folder. Use the Python script to visualize them.

**For 2D Datasets:**

```bash
python3 scripts/plot_results.py data/mixed_dataset.csv data/mixed_dataset_labels_mpi.csv
```

**For High-Dimensional Data:** The script automatically detects dimensions > 2 and applies PCA + t-SNE for visualization.

```bash
python3 scripts/plot_results.py data/dataset_mca_20k.csv data/dataset_mca_20k_labels.csv
```

## ��� Authors

* Antonio Di Lauro
* Juliya Sharipova

Course: High Performance Computing Systems - University of Trento.
