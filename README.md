# High Performance Spectral Clustering
![Result Preview](mixed_dataset_labels_mpi_plot.png)
![C++](https://img.shields.io/badge/C++-17-blue.svg)
![MPI](https://img.shields.io/badge/MPI-OpenMP-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An optimized hybrid parallel implementation of the **Spectral Clustering** algorithm.
This project combines **MPI** (Distributed Memory) and **OpenMP** (Shared Memory) to efficiently cluster large datasets.

##  Project Structure

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
│   ── performance_scalability/ # Scalability tests
│       ├── run_hybrid_strong.sh
│       ├── run_hybrid_weak.sh
│       ├── run_mpi_strong.sh
│       └── run_mpi_weak.sh
│
├── data/                       # Datasets & Results
│   ├── mixed_dataset.csv             # Input: Synthetic Benchmark (Spirals/Circles) for validation
│   ├── mixed_dataset_labels_mpi.csv  # Output: Cluster labels for the synthetic dataset
│   ├── dataset_mca_20k.csv           # Input: Mouse Cell Atlas (20k cells) for scalability testing
│   └── dataset_mca_20k_labels.csv    # Output: Cluster labels for the biological dataset
│   └── others    # for the Strong+Weak scalability
│
├── old_base_code/              # Oldest try and implementations
├── eigen_local/                # (Optional) Local copy of Eigen 3 library
├── Makefile                    # Build automation
└── README.md                   # Documentation
```

## ️ Prerequisites

To build and run this project, you need:

* **C++ Compiler** with C++17 support (e.g., GCC 9+).
* **MPI Implementation** (MPICH or OpenMPI).
* **Eigen 3 Library:**
  * If not installed globally (`/usr/include/eigen3`), place the library files in a folder named `eigen_local` in the project root.
* **Python 3** (Optional, for visualization) with pandas, matplotlib, and scikit-learn.

##  Compilation

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

##  Usage

### 1. Running Locally (Interactive)

**Sequential Run:**

```bash
# Syntax: ./spectral_seq <file> <k> <knn> <sigma> <runs>
./spectral_seq data/mixed_dataset.csv 6 10 0.05 1
```

**Parallel Run (MPI):**

```bash
# Syntax: mpirun -np <ranks> ./spectral_mpi <file> <k> <knn> <sigma> <runs>
# Example: Run on 4 processes
mpirun -np 4 ./spectral_mpi data/mixed_dataset.csv 6 10 0.05 1
```

**Parameters:**

* `<file>`: Path to CSV dataset (x, y coordinates).
* `<k>`: Number of clusters to find.
* `<knn>`: Number of nearest neighbors (used in Sequential mode).
* `<sigma>`: Gaussian kernel width. 
    * *Note:* Use `-1` for auto-tuning (Zelnik-Manor heuristic) **only in Sequential mode**. The MPI implementation requires a fixed sigma value.

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

**Submit the other Job**

```bash
qsub scripts/performance_scalability/run_hybrid_strong.sh
qsub scripts/performance_scalability/run_hybrid_weak.sh
qsub scripts/performance_scalability/run_mpi_strong.sh
qsub scripts/performance_scalability/run_mpi_weak.sh
```
You can customize the run configuration (nodes, cores, dataset) by editing the variables inside the `.sh` files.

##  Visualization

After running the simulation, result labels are saved in the `data/` folder. Use the Python script to visualize them.

**For 2D Datasets:**

```bash
python3 scripts/plot_results.py data/mixed_dataset.csv data/mixed_dataset_labels_mpi.csv
```

**For High-Dimensional Data:** The script automatically detects dimensions > 2 and applies PCA + t-SNE for visualization.

```bash
python3 scripts/plot_results.py data/dataset_mca_20k.csv data/dataset_mca_20k_labels.csv
```
> **Note on Implementation:**
> * **Sequential:** Uses full similarity matrix construction (exact spectral clustering).
> * **MPI:** Uses **Distributed Nyström Approximation** to handle large datasets and reduce memory complexity.

##  Authors

* Antonio Di Lauro
* Juliya Sharipova

Course: High Performance Computing - University of Trento.
