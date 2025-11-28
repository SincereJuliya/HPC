# Spectral Clustering with MPI in C++

## Project Overview
This project implements **Spectral Clustering** in **C++** with **MPI** for parallel processing for the course 'HPC for DS' at the University of Trento.
The goal is to cluster data efficiently using high-performance computing resources, such as the university cluster.

## Features
- Generate **synthetic datasets** or load **real-world datasets** (e.g., gene expression matrices).  
- Compute **similarity matrix** using RBF kernel.  
- Construct **graph Laplacian** and compute its **eigenvectors**.  
- Perform **K-means clustering** on the spectral embeddings.  
- Parallelized computations using **MPI** to scale on large datasets.

## 🔧 Build Instructions

### 1. To build and run the code, use the provided scripts

**MPI version:**
```bash
./mpi.sh
```

**Non-MPI version:**
```bash
./nonmpi.sh
```

### 2. Output

- Plots will be saved in the spectral_clustering/plots folder.

- Tables with results and cluster labels will be saved in the spectral_clustering/data folder.


