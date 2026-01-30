**Project Documentation: Hybrid Spectral Clustering**

**1. Project Overview**

This project implements **Self-Tuning Spectral Clustering** in **C++** using a **Hybrid Parallel approach (MPI + OpenMP)**. Developed for the *HPC for Data Science* course at the University of Trento, the software is designed to efficiently process complex datasets on high-performance computing clusters.

The core objective was to overcome the limitations of standard spectral clustering when dealing with **multi-density datasets**. By integrating **Local Scaling (Zelnik-Manor & Perona)**, the algorithm adapts to the local density of data points, allowing it to correctly segment dense structures (like Gaussian blobs) and sparse, non-convex structures (like concentric circles) simultaneously.

**2. Key Features**

**Hybrid Parallelization**

To maximize resource utilization on the cluster, the project employs a hybrid model:

- **MPI (Inter-node):** Manages data distribution and communication between different computing nodes.
- **OpenMP (Intra-node):** Accelerates computationally intensive tasks—specifically the construction of the Similarity Matrix and distance calculations—by utilizing multi-threading within each MPI process.

**Self-Tuning Similarity Kernel**

Standard RBF kernels use a fixed $\sigma$ (sigma) parameter, which leads to failure in hybrid datasets: a small sigma breaks dense blobs, while a large sigma merges sparse circles.

This implementation uses **Local Scaling**:

$$W\_{ij} = \exp\left(-\frac{d(x\_i, x\_j)^2}{\sigma\_i \sigma\_j}\right)$$

Where $\sigma\_i$ is the distance to the $7^{th}$ nearest neighbor of point $i$. This allows the kernel to "tighten" connections in dense regions and "relax" them in sparse regions automatically.

**Robust Distributed K-Means**

The final clustering step is performed using a custom MPI-based K-Means algorithm. It includes a **multi-run mechanism** (default: 100 restarts) to mitigate the risk of initialization bias and ensure the global optimum is found.

**3. HPC Cluster Usage & Execution**

**Environment Setup**

The project relies on the following modules available on the Unitn Cluster:

- gcc91
- mpich-3.2.1--gcc-9.1.0
- python-3.10.14

**Compilation**

The code is compiled with O3 optimization and OpenMP support:

Bash

mpic++ -fopenmp -O3 src/SpectralClusteringMPI.cpp src/main\_mpi.cpp -o build/spectral\_hybrid

**Execution (PBS Script)**

The job is submitted via the PBS scheduler. The execution command uses specific parameters tuned for the hybrid dataset:

Bash

mpirun -np 1 ./build/spectral\_hybrid "scripts/data/mixed\_dataset.csv" 9 7 0.05 100

**Parameters Explanation:**

1. **Dataset:** Path to the CSV file.
1. **K=9:** Number of clusters (Explained in Results).
1. **KNN=7:** The neighbor index used for local scaling (Zelnik-Manor standard).
1. **Sigma=0.05:** Placeholder (ignored in Self-Tuning mode).
1. **Runs=100:** Number of K-Means retries.

**4. Experimental Results**

The algorithm was tested on a challenging "Hybrid Dataset" containing 9 distinct topological components with varying densities: 3 dense blobs, 2 non-convex moons, 2 intertwined spirals, and 2 concentric circles.

**The Challenge**

Initial tests using standard Spectral Clustering (fixed sigma) failed to segment the dataset correctly.

- **Low Sigma:** Successfully separated the spirals but fragmented the dense blobs into multiple clusters.
- **High Sigma:** Kept blobs intact but merged the concentric circles into a single cluster due to "bridging" effects in sparse regions.

**The Solution**

By implementing the **Self-Tuning** algorithm and adjusting the target clusters to **K=9**, we achieved a perfect segmentation.

**1. Resolution of Multi-Density Structures**

The **Local Scaling** mechanism proved decisive for the Concentric Circles. By calculating $\sigma\_i$ dynamically, the algorithm recognized that the "gap" between the inner and outer ring was significant relative to the local density of the rings themselves. This effectively cut the graph connections between the two rings, separating them into distinct clusters (Dark Blue and Teal in the final plot) without affecting the dense blobs.

**2. Resolution of Intertwined Spirals**

Even with local scaling, initial runs with $K=8$ resulted in the fusion of the two spiral arms. This was identified as a topological constraint: the dataset naturally contains **9 components** (3 blobs + 2 moons + 2 circles + 2 spirals). Forcing the algorithm to find only 8 clusters compelled it to merge the two most geometrically similar structures (the spirals).

Increasing the parameter to **$K=9$** provided the K-Means step with the necessary degree of freedom to assign a unique label to each spiral arm.

**Final Outcome**

The final visualization confirms the robustness of the approach:

- **Concentric Circles:** Perfectly separated.
- **Spirals:** Distinctly clustered, tracing the non-linear manifold correctly.
- **Blobs & Moons:** Preserved as coherent, dense clusters with no fragmentation.

Output:
![alt text](screenshots/Figure_new_clust.png)

