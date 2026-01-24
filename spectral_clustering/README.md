Hybrid Spectral Clustering with MPI & OpenMPProject OverviewThis project implements Self-Tuning Spectral Clustering in C++ using a Hybrid Parallel approach (MPI + OpenMP). developed for the course HPC for DS at the University of Trento.It is designed to handle multi-density datasets (e.g., dense blobs mixed with sparse concentric circles) by implementing Local Scaling (Zelnik-Manor & Perona), efficiently distributing computations across high-performance computing resources.🚀 Key FeaturesHybrid Parallelization: Uses MPI for inter-node communication and OpenMP for intra-node multi-threading (accelerating distance calculations and matrix operations).Self-Tuning Similarity: Implements Local Scaling ($\sigma_i$) based on the $7^{th}$ neighbor, replacing the standard fixed-sigma RBF kernel. This allows correct clustering of hybrid datasets containing both dense and sparse structures.Robustness: Includes a multi-run mechanism for the distributed K-Means step to avoid local minima.Synthetic & Real Data: Supports generation of complex synthetic shapes (circles, moons, spirals) and loading of CSV datasets.🔧 Build & Run Instructions1. CompilationThe project uses CMake or a custom Makefile. Ensure you compile with OpenMP support.Hybrid MPI+OpenMP version:Bashmpic++ -fopenmp -O3 src/SpectralClusteringMPI.cpp src/main.cpp -o build/spectral_hybrid
2. OutputPlots: Saved in the plots/ folder (e.g., Figure_new_clust.png).Labels: Cluster assignments are saved in data/mixed_dataset_labels_mpi.csv.🔧 HPC / Cluster Usage1. Environment Setup (Unitn Cluster)Before running the scripts, load the required modules:Bashmodule load gcc91
module load mpich-3.2.1--gcc-9.1.0
module load python-3.10.14
Check dependencies:Bashmpic++ --version
python3 --version
2. Python Dependencies (Visualization)If you need to reproduce the plots locally or on the cluster:Bashpython3 -m pip install --user pandas matplotlib seaborn
3. Running the Job (PBS Script)To run the Hybrid version with Self-Tuning, use the provided PBS script (run_hybrid_final.pbs).Example PBS Configuration:Bash#!/bin/bash
#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=00:10:00
#PBS -q short_cpuQ
module load gcc91
module load mpich-3.2.1--gcc-9.1.0
# Set OpenMP threads (Hybrid Mode)
export OMP_NUM_THREADS=4
cd $PBS_O_WORKDIR
# Compile
mpic++ -fopenmp -O3 src/SpectralClusteringMPI.cpp src/main_mpi.cpp -o build/spectral_hybrid
# Run Parameters:
# 1. Dataset Path
# 2. Clusters (K=9 covers all topological components: blobs, circles, moons, spirals)
# 3. KNN (7 is the standard for Zelnik-Manor local scaling)
# 4. Sigma (Ignored in Self-Tuning mode, kept for compatibility)
# 5. K-Means Runs (100 retries for stability)
mpirun -np 1 ./build/spectral_hybrid "scripts/data/mixed_dataset.csv" 9 7 0.05 100
Submit the job:Bashqsub run_hybrid_final.pbs
4. ResultsThe final implementation successfully segments complex topologies that standard Spectral Clustering fails on:Concentric Circles: Separated (thanks to local scaling).Spirals: Distinctly clustered (using K=9).Blobs: Preserved as compact clusters.