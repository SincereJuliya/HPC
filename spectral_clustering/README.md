# Spectral Clustering with MPI in C++

## Project Overview
This project implements **Spectral Clustering** in **C++** with **MPI** for parallel processing, developed for the course *HPC for DS* at the University of Trento.  
It efficiently clusters data using high-performance computing resources, such as the university cluster.

---

## Features
- Generate **synthetic datasets** or load **real-world datasets** (e.g., gene expression matrices).  
- Compute **similarity matrix** using RBF kernel.  
- Construct **graph Laplacian** and compute its **eigenvectors**.  
- Perform **K-means clustering** on the spectral embeddings.  
- Parallelized computations using **MPI** for large datasets.

---

## 🔧 Build & Run Instructions

### 1. Using the provided scripts

**MPI version:**
```bash
./mpi.sh
```
**Non-MPI version:**
```bash
./nonmpi.sh
```

The MPI version will distribute computations across multiple processes for faster execution.
The Non-MPI version runs everything on a single process. And basically we dont run Non-MPI - it is the initial code for parallelizing.

### 2.Output

Plots are saved in the plots/ folder.

Cluster labels and result tables are saved in the data/ folder.

---

## 🔧 HPC / Cluster Usage

### 1. Check the environment

Before running the scripts on the cluster, load the required modules:
```bash
module load gcc91
module load mpich-3.2.1--gcc-9.1.0
module load python-3.10.14
```
Check Python version:
```bash
which python3
python3 --version
```

Check pip:
```bash
python3 -m pip --version
```

### 2. Python Libraries (User Install)

Required Python packages can be installed for the current user without root access:
```bash
python3 -m pip install --user pandas matplotlib seaborn
```
Check installed packages:
```bash
python3 -m pip list --user
```
Example output:
![alt text](spectral_clustering/screenshots/image.png)

### 3. HPC Job Example (PBS)

PBS script to run the MPI job on the cluster:
```bash
qsub job_spectral.sh
```
The PBS script will:

  -  Load GCC, MPI, and Python modules

  -  Generate the dataset

  -  Compile the MPI binary

  -  Run the MPI spectral clustering job