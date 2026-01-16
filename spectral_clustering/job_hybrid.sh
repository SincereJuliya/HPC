#!/bin/bash
#PBS -N SpectralHybridTest
#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=00:10:00
#PBS -q short_cpuQ

cd $PBS_O_WORKDIR

# 1. Load modules
module purge
module load gcc91
module load mpich-3.2.1--gcc-9.1.0
module load python-3.10.14

echo "==== Modules loaded ===="

# 2. Generate dataset (Increased points to 8000 to see the OpenMP impact)
echo "==== Generating data (тотал 4800 point) ===="
python3 scripts/synt_data.py --points 900 --type mixed --name mixed_dataset_2
if [ $? -ne 0 ]; then
    echo "Error: Data generation failed."
    exit 1
fi

# 3. Compile with OpenMP support
# Adding -fopenmp flag allows Eigen and #pragma loops to use multi-threading
echo "==== Compiling Hybrid MPI + OpenMP version ===="
mkdir -p build
mpicxx -std=c++17 -O3 -fopenmp -I ./eigen_local \
    src/mainMPI.cpp src/SpectralClusteringMPI.cpp \
    -o build/spectral_hybrid

if [ $? -ne 0 ]; then
    echo "Error: Compilation failed."
    exit 1
fi

# 4. Hybrid Execution Strategy (1 MPI Rank x 4 Threads)
echo "==== Running Hybrid version (1 MPI Rank x 4 OpenMP Threads) ===="
export OMP_NUM_THREADS=4
# -np 1 means only the Master process runs, but it will use all 4 cores for Eigen
mpirun -np 1 ./build/spectral_hybrid data/mixed_dataset_2.csv

if [ $? -ne 0 ]; then
    echo "Error: Execution failed."
    exit 1
fi

# 5. Visualization for Hybrid version
echo "==== Generating visualization (Hybrid) ===="
python3 scripts/visualize.py --data data/mixed_dataset_2.csv --labels data/mixed_dataset_2_labels.csv

echo "==== Job completed successfully ===="