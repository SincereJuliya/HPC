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
echo "==== Generating data (1800 point) ===="
python3 scripts/synt_data.py --points 1800 --type mixed
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

# 4. Hybrid Execution Strategy
# Logic: We have 4 CPUs total. 
# To let Eigen use all of them, we launch 1 MPI rank and set OMP_NUM_THREADS to 4.
# This avoids "oversubscription" (where threads fight for the same core).

echo "==== Running Hybrid version (1 MPI Rank x 4 OpenMP Threads) ===="

export OMP_NUM_THREADS=4
# -np 1 means only the Master process runs, but it will use all 4 cores for Eigen
mpirun -np 1 ./build/spectral_hybrid scripts/data/mixed_dataset.csv

if [ $? -ne 0 ]; then
    echo "Error: Execution failed."
    exit 1
fi

echo "==== Job completed successfully ===="