#!/bin/bash
#PBS -N SpectralFinal
#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=00:10:00
#PBS -q short_cpuQ

cd $PBS_O_WORKDIR

# 1. Load modules (GCC 9 + MPI under GCC 9 + Python)
module purge
module load gcc91
module load mpich-3.2.1--gcc-9.1.0
module load python-3.10.14

echo "==== Modules loaded ===="
module list
which mpicxx
which mpirun

# 2. Generate dataset via Python
echo "==== Generating data ===="
python3 scripts/synt_data.py --points 900 --type mixed --name mixed_dataset_1
if [ $? -ne 0 ]; then
    echo "Error: Data generation failed."
    exit 1
fi

# 3. Compile MPI version
echo "==== Compiling MPI version ===="
mkdir -p build
mpicxx -std=c++17 -O3 -I ./eigen_local \
    src/mainMPI.cpp src/SpectralClusteringMPI.cpp \
    -o build/spectral_mpi
if [ $? -ne 0 ]; then
    echo "Error: MPI compilation failed."
    exit 1
fi

# 4. Run MPI binary with 4 processes
# Force 1 thread per process to simulate sequential Eigen execution
export OMP_NUM_THREADS=1
echo "==== Running MPI version ===="
mpirun -np 4 ./build/spectral_mpi data/mixed_dataset_1.csv
if [ $? -ne 0 ]; then
    echo "Error: MPI execution failed."
    exit 1
fi

# 5. Visualization for MPI version
echo "==== Generating visualization (MPI) ===="
python3 scripts/plot_clusters.py --data data/mixed_dataset_1.csv --labels data/mixed_dataset_1_labels.csv

echo "==== Job completed successfully ===="
