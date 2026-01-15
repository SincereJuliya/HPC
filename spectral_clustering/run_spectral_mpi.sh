#!/bin/bash
#PBS -N SpectralMPI
#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=00:10:00
#PBS -q short_cpuQ

cd $PBS_O_WORKDIR

module purge
module load gcc91
module load mpich-3.2

echo "==== Compiling MPI version ===="
mkdir -p build
mpicxx -O3 -std=c++17 -I ./eigen_local \
    src/mainMPI.cpp src/SpectralClusteringMPI.cpp \
    -o build/spectral_mpi

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "==== Running MPI version ===="
mpirun -np 4 ./build/spectral_mpi scripts/data/mixed_dataset.csv

echo "==== Job completed successfully ===="
