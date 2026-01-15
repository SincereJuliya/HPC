#!/bin/bash

# 1. Load MPI (if available)
if command -v module &> /dev/null; then
    module load mpich-3.2 || true
fi

# 2. Define variables
NP=4
DATASET="scripts/data/mixed_dataset.csv"

echo "==== Compiling MPI version ===="
mkdir -p build

# Check compiler
if [ -z "$MPICH_CXX" ]; then
    export MPICH_CXX=g++
fi
echo "Using C++ compiler: $MPICH_CXX"
$MPICH_CXX --version

# Compilation
mpicxx -O3 -std=c++17 -I ./eigen_local \
    src/mainMPI.cpp src/SpectralClusteringMPI.cpp \
    -o build/spectral_mpi

if [ $? -ne 0 ]; then
    echo "Error: MPI compilation failed."
    exit 1
fi

echo "==== Running MPI version ===="
mpirun -np "$NP" ./build/spectral_mpi "$DATASET"

if [ $? -ne 0 ]; then
    echo "Error: MPI execution failed."
    exit 1
fi

echo "==== Execution completed successfully ===="