#!/bin/bash
# Ricarico MPI per sicurezza
module load mpich-3.2 || true
# Definisco dataset e processi
NP=4
DATASET="scripts/data/mixed_dataset.csv"

echo "==== Building MPI version ===="
mkdir -p build

# DEBUG: Stampo quale compilatore sto per usare
echo "Compilatore C++ in uso: $MPICH_CXX"
$MPICH_CXX --version

# COMPILAZIONE
# Nota: Uso mpicxx ma lui userà il compilatore definito nella variabile ambiente
mpicxx -O3 -std=c++17 -I ./eigen_local \
    src/mainMPI.cpp src/SpectralClusteringMPI.cpp \
    -o build/spectral_mpi

if [ $? -ne 0 ]; then
    echo "MPI build failed!"
    exit 1
fi

echo ""
echo "==== Running MPI spectral clustering ===="
mpirun -np "$NP" ./build/spectral_mpi "$DATASET"