#!/bin/bash

# Default number of processes
NP=4

# Default dataset
DATASET="scripts/data/mixed_dataset.csv"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --np)
            NP="$2"
            shift 2
            ;;
        --data)
            DATASET="$2"
            shift 2
            ;;
        *)
            echo "Unknown arg: $1"
            exit 1
            ;;
    esac
done

echo "==== Building MPI version ===="
mkdir -p build

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

if [ $? -ne 0 ]; then
    echo "MPI program failed!"
    exit 1
fi

echo ""
echo "==== Running visualization ===="
python3 scripts/visualize.py \
    --data scripts/data/mixed_dataset.csv \
    --labels data/mixed_dataset_labels_mpi.csv \
    --output plots/mpi_visualization.png
