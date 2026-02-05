#!/bin/bash
#PBS -N SpectralMPI_Strong
#PBS -l select=1:ncpus=64:mem=32gb
#PBS -l walltime=01:00:00
#PBS -q short_cpuQ

cd $PBS_O_WORKDIR

module purge
module load gcc91
module load mpich-3.2.1--gcc-9.1.0

# Each MPI process must use exactly 1 core for pure MPI test
export OMP_NUM_THREADS=1

echo "==== Compiling MPI Version ===="
mpicxx -O3 -std=c++17 -fopenmp -I ./eigen_local \
    src/mainMPI.cpp src/SpectralClusteringMPI.cpp \
    -o spectral_mpi

OUTPUT_DIR="results/mpi_strong"
mkdir -p $OUTPUT_DIR

DATASETS=("data/mixed_dataset_10k.csv" "data/mixed_dataset_100k.csv" "data/mixed_dataset_1M.csv")
RANKS=(1 2 4 8 16 32 64)

for DATASET in "${DATASETS[@]}"; do
    BASENAME=$(basename "$DATASET" .csv)
    for r in "${RANKS[@]}"; do
        echo "==== Running MPI Strong: $r ranks on $DATASET ===="
        mpirun -np $r ./spectral_mpi "$DATASET" 6 10 0.05 5 \
            > "$OUTPUT_DIR/log_${BASENAME}_${r}ranks.txt"
    done
done