#!/bin/bash
#PBS -N SpectralHybrid_Strong
#PBS -l select=1:ncpus=64:mem=32gb
#PBS -l walltime=01:00:00
#PBS -q short_cpuQ

cd $PBS_O_WORKDIR

module purge
module load gcc91
module load mpich-3.2.1--gcc-9.1.0

# Hybrid strategy: 4 threads per rank
export OMP_NUM_THREADS=4

echo "==== Compiling Hybrid Version ===="
mpicxx -O3 -std=c++17 -fopenmp -I ./eigen_local \
    src/mainMPI.cpp src/SpectralClusteringMPI.cpp \
    -o spectral_hybrid

OUTPUT_DIR="results/hybrid_strong"
mkdir -p $OUTPUT_DIR

DATASETS=("data/mixed_dataset_10k.csv" "data/mixed_dataset_100k.csv" "data/mixed_dataset_1M.csv")
# 16 ranks * 4 threads = 64 cores used
RANKS=(1 2 4 8 16)

for DATASET in "${DATASETS[@]}"; do
    BASENAME=$(basename "$DATASET" .csv)
    for r in "${RANKS[@]}"; do
        echo "==== Running Hybrid Strong: $r Ranks x 4 Threads on $DATASET ===="
        mpirun -np $r ./spectral_hybrid "$DATASET" 6 10 0.05 5 \
            > "$OUTPUT_DIR/log_${BASENAME}_${r}ranks.txt"
    done
done