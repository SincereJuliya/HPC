#!/bin/bash
#PBS -N SpectralMPI_Weak
#PBS -l select=1:ncpus=64:mem=32gb
#PBS -l walltime=01:00:00
#PBS -q short_cpuQ

cd $PBS_O_WORKDIR

module purge
module load gcc91
module load mpich-3.2.1--gcc-9.1.0

export OMP_NUM_THREADS=1

echo "==== Compiling MPI Version ===="
mpicxx -O3 -std=c++17 -fopenmp -I ./eigen_local \
    src/mainMPI.cpp src/SpectralClusteringMPI.cpp \
    -o spectral_mpi

OUTPUT_DIR="results/mpi_weak"
mkdir -p $OUTPUT_DIR

# Ensure execution permission
chmod +x ./spectral_mpi

RANKS=(1 2 4 8 16 32 64)
DATASETS=("data/weak_dataset_10k_1rank.csv" \
          "data/weak_dataset_20k_2rank.csv" \
          "data/weak_dataset_40k_4rank.csv" \
          "data/weak_dataset_80k_8rank.csv" \
          "data/weak_dataset_160k_16rank.csv" \
          "data/weak_dataset_320k_32rank.csv" \
          "data/weak_dataset_640k_64rank.csv")

for i in "${!DATASETS[@]}"; do
    r=${RANKS[$i]}
    DATASET=${DATASETS[$i]}
    BASENAME=$(basename "$DATASET" .csv)
    echo "==== Running MPI Weak: $r ranks on $DATASET ===="
    mpirun -np $r ./spectral_mpi "$DATASET" 6 10 0.05 5 \
        > "$OUTPUT_DIR/log_${BASENAME}_${r}ranks.txt"
done