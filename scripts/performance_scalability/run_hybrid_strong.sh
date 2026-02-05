#!/bin/bash
#PBS -N SpectralHybrid_Strong
#PBS -l select=16:ncpus=64:mem=4gb
#PBS -l walltime=02:00:00
#PBS -q short_cpuQ

# Move to working directory
cd $PBS_O_WORKDIR

# Load modules
module purge
module load gcc91
module load mpich-3.2.1--gcc-9.1.0

# Compile hybrid MPI + OpenMP program
echo "==== Compiling Hybrid MPI Version ===="
mpicxx -O3 -std=c++17 -fopenmp -I ./eigen_local \
    src/mainMPI.cpp src/SpectralClusteringMPI.cpp \
    -o spectral_hybrid

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

# --- Datasets list ---
DATASETS=("data/mixed_dataset_10k.csv" \
          "data/mixed_dataset_50k.csv" \
          "data/mixed_dataset_100k.csv" \
          "data/mixed_dataset_500k.csv" \
          "data/mixed_dataset_1M.csv")

OUTPUT_DIR="results/hybrid_strong"
mkdir -p $OUTPUT_DIR

# --- MPI ranks and OpenMP threads ---
RANKS_SMALL=(1 2 4 8 16)  # safe for small datasets
RANKS_LARGE=(1 2 4 8)     # avoid large ranks for big datasets
OMP_THREADS=4
export OMP_NUM_THREADS=$OMP_THREADS

# Run all datasets
for DATASET in "${DATASETS[@]}"; do
    BASENAME=$(basename "$DATASET" .csv)

    if [ ! -f "$DATASET" ]; then
        echo "Warning: Dataset $DATASET not found. Skipping..."
        continue
    fi

    # Choose ranks depending on dataset size
    if [[ "$DATASET" == *"500k"* || "$DATASET" == *"1M"* ]]; then
        RANKS_TO_USE=("${RANKS_LARGE[@]}")
    else
        RANKS_TO_USE=("${RANKS_SMALL[@]}")
    fi

    for r in "${RANKS_TO_USE[@]}"; do
        echo "==== Running Hybrid Strong Scaling: $r MPI ranks x $OMP_THREADS threads on $DATASET ===="
        mpirun -np $r ./spectral_hybrid "$DATASET" 6 10 0.05 5 \
            > "$OUTPUT_DIR/log_${BASENAME}_${r}ranks.txt"
    done
done

echo "==== Hybrid Strong Scaling Finished. Logs in $OUTPUT_DIR ===="