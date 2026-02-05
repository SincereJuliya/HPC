#!/bin/bash
#PBS -N SpectralMPI_Strong
#PBS -l select=1:ncpus=64:mem=4gb
#PBS -l walltime=01:00:00
#PBS -q short_cpuQ

cd $PBS_O_WORKDIR

module purge
module load gcc91
module load mpich-3.2.1--gcc-9.1.0

# Compile MPI program
echo "==== Compiling MPI Version ===="
mpicxx -O3 -std=c++17 -I ./eigen_local \
    src/mainMPI.cpp src/SpectralClusteringMPI.cpp \
    -o spectral_mpi

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

OUTPUT_DIR="results/mpi_strong"
mkdir -p $OUTPUT_DIR

# List of datasets
DATASETS=("data/mixed_dataset_10k.csv" \
          "data/mixed_dataset_50k.csv" \
          "data/mixed_dataset_100k.csv" \
          "data/mixed_dataset_500k.csv" \
          "data/mixed_dataset_1M.csv")

# List of MPI ranks
RANKS=(1 2 4 8 16 32 64)

for DATASET in "${DATASETS[@]}"; do
    BASENAME=$(basename "$DATASET" .csv)

    if [ ! -f "$DATASET" ]; then
        echo "Warning: Dataset $DATASET not found. Skipping..."
        continue
    fi

    for r in "${RANKS[@]}"; do
        # --- Safety check for large datasets ---
        if [[ "$DATASET" == *"500k"* ]] && [ $r -gt 16 ]; then
            echo "Skipping $r ranks for $DATASET to prevent MPI_Reduce truncation."
            continue
        fi
        if [[ "$DATASET" == *"1M"* ]] && [ $r -gt 8 ]; then
            echo "Skipping $r ranks for $DATASET to prevent MPI_Reduce truncation."
            continue
        fi

        echo "==== Running MPI Strong Scaling: $r ranks on $DATASET ===="
        mpirun -np $r ./spectral_mpi "$DATASET" 6 10 0.05 5 \
            > "$OUTPUT_DIR/log_${BASENAME}_${r}ranks.txt"
    done
done

echo "==== MPI Strong Scaling Finished. Logs in $OUTPUT_DIR ===="