#!/bin/bash
#PBS -N SpectralMPI_Weak
#PBS -l select=64:ncpus=64:mem=4gb
#PBS -l walltime=02:00:00
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

OUTPUT_DIR="results/mpi_weak"
mkdir -p $OUTPUT_DIR

# --- Weak scaling: MPI ranks & corresponding dataset ---
RANKS=(1 2 4 8 16 32 64)
DATASETS=("data/weak_dataset_10k_1rank.csv" \
          "data/weak_dataset_20k_2rank.csv" \
          "data/weak_dataset_40k_4rank.csv" \
          "data/weak_dataset_80k_8rank.csv" \
          "data/weak_dataset_160k_16rank.csv" \
          "data/weak_dataset_320k_32rank.csv" \
          "data/weak_dataset_640k_64rank.csv")

for i in "${!DATASETS[@]}"; do
    DATASET=${DATASETS[$i]}
    r=${RANKS[$i]}
    BASENAME=$(basename "$DATASET" .csv)

    if [ ! -f "$DATASET" ]; then
        echo "Warning: Dataset $DATASET not found. Skipping..."
        continue
    fi

    # --- Safety limit: avoid too many ranks for small datasets ---
    if [[ "$DATASET" == *"10k"* ]] && [ $r -gt 4 ]; then
        echo "Skipping $r ranks for $DATASET to prevent MPI_Reduce truncation."
        continue
    fi
    if [[ "$DATASET" == *"20k"* ]] && [ $r -gt 8 ]; then
        echo "Skipping $r ranks for $DATASET to prevent MPI_Reduce truncation."
        continue
    fi
    if [[ "$DATASET" == *"40k"* ]] && [ $r -gt 16 ]; then
        echo "Skipping $r ranks for $DATASET to prevent MPI_Reduce truncation."
        continue
    fi
    # bigger datasets can run all ranks

    echo "==== Running MPI Weak Scaling with $r ranks on $DATASET ===="
    mpirun -np $r ./spectral_mpi "$DATASET" 6 10 0.05 5 \
        > "$OUTPUT_DIR/log_${BASENAME}_${r}ranks.txt"
done

echo "==== MPI Weak Scaling Finished. Logs in $OUTPUT_DIR ===="