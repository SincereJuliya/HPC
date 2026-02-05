#!/bin/bash
#PBS -N SpectralHybrid_Weak
#PBS -l select=16:ncpus=64:mem=4gb
#PBS -l walltime=02:00:00
#PBS -q short_cpuQ

cd $PBS_O_WORKDIR

module purge
module load gcc91
module load mpich-3.2.1--gcc-9.1.0

# Compile program
echo "==== Compiling Hybrid MPI Version ===="
mpicxx -O3 -std=c++17 -fopenmp -I ./eigen_local \
    src/mainMPI.cpp src/SpectralClusteringMPI.cpp \
    -o spectral_hybrid

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

OUTPUT_DIR="results/hybrid_weak"
mkdir -p $OUTPUT_DIR

# Set OpenMP threads
OMP_THREADS=4
export OMP_NUM_THREADS=$OMP_THREADS

# --- Weak scaling datasets & MPI ranks ---
DATASETS=("data/weak_dataset_10k_1rank.csv" \
          "data/weak_dataset_20k_2rank.csv" \
          "data/weak_dataset_40k_4rank.csv" \
          "data/weak_dataset_80k_8rank.csv" \
          "data/weak_dataset_160k_16rank.csv")

RANKS=(1 2 4 8 16)

# Loop over datasets
for i in "${!DATASETS[@]}"; do
    DATASET=${DATASETS[$i]}
    r=${RANKS[$i]}
    BASENAME=$(basename "$DATASET" .csv)

    if [ ! -f "$DATASET" ]; then
        echo "Warning: Dataset $DATASET not found. Skipping..."
        continue
    fi

    # For very large datasets, reduce max MPI ranks to avoid MPI_Reduce truncation
    if [[ "$DATASET" == *"160k"* ]]; then
        r=8  # safe for large dataset
        echo "Adjusting ranks to $r for $DATASET to prevent MPI errors."
    fi

    echo "==== Running Hybrid Weak Scaling: $r MPI ranks x $OMP_THREADS threads on $DATASET ===="
    mpirun -np $r ./spectral_hybrid "$DATASET" 6 10 0.05 5 \
        > "$OUTPUT_DIR/log_${BASENAME}_${r}ranks.txt"
done

echo "==== Hybrid Weak Scaling Finished. Logs in $OUTPUT_DIR ===="