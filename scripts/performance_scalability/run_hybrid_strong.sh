#!/bin/bash
#PBS -N SpectralHybrid_Strong
#PBS -l select=1:ncpus=64:mem=32gb
#PBS -l walltime=02:00:00
#PBS -q short_cpuQ

# Move to the directory where the job was submitted
cd $PBS_O_WORKDIR

# Clean environment and load required modules
module purge
module load gcc91
module load mpich-3.2.1--gcc-9.1.0

# Define the number of OpenMP threads per MPI rank
export OMP_NUM_THREADS=4

echo "==== Compiling Hybrid Version (MPI + OpenMP) ===="
mpicxx -O3 -std=c++17 -fopenmp -I ./eigen_local \
    src/mainMPI.cpp src/SpectralClusteringMPI.cpp \
    -o spectral_hybrid

# Create directory for results if it doesn't exist
OUTPUT_DIR="results/hybrid_strong"
mkdir -p $OUTPUT_DIR

# List of datasets to test scalability
DATASETS=(
    "data/mixed_dataset_10k.csv" 
    "data/mixed_dataset_50k.csv" 
    "data/mixed_dataset_100k.csv" 
    "data/mixed_dataset_500k.csv" 
    "data/mixed_dataset_1M.csv"
    "data/mixed_dataset_2M.csv"
)

# MPI Ranks: total cores will be r * 4 (e.g., 16 ranks = 64 cores)
RANKS=(1 2 4 8 12 16)

for DATASET in "${DATASETS[@]}"; do
    # Check if dataset file exists before running
    if [ ! -f "$DATASET" ]; then
        echo "Warning: $DATASET not found. Skipping..."
        continue
    fi
    
    BASENAME=$(basename "$DATASET" .csv)
    
    for r in "${RANKS[@]}"; do
        TOTAL_CORES=$((r * 4))
        echo "==== Running Hybrid: $r Ranks ($TOTAL_CORES Cores) on $BASENAME ===="
        
        # Parameters: dataset, clusters=6, max_iter=10, sigma=0.05, steps=5
        mpirun -np $r ./spectral_hybrid "$DATASET" 6 10 0.05 5 \
            > "$OUTPUT_DIR/log_${BASENAME}_${TOTAL_CORES}cores_hybrid.txt"
    done
done

echo "==== All Hybrid jobs completed ===="