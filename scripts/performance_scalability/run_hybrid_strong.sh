#!/bin/bash
#PBS -N SpectralHybrid_Strong
#PBS -l select=1:ncpus=64:mem=32gb
#PBS -l walltime=02:00:00
#PBS -q short_cpuQ

cd $PBS_O_WORKDIR

# Clean environment and load modules
module purge
module load gcc91
module load mpich-3.2.1--gcc-9.1.0

# Define the number of OpenMP threads per MPI rank
export OMP_NUM_THREADS=4

# Create results directory
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

# MPI Ranks grid (r * 4 threads = Total Cores)
RANKS=(1 2 4 8 12 16)

for DATASET in "${DATASETS[@]}"; do
    if [ ! -f "$DATASET" ]; then
        echo "Warning: $DATASET not found. Skipping..."
        continue
    fi
    
    BASENAME=$(basename "$DATASET" .csv)
    
    for r in "${RANKS[@]}"; do
        TOTAL_CORES=$((r * 4))
        echo "==== Running Hybrid Strong: $r Ranks ($TOTAL_CORES Cores) on $BASENAME ===="
        
        # Use the pre-compiled hybrid binary
        mpirun -np $r ./spectral_hybrid "$DATASET" 6 10 0.05 5 \
            > "$OUTPUT_DIR/log_${BASENAME}_${TOTAL_CORES}cores_hybrid.txt"
    done
done

echo "==== All Hybrid Strong jobs completed ===="