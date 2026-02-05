#!/bin/bash
#PBS -N SpectralHybrid_Weak
#PBS -l select=1:ncpus=64:mem=32gb
#PBS -l walltime=01:00:00
#PBS -q short_cpuQ

cd $PBS_O_WORKDIR

# Clean environment and load modules
module purge
module load gcc91
module load mpich-3.2.1--gcc-9.1.0

# Fixed 4 threads per MPI rank
export OMP_NUM_THREADS=4
    
# Create results directory
OUTPUT_DIR="results/hybrid_weak"
mkdir -p $OUTPUT_DIR

RANKS=(1 2 4 8 16)
DATASETS=("data/weak_dataset_10k_1rank.csv" \
          "data/weak_dataset_20k_2rank.csv" \
          "data/weak_dataset_40k_4rank.csv" \
          "data/weak_dataset_80k_8rank.csv" \
          "data/weak_dataset_160k_16rank.csv")

for i in "${!DATASETS[@]}"; do
    r=${RANKS[$i]}
    DATASET=${DATASETS[$i]}
    
    if [ ! -f "$DATASET" ]; then
        echo "Warning: $DATASET not found. Skipping..."
        continue
    fi
    
    BASENAME=$(basename "$DATASET" .csv)
    echo "==== Running Hybrid Weak: $r Ranks x 4 Threads on $DATASET ===="
    
    # Run using the pre-compiled hybrid binary
    mpirun -np $r ./spectral_hybrid "$DATASET" 6 10 0.05 5 \
        > "$OUTPUT_DIR/log_${BASENAME}_${r}ranks.txt"
done

echo "==== All Hybrid Weak jobs completed ===="