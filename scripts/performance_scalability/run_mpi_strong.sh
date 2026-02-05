#!/bin/bash
#PBS -N SpectralMPI_Strong
#PBS -l select=1:ncpus=64:mem=32gb
#PBS -l walltime=02:00:00
#PBS -q short_cpuQ

# Move to the directory where the job was submitted
cd $PBS_O_WORKDIR

# Clean environment and load required modules
module purge
module load gcc91
module load mpich-3.2.1--gcc-9.1.0

# Ensure each MPI process uses only 1 thread
export OMP_NUM_THREADS=1

echo "==== Compiling Pure MPI Version ===="
mpicxx -O3 -std=c++17 -I ./eigen_local \
    src/mainMPI.cpp src/SpectralClusteringMPI.cpp \
    -o spectral_mpi

# Create directory for results
OUTPUT_DIR="results/mpi_strong"
mkdir -p $OUTPUT_DIR

# Ensure execution permission
chmod +x ./spectral_mpi

# List of datasets (same as in hybrid script)
DATASETS=(
    "data/mixed_dataset_10k.csv" 
    "data/mixed_dataset_50k.csv" 
    "data/mixed_dataset_100k.csv" 
    "data/mixed_dataset_500k.csv" 
    "data/mixed_dataset_1M.csv"
    "data/mixed_dataset_2M.csv"
)

# Ranks correspond directly to total cores used
RANKS=(1 2 4 8 16 24 32 48 64)

for DATASET in "${DATASETS[@]}"; do
    if [ ! -f "$DATASET" ]; then
        echo "Warning: $DATASET not found. Skipping..."
        continue
    fi
    
    BASENAME=$(basename "$DATASET" .csv)
    
    for r in "${RANKS[@]}"; do
        echo "==== Running Pure MPI: $r Cores on $BASENAME ===="
        
        # Parameters: dataset, clusters=6, max_iter=10, sigma=0.05, steps=5
        mpirun -np $r ./spectral_mpi "$DATASET" 6 10 0.05 5 \
            > "$OUTPUT_DIR/log_${BASENAME}_${r}cores_mpi.txt"
    done
done

echo "==== All MPI jobs completed ===="