#!/bin/bash
#PBS -N SpectralHybrid_Strong
#PBS -l select=16:ncpus=64:mem=4gb
#PBS -l walltime=02:00:00
#PBS -q short_cpuQ

cd $PBS_O_WORKDIR

module purge
module load gcc91
module load mpich-3.2.1--gcc-9.1.0

mpicxx -O3 -std=c++17 -fopenmp -I ./eigen_local \
    src/mainMPI.cpp src/SpectralClusteringMPI.cpp \
    -o spectral_hybrid

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

DATASET="data/mixed_dataset_1M.csv"
OUTPUT_DIR="results/hybrid_strong"
mkdir -p $OUTPUT_DIR

RANKS=(1 2 4 8 16)
OMP_THREADS=4
export OMP_NUM_THREADS=$OMP_THREADS

for r in "${RANKS[@]}"; do
    echo "==== Running Hybrid Strong Scaling: $r MPI ranks x $OMP_THREADS threads ===="
    mpirun -np $r ./spectral_hybrid "$DATASET" 6 10 0.05 5 \
        > "$OUTPUT_DIR/log_${r}ranks.txt"
done

echo "==== Hybrid Strong Scaling Finished. Logs in $OUTPUT_DIR ===="