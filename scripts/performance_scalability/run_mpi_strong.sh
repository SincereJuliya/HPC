#!/bin/bash
#PBS -N SpectralMPI_Strong
#PBS -l select=1:ncpus=64:mem=4gb
#PBS -l walltime=01:00:00
#PBS -q short_cpuQ

cd $PBS_O_WORKDIR

module purge
module load gcc91
module load mpich-3.2.1--gcc-9.1.0

mpicxx -O3 -std=c++17 -I ./eigen_local \
    src/mainMPI.cpp src/SpectralClusteringMPI.cpp \
    -o spectral_mpi

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

DATASET="data/mixed_dataset_100k.csv"
OUTPUT_DIR="results/mpi_strong"
mkdir -p $OUTPUT_DIR

# List of MPI ranks to test
RANKS=(1 2 4 8 16 32 64)

for r in "${RANKS[@]}"; do
    echo "==== Running MPI Strong Scaling with $r ranks ===="
    mpirun -np $r ./spectral_mpi "$DATASET" 6 10 0.05 5 \
        > "$OUTPUT_DIR/log_${r}ranks.txt"
done

echo "==== MPI Strong Scaling Finished. Logs in $OUTPUT_DIR ===="