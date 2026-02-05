#!/bin/bash
#PBS -N SpectralMPI_Weak
#PBS -l select=64:ncpus=64:mem=4gb
#PBS -l walltime=02:00:00
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

OUTPUT_DIR="results/mpi_weak"
mkdir -p $OUTPUT_DIR

# MPI ranks to test
RANKS=(1 2 4 8 16 32 64)

for r in "${RANKS[@]}"; do
    DATASET="data/mixed_dataset_100k_${r}.csv"   # dataset scaled with r
    if [ ! -f "$DATASET" ]; then
        echo "Dataset $DATASET not found! Skip."
        continue
    fi
    echo "==== Running MPI Weak Scaling with $r ranks ===="
    mpirun -np $r ./spectral_mpi "$DATASET" 6 10 0.05 5 \
        > "$OUTPUT_DIR/log_${r}ranks.txt"
done

echo "==== MPI Weak Scaling Finished. Logs in $OUTPUT_DIR ===="