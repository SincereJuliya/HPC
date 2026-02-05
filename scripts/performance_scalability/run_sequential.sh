#!/bin/bash
#PBS -N SpectralSerial
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=00:10:00
#PBS -q short_cpuQ

cd $PBS_O_WORKDIR

module purge
module load gcc91
module load mpich-3.2.1--gcc-9.1.0

echo "==== Compiling Serial Version ===="
mpicxx -O3 -std=c++17 -I ./eigen_local \
    src/main.cpp src/SpectralClustering.cpp \
    -o spectral_seq

if [ $? -ne 0 ]; then
    echo "!!!! COMPILATION FAILED !!!!"
    exit 1
fi

echo "==== Running Serial Version ===="
DATASET="data/mixed_dataset_100k.csv"
OUTPUT_DIR="results/sequential"
mkdir -p $OUTPUT_DIR

if [ -f "$DATASET" ]; then
    ./spectral_seq "$DATASET" 6 10 0.05 5 > "$OUTPUT_DIR/log_serial.txt"
else
    echo "Error: Dataset $DATASET not found."
    exit 1
fi

echo "==== Serial Job completed. Output in $OUTPUT_DIR ===="