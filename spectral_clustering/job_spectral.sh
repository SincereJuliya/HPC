#!/bin/bash
#PBS -N SpectralFinal
#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=00:10:00
#PBS -q short_cpuQ

cd $PBS_O_WORKDIR

# 1. Load modules (GCC 9 + MPI under GCC 9)
module purge
module load gcc91
module load mpich-3.2.1--gcc-9.1.0
module load python-3.10.14

echo "==== Modules loaded ===="
module list
which mpicxx

# 2. Generate data
echo "==== Generating data ===="
python3 scripts/synt_data.py --points 1000 --type mixed
if [ $? -ne 0 ]; then
    echo "Error: Data generation failed."
    exit 1
fi

# 3. Compile MPI version
echo "==== Compiling MPI version ===="
mpicxx -std=c++17 -O3 -o spectral_clustering_mpi src/main.cpp
if [ $? -ne 0 ]; then
    echo "Error: MPI compilation failed."
    exit 1
fi

# 4. Run MPI binary
echo "==== Running MPI version ===="
/opt/pbs/bin/mpiexe -n 4 ./spectral_clustering_mpi
if [ $? -ne 0 ]; then
    echo "Error: MPI execution failed."
    exit 1
fi

echo "==== Job completed successfully ===="
