#!/bin/bash
#PBS -N SpectralFinal
#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=00:10:00
#PBS -q short_cpuQ

cd $PBS_O_WORKDIR

# 1. Load modules
module load gcc91
module load mpich-3.2
module load python/3.x  # Replace with the correct Python version

# Ensure correct GCC version is used
export PATH=/apps/gcc-9.1.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/gcc-9.1.0/lib64:$LD_LIBRARY_PATH

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

# 3. Compile and run MPI version
echo "==== Compiling and running ===="
chmod +x mpi.sh
./mpi.sh
if [ $? -ne 0 ]; then
    echo "Error: MPI execution failed."
    exit 1
fi

echo "==== Job completed successfully ===="