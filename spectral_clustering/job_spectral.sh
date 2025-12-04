#!/bin/bash
#PBS -N SpectralFinal
#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=00:10:00
#PBS -q short_cpuQ

cd $PBS_O_WORKDIR

# 1. Carico il compilatore moderno (GCC 9.1 supporta C++17)
module load gcc91

# 2. Carico MPI (dopo il compilatore)
module load mpich-3.2

echo "Ambiente:"
module list
which mpicxx

# 3. Genero i dati (per sicurezza, nel caso mancassero)
# Nota: Assicurati che python3 funzioni, altrimenti 'module load python-3.x'
python3 scripts/synt_data.py --points 1000 --type mixed

echo "Compilazione ed Esecuzione..."
chmod +x mpi.sh
./mpi.sh --np 4