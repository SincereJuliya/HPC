#!/bin/bash
#PBS -N mpi_bandwidth_spread
#PBS -l select=2:ncpus=1:mem=2gb
#PBS -l walltime=0:10:00
#PBS -q short_cpuQ

cd $PBS_O_WORKDIR
module load mpich-3.2
mpicc -O2 -o mpi_bandwidth mpi_bandwidth.cpp

# Запуск 2 процессов на двух узлах (spread)
mpirun.actual -np 2 ./mpi_bandwidth > mpi_results_spread.txt
