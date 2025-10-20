#!/bin/bash
#PBS -N mpi_bandwidth_pack
#PBS -l select=1:ncpus=2:mem=2gb
#PBS -l walltime=0:10:00
#PBS -q short_cpuQ

cd $PBS_O_WORKDIR
module load mpich-3.2
mpicc -O2 -o mpi_bandwidth mpi_bandwidth.c

mpirun.actual -np 2 ./mpi_bandwidth > mpi_results_pack.txt
