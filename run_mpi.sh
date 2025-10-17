#!/bin/bash
#PBS -N hello_mpi
#PBS -l nodes=1:ppn=4
#PBS -l walltime=00:05:00
#PBS -j oe
#PBS -o output.log

cd $PBS_O_WORKDIR
module load mpich-3.2.1--gcc-9.1.0
mpirun -np 4 ./hello_mpi
