test1 : 
    just qsub

test2 :
    module load mpich-3.2
    mpicc hello_world.cpp -o hello_world
    and then qsub