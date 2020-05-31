# DD2356 Project

There are three source files. matmul_mpi.c will return the execution time of the entire program, including allocation, distribution, computation, collection. matmul_serial.c does exactly the same as the former but sequentially. matmul_ptr.c returns the computation time (matmul only) for both fox algorithm and sequential matmul. 

## To compile the code on Beskow:
    cc -o xxx.out matmul_xxx.c -lm -fopenmp

## To run the program:
    srun -n P xxx.out D

where P is the number of processes and D is the dimension of matrices.
