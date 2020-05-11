#include <mpi.h>
#include <stdio.h>

int main(int argc, char**argv){
    int rank, size, i, provided;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    printf("Hello world from rank %d from %d processes!\n", rank, size);
    
    MPI_Finalize();
}

/*
1. Compile with mpicc, run with mpiexec -n PROCESSES a.out
2. srun -n PROCESSES a.out
3. Use -n
4.  MPI_Comm_size(MPI_COMM_WORLD, &size)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank)
5. OpenMPI and MPICH
*/