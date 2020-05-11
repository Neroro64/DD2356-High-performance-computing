#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define SEED     921
#define NUM_ITER 1000000000

int main(int argc, char* argv[])
{
    int rank, size, provided;
    float start, end;
    MPI_Win win;
    int *counts;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Win_allocate(size*sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &counts, &win);

    start = MPI_Wtime();
   
    int count = 0;
    double x, y, z, pi;
    int iterations = NUM_ITER/size;
    srand(SEED*rank); // Important: Multiply SEED by "rank" when you introduce MPI!
    
    // Calculate PI following a Monte Carlso method
    for (int iter = 0; iter < iterations; iter++)
    {
        // Generate random (X,Y) points
        x = (double)random() / (double)RAND_MAX;
        y = (double)random() / (double)RAND_MAX;
        z = sqrt((x*x) + (y*y));
        
        // Check if point is in unit circle
        if (z <= 1.0)
        {
            count++;
        }
    }
    MPI_Win_fence(0, win);
    MPI_Accumulate(&count, 1, MPI_INT, 0, 0, 1, MPI_INT, MPI_SUM, win);
    MPI_Win_fence(0, win);

    if (rank==0){
        end = MPI_Wtime();
        // Estimate Pi and display the result
        pi = ((double)counts[0] / (double)NUM_ITER) * 4.0;
        printf("The result is %f\n", pi);
        printf("The elapsed time is %f\n", end-start);
    }
    MPI_Win_free(&win);
    MPI_Finalize();
    
    return 0;
}


