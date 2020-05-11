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
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
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
    int recv_buf;
    MPI_Reduce(&count, &recv_buf, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank==0){
        end = MPI_Wtime();
        // Estimate Pi and display the result
        pi = ((double)recv_buf / (double)NUM_ITER) * 4.0;
        printf("The result is %f\n", pi);
        printf("The elapsed time is %f\n", end-start);
    }
    MPI_Finalize();
    
    return 0;
}


