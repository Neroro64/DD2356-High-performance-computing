
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
    
    // Calculate PI following a Monte Carlo method
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

    if (rank==0) {
        int c;
        MPI_Recv(&c, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        count +=c;
        MPI_Recv(&c, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        count +=c;
        
        end = MPI_Wtime();
        pi = ((double)count / (double)NUM_ITER) * 4.0;
        printf("The result is %f\n", pi);
        printf("The elapsed time is %f\n", end-start);
    }
    else{
        int r = rank*2;
        int c;
            
        if (r < size-1){
            MPI_Recv(&c, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            count += c;
        }
        if (r+2 < size-1){
            MPI_Recv(&c, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            count += c;
        }
        MPI_Send(&count, 1, MPI_INT, (int)((rank-1)/2), 0, MPI_COMM_WORLD);
    }
   
    // Estimate Pi and display the result
    MPI_Finalize();
    
    return 0;
}
