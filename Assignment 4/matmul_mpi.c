#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

void matmul(int size, unsigned long **A, unsigned long **B, unsigned long **C);
void matmul_serial(int size, unsigned long **A, unsigned long **B, unsigned long **C);
void tiling(int size, int tile_size, unsigned long **tiles[size], unsigned long **A);
void untiling(int size, int tile_size, unsigned long **tiles[size], unsigned long **A);
unsigned long **alloc_mat(int rows);
void free_mat(unsigned long **);
void free_tiles(int, unsigned long **[]);
void cpy(int, unsigned long **, unsigned long **);

int main(int argc, char* argv[]){
    
    int mat_size = atoi(argv[1]);
    
    int rank, size, provided;
    double start,end;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int sqrtS = sqrt(size);
    int tile_size = mat_size / sqrtS;
    int sqrT = tile_size*tile_size;
    assert(mat_size % tile_size == 0);

    int row = (int) (rank / sqrtS);
    int col = rank % (mat_size / tile_size);
    
    unsigned long **A_buffer = alloc_mat(tile_size);
    unsigned long **B_buffer = alloc_mat(tile_size);
    unsigned long **C_buffer = alloc_mat(tile_size);
    unsigned long **T_buffer = alloc_mat(tile_size);

    unsigned long ** A, **B, **C, **C_ord;

    MPI_Comm row_comm, col_comm;

    MPI_Comm_split(MPI_COMM_WORLD, row, col, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, col, row, &col_comm);

    if (rank == 0){
        start = MPI_Wtime();
        srand(0);
        A = alloc_mat(mat_size);
        B = alloc_mat(mat_size);
        C = alloc_mat(mat_size);
        C_ord = alloc_mat(mat_size);
        for (int i = 0; i < mat_size; i++){            
            for (int j = 0; j < mat_size; j++){
                A[i][j] = rand() % 10;
                B[i][j] = rand() % 10;
            }
        }

        unsigned long **A_tiles[size];
        unsigned long **B_tiles[size];
        for (int i = 0; i < size; i++){
            A_tiles[i] = alloc_mat(tile_size);
            B_tiles[i] = alloc_mat(tile_size);
        }

        tiling(mat_size, tile_size, A_tiles, A);
        tiling(mat_size, tile_size, B_tiles, B);
        for (int i = 1; i < size; i++){
            MPI_Send(&A_tiles[i][0][0], sqrT, MPI_UNSIGNED_LONG, i, 0, MPI_COMM_WORLD);
            MPI_Send(&B_tiles[i][0][0], sqrT, MPI_UNSIGNED_LONG, i, 0, MPI_COMM_WORLD);
        }
        
        cpy(tile_size, A_buffer, A_tiles[0]);
        cpy(tile_size, B_buffer, B_tiles[0]);
        
        free_tiles(size, A_tiles);
        free_tiles(size, B_tiles);
    }
    else{
        MPI_Recv(&A_buffer[0][0], sqrT, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&B_buffer[0][0], sqrT, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    int col_offset = 0;
    unsigned long **temp;
    for (int i = 0; i < sqrtS; i++){
        col_offset = (row+i) % sqrtS;
        if (col == col_offset){
            MPI_Bcast(&A_buffer[0][0], sqrT, MPI_UNSIGNED_LONG, col, row_comm);
            matmul(tile_size, A_buffer, B_buffer, C_buffer);   
        }
        else{
            MPI_Bcast(&T_buffer[0][0], sqrT, MPI_UNSIGNED_LONG, col_offset, row_comm);
            matmul(tile_size, T_buffer, B_buffer, C_buffer);   
        }
        
        int dest = row - 1;
        if (dest < 0)
            dest = sqrtS-1; 

        MPI_Sendrecv(&B_buffer[0][0], sqrT, MPI_UNSIGNED_LONG, dest, 0, 
            &T_buffer[0][0], sqrT, MPI_UNSIGNED_LONG, MPI_ANY_SOURCE, 0, col_comm, MPI_STATUS_IGNORE);
       
        temp = T_buffer;
        T_buffer = B_buffer;
        B_buffer = temp;
    }

    if (rank == 0){
        unsigned long **C_tiles[size];
        for (int i = 0; i < size; i++){
            C_tiles[i] = alloc_mat(tile_size);
        }
        for (int i = 1; i < size; i++)
            MPI_Recv(&C_tiles[i][0][0], sqrT, MPI_UNSIGNED_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        cpy(tile_size, C_tiles[0], C_buffer);
        untiling(mat_size, tile_size, C_tiles, C);
        
        free_mat(A);
        free_mat(B);
        free_mat(C);

        free_tiles(size, C_tiles);

        end = MPI_Wtime();
        printf("---------------------------------\n");
        printf("MATMUL: \n");

        printf("MPI TIME: %f\n", end-start);
        printf("---------------------------------\n");
    }
    else{
        MPI_Send(&C_buffer[0][0], sqrT, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD);
    }

    free_mat(A_buffer);
    free_mat(B_buffer);
    free_mat(C_buffer);
    free_mat(T_buffer);


    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Finalize();
    
}

void matmul(int size, unsigned long **A, unsigned long **B, unsigned long **C){
    #pragma omp parallel for schedule(auto)
    for (int i = 0; i < size; i+=2){
        for (int j = 0; j < size; j+=2){
            for (int k = 0; k < size; k++){
                C[i][j] += A[i][k] * B[k][j];
                C[i+1][j] += A[i+1][k] * B[k][j];
                C[i][j+1] += A[i][k] * B[k][j+1];
                C[i+1][j+1] += A[i+1][k] * B[k][j+1];
            }
        }
    }
}
void matmul_serial(int size, unsigned long **A, unsigned long **B, unsigned long **C){
    for (int i = 0; i < size; i+=2){
        for (int j = 0; j < size; j+=2){
            for (int k = 0; k < size; k++){
                C[i][j] += A[i][k] * B[k][j];
                C[i+1][j] += A[i+1][k] * B[k][j];
                C[i][j+1] += A[i][k] * B[k][j+1];
                C[i+1][j+1] += A[i+1][k] * B[k][j+1];
            }
        }
    }
}

void tiling(int size, int tile_size, unsigned long **tiles[size], unsigned long **A){
    int rp, cp, t;
    t = 0;
    for (rp = 0; rp < size; rp+=tile_size){
        for (cp = 0; cp < size; cp+=tile_size){
            #pragma omp parallel for schedule(auto) 
            for (int i = 0; i < tile_size; i++){
                for (int j = 0; j < tile_size; j++){
                    tiles[t][i][j] = A[rp+i][cp+j];
                }
            }
            t++;
        }
    } 
}
void untiling(int size, int tile_size, unsigned long **tiles[size], unsigned long **A){
    int rp, cp, t;
    t = 0;
    for (rp = 0; rp < size; rp+=tile_size){
        for (cp = 0; cp < size; cp+=tile_size){
            #pragma omp parallel for schedule(auto) 
            for (int i = 0; i < tile_size; i++){
                for (int j = 0; j < tile_size; j++){
                    A[rp+i][cp+j] = tiles[t][i][j];
                }
            }
            t++;
        }
    } 
}

unsigned long **alloc_mat(int rows) {
    unsigned long *elems = (unsigned long *)malloc(rows*rows*sizeof(unsigned long));
    unsigned long **mat= (unsigned long **)malloc(rows*sizeof(unsigned long*));
    for (int i=0; i<rows; i++){
        mat[i] = &(elems[rows*i]);
    }
    for (int i = 0; i < rows*rows; i++)
        elems[i] = 0;
    return mat;
}

void free_mat(unsigned long ** mat) {
    free(mat[0]);
    free(mat);
}

void free_tiles(int size, unsigned long **tiles[size]){
    for (int i = 0; i < size; i++)
        free_mat(tiles[i]);
}
void cpy(int s, unsigned long **dest, unsigned long **src){
    for (int i = 0; i < s; i++){
        for (int j = 0; j < s; j++)
            dest[i][j] = src[i][j];

    }
}
