#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

void matmul_serial(int size, unsigned long **A, unsigned long **B, unsigned long **C);
unsigned long **alloc_mat(int rows);
void free_mat(unsigned long ** mat);
double mysecond(){
  struct timeval tp;
  struct timezone tzp;
  int i;

  i = gettimeofday(&tp,&tzp);
  return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

int main(int argc, char* argv[]){
    
    int mat_size = atoi(argv[1]);
    
    double start,end;

    unsigned long ** A, **B, **C, **C_ord;

    start = mysecond();
    srand(0);
    A = alloc_mat(mat_size);
    B = alloc_mat(mat_size);
    C = alloc_mat(mat_size);
    for (int i = 0; i < mat_size; i++){            
        for (int j = 0; j < mat_size; j++){
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
        }
    }

    matmul_serial(mat_size, A, B, C);

    end = mysecond();
    printf("ORD: \n");
    printf("Serial: TIME: %f\n", end-start);
    printf("---------------------------\n");

    free_mat(A);
    free_mat(B);
    free_mat(C);
    
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
