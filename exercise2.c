#include <omp.h>
#include <stdlib.h>
#include <sys/time.h>

#define N 1000000

#define THREADS 12
#define PAD 3

void serial(double x[]);
void parallel(double x[]);
double mysecond();
void output(double max, int maxloc, double time);

int main(){
    srand(0);
    double x[N];
    for (int i = 0; i < N; i++){ 
        x[i] = ((double)(rand()) / RAND_MAX)*((double)(rand()) / RAND_MAX)*((double)(rand()) / RAND_MAX)*1000;
    }

    // serial(x);
    // parallel1(x);
    parallel2(x);
    return 0;
}



double mysecond(){
  struct timeval tp;
  struct timezone tzp;
  int i;
  

  i = gettimeofday(&tp,&tzp);
  return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

void output(double max, int maxloc, double time){
    printf("Max: %f ; Loc: %d \n", max, maxloc);
    printf("Time: %f s\n", time);
}

void serial(double x[]){
    double t = omp_get_wtime();
    double max = 0.0;
    int maxloc = 0;
    for (int i = 0; i < N; i++){
        if (x[i] > max){
            max = x[i];
            maxloc = i;
        }
    } 
    double t2 = omp_get_wtime();

    output(max, maxloc, t2-t);
}

void parallel1(double x[]){
    double t = mysecond();
    double max = 0.0;
    int maxloc = 0;
    // omp_set_num_threads(THREADS);
    #pragma omp parallel for num_threads(THREADS) schedule(auto)
    for (int i = 0; i < N; i++){
        if (x[i] > max){
            #pragma omp critical 
            {
                if (x[i] > max){
                    max = x[i];
                    maxloc = i;
                }
            }
                
        }
    } 
    double t2 = mysecond();

    output(max, maxloc, t2-t);
}

void parallel2(double x[]){
    double max[THREADS*PAD], maxF;
    int maxloc[THREADS*PAD], mlocF;
    memset(max, 0, THREADS*PAD*sizeof(double));
    memset(maxloc, 0, THREADS*PAD*sizeof(int));
    double t = omp_get_wtime();
    #pragma omp parallel for num_threads(THREADS) schedule(auto)
        for (int i = 0; i < N; i++){
            if (x[i] > max[omp_get_thread_num()*PAD]){
                max[omp_get_thread_num()*PAD] = x[i];
                maxloc[omp_get_thread_num()*PAD] = i;
            }
            
        }
    double t2 = omp_get_wtime();
    maxF = 0.0;
    mlocF = 0;
    for (int i = 0; i < THREADS*PAD; i+=PAD){
        if (max[i] > maxF){
            maxF = max[i];
            mlocF = maxloc[i];
        }
    }

    output(maxF, mlocF, t2-t);
}
