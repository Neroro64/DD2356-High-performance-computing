#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#define D 2
#define X 0
#define Y 1
#define G 1

typedef double vec[D];

void init(vec *pos, vec *oP, vec *vel, double *m, int p);
void simple(vec *f, vec *pos, vec *oP, vec *vel, double *m, int p, int t, double dT);
void reduced(vec *f, vec *pos, vec *oP, vec *vel, double *m, int p, int t, double dT);
double mysecond();

double mysecond(){
  struct timeval tp;
  struct timezone tzp;
  int i;
  

  i = gettimeofday(&tp,&tzp);
  return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

int main(int argc, char** argv){
    int particles = atoi(argv[1]);
    int timesteps = atoi(argv[2]);
    double dT = atof(argv[3]);

    vec *forces = malloc(particles*sizeof(vec));
    vec *pos = malloc(particles * sizeof(vec));
    vec *old_pos = malloc(particles * sizeof(vec));
    vec *vel = malloc(particles * sizeof(vec));
    double *masses = malloc(particles * sizeof(double));

    init(pos, old_pos, vel, masses, particles);
    double t = omp_get_wtime();
    simple(forces, pos, old_pos, vel, masses, particles, timesteps, dT);
    // reduced(forces, pos, old_pos, vel, masses, particles, timesteps, dT);
    t = omp_get_wtime() - t;

    printf("Elapsed time: %f\n", t);
    
    free(forces);
    free(pos);
    free(old_pos);
    free(vel);
}


void init(vec *pos, vec *old_pos, vec *vel, double *masses, int p){
    for (int i = 0; i < p; i++){
        pos[i][X] = (rand() / (double)(RAND_MAX)) * 2 - 1;
        pos[i][Y] = (rand() / (double)(RAND_MAX)) * 2 - 1;
        
        old_pos[i][X] = pos[i][X];
        old_pos[i][Y] = pos[i][Y];

        vel[i][X] = (rand() / (double)(RAND_MAX)) * 2 -1;
        vel[i][Y] = (rand() / (double)(RAND_MAX)) * 2 -1;

        masses[i] = fabs((rand() / (double)(RAND_MAX)) * 2);
    }   
}

void simple(vec *f, vec *pos, vec *oP, vec *vel, double *m, int p, int t, double dT){
    double x_diff, y_diff, dist, dist_cubed, temp;

    #pragma omp parallel private(x_diff, y_diff, dist, dist_cubed, temp)
    {
        for (int step = 0; step < t; step++){
            // Reset the force matrix
            f = memset(f, 0, p*sizeof(vec));
            
            //Calc force
            #pragma omp for schedule(dynamic, 32) 
            for (int i = 0; i < p; i++){
                for (int j = 0; j < p; j++){
                    if (j == i)
                        continue;
                    
                    x_diff = pos[i][X] - pos[j][X];
                    y_diff = pos[i][Y] - pos[j][Y];
                    dist = sqrt(x_diff*x_diff + y_diff*y_diff);
                    dist_cubed = dist*dist*dist;
                    temp = G*m[i]*m[j] / dist_cubed;
                    f[i][X] -= temp * x_diff;
                    f[i][Y] -= temp * y_diff;

                }
            }
            
            #pragma omp barrier
            //Update positions and velocities
            #pragma omp for schedule(dynamic, 32) 
            for (int i = 0; i < p; i++){
                for (int j = 0; j < p; j++){
                    pos[i][X] += dT * vel[i][X];
                    pos[i][Y] += dT * vel[i][Y];
                    vel[i][X] += dT / m[i] * f[i][X];
                    vel[i][Y] += dT / m[i] * f[i][Y];
                }
            }
        }
    }
}

void reduced(vec *f, vec *pos, vec *oP, vec *vel, double *m, int p, int t, double dT){
    double x_diff, y_diff, dist, dist_cubed, f_qkX, f_qkY, temp;

    #pragma omp parallel private(x_diff, y_diff, dist, dist_cubed, f_qkX, f_qkY, temp)
    {
        for (int step = 0; step < t; step++){
            // Reset the force matrix
            f = memset(f, 0, p*sizeof(vec));
            
            //Calc force
            #pragma omp for schedule(dynamic, 32)
            for (int i = 0; i < p; i++){
                for (int j = i+1; j < p; j++){
                    
                    x_diff = pos[i][X] - pos[j][X];
                    y_diff = pos[i][Y] - pos[j][Y];
                    dist = sqrt(x_diff*x_diff + y_diff*y_diff);
                    dist_cubed = dist*dist*dist;
                    temp = G*m[i]*m[j] / dist_cubed;
                    f_qkX = temp * x_diff;
                    f_qkY = temp * y_diff;
                    f[i][X] += f_qkX;
                    f[i][Y] += f_qkY;
                    f[j][X] -= f_qkX;
                    f[j][Y] -= f_qkY;

                }
            }
            
            #pragma omp barrier

            //Update positions and velocities
            #pragma omp for schedule(dynamic, 32)
            for (int i = 0; i < p; i++){
                for (int j = 0; j < p; j++){
                    pos[i][X] += dT * vel[i][X];
                    pos[i][Y] += dT * vel[i][Y];
                    vel[i][X] += dT / m[i] * f[i][X];
                    vel[i][Y] += dT / m[i] * f[i][Y];
                }
            }
        }
    }
}