#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define NUMSTEPS 1000000000

double area_approx(int s, int p, int r);

int main(void) {
    int size = 0, part = 0, rank = 0;
    
    struct timespec start, end;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    part = (int) (NUMSTEPS / size);
    double total_sum;
    double local_sum = area_approx(size, part, rank); 
    
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
     
    if (rank == 0) 
	printf("%.20f\n", total_sum);

    MPI_Finalize();
    clock_gettime(CLOCK_MONOTONIC, &end);
    u_int64_t diff = 1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
    
    printf("elapsed time = %llu nanoseconds\n", (long long unsigned int) diff);



}

double area_approx(int size, int part, int rank) {
        int i;
	int start = rank * part;
	int end = (size-1 == rank) ? NUMSTEPS : start + part; 
        double x, pi, sum = 0.0;
        double step = (1.0 - 0.0) / (double) NUMSTEPS; // b-a/n, where n is the # of partitions and b is the upperbound and a is the lowerbound
        
	x = 0.5 * step; // midpoint that intersects the figure
	if (rank > 0) 
	    x = (step * start) + step;
	
        for (i = start; i <= end; i++) {
                x+=step; // advance to the next midpoint
                sum += 4.0/(1.0+x*x); // calculate the value at the midpoint
        }

	return sum * step;
}
