#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef MATRIX_DIM
#define N MATRIX_DIM
#else
#define N 2048
#endif

#ifdef _OPENMP
#include <omp.h>
#endif
#define FactorIntToDouble 1.1;

double firstMatrix[N][N] = {0.0};
double secondMatrix[N][N] = {0.0};
double matrixMultiResult[N][N] = {0.0};

void matrixMulti() {
#pragma omp parallel for
  for (int row = 0; row < N; row++) {
    for (int col = 0; col < N; col++) {
      double resultValue = 0;
      for (int transNumber = 0; transNumber < N; transNumber++) {
        resultValue +=
            firstMatrix[row][transNumber] * secondMatrix[transNumber][col];
      }
      matrixMultiResult[row][col] = resultValue;
    }
  }
}

void matrixInit() {
  for (int row = 0; row < N; row++) {
    for (int col = 0; col < N; col++) {
      srand(row + col);
      firstMatrix[row][col] = (rand() % 10) * FactorIntToDouble;
      secondMatrix[row][col] = (rand() % 10) * FactorIntToDouble;
    }
  }
}

int main(char argc, char **argv) {
  struct timespec start, finish;
  int NUM_THREAD = (argc == 1) ? 1 : strtol(argv[1], NULL, 10);
  printf("Value of N: %d\n# of threads: %d\n", N, NUM_THREAD);

  matrixInit();
  omp_set_num_threads(NUM_THREAD);

  printf("[\x1B[33m!\x1B[0m] Starting matrix multiplication\n");
  clock_gettime(CLOCK_MONOTONIC, &start);
  matrixMulti();
  clock_gettime(CLOCK_MONOTONIC, &finish);

  long double diff = (finish.tv_sec - start.tv_sec) +
                     ((finish.tv_nsec - start.tv_nsec) / 1000000000.0L);

  printf("time: %Lf seconds\n", diff);
  return 0;
}
