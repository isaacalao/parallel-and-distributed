#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef MATRIX_DIM
#define N MATRIX_DIM
#else
#define N 2048
#endif

#define CHKSZ 10
#define FactorIntToDouble 1.1;

typedef enum { false, true } bool;

double firstMatrix[N][N] = {0.0};
double secondMatrix[N][N] = {0.0};
double matrixMultiResult[N][N] = {0.0};

void matrixMulti(bool transpose) {
  int chksz = CHKSZ;
  for (int row = 0; row < N;
       row += ((row + chksz >= N)) + ((row + chksz < N) * (chksz))) {
    for (int col = 0; col < N;
         col += ((col + chksz >= N)) + ((col + chksz < N) * (chksz))) {
      int max = (row + chksz >= N ? N : row + chksz);
#pragma omp parallel for
      for (int rowchk = row; rowchk < max; rowchk++) {
        int max2 = (col + chksz >= N ? N : col + chksz);
#pragma omp parallel for schedule(static, max)
        for (int colchk = col; colchk < max2; colchk++) {
          double resultValue = 0;
          for (int transNumber = 0; transNumber < N; transNumber++) {
            if (transpose) {
              resultValue += firstMatrix[rowchk][transNumber] *
                             secondMatrix[colchk][transNumber];
            } else {
              resultValue += firstMatrix[rowchk][transNumber] *
                             secondMatrix[transNumber][colchk];
            }
          }
          matrixMultiResult[rowchk][colchk] = resultValue;
        }
      }
    }
  }
}

void matrixInit(bool transpose) {
  printf((transpose) ? "Transposed [B] to enforce row-major order form.\n"
                     : "No matrix was transposed.\n");
  for (int row = 0; row < N; row++) {
    for (int col = 0; col < N; col++) {
      srand(row + col);
      firstMatrix[row][col] = (rand() % 10) * FactorIntToDouble;
      if (transpose) {
        secondMatrix[col][row] = (rand() % 10) * FactorIntToDouble;
      } else {
        secondMatrix[row][col] = (rand() % 10) * FactorIntToDouble;
      }
    }
  }
}

int main(char argc, char **argv) {
  bool transpose = true;
  struct timespec start, finish;
  int NUM_THREAD = (argc == 1) ? 1 : strtol(argv[1], NULL, 10);
  printf("Value of N: %d\n# of threads: %d\n", N, NUM_THREAD);

  matrixInit(transpose);
  omp_set_num_threads(NUM_THREAD);

  printf("[\x1B[33m!\x1B[0m] Starting matrix multiplication\n");
  clock_gettime(CLOCK_MONOTONIC, &start);
  matrixMulti(transpose);
  clock_gettime(CLOCK_MONOTONIC, &finish);

  long double diff = (finish.tv_sec - start.tv_sec) +
                     ((finish.tv_nsec - start.tv_nsec) / 1000000000.0L);
  printf("time: %Lf seconds\n", diff);

  return 0;
}
