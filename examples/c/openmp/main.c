#include <stdlib.h>
#include<stdio.h>
// #include <openacc.h>
#include<omp.h>
#include <time.h>


#define VECTOR_SIZE 1024*1024*1024

void vec_add(float *a, float *b, float *c) {
  printf("Let's go...\n");

  #pragma omp target enter data map(to: a[0:VECTOR_SIZE])
  #pragma omp target enter data map(to: b[0:VECTOR_SIZE])
  #pragma omp target enter data map(to: c[0:VECTOR_SIZE])

  clock_t start = clock();

  #pragma omp target
  {
    #pragma omp parallel for
    for (unsigned int i = 0; i < VECTOR_SIZE; i++ ) {
      c[i] = a[i] + b[i];

    }

    #pragma omp parallel for
    for (unsigned int i = VECTOR_SIZE - 1; i > 0; i-- ) {
      b[i] = a[i] + b[i];
    }
  }

  // #pragma omp parallel for
  // for (unsigned int i = vec_size - 1; i > 0; i-- ) {
  //   a[i] = a[i] + b[i];
  // }

  clock_t end = clock();
  float time_used = (float)(end - start) / CLOCKS_PER_SEC;
  printf("time = %f\n", time_used);

  #pragma omp target exit data map(from: a[0:VECTOR_SIZE])
  #pragma omp target exit data map(from: b[0:VECTOR_SIZE])
  #pragma omp target exit data map(from: c[0:VECTOR_SIZE])

  printf("method done..\n");
}

int main(void) {
	float * a = (float *) malloc(VECTOR_SIZE * sizeof(float));
	float * b = (float *) malloc(VECTOR_SIZE * sizeof(float));
	float * c = (float *) malloc(VECTOR_SIZE * sizeof(float));

  vec_add(a, b, c);

  printf("Done...\n");

	free(a);
	free(b);
	free(c);
}