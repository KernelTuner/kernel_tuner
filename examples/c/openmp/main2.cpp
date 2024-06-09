#include <stdlib.h>
#include <omp.h>
#define VECTOR_SIZE 100000000

#include <chrono>
extern "C" float vector_add(float * restrict a, float * restrict b, float * restrict c, int size){
  #pragma omp target enter data map(to: a[0:100000000])
  #pragma omp target enter data map(to: b[0:100000000])
  #pragma omp target enter data map(to: c[0:100000000])
  auto kt_timing_start = std::chrono::steady_clock::now();
    #pragma omp target  parallel for num_threads(nthreads)
    for ( int i = 0; i < size; i++ ) {
      c[i] = a[i] + b[i];
    }

  auto kt_timing_end = std::chrono::steady_clock::now();
  std::chrono::duration<float, std::milli> elapsed_time = kt_timing_end - kt_timing_start;
  #pragma omp target exit data map(from: a[0:100000000])
  #pragma omp target exit data map(from: b[0:100000000])
  #pragma omp target exit data map(from: c[0:100000000])

  return elapsed_time.count();

}

extern "C" float vector_add(float * restrict a, float * restrict b, float * restrict c, int size);

int main() {
  float a[VECTOR_SIZE];
  float b[VECTOR_SIZE];
  float c[VECTOR_SIZE];

  for (int i =0; i < VECTOR_SIZE; i++) {
    a[i] = b[i] = c[i] = 0;
  }

  vector_add(a, b, c, VECTOR_SIZE);

  // std::cout<< "Done...\n";

	free(a);
	free(b);
	free(c);
}