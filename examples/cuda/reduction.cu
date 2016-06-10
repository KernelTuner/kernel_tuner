#include <stdio.h>

#ifndef vector
#define vector 1
#endif 

#if (vector==1)
#define floatvector float
#elif (vector == 2)
#define floatvector float2
#elif (vector == 4)
#define floatvector float4
#endif

#if use_shuffle == 1
#define stop_loop 16
#elif use_shuffle == 0
#define stop_loop 0
#endif

__global__ void sum_floats(float *sum_global, floatvector *array, int n) {
    int ti = threadIdx.x;
    int x = blockIdx.x * block_size_x + threadIdx.x;
    int step_size = gridDim.x * block_size_x;
    float sum = 0.0f;

    //cooperatively iterate over input array with all thread blocks
    for (int i=x; i<n/vector; i+=step_size) {
        floatvector v = array[i];
        #if vector == 1
        sum += v;
        #elif vector == 2
        sum += v.x + v.y;
        #elif vector == 4
        sum += v.x + v.y + v.z + v.w;
        #endif
    }
    
    //reduce sum to single value (or last 32 in case of use_shuffle)
    __shared__ float sh_mem[block_size_x];
    sh_mem[ti] = sum;
    __syncthreads();
    #pragma unroll
    for (unsigned int s=block_size_x/2; s>stop_loop; s>>=1) {
        if (ti < s) {
            sh_mem[ti] += sh_mem[ti + s];
        }
        __syncthreads();
    }

    //reduce last 32 values to single value using warp shuffle instructions
    #if use_shuffle == 1
    if (ti < 32) {
        sum = sh_mem[ti];
        #pragma unroll
        for (unsigned int s=16; s>0; s>>=1) {
            sum += __shfl_down(sum, s);        
        }
    }
    #else
    sum = sh_mem[0];
    #endif

    //write back one value per thread block, run kernel again with one tread block
    if (ti == 0) {
        sum_global[blockIdx.x] = sum;
    }
}
