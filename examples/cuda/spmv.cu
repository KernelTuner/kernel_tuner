#ifndef block_size_x
    #define block_size_x 64
#endif
#ifndef threads_per_row
    #define threads_per_row 32
#endif
#ifndef read_only
    #define read_only 1
#endif


#define USE_READ_ONLY_CACHE read_only
#if USE_READ_ONLY_CACHE == 1
#define LDG(x) __ldg(x)
#elif USE_READ_ONLY_CACHE == 0
#define LDG(x) *(x)
#endif

#define warp_size threads_per_row

__global__ void spmv_kernel(float *y, int *rows, int *cols, float* values, float *__restrict__ x, int nrows) {

    //global warp index and within warp
    int i = (blockIdx.x * block_size_x + threadIdx.x)/warp_size;
    int tx = threadIdx.x & (warp_size-1);

    float local_y = 0.0;
    if (i < nrows) {
        //for each element on row
        int start = rows[i]+tx;
        int end = rows[i+1];

        //computation with somewhat improved memory access pattern
        for (int j = start; j < end; j+=threads_per_row) {
            local_y += values[j] * LDG(x + cols[j]);
        }

        #if (threads_per_row == 32)
        //reduce result to single value per warp
        #pragma unroll
        for (int s=warp_size/2; s>0; s>>=1) {
            local_y += __shfl_xor_sync(0xffffffff, local_y, s);
        }
        #endif

        //write result
        if (tx == 0) {
            y[i] = local_y;
        }

    }
}



