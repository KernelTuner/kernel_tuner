__global__ void matmul_kernel(float *C, float *A, float *B) {

    __shared__ float sA[block_size][block_size];
    __shared__ float sB[block_size][block_size];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * block_size + tx;
    int y = blockIdx.y * block_size + ty;

    float sum = 0.0;
    int k,kb;

    for (k=0; k<WIDTH; k+=block_size) {
        __synchthreads();
        sA[ty][tx] = A[y*WIDTH+k+tx];
        sB[ty][tx] = B[(k+ty)*WIDTH+x];
        __synchthreads();

        for (kb=0; kb<block_size; kb++) {
            sum += sA[ty][kb] * sB[kb][tx];
        }

    }

    C[y*WIDTH+x] = sum;
}

