#define WIDTH 4096

__global__ void matmul_kernel(float *C, float *A, float *B) {
    int x = blockIdx.x * block_size_x + threadIdx.x;
    int y = blockIdx.y * block_size_y + threadIdx.y;
    float sum = 0.0;

    for (int k=0; k<WIDTH; k++) {
        sum += A[y*WIDTH+k] * B[k*WIDTH+x];
    }

    C[y*WIDTH+x] = sum;
}
