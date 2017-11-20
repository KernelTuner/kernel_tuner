#define WIDTH 4096

__global__ void matmul_kernel(float *C, float *A, float *B) {

    __shared__ float sA[block_size_y*tile_size_y][block_size_x];
    __shared__ float sB[block_size_y*tile_size_y][block_size_x * tile_size_x];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * block_size_x * tile_size_x + threadIdx.x;
    int y = blockIdx.y * block_size_y * tile_size_y + threadIdx.y;
    int k, kb;

    float sum[tile_size_y][tile_size_x];

    for (k = 0; k < WIDTH; k += block_size_x) {

        __syncthreads ();
        #pragma unroll
        for (int i = 0; i < tile_size_y; i++) {
            sA[ty + block_size_y * i][tx] = A[y * WIDTH + block_size_y * i * WIDTH + k + tx];

            #pragma unroll
            for (int j = 0; j < tile_size_x; j++) {
                sB[ty + block_size_y * i][tx + j * block_size_x] = B[(k + ty + block_size_y * i) * WIDTH + x + j * block_size_x];
            }
        }
        __syncthreads ();

        //compute
        #pragma unroll
        for (kb = 0; kb < block_size_x; kb++) {

            #pragma unroll
            for (int i = 0; i < tile_size_y; i++) {
                #pragma unroll
                for (int j = 0; j < tile_size_x; j++) {
                        sum[i][j] += sA[ty + block_size_y * i][kb] * sB[kb][tx + j * block_size_x];
                    }
            }

        }

    }

    //store result
    #pragma unroll
    for (int i = 0; i < tile_size_y; i++) {
        #pragma unroll
        for (int j = 0; j < tile_size_x; j++) {
            C[y * WIDTH + x + block_size_y * i * WIDTH + j * block_size_x] = sum[i][j];
        }
    }

}
