/**
 * The kernel is assumed to be tuned to each device by selecting
 * the best performing combination of thread block dimensions 
 * and tiling factors in X and Y. In this implementation tiling
 * in X increases the amount of work per thread block and tiling
 * in Y increases the amount of work per thread within the block. 
 * 
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * 
 */

#ifndef WIDTH
#define WIDTH 4096
#endif
/*
 * Optimized OpenCL kernel for matrix multiplication
 *
 * This kernel is optimized according to the directions given
 * in: "Better performance at lower occupancy" by V. Volkov,
 * GPU Technology Conference, GTC 2010.
 *
 * The thread block dimensions (block_size_x, block_size_y) 
 * and tiling factors (tile_size_x, tile_size_y) are to be
 * tuned towards each GPU. This kernel assumes that
 * block_size_x = block_size_y * tile_size_y.
 *
 * The kernel computes C=A*B, where A, B, and C are square
 * matrices with height and width equal to WIDTH
 */
#ifndef block_size_x
#define block_size_x 16
#endif
#ifndef block_size_y
#define block_size_y 16
#endif
#ifndef tile_size_x
#define tile_size_x 1
#endif
#ifndef tile_size_y
#define tile_size_y 1
#endif


__kernel void matmul_kernel(__global float *C, __global float *A, __global float *B) {

    __local float sA[block_size_y*tile_size_y][block_size_x];
    __local float sB[block_size_y*tile_size_y][block_size_x * tile_size_x];

    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int x = get_group_id(0) * block_size_x * tile_size_x + tx;
    int y = get_group_id(1) * block_size_y * tile_size_y + ty;
    int k, kb;

    float sum[tile_size_y][tile_size_x];

    for (k = 0; k < WIDTH; k += block_size_x) {

        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int i = 0; i < tile_size_y; i++) {
            sA[ty + block_size_y * i][tx] = A[y * WIDTH + block_size_y * i * WIDTH + k + tx];

            #pragma unroll
            for (int j = 0; j < tile_size_x; j++) {
            	sB[ty + block_size_y * i][tx + j * block_size_x] = B[(k + ty + block_size_y * i) * WIDTH + x + j * block_size_x];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

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
