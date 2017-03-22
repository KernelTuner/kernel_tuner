#include <stdio.h>
#include <cuda.h>

#include "pnpoly.cu"

#ifndef grid_size_x
    #define grid_size_x 1
#endif
#ifndef grid_size_y
    #define grid_size_y 1
#endif

/*
 * This function contains the host code for benchmarking the cn_pnpoly CUDA kernel
 * Including the time spent on data transfers between host and device memory
 *
 * This host code uses device mapped host memory to overlap communication
 * between host and device with kernel execution on the GPU. Because each input
 * is read only once and each output is written only once, this implementation
 * almost fully overlaps all communication and the kernel execution time dominates
 * the total execution time.
 *
 * The code has the option to precompute all polygon line slopes on the CPU and
 * reuse those results on the GPU, instead of recomputing them on the GPU all
 * the time. The time spent on precomputing these values on the CPU is also 
 * taken into account by the time measurement in the code below. 
 *
 * This code was written for use with the Kernel Tuner. See: 
 *      https://github.com/benvanwerkhoven/kernel_tuner
 *
 * Author: Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 */
extern "C" float cn_pnpoly_host(int* bitmap, float2* points, float2* vertices, int n) {

    cudaError_t err;

    #if use_precomputed_slopes == 1
    float *h_slopes;
    err = cudaHostAlloc((void **)&h_slopes, VERTICES*sizeof(float), cudaHostAllocMapped);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaHostAlloc: %s\n", cudaGetErrorString(err));
    }
    #endif

    //create CUDA streams and events
    cudaStream_t stream[1];
    err = cudaStreamCreate(&stream[0]);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaStreamCreate: %s\n", cudaGetErrorString(err));
    }
    cudaEvent_t start;
    err = cudaEventCreate(&start);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaEventCreate: %s\n", cudaGetErrorString(err));
    }

    cudaEvent_t stop;
    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaEventCreate: %s\n", cudaGetErrorString(err));
    }

    //kernel parameters
    dim3 threads(block_size_x, block_size_y, block_size_z);
    dim3 grid(grid_size_x, grid_size_y);

    //start measuring time
    cudaDeviceSynchronize();
    cudaEventRecord(start, stream[0]);

    //transfer vertices to d_vertices
    err = cudaMemcpyToSymbolAsync(d_vertices, vertices, VERTICES*sizeof(float2), 0, cudaMemcpyHostToDevice, stream[0]);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaMemcpyToSymbolAsync: %s\n", cudaGetErrorString(err));
    }

    #if use_precomputed_slopes == 1
    //precompute the slopes and transfer to symbol d_slopes
    h_slopes[0] = (vertices[VERTICES-1].x - vertices[0].x) / (vertices[VERTICES-1].y - vertices[0].y);
    for (int i=1; i<VERTICES; i++) {
        h_slopes[i] = (vertices[i-1].x - vertices[i].x) / (vertices[i-1].y - vertices[i].y);
    }
    err = cudaMemcpyToSymbolAsync(d_slopes, h_slopes, VERTICES*sizeof(float), 0, cudaMemcpyHostToDevice, stream[0]);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaMemcpyToSymbolAsync: %s\n", cudaGetErrorString(err));
    }
    #endif

    //call the kernel
    cn_pnpoly<<<grid, threads, 0, stream[0]>>>(bitmap, points, n);  //using mapped memory

    //stop time measurement
    cudaEventRecord(stop, stream[0]);
    cudaDeviceSynchronize();
    float time = 0.0;
    cudaEventElapsedTime(&time, start, stop);

    //cleanup
    #if use_precomputed_slopes == 1
    cudaFreeHost(h_slopes);
    #endif
    cudaStreamDestroy(stream[0]);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *error_string = cudaGetErrorString(err);
        if (strncmp("too many resources requested for launch", error_string, 10) == 0) {
            time = -1.0;
        } else {
            fprintf(stderr, "Error after CUDA kernel: %s\n", error_string);
            exit(1);
        }
    }

    return time; //ms
}
