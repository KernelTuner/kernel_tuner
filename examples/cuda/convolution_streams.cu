/*
 * Example program to demonstrate how to use the kernel tuner to tune
 * parameters in the host code of GPU programs, such as the number of 
 * streams
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#ifndef num_streams
    #define num_streams 1
#endif
#ifndef grid_size_x
    #define grid_size_x 1
#endif
#ifndef grid_size_y
    #define grid_size_y 1
#endif


#include "convolution.cu"



float convolution_streams(float *output, float *input, float *filter) {

    float *h_output;
    float *h_input;
    float *d_output;
    float *d_input;
    cudaError_t err;

    err = cudaHostAlloc((void **)&h_input, input_width*input_height*sizeof(float), cudaHostAllocDefault);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaHostAlloc: %s\n", cudaGetErrorString(err));
    }
    err = cudaHostAlloc((void **)&h_output, image_width*image_height*sizeof(float), cudaHostAllocDefault);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaHostAlloc: %s\n", cudaGetErrorString(err));
    }
    memcpy(h_input, input, input_width*input_height*sizeof(float));
    memcpy(h_output, output, image_width*image_height*sizeof(float));


    err = cudaMalloc((void **)&d_output, image_width*image_height*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaMalloc: %s\n", cudaGetErrorString(err));
    }
    err = cudaMalloc((void **)&d_input, input_width*input_height*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaMalloc: %s\n", cudaGetErrorString(err));
    }

    cudaStream_t stream[num_streams];
    cudaEvent_t event_htod[num_streams];
    for (int i=0; i<num_streams; i++) {
        err = cudaStreamCreate(&stream[i]);
        if (err != cudaSuccess) {
            fprintf(stderr, "Error in cudaStreamCreate: %s\n", cudaGetErrorString(err));
        }
        err = cudaEventCreate(&event_htod[i]);
        if (err != cudaSuccess) {
            fprintf(stderr, "Error in cudaEventCreate: %s\n", cudaGetErrorString(err));
        }
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

    dim3 threads(block_size_x, block_size_y, block_size_z);
    dim3 grid(grid_size_x, grid_size_y);

    //lines per stream, input data per stream, and border size
    int lps = (image_height / num_streams);
    int dps = lps * input_width;
    int border = border_height * input_width;

    //start timing
    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);

    err = cudaMemcpyToSymbolAsync(d_filter, filter, filter_width*filter_height*sizeof(float), 0, cudaMemcpyHostToDevice, stream[0]);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaMemcpyToSymbolAsync: %s\n", cudaGetErrorString(err));
    }

    //streamed copy of input data with strict order among streams, stream[0] also copies border
    for (int k=0; k<num_streams; k++) {
        if (k == 0) {
            err = cudaMemcpyAsync(d_input, h_input, border + dps*sizeof(float), cudaMemcpyHostToDevice, stream[k]);
        }
        else {
            err = cudaStreamWaitEvent(stream[k], event_htod[k-1], 0);
            if (err != cudaSuccess) {
                fprintf(stderr, "Error in cudaStreamWaitEvent htod k-1: %s\n", cudaGetErrorString(err));
            }
            err = cudaMemcpyAsync(d_input +border+k*dps, h_input +border+k*dps, dps*sizeof(float), cudaMemcpyHostToDevice, stream[k]);
        }
        if (err != cudaSuccess) {
            fprintf(stderr, "Error in cudaMemcpyHostToDevice: %s\n", cudaGetErrorString(err));
        }

        err = cudaEventRecord(event_htod[k], stream[k]);
        if (err != cudaSuccess) {
            fprintf(stderr, "Error in cudaEventRecord htod: %s\n", cudaGetErrorString(err));
        }
    }

    //start the kernel in each stream
    for (int k=0; k<num_streams; k++) {
        convolution_kernel<<<grid, threads, 0, stream[k]>>>(d_output+k*lps*image_width, d_input +k*dps, filter);
    }

    //streamed copy of the output data back to the host
    for (int k=0; k<num_streams; k++) {
        err = cudaMemcpyAsync(h_output + k*lps*image_width, d_output + k*lps*image_width, lps*image_width*sizeof(float), cudaMemcpyDeviceToHost, stream[k]);
        if (err != cudaSuccess) {
            fprintf(stderr, "Error in cudaMemcpyDeviceToHost: %s\n", cudaGetErrorString(err));
        }
    }    

    //mark the end of the computation
    cudaEventRecord(stop, 0);

    //wait for all to finish and get time
    cudaDeviceSynchronize();
    float time = 0.0;
    cudaEventElapsedTime(&time, start, stop);

    memcpy(output, h_output, image_width*image_height*sizeof(float));

    //make sure there have been no errors
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error at the end of convolution_streams: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    //cleanup
    cudaFreeHost(h_output);
    cudaFreeHost(h_input);
    cudaFree(d_output);
    cudaFree(d_input);
    for (int k=0; k<num_streams; k++) {
        cudaStreamDestroy(stream[k]);
        cudaEventDestroy(event_htod[k]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time;
}

