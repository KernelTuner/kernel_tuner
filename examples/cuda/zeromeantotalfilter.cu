/*
* Copyright 2015 Netherlands eScience Center, VU University Amsterdam, and Netherlands Forensic Institute
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

/**
 * This file contains CUDA kernels for applying a zero-mean total
 * filter to a PRNU pattern, as proposed by:
 * M. Chen et al. "Determining image origin and integrity using sensor
 * noise", IEEE Trans. Inf. Forensics Secur. 3 (2008) 74-90.
 *
 * The Zero Mean filter ensures that even and uneven subsets of columns
 * and rows in a checkerboard pattern become zero to remove any linear
 * patterns in the input.
 *
 * To apply the complete filter:
 *  computeMeanVertically(h, w, input);
 *  transpose(h, w, input);
 *  computeMeanVertically(h, w, input);
 *  transpose(h, w, input);
 *
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * @version 0.1
 */

//function interfaces to prevent C++ garbling the kernel names
extern "C" {
	__global__ void computeMeanVertically(int h, int w, float* input);
	__global__ void transpose(int h, int w, float* output, float* input);
}


/*
 * This function applies the Zero Mean filter vertically.
 *
 * Setup this kernel as follows:
 * gridDim.x = ceil ( w / (blockDim.x) )
 * gridDim.y = 1
 *
 * blockDim.x (block_size_y) = multiple of 32
 * blockDim.y (block_size_y) = power of 2
 */
//#define block_size_y 1
//#define block_size_y 256
__global__ void computeMeanVertically(int h, int w, float* input) {
	int j = threadIdx.x + blockIdx.x * block_size_x;
	int ti = threadIdx.y;
	int tj = threadIdx.x;

	if (j < w) {
		float sumEven = 0.0f;
		float sumOdd = 0.0f;

		//iterate over vertical domain
		for (int i = 2*ti; i < h-1; i += 2*block_size_y) {
			sumEven += input[i*w+j];
			sumOdd += input[(i+1)*w+j];
		}
		if (ti == 0 && h & 1) { //if h is odd
			sumEven += input[(h-1)*w+j];
		}

		//write local sums into shared memory
		__shared__ float shEven[block_size_y][block_size_x];
		__shared__ float shOdd[block_size_y][block_size_x];

		shEven[ti][tj] = sumEven;
		shOdd[ti][tj] = sumOdd;
		__syncthreads();

		//reduce local sums
		for (unsigned int s=block_size_y/2; s>0; s>>=1) {
			if (ti < s) {
				shEven[ti][tj] += shEven[ti + s][tj];
				shOdd[ti][tj] += shOdd[ti + s][tj];
			}
			__syncthreads();
		}

		//compute means
		float meanEven = shEven[0][tj] / ((h + 1) / 2);
		float meanOdd = shOdd[0][tj] / (h / 2);

		//iterate over vertical domain
		for (int i = 2*ti; i < h-1; i += 2*block_size_y) {
			input[i*w+j] -= meanEven;
			input[(i+1)*w+j] -= meanOdd;
		}
		if (ti == 0 && h & 1) { //if h is odd
			input[(h-1)*w+j] -= meanEven;
		}
	}
}
__global__ void computeMeanVertically_naive(int h, int w, float* input) {
	int j = threadIdx.x + blockIdx.x * blockDim.x;

	if (j < w) {
		float sumEven = 0.0f;
		float sumOdd = 0.0f;

		//iterate over vertical domain
		for (int i = 0; i < h-1; i += 2) {
			sumEven += input[i*w+j];
			sumOdd += input[(i+1)*w+j];
		}
		if (h & 1) { //if h is odd
			sumEven += input[(h-1)*w+j];
		}

		//compute means
		float meanEven = sumEven / ((h + 1) / 2);
		float meanOdd = sumOdd / (h / 2);

		//iterate over vertical domain
		for (int i = 0; i < h-1; i += 2) {
			input[i*w+j] -= meanEven;
			input[(i+1)*w+j] -= meanOdd;
		}
		if (h & 1) { //if h is odd
			input[(h-1)*w+j] -= meanEven;
		}
	}
}



/*
 * Naive transpose kernel
 *
 * gridDim.x = w / blockDim.x  (ceiled)
 * gridDim.y = h / blockDim.y  (ceiled)
 */
__global__ void transpose(int h, int w, float* output, float* input) {
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;

	if (j < w && i < h) {
		output[j*h+i] = input[i*w+j];
	}
}






























