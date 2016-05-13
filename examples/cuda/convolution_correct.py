#!/usr/bin/env python
import numpy
import kernel_tuner

with open('convolution.cu', 'r') as f:
    kernel_string = f.read()

problem_size = (4096, 4096)
size = numpy.prod(problem_size)
input_size = ((problem_size[0]+16) * (problem_size[1]+16))

output = numpy.zeros(size).astype(numpy.float32)
input = numpy.random.randn(input_size).astype(numpy.float32)
input = input.reshape((problem_size[0]+16), (problem_size[1]+16))

#filter = numpy.random.randn(17*17).astype(numpy.float32)
filter = numpy.ones(17*17).astype(numpy.float32)

#from matplotlib import pyplot

input[:] = 0.0

for i in range(problem_size[0]):
    for j in range(problem_size[0]):
        input[i,j] = (i/50)%2 != (j/50)%2


cmem_args= {'d_filter': filter }

args = [output, input, filter]
tune_params = dict()
tune_params["block_size_x"] = [16*i for i in range(1,9)]
tune_params["block_size_y"] = [2**i for i in range(6)]

tune_params["tile_size_x"] = [2**i for i in range(3)]
tune_params["tile_size_y"] = [2**i for i in range(3)]

grid_div_x = ["block_size_x", "tile_size_x"]
grid_div_y = ["block_size_y", "tile_size_y"]

#compute answer using naive kernel
import pycuda.driver as drv
from pycuda.autoinit import context
from kernel_tuner.cuda import CudaFunctions

dev = CudaFunctions(device=0)
gpu_args = dev.create_gpu_args(args)
defs = "#define block_size_x 16 \n #define block_size_y 16 \n #define tile_size_x 1 \n #define tile_size_y 1 \n"
func = dev.compile("convolution_naive", defs+ kernel_string)
func(*gpu_args, block=(16,16,1), grid=(int(numpy.ceil(problem_size[0]/16.)), int(numpy.ceil(problem_size[1]/16.))))
output_image = numpy.zeros_like(output)
drv.memcpy_dtoh(output_image, gpu_args[0])
output_image = output_image.reshape(problem_size)
answer = [output_image, None, None]

#pyplot.imshow(output_image, cmap=pyplot.cm.bone)
#pyplot.show()


kernel_tuner.tune_kernel("convolution_kernel", kernel_string,
    problem_size, args, tune_params,
    grid_div_y=grid_div_y, grid_div_x=grid_div_x, verbose=True, cmem_args=cmem_args, answer=answer)

