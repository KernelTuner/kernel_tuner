#!/usr/bin/env python

import kernel_tuner
import numpy
from collections import OrderedDict

_rotation_kernel_source = """
// This kernel is adapted from the PyCuda "Rotate" example.

texture<float, 2> tex;

__global__ void copy_texture_kernel(
    const float resize_val, 
    const float alpha,
    unsigned char* data) {

        // calculate pixel idx
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
        
        // We might be outside the reachable pixels. Don't do anything
        if( (x >= newiw) || (y >= newih) )
            return;
        
        // calculate offset into destination array
        unsigned int didx = y * newiw + x;
        
        // calculate offset into source array (be aware of rotation and scaling)
        float xmiddle = (x-newiw/2.) / resize_val;
        float ymiddle = (y-newih/2.) / resize_val;
        float sx = ( xmiddle*cos(alpha)+ymiddle*sin(alpha) + oldiw/2.) ;
        float sy = ( -xmiddle*sin(alpha)+ymiddle*cos(alpha) + oldih/2.);
        
        data[didx] = tex2D(tex, sx, sy);
    }
"""

def tune():
    kernel_string = _rotation_kernel_source
    problem_size = (1024, 1024)
    x = 128 * numpy.ones(problem_size, dtype=numpy.float32)
    out = numpy.zeros(problem_size, dtype=numpy.uint8)

    tune_params = OrderedDict()
    tune_params["block_size_x"] = [ 16, 32 ]
    tune_params["block_size_y"] = [ 16, 32 ]
    tune_params["oldiw"] = [ problem_size[0] ]
    tune_params["oldih"] = [ problem_size[1] ]
    tune_params["newiw"] = [ problem_size[0] ]
    tune_params["newih"] = [ problem_size[1] ]

    args = [ numpy.float32(0.5), numpy.float32(20), out ]

    return kernel_tuner.tune_kernel("copy_texture_kernel", kernel_string, problem_size, args, tune_params, texmem_args = { 'tex': { 'array': x, 'address_mode': 'border' } })

if __name__ == "__main__":
    tune()
