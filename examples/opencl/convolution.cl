#define image_height 4096
#define image_width 4096

#ifndef filter_height
    #define filter_height 17
#endif
#ifndef filter_width
    #define filter_width 17
#endif

#define border_height ((filter_height/2)*2)
#define border_width ((filter_width/2)*2)
#define input_height (image_height + border_height)
#define input_width (image_width + border_width)

#ifndef block_size_x
    #define block_size_x 16
#endif
#ifndef block_size_y
    #define block_size_y 16
#endif
#ifndef block_size_z
    #define block_size_z 1
#endif
#ifndef tile_size_x
    #define tile_size_x 1
#endif
#ifndef tile_size_y
    #define tile_size_y 1
#endif

#define i_end min(block_size_y*tile_size_y+border_height, input_height)
#define j_end min(block_size_x*tile_size_x+border_width, input_width)

/*
 * If use_padding == 1, we introduce (only when necessary) a number of padding
 * columns in shared memory to avoid shared memory bank conflicts
 *
 * padding columns are only inserted when block_size_x is not a multiple of 32 (the assumed number of memory banks)
 * and when the width of the data needed is not a multiple of 32. The latter is because some filter_widths never
 * cause bank conflicts.
 * 
 * If not passed as a tunable parameter, padding is on by default
 */
#define shared_mem_width (block_size_x*tile_size_x+border_width)
#ifndef use_padding
    #define use_padding 1
#endif
#if use_padding == 1
    #if (((block_size_x % 32)!=0) && (((shared_mem_width-block_size_x)%32) != 0))
        // next line uses &31 instead of %32, because % in C is remainder not modulo
        #define padding_columns ((32 - (border_width + block_size_x*tile_size_x - block_size_x)) & 31)
        #undef shared_mem_width
        #define shared_mem_width (block_size_x*tile_size_x+border_width+padding_columns)
    #endif
#endif


__kernel void convolution_kernel(__global float *output, __global float *input, __global float *filter) {
    int ty = get_local_id(1);
    int tx = get_local_id(0);
    int by = get_group_id(1) * block_size_y * tile_size_y;
    int bx = get_group_id(0) * block_size_x * tile_size_x;

    //shared memory to hold all input data need by this thread block
    __local float sh_input[block_size_y*tile_size_y+border_height][shared_mem_width];

    //load all input data needed by this thread block into shared memory
    #pragma unroll
    for (int i=ty; i<i_end; i+=block_size_y) {
        #pragma unroll
        for (int j=tx; j<j_end; j+=block_size_x) {
            #if ((image_height%(block_size_y*tile_size_y)!=0) || (image_width%(block_size_x*tile_size_x)!=0))
            int y = by+i;
            int x = bx+j;
            if (y < input_height && x < input_width) {
                sh_input[i][j] = input[y*input_width+x];
            }
            #else
                sh_input[i][j] = input[(by+i)*input_width + (bx+j)];
            #endif
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //thread-local registers to hold local sums
    float sum[tile_size_y][tile_size_x];
    #pragma unroll
    for (int yi=0; yi<tile_size_y; yi++) {
        #pragma unroll
        for (int xi=0; xi<tile_size_x; xi++) {
             sum[yi][xi] = 0.0f;
        }
    }

    //for each filter weight
    #pragma unroll
    for (int i=0; i < filter_height; i++) {
        #pragma unroll
        for (int j=0; j < filter_width; j++) {

            #pragma unroll
            for (int yi=0; yi<tile_size_y; yi++) {   
                #pragma unroll
                for (int xi=0; xi<tile_size_x; xi++) {
                    sum[yi][xi] += sh_input[ty+yi*block_size_y+i][tx+xi*block_size_x+j] * filter[i*filter_width+j];
                }
            }

        }
    }

    //store results to global memory
    #pragma unroll
    for (int yi=0; yi<tile_size_y; yi++) {   
        #pragma unroll
        for (int xi=0; xi<tile_size_x; xi++) {
            #if ((image_height%(block_size_y*tile_size_y)!=0) || (image_width%(block_size_x*tile_size_x)!=0))
            int y = by+ty+yi*block_size_y;
            int x = bx+tx+xi*block_size_x;
            if (y < image_height && x < image_width) {
                output[y * image_width + x] = sum[yi][xi];
            }
            #else
                output[(by+ty+yi*block_size_y) * image_width + bx+tx+xi*block_size_x] = sum[yi][xi];
            #endif
        }
    }

}





__kernel void convolution_naive(__global float *output, __global float *input, __global float *filter) {

    int x = get_group_id(0) * get_local_size(0) + get_local_id(0);
    int y = get_group_id(1) * get_local_size(1) + get_local_id(1);
    int i, j;
    float sum = 0.0;

    if (y < image_height && x < image_width) {

        for (j = 0; j < filter_height; j++) {
            for (i = 0; i < filter_width; i++) {
                sum += input[(y + j) * input_width + (x + i)] * filter[j * filter_width + i];
            }
        }

        output[y * image_width + x] = sum;
    }
}

