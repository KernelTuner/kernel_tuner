#define image_height 4096
#define image_width 4096
#define filter_height 17
#define filter_width 17

#define border_height ((filter_height/2)*2)
#define border_width ((filter_width/2)*2)
#define input_height (image_height + border_height)
#define input_width (image_width + border_width)

__constant__ float d_filter[filter_height*filter_width];

__global__ void convolution_kernel(float *output, float *input, float *filter) {
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int by = blockIdx.y * block_size_y;
    int bx = blockIdx.x * block_size_x;
    int y = by+ty;
    int x = bx+tx;

    //thread-local registers to hold local sums
    float sum[tile_size_y][tile_size_x];

    //shared memory to hold all input data need by this thread block
    __shared__ float sh_input[block_size_y*tile_size_y+border_height][block_size_x*tile_size_x+border_width];

    //load all input data needed by this thread block into shared memory
    #pragma unroll
    for (int i=ty; i<block_size_y*tile_size_y+border_height; i+=block_size_y) {
        #pragma unroll
        for (int j=tx; j<block_size_x*tile_size_x+border_width; j+=block_size_x) {
            sh_input[i][j] = input[(by+i)*input_width + (bx+j)];
        }
    }
    __syncthreads();

    //for each filter weight
    #pragma unroll
    for (int i=0; i < filter_height; i++) {
        #pragma unroll
        for (int j=0; j < filter_width; j++) {

            #pragma unroll
            for (int yi=0; yi<tile_size_y; yi++) {   
                #pragma unroll
                for (int xi=0; xi<tile_size_x; xi++) {
                    sum[yi][xi] += sh_input[ty+yi*block_size_y+i][tx+xi*block_size_x+j] * d_filter[i*filter_width+j];
                }
            }

        }
    }

    //store results to global memory
    #pragma unroll
    for (int yi=0; yi<tile_size_y; yi++) {   
        #pragma unroll
        for (int xi=0; xi<tile_size_x; xi++) {
             output[(y+yi*block_size_x)*image_width+x+xi*block_size_x] = sum[yi][xi];
        }
    }

}

