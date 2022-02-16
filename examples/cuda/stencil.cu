
#define domain_width    4096
#define domain_height   2048

__global__ void stencil_kernel(float *x_new, float *x_old) {
    int x = blockIdx.x * block_size_x + threadIdx.x;
    int y = blockIdx.y * block_size_y + threadIdx.y;

    if (y>0 && y<domain_height-1 && x>0 && x<domain_width-1) {

    x_new[y*domain_width+x] = ( x_old[ (y  ) * domain_width + (x  ) ] +
                                x_old[ (y  ) * domain_width + (x-1) ] +
                                x_old[ (y  ) * domain_width + (x+1) ] +
                                x_old[ (y+1) * domain_width + (x  ) ] +
                                x_old[ (y-1) * domain_width + (x  ) ] ) / 5.0f;

    }
}
