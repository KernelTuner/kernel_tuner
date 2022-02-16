
#define domain_width    4096
#define domain_height   2048

__kernel void stencil_kernel(__global float *x_new, __global float *x_old) {
    int x = get_group_id(0) * block_size_x + get_local_id(0);
    int y = get_group_id(1) * block_size_y + get_local_id(1);

    if (y>0 && y<domain_height-1 && x>0 && x<domain_width-1) {

    x_new[y*domain_width+x] = ( x_old[ (y  ) * domain_width + (x  ) ] +
                                x_old[ (y  ) * domain_width + (x-1) ] +
                                x_old[ (y  ) * domain_width + (x+1) ] +
                                x_old[ (y+1) * domain_width + (x  ) ] +
                                x_old[ (y-1) * domain_width + (x  ) ] ) / 5.0f;

    }
}

