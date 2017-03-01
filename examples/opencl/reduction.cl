#ifndef vector
#define vector 1
#endif 

#if (vector==1)
#define floatvector float
#elif (vector == 2)
#define floatvector float2
#elif (vector == 4)
#define floatvector float4
#endif

__kernel void sum_floats(__global float *sum_global, __global floatvector *array, int n) {
    int ti = get_local_id(0);
    int x = get_group_id(0) * block_size_x + get_local_id(0);
    int step_size = num_blocks * block_size_x;
    float sum = 0.0f;

    //cooperatively iterate over input array with all thread blocks
    for (int i=x; i<n/vector; i+=step_size) {
        floatvector v = array[i];
        #if vector == 1
        sum += v;
        #elif vector == 2
        sum += v.x + v.y;
        #elif vector == 4
        sum += v.x + v.y + v.z + v.w;
        #endif
    }
    
    //reduce sum to single value (or last 32 in case of use_shuffle)
    __local float sh_mem[block_size_x];

    sh_mem[ti] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (unsigned int s=block_size_x/2; s>0; s>>=1) {
        if (ti < s) {
            sh_mem[ti] += sh_mem[ti + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    //write back one value per thread block, run kernel again with one tread block
    if (ti == 0) {
        sum_global[get_group_id(0)] = sh_mem[0];
    }
}

