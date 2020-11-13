#include <helper_math.h>

extern "C"
{
__global__ void vector_add(const {{ real_type }} * a,const {{ real_type }} * b, {{ real_type }} * c, int size)
{
    int i = (blockIdx.x * block_size_x) + threadIdx.x;
    if ( i < (size / {{ vector_size }}) )
    {
        c[i] = a[i] + b[i];
    }
}
}