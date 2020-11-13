#include <helper_math.h>

extern "C"
{
__global__ void vector_add(const {{ real_type }} * a,const {{ real_type }} * b, {{ real_type }} * c, int size)
{
    int index = (blockIdx.x * {{ block_size_x }} * {{ tiling_x }}) + threadIdx.x;

    {% for tile in range(tiling_x) %}
    {% set offset = block_size_x * tile %}
    if ( index + {{ offset }} < (size / {{ vector_size }}) )
    {
        c[index + {{ offset }}] = a[index + {{ offset }}] + b[index + {{ offset }}];
    }
    {% endfor %}
}
}