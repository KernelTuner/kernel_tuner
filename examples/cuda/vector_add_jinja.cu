__global__ void vector_add(const {{ real_type }}{{ vector_size }} * a,const {{ real_type }}{{ vector_size }} * b, {{ real_type }}{{ vector_size }} * c, int size)
{
    int i = (blockIdx.x * block_size_x) + threadIdx.x;
    if ( i < size )
    {
        c[i] = a[i] + b[i];
    }
}