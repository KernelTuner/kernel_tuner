
__global__ void vector_add(const {{ real_type }} * __restrict__ a, const {{ real_type }} * __restrict__ b, {{ real_type }} * __restrict__ c, int size)
{
    int index = (blockIdx.x * {{ block_size_x }} * {{ tiling_x }}) + threadIdx.x;

    {% for tile in range(tiling_x) %}
    {% set offset = block_size_x * tile %}
    if ( index + {{ offset }} < (size / {{ vector_size }}) )
    {
        {{ real_type }} item_a = a[index + {{ offset }}];
        {{ real_type }} item_b = b[index + {{ offset }}];

        {% if vector_size == 1 %}
        c[index + {{ offset }}] =  item_a + item_b;
        {% elif vector_size == 2 %}
        c[index + {{ offset }}] =  make_{{ real_type }}(item_a.x + item_b.x, item_a.y + item_b.y);
        {% elif vector_size == 4 %}
        c[index + {{ offset }}] =  make_{{ real_type }}(item_a.x + item_b.x, item_a.y + item_b.y, item_a.z + item_b.z, item_a.w + item_b.w);
        {% endif %}
    }
    {% endfor %}
}