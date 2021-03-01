{% set real_vector = real_type ~ vector_size %}
#define float1 float

__global__ void vector_add(const {{ real_vector }} * __restrict__ a, const {{ real_vector }} * __restrict__ b, {{ real_vector }} * __restrict__ c, int size)
{
    int index = (blockIdx.x * {{ block_size_x }} * {{ tiling_x }}) + threadIdx.x;

    {% for tile in range(tiling_x) %}
    {% set offset = block_size_x * tile %}
    if ( index + {{ offset }} < (size / {{ vector_size }}) )
    {
        {{ real_vector }} item_a = a[index + {{ offset }}];
        {{ real_vector }} item_b = b[index + {{ offset }}];

        {% if vector_size == 1 %}
        c[index + {{ offset }}] =  item_a + item_b;
        {% elif vector_size == 2 %}
        c[index + {{ offset }}] =  make_{{ real_vector }}(item_a.x + item_b.x, item_a.y + item_b.y);
        {% elif vector_size == 4 %}
        c[index + {{ offset }}] =  make_{{ real_vector }}(item_a.x + item_b.x, item_a.y + item_b.y, item_a.z + item_b.z, item_a.w + item_b.w);
        {% endif %}
    }
    {% endfor %}
}
