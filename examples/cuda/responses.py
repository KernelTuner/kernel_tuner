responses1 = [
    """
    __global__ void add_vectors(float* a, float* b, float* c, int n) {
        int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x * 2;
        if (i < n) {
            c[i] = a[i] + b[i];
            if (i + 1 < n) {
                c[i + 1] = a[i + 1] + b[i + 1];
            }
        }
    }
    """
]

responses2 = [
    """
    __global__ void add_vectors(float* a, float* b, float* c, int n) {
        int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
        if (i < n) {
            c[i] = a[i] + b[i];
            c[i+1] = a[i+1] + b[i+1];
        }
    }
    """,
    """
    __global__ void add_vectors(float* a, float* b, float* c, int n) {
        int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        if (i < n) {
            c[i] = a[i] + b[i];
            if (i + 1 < n) {
                c[i+1] = a[i+1] + b[i+1];
            }
        }
    }
    """
]

responses3 = [
    """
    __global__ void add_vectors(float* a, float* b, float* c, int n) {
        int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        if (i < n) {
            c[i] = a[i] + b[i];
            c[i+1] = a[i+1] + b[i+1];
        }
    }
    """,
]

responses4 = [
    """
    __global__ void add_vectors(float* a, float* b, float* c, int n) {
        int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;  // Multiply by 2 to process 2 elements per thread
        if (i < n - 1) {  // Check if both elements are within the range of the arrays
            c[i] = a[i] + b[i];
            c[i+1] = a[i+1] + b[i+1];
        }
        else if (i == n - 1) {  // If n is odd, process the last element with a single thread
            c[i] = a[i] + b[i];
        }
    }
    """
]


responses5 = [
    """
    __global__ void add_vectors(float* a, float* b, float* c, int n) {
        int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

        if (i < n) {
            c[i] = a[i] + b[i];
            c[i+1] = a[i+1] + b[i+1];
        }
    }
    """,
    """
    __global__ void add_vectors(float* a, float* b, float* c, int n) {
        int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

        if (i < n) {
            c[i] = a[i] + b[i];
            if (i+1 < n) {
                c[i+1] = a[i+1] + b[i+1];
            }
        }
    }
    """
]

# Deze is correct maar doet dubbel werk...
responses6 = [
    """
    __global__ void add_vectors(float* a, float* b, float* c, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n - 1) {
            c[i] = a[i] + b[i];
            c[i+1] = a[i+1] + b[i+1];
        } else if (i == n - 1) {
            c[i] = a[i] + b[i];
        }
    }
    """,
]

# Wtf is dit lol, wederom dubbel werk
responses7 = [
    """
    __global__ void add_vectors(float* a, float* b, float* c, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x * 2;
        for (int j = i; j < n; j += stride) {
            int index1 = j;
            int index2 = j + blockDim.x;
            if (index1 < n) {
                c[index1] = a[index1] + b[index1];
            }
            if (index2 < n) {
                c[index2] = a[index2] + b[index2];
            }
        }
    }
    """,
]

responses8 = [
    """
    __global__ void add_vectors(float* a, float* b, float* c, int n) {
        int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
        if (i < n) {
            c[i] = a[i] + b[i];
            int j = i + blockDim.x;
            if (j < n) {
                c[j] = a[j] + b[j];
            }
        }
    }
    """,
    """
    __global__ void add_vectors(float* a, float* b, float* c, int n) {
        int i = blockIdx.x * (blockDim.x * tile_size_x) + threadIdx.x;
        for (int j = 0; j < tile_size_x && i < n; j++, i += blockDim.x) {
            c[i] = a[i] + b[i];
        }
    }
    """,
]

responses9 = [
    """
    __global__ void add_vectors(float* a, float* b, float* c, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x * 2;
        if (i < n) {
            c[i] = a[i] + b[i];
            if (i + 1 < n) {
                c[i + 1] = a[i + 1] + b[i + 1];
            }
        }
    }
    """,
    """
    __global__ void add_vectors(float* a, float* b, float* c, int n) {
        int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        if (i < n) {
            c[i] = a[i] + b[i];
            if (i + 1 < n) {
                c[i + 1] = a[i + 1] + b[i + 1];
            } else {
                c[i + 1] = 0; // Set the last element of the result vector to 0
            }
        }
    }
    """,
    """
    __global__ void add_vectors(float* a, float* b, float* c, int n) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        for (int i = tid * tile_size_x; i < min(n, (tid + 1) * tile_size_x); i++) {
            c[i] = a[i] + b[i];
        }
    }
    """,
]

responses10 = [
    """
    __global__ void add_vectors(float* a, float* b, float* c, int n) {
        int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

        // Each thread processes 2 elements
        if (i < n) {
            c[i] = a[i] + b[i];
        }
        if (i + blockDim.x < n) {
            c[i + blockDim.x] = a[i + blockDim.x] + b[i + blockDim.x];
        }
    }
    """,
]
