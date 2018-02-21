
template<typename T, int sz>
void multiply_matrix(T (&output)[sz], const T (&a)[sz], const T (&b)[sz], int s) {
    // calculates matrix product of two square matrices
    // out=A*B
    for (int i=0; i<sz; i++) {
        output[i] = 0;
    }
    for (int i=0; i<s; i++) {
        for (int j=0; j<s; j++) {
            for (int k=0; k<s; k++) {
                output[i*s+j] += a[i*s+k] * b[k*s+j];
            }
    }
    }
}

