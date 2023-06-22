from kernel_tuner.utils.directives import *


def test_extract_directive_code():
    code = """
        #include <stdlib.h>

        #define VECTOR_SIZE 65536

        int main(void) {
            int size = VECTOR_SIZE;
            __restrict float * a = (float *) malloc(VECTOR_SIZE * sizeof(float));
            __restrict float * b = (float *) malloc(VECTOR_SIZE * sizeof(float));
            __restrict float * c = (float *) malloc(VECTOR_SIZE * sizeof(float));

            #pragma tuner start initialize
            #pragma acc parallel
            #pragma acc loop
            for ( int i = 0; i < size; i++ ) {
                    a[i] = i;
                    b[i] = i + 1;
            }
            #pragma tuner stop

            #pragma tuner start vector_add
            #pragma acc parallel
            #pragma acc loop
            for ( int i = 0; i < size; i++ ) {
                    c[i] = a[i] + b[i];
            }
            #pragma tuner stop

            free(a);
            free(b);
            free(c);
    }
    """
    expected_one = """            #pragma acc parallel
            #pragma acc loop
            for ( int i = 0; i < size; i++ ) {
                    a[i] = i;
                    b[i] = i + 1;
            }"""
    expected_two = """            #pragma acc parallel
            #pragma acc loop
            for ( int i = 0; i < size; i++ ) {
                    c[i] = a[i] + b[i];
            }"""
    returns = extract_directive_code(code)
    assert len(returns) == 2
    assert expected_one in returns["initialize"]
    assert expected_two in returns["vector_add"]
    assert expected_one not in returns["vector_add"]
    returns = extract_directive_code(code, "vector")
    assert len(returns) == 0

    code = """
    !$tuner start vector_add
    !$acc parallel loop num_gangs(ngangs) vector_length(vlength)
    do i = 1, N
      C(i) = A(i) + B(i)
    end do
    !$acc end parallel loop
    !$tuner stop
    """
    expected = """    !$acc parallel loop num_gangs(ngangs) vector_length(vlength)
    do i = 1, N
      C(i) = A(i) + B(i)
    end do
    !$acc end parallel loop"""
    returns = extract_directive_code(code, "vector_add")
    assert len(returns) == 1
    assert expected in returns["vector_add"]


def test_extract_preprocessor():
    code = """
        #include <stdlib.h>

        #define VECTOR_SIZE 65536

        int main(void) {
            int size = VECTOR_SIZE;
            __restrict float * a = (float *) malloc(VECTOR_SIZE * sizeof(float));
            __restrict float * b = (float *) malloc(VECTOR_SIZE * sizeof(float));
            __restrict float * c = (float *) malloc(VECTOR_SIZE * sizeof(float));

            #pragma tuner start
            #pragma acc parallel
            #pragma acc loop
            for ( int i = 0; i < size; i++ ) {
                    a[i] = i;
                    b[i] = i + 1;
            }
            #pragma tuner stop

            #pragma tuner start
            #pragma acc parallel
            #pragma acc loop
            for ( int i = 0; i < size; i++ ) {
                    c[i] = a[i] + b[i];
            }
            #pragma tuner stop

            free(a);
            free(b);
            free(c);
    }
    """
    expected = ["        #include <stdlib.h>", "        #define VECTOR_SIZE 65536"]
    results = extract_preprocessor(code)
    assert len(results) == 2
    for item in expected:
        assert item in results


def test_wrap_timing():
    code = "#pragma acc\nfor ( int i = 0; i < size; i++ ) {\nc[i] = a[i] + b[i];\n}"
    wrapped = wrap_timing(code)
    assert (
        wrapped
        == "auto start = std::chrono::steady_clock::now();\n#pragma acc\nfor ( int i = 0; i < size; i++ ) {\nc[i] = a[i] + b[i];\n}\nauto end = std::chrono::steady_clock::now();\nstd::chrono::duration<float, std::milli> elapsed_time = end - start;\nreturn elapsed_time.count();"
    )


def test_extract_directive_signature():
    code = "#pragma tuner start vector_add a(float*:VECTOR_SIZE) b(float*:VECTOR_SIZE) c(float*:VECTOR_SIZE) size(int:VECTOR_SIZE)  \n#pragma acc"
    signatures = extract_directive_signature(code)
    assert len(signatures) == 1
    assert (
        "float vector_add(float * restrict a, float * restrict b, float * restrict c, int size)"
        in signatures["vector_add"]
    )
    signatures = extract_directive_signature(code, "vector_add")
    assert len(signatures) == 1
    assert (
        "float vector_add(float * restrict a, float * restrict b, float * restrict c, int size)"
        in signatures["vector_add"]
    )
    signatures = extract_directive_signature(code, "vector_add_ext")
    assert len(signatures) == 0


def test_extract_directive_data():
    code = "#pragma tuner start vector_add a(float*:VECTOR_SIZE) b(float*:VECTOR_SIZE) c(float*:VECTOR_SIZE) size(int:VECTOR_SIZE)\n#pragma acc"
    data = extract_directive_data(code)
    assert len(data) == 1
    assert len(data["vector_add"]) == 4
    assert "float*" in data["vector_add"]["b"]
    assert "int" not in data["vector_add"]["c"]
    assert "VECTOR_SIZE" in data["vector_add"]["size"]
    data = extract_directive_data(code, "vector_add_double")
    assert len(data) == 0
