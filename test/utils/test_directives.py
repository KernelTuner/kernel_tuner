from kernel_tuner.utils.directives import *


def test_is_openacc():
    assert is_openacc(OpenACC())
    assert not is_openacc(None)


def test_is_cxx():
    assert is_cxx(Cxx())
    assert not is_cxx(Fortran())
    assert not is_cxx(None)


def test_is_fortran():
    assert is_fortran(Fortran())
    assert not is_fortran(Cxx())
    assert not is_fortran(None)


def test_line_contains_openacc_directive():
    cxx_code = "int main(void) {\n#pragma acc parallel}"
    f90_code = "!$acc parallel"
    assert line_contains_openacc_directive(cxx_code, Cxx())
    assert not line_contains_openacc_directive(f90_code, Cxx())
    assert line_contains_openacc_directive(f90_code, Fortran())
    assert not line_contains_openacc_directive(cxx_code, Fortran())
    assert not line_contains_openacc_directive(cxx_code, None)


def test_line_contains_openacc_parallel_directive():
    assert line_contains_openacc_parallel_directive("#pragma acc parallel wait", Cxx())
    assert line_contains_openacc_parallel_directive("!$acc parallel", Fortran())
    assert not line_contains_openacc_parallel_directive("#pragma acc loop", Cxx())
    assert not line_contains_openacc_parallel_directive("!$acc loop", Fortran())
    assert not line_contains_openacc_parallel_directive("!$acc parallel", None)


def test_openacc_directive_contains_data_clause():
    assert openacc_directive_contains_data_clause("#pragma acc parallel present(A[:1089])")
    assert not openacc_directive_contains_data_clause("#pragma acc parallel for")


def test_create_data_directive():
    size = ArraySize()
    size.add(1024)
    assert (
        create_data_directive_openacc("array", size, Cxx())
        == "#pragma acc enter data create(array[:1024])\n#pragma acc update device(array[:1024])\n"
    )
    size.clear()
    size.add(35)
    size.add(16)
    assert (
        create_data_directive_openacc("matrix", size, Fortran())
        == "!$acc enter data create(matrix(:35,:16))\n!$acc update device(matrix(:35,:16))\n"
    )
    assert create_data_directive_openacc("array", size, None) == ""


def test_exit_data_directive():
    size = ArraySize()
    size.add(1024)
    assert exit_data_directive_openacc("array", size, Cxx()) == "#pragma acc exit data copyout(array[:1024])\n"
    size.clear()
    size.add(35)
    size.add(16)
    assert exit_data_directive_openacc("matrix", size, Fortran()) == "!$acc exit data copyout(matrix(:35,:16))\n"
    assert exit_data_directive_openacc("matrix", size, None) == ""


def test_correct_kernel():
    assert correct_kernel("vector_add", "tuner start vector_add")
    assert correct_kernel("vector_add", "tuner start vector_add a(float:size)")
    assert not correct_kernel("vector_add", "tuner start gemm")
    assert not correct_kernel("vector_add", "tuner start gemm a(float:size) b(float:size)")


def test_parse_size():
    assert parse_size(128).get() == 128
    assert parse_size("16").get() == 16
    assert parse_size("test").get() == 0
    assert parse_size("n", ["#define n 1024\n"]).get() == 1024
    assert parse_size("n,m", ["#define n 16\n", "#define m 32\n"]).get() == 512
    assert parse_size("n", ["#define size 512\n"], {"n": 32}).get() == 32
    assert parse_size("m", ["#define size 512\n"], {"n": 32}).get() == 0
    assert parse_size("rows,cols", dimensions={"rows": 16, "cols": 8}).get() == 128
    assert parse_size("n_rows,n_cols", ["#define n_cols 16\n", "#define n_rows 32\n"]).get() == 512
    assert parse_size("rows,cols", [], dimensions={"rows": 16, "cols": 8}).get() == 128


def test_wrap_timing():
    code = "#pragma acc\nfor ( int i = 0; i < size; i++ ) {\nc[i] = a[i] + b[i];\n}"
    wrapped = wrap_timing(code, Cxx())
    assert (
        wrapped
        == "auto kt_timing_start = std::chrono::steady_clock::now();\n#pragma acc\nfor ( int i = 0; i < size; i++ ) {\nc[i] = a[i] + b[i];\n}\nauto kt_timing_end = std::chrono::steady_clock::now();\nstd::chrono::duration<float, std::milli> elapsed_time = kt_timing_end - kt_timing_start;\nreturn elapsed_time.count();\n"
    )


def test_wrap_data():
    acc_cxx = Code(OpenACC(), Cxx())
    acc_f90 = Code(OpenACC(), Fortran())
    code_cxx = "// this is a comment\n"
    code_f90 = "! this is a comment\n"
    data = {"array": ["int*", "size"]}
    preprocessor = ["#define size 42"]
    expected_cxx = "#pragma acc enter data create(array[:42])\n#pragma acc update device(array[:42])\n\n// this is a comment\n\n#pragma acc exit data copyout(array[:42])\n"
    assert wrap_data(code_cxx, acc_cxx, data, preprocessor, None) == expected_cxx
    expected_f90 = "!$acc enter data create(array(:42))\n!$acc update device(array(:42))\n\n! this is a comment\n\n!$acc exit data copyout(array(:42))\n"
    assert wrap_data(code_f90, acc_f90, data, preprocessor, None) == expected_f90
    data = {"matrix": ["float*", "rows,cols"]}
    preprocessor = ["#define rows 42", "#define cols 84"]
    expected_f90 = "!$acc enter data create(matrix(:42,:84))\n!$acc update device(matrix(:42,:84))\n\n! this is a comment\n\n!$acc exit data copyout(matrix(:42,:84))\n"
    assert wrap_data(code_f90, acc_f90, data, preprocessor, None) == expected_f90
    dimensions = {"rows": 42, "cols": 84}
    assert wrap_data(code_f90, acc_f90, data, user_dimensions=dimensions) == expected_f90
    assert wrap_data(code_f90, acc_f90, data, preprocessor=[], user_dimensions=dimensions) == expected_f90


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
    acc_cxx = Code(OpenACC(), Cxx())
    returns = extract_directive_code(code, acc_cxx)
    assert len(returns) == 2
    assert expected_one in returns["initialize"]
    assert expected_two in returns["vector_add"]
    assert expected_one not in returns["vector_add"]
    returns = extract_directive_code(code, acc_cxx, "vector")
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
    returns = extract_directive_code(code, Code(OpenACC(), Fortran()), "vector_add")
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


def test_extract_directive_signature():
    acc_cxx = Code(OpenACC(), Cxx())
    code = "#pragma tuner start vector_add a(float*:VECTOR_SIZE) b(float*:VECTOR_SIZE) c(float*:VECTOR_SIZE) size(int:VECTOR_SIZE)  \n#pragma acc"
    signatures = extract_directive_signature(code, acc_cxx)
    assert len(signatures) == 1
    assert (
        "float vector_add(float * restrict a, float * restrict b, float * restrict c, int size)"
        in signatures["vector_add"]
    )
    signatures = extract_directive_signature(code, acc_cxx, "vector_add")
    assert len(signatures) == 1
    assert (
        "float vector_add(float * restrict a, float * restrict b, float * restrict c, int size)"
        in signatures["vector_add"]
    )
    signatures = extract_directive_signature(code, acc_cxx, "vector_add_ext")
    assert len(signatures) == 0
    code = "!$tuner start vector_add A(float*:VECTOR_SIZE) B(float*:VECTOR_SIZE) C(float*:VECTOR_SIZE) n(int:VECTOR_SIZE)\n!$acc"
    signatures = extract_directive_signature(code, Code(OpenACC(), Fortran()))
    assert len(signatures) == 1
    assert "function vector_add(A, B, C, n)" in signatures["vector_add"]


def test_extract_directive_data():
    acc_cxx = Code(OpenACC(), Cxx())
    code = "#pragma tuner start vector_add a(float*:VECTOR_SIZE) b(float*:VECTOR_SIZE) c(float*:VECTOR_SIZE) size(int:VECTOR_SIZE)\n#pragma acc"
    data = extract_directive_data(code, acc_cxx)
    assert len(data) == 1
    assert len(data["vector_add"]) == 4
    assert "float*" in data["vector_add"]["b"]
    assert "int" not in data["vector_add"]["c"]
    assert "VECTOR_SIZE" in data["vector_add"]["size"]
    data = extract_directive_data(code, acc_cxx, "vector_add_double")
    assert len(data) == 0
    acc_f90 = Code(OpenACC(), Fortran())
    code = "!$tuner start vector_add A(float*:VECTOR_SIZE) B(float*:VECTOR_SIZE) C(float*:VECTOR_SIZE) n(int:VECTOR_SIZE)\n!$acc"
    data = extract_directive_data(code, acc_f90)
    assert len(data) == 1
    assert len(data["vector_add"]) == 4
    assert "float*" in data["vector_add"]["B"]
    assert "int" not in data["vector_add"]["C"]
    assert "VECTOR_SIZE" in data["vector_add"]["n"]
    code = (
        "!$tuner start matrix_add A(float*:N_ROWS,N_COLS) B(float*:N_ROWS,N_COLS) nr(int:N_ROWS) nc(int:N_COLS)\n!$acc"
    )
    data = extract_directive_data(code, acc_f90)
    assert len(data) == 1
    assert len(data["matrix_add"]) == 4
    assert "float*" in data["matrix_add"]["A"]
    assert "N_ROWS,N_COLS" in data["matrix_add"]["B"]


def test_allocate_signature_memory():
    code = "#pragma tuner start vector_add a(float*:VECTOR_SIZE) b(float*:VECTOR_SIZE) c(float*:VECTOR_SIZE) size(int:VECTOR_SIZE)\n#pragma acc"
    data = extract_directive_data(code, Code(OpenACC(), Cxx()))
    args = allocate_signature_memory(data["vector_add"])
    assert args[3] == 0
    preprocessor = ["#define VECTOR_SIZE 1024\n"]
    args = allocate_signature_memory(data["vector_add"], preprocessor)
    assert type(args[0]) is np.ndarray
    assert type(args[1]) is not np.float64
    assert args[2].dtype == "float32"
    assert type(args[3]) is np.int32
    assert args[3] == 1024
    user_values = dict()
    user_values["VECTOR_SIZE"] = 1024
    args = allocate_signature_memory(data["vector_add"], user_dimensions=user_values)
    assert type(args[0]) is np.ndarray
    assert type(args[1]) is not np.float64
    assert args[2].dtype == "float32"
    assert type(args[3]) is np.int32
    code = (
        "!$tuner start matrix_add A(float*:N_ROWS,N_COLS) B(float*:N_ROWS,N_COLS) nr(int:N_ROWS) nc(int:N_COLS)\n!$acc"
    )
    data = extract_directive_data(code, Code(OpenACC(), Fortran()))
    preprocessor = ["#define N_ROWS 128\n", "#define N_COLS 512\n"]
    args = allocate_signature_memory(data["matrix_add"], preprocessor)
    assert args[2] == 128
    assert len(args[0]) == (128 * 512)
    user_values = dict()
    user_values["N_ROWS"] = 32
    user_values["N_COLS"] = 16
    args = allocate_signature_memory(data["matrix_add"], user_dimensions=user_values)
    assert args[3] == 16
    assert len(args[1]) == 512


def test_extract_initialization_code():
    code_cpp = "#pragma tuner initialize\nconst int value = 42;\n#pragma tuner stop\n"
    code_f90 = "!$tuner initialize\ninteger :: value\n!$tuner stop\n"
    assert extract_initialization_code(code_cpp, Code(OpenACC(), Cxx())) == "const int value = 42;\n"
    assert extract_initialization_code(code_f90, Code(OpenACC(), Fortran())) == "integer :: value\n"


def test_extract_deinitialization_code():
    code_cpp = "#pragma tuner deinitialize\nconst int value = 42;\n#pragma tuner stop\n"
    code_f90 = "!$tuner deinitialize\ninteger :: value\n!$tuner stop\n"
    assert extract_deinitialization_code(code_cpp, Code(OpenACC(), Cxx())) == "const int value = 42;\n"
    assert extract_deinitialization_code(code_f90, Code(OpenACC(), Fortran())) == "integer :: value\n"


def test_add_present_openacc():
    acc_cxx = Code(OpenACC(), Cxx())
    acc_f90 = Code(OpenACC(), Fortran())
    code_cxx = "#pragma acc parallel num_gangs(32)\n#pragma acc\n"
    code_f90 = "!$acc parallel async num_workers(16)\n"
    data = {"array": ["int*", "size"]}
    preprocessor = ["#define size 42"]
    expected_cxx = "#pragma acc parallel num_gangs(32) present(array[:42])\n#pragma acc\n"
    assert add_present_openacc(code_cxx, acc_cxx, data, preprocessor, None) == expected_cxx
    expected_f90 = "!$acc parallel async num_workers(16) present(array(:42))\n"
    assert add_present_openacc(code_f90, acc_f90, data, preprocessor, None) == expected_f90
    code_f90 = "!$acc parallel async num_workers(16) copy(array(:42))\n"
    assert add_present_openacc(code_f90, acc_f90, data, preprocessor, None) == code_f90
    code_cxx = "#pragma acc parallel num_gangs(32)\n\t#pragma acc loop\n\t//for loop\n"
    expected_cxx = "#pragma acc parallel num_gangs(32) present(array[:42])\n\t#pragma acc loop\n\t//for loop\n"
    assert add_present_openacc(code_cxx, acc_cxx, data, preprocessor, None) == expected_cxx
    code_f90 = "!$acc parallel async num_workers(16)\n"
    data = {"matrix": ["float*", "rows,cols"]}
    preprocessor = ["#define cols 18\n", "#define rows 14\n"]
    expected_f90 = "!$acc parallel async num_workers(16) present(matrix(:14,:18))\n"
    assert add_present_openacc(code_f90, acc_f90, data, preprocessor, None) == expected_f90
    dimensions = {"cols": 18, "rows": 14}
    assert add_present_openacc(code_f90, acc_f90, data, user_dimensions=dimensions) == expected_f90
    assert add_present_openacc(code_f90, acc_f90, data, preprocessor=[], user_dimensions=dimensions) == expected_f90
