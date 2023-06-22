#!/usr/bin/env python
"""This is a simple example for tuning Fortran OpenACC code with the kernel tuner"""

from kernel_tuner import tune_kernel
from kernel_tuner.utils.directives import (
    extract_directive_signature,
    extract_directive_code,
    extract_preprocessor,
    generate_directive_function,
    extract_directive_data,
    allocate_signature_memory,
)
from collections import OrderedDict

code = """
#define VECTOR_SIZE 65536

subroutine vector_add(C, A, B, n)
    use iso_c_binding
    real (c_float), intent(out), dimension(VECTOR_SIZE) :: C
    real (c_float), intent(in), dimension(VECTOR_SIZE) :: A, B
    integer (c_int), intent(in) :: n

    !$tuner start vector_add A(float*:VECTOR_SIZE) B(float*:VECTOR_SIZE) C(float*:VECTOR_SIZE) n(int:VECTOR_SIZE)
    !$acc parallel loop num_gangs(ngangs) vector_length(vlength)
    do i = 1, N
      C(i) = A(i) + B(i)
    end do
    !$acc end parallel loop
    !$tuner stop

end subroutine vector_add
"""

# Extract tunable directive and generate kernel_string
preprocessor = extract_preprocessor(code)
signature = extract_directive_signature(code)
body = extract_directive_code(code)
kernel_string = generate_directive_function(
    preprocessor, signature["vector_add"], body["vector_add"]
)

# Allocate memory on the host
data = extract_directive_data(code)
args = allocate_signature_memory(data["vector_add"], preprocessor)

tune_params = OrderedDict()
tune_params["ngangs"] = [2**i for i in range(0, 15)]
tune_params["vlength"] = [2**i for i in range(0, 11)]

answer = [None, None, args[0] + args[1], None]

tune_kernel(
    "vector_add",
    kernel_string,
    0,
    args,
    tune_params,
    answer=answer,
    compiler_options=["-fast", "-acc=gpu"],
    compiler="nvfortran",
)
