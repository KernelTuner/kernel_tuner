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

subroutine vector_add(A, B, C, n)
    use iso_c_binding
    real (c_float), intent(out), dimension(VECTOR_SIZE) :: C
    real (c_float), intent(in), dimension(VECTOR_SIZE) :: A, B
    integer (c_int), intent(in) :: n

    !$tuner start vector_add A(float*:VECTOR_SIZE) B(float*:VECTOR_SIZE) C(float*:VECTOR_SIZE) n(int:VECTOR_SIZE) i(int:VECTOR_SIZE)
    !$acc parallel loop vector_length(nthreads)
    do i = 1, n
      C(i) = A(i) + B(i)
    end do
    !$acc end parallel loop
    !$tuner stop

end subroutine vector_add
"""

# Extract tunable directive
preprocessor = extract_preprocessor(code)
signature = extract_directive_signature(code)
body = extract_directive_code(code)
# Allocate memory on the host
data = extract_directive_data(code)
args = allocate_signature_memory(data["vector_add"], preprocessor)
# Generate kernel string
kernel_string = generate_directive_function(preprocessor, signature["vector_add"], body["vector_add"], data=data)

tune_params = OrderedDict()
tune_params["nthreads"] = [32 * i for i in range(1, 33)]

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
