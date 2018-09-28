#ifndef N
 #define N 1024
#endif

module vector
use iso_c_binding

contains

subroutine vector_add(C, A, B, n)
    use iso_c_binding
    real (c_float), intent(out), dimension(N) :: C
    real (c_float), intent(in), dimension(N) :: A, B
    integer (c_int), intent(in) :: n

    C = A + B

end subroutine vector_add

end module vector
