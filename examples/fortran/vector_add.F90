#ifndef N
#define N 1024
#endif
#ifndef NTHREADS
#define NTHREADS 1
#endif

module vector
use iso_c_binding
use omp_lib

contains

subroutine vector_add(C, A, B, n)
    use iso_c_binding
    real (c_float), intent(out), dimension(N) :: C
    real (c_float), intent(in), dimension(N) :: A, B
    integer (c_int), intent(in) :: n

    !$omp parallel do
    do i = 1, N
      C(i) = A(i) + B(i)
    end do
    !$omp end parallel do

end subroutine vector_add



function time_vector_add(C, A, B, n) result(time)
    use iso_c_binding
    real (c_float), intent(out), dimension(N) :: C
    real (c_float), intent(in), dimension(N) :: A, B
    integer (c_int), intent(in) :: n
    real (c_float) :: time
    real (c_double) start_time, end_time

    CALL OMP_SET_NUM_THREADS(NTHREADS);
    start_time = omp_get_wtime()

    call vector_add(C, A, B, n)

    end_time = omp_get_wtime()
    time = (end_time - start_time)*1e3

end function time_vector_add



end module vector
