// Adapted from Codeplay's ComputeCpp SDK simple-vector-add.cpp

#ifndef block_size_x
 #define block_size_x 256
#endif

#include <CL/sycl.hpp>
#include <chrono>
#include <vector>
#include <iostream>
#include <math.h>

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

/* This is the class used to name the kernel for the runtime.
 * This must be done when the kernel is expressed as a lambda. */
template <typename T>
class SimpleVadd;


class vector_add_kernel {
  template <typename T>
  using read_global_accessor =
      sycl::accessor<T, 1, sycl::access::mode::read,
                     sycl::access::target::global_buffer>;
  template <typename T>
  using write_global_accessor =
      sycl::accessor<T, 1, sycl::access::mode::write,
                     sycl::access::target::global_buffer>;

  public:
    vector_add_kernel(
      read_global_accessor<cl::sycl::cl_float> A_ptr,
      read_global_accessor<cl::sycl::cl_float> B_ptr,
      write_global_accessor<cl::sycl::cl_float> C_ptr) : 
      m_A(A_ptr), m_B(B_ptr), m_C(C_ptr) {}

  //void operator()(sycl::nd_item<1> item) {  // ptxas fatal   : Unresolved extern function '_Z17get_global_offsetj'
  //  size_t gid = item.get_global_id(0);
  void operator()(cl::sycl::id<1> gid) {
    m_C[gid] = m_A[gid] + m_B[gid];
  }

  private:
    read_global_accessor<cl::sycl::cl_float> m_A;
    read_global_accessor<cl::sycl::cl_float> m_B;
    write_global_accessor<cl::sycl::cl_float> m_C;
};



template <typename T>
float simple_vadd(const std::vector<T>& VA, const std::vector<T>& VB,
                 std::vector<T>& VC) {

  cl::sycl::gpu_selector device_selector;
  cl::sycl::property_list prop_list{cl::sycl::property::queue::enable_profiling()};
  cl::sycl::queue queue(device_selector, [=](cl::sycl::exception_list eL) {
      try {
        for (auto& e : eL) {
          std::rethrow_exception(e);
        }
      } catch (cl::sycl::exception ex) {
        std::cout << " There is an exception in the kernel"
                  << std::endl;
        std::cout << ex.what() << std::endl;
      }
    }, prop_list);

  std::cout << "Running on "
        << queue.get_device().template get_info<sycl::info::device::name>()
        << "\n";

  auto context = queue.get_context();
  sycl::program program(context);
  try {
    program.build_with_kernel_type<vector_add_kernel>();
  } catch (cl::sycl::exception ex) {
    std::cout << " There is an exception in compiling the kernel" << std::endl;
    std::cout << ex.what() << std::endl;
  }
  auto kernel = program.get_kernel<vector_add_kernel>();


  cl::sycl::range<1> numOfItems = VC.size();

  cl::sycl::nd_range<1> ndrange_info(VC.size(), block_size_x);

  //const cl::sycl::property_list props = {cl::sycl::property::buffer::use_host_ptr()}; 
  //strangely enough using the use_host_ptr property does not affect our observed runtime
  cl::sycl::buffer<T, 1> bufferA(VA.data(), numOfItems); //, props);
  cl::sycl::buffer<T, 1> bufferB(VB.data(), numOfItems); //, props);
  cl::sycl::buffer<T, 1> bufferC(VC.data(), numOfItems); //, props);

  cl::sycl::event curEvent;

//this loop can be activated to rule out any startup overhead from runtime compilation and/or data allocation/movement
for (int i=0; i<20; i++) {

  auto start = std::chrono::steady_clock::now();
  curEvent = queue.submit([&](cl::sycl::handler& cgh) {
    auto accessorA = bufferA.template get_access<sycl_read>(cgh);
    auto accessorB = bufferB.template get_access<sycl_read>(cgh);
    auto accessorC = bufferC.template get_access<sycl_write>(cgh);

    //not actually using block_size_x for anything now as I can't get the following code to run on my GPU
    //due to the Unresolved extern function '_Z17get_global_offsetj' error
    /*
    auto kern = [=](cl::sycl::nd_item<1> item_id) {
        auto gid = item_id.get_global_id();
       if (gid[0] < numOfItems) {
        accessorC[gid] = accessorA[gid] + accessorB[gid];
        }
    };
    //cgh.parallel_for<class SimpleVadd<T>>(ndrange_info, kern);
    */

    //this version works but doesn't use block_size_x
    //auto kern = [=](cl::sycl::id<1> wiID) {
    //  accessorC[wiID] = accessorA[wiID] + accessorB[wiID];
    //};
    //cgh.parallel_for<class SimpleVadd<T>>(numOfItems, kern);

    //cgh.parallel_for(kernel, ndrange_info, 
    cgh.parallel_for(kernel, numOfItems, 
        vector_add_kernel(accessorA, accessorB, accessorC)
    );
  });

  curEvent.wait();
  auto end = std::chrono::steady_clock::now();

  auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  std::cout << "sycl::event " << ((float) (curEvent.template get_profiling_info<cl::sycl::info::event_profiling::command_end>() -
                  curEvent.template get_profiling_info<cl::sycl::info::event_profiling::command_start>() ) * 1e-6  ) << std::endl; //ms

  std::cout << "std::chrono " << time << std::endl;


    } // end of for-loop around kernel call


  return ((float) (curEvent.template get_profiling_info<cl::sycl::info::event_profiling::command_end>() -
                  curEvent.template get_profiling_info<cl::sycl::info::event_profiling::command_start>() ) * 1e-6  ); //ms

}




extern "C"
float vector_add(float *c, const float *a, const float *b, int n) {

  float time;

  const std::vector<cl::sycl::cl_float>A(a,a+n);
  const std::vector<cl::sycl::cl_float>B(b,b+n);
  std::vector<cl::sycl::cl_float>C(c,c+n);

  time = simple_vadd(A, B, C);

  std:copy(C.begin(), C.end(), c);

  return time;
}

int main() {
    vector_add((float *)0, (float *)0, (float *)0, 0);

}
