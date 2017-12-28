#include <array>
#include <random>

#ifdef __APPLE__
  #include "OpenCL/opencl.h"
#else
  #include "CL/cl.h"
#endif

__kernel void vector_add(__global float *c, __global const float *a, __global const float *b, int n);

int main (int argc, const char * argv[]) {
  // try to get a GPU queue
  dispatch_queue_t queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_GPU, NULL);
 
  // ...or if we don't have a compatible gpu, get a cpu queue
  if (queue == NULL) {
    queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_CPU, NULL);
  }

  // initialize arrays
  int vector_size = 72*1024*1024;
  std::array<float, vector_size> a, b, c;

  // random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> dis();

  // fill arrays
  for (int i = 0; i < vector_size; ++i) {
    a[i] = dis(gen);
    b[i] = dis(gen);
    c[i] = 0;
  }


  dispatch_sync(queue, ^{
        size_t wgs;
        gcl_get_kernel_block_workgroup_info(square_kernel,
                                            CL_KERNEL_WORK_GROUP_SIZE,
                                            sizeof(wgs), &wgs, NULL);
 
        cl_ndrange range = {1, {0, 0, 0}, {NUM_VALUES, 0, 0}, {wgs, 0, 0}            // The local size of each workgroup.  This
                                   // determines the number of work items per
                                   // workgroup.  It indirectly affects the
                                   // number of workgroups, since the global
                                   // size / local size yields the number of
                                   // workgroups.  In this test case, there are
                                   // NUM_VALUE / wgs workgroups.
        };
        // Calling the kernel is easy; simply call it like a function,
        // passing the ndrange as the first parameter, followed by the expected
        // kernel parameters.  Note that we case the 'void*' here to the
        // expected OpenCL types.  Remember, a 'float' in the
        // kernel, is a 'cl_float' from the application's perspective.   // 8
 
        square_kernel(&range,(cl_float*)mem_in, (cl_float*)mem_out);
 
        // Getting data out of the device's memory space is also easy;
        // use gcl_memcpy.  In this case, gcl_memcpy takes the output
        // computed by the kernel and copies it over to the
        // application's memory space.                                   // 9
 
        gcl_memcpy(test_out, mem_out, sizeof(cl_float) * NUM_VALUES);
 
    });


  // vector_add(c, a, b, vector_size);
}