#define VERTICES 600

__constant__ float2 d_vertices[VERTICES];
__constant__ float d_slopes[VERTICES];

/*
 * This file contains the implementation of a CUDA Kernel for the
 * point-in-polygon problem using the crossing number algorithm
 *
 * The kernel cn_pnpoly is can be tuned using the following parameters:
 *    * block_size_x                any sensible thread block size
 *    * tile_size                   any sensible tile size value
 *    * between_method              any of [0, 1, 2, 3]
 *    * use_precomputed_slopes      enable or disable [0, 1]
 *    * use_method                  any of [0, 1]
 *
 * The kernel cn_pnpoly_naive is used for correctness checking.
 *
 * The algorithm used here is adapted from: 
 *     'Inclusion of a Point in a Polygon', Dan Sunday, 2001
 *     (http://geomalgorithms.com/a03-_inclusion.html)
 *
 * Author: Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 */

#ifndef block_size_x
    #define block_size_x 256
#endif
#ifndef block_size_y
    #define block_size_y 1
#endif
#ifndef block_size_z
    #define block_size_z 1
#endif

#ifndef tile_size
    #define tile_size 1
#endif

#ifndef between_method
    #define between_method 0
#endif

#ifndef use_precomputed_slopes
    #define use_precomputed_slopes 0
#endif

#ifndef use_method
    #define use_method 0
#endif



/*
 * The is_between method returns a boolean that is True when the a is between c and b.
 * Since the kernel is instruction bound, the exact way in which you compute is_between
 * can have a dramatic effect on performance.
 * Note that the way the different methods handle coincidents of a with b and c differs slightly.
 */
__device__ __forceinline__ int is_between(float a, float b, float c) {
    #if between_method == 0
        return (b > a) != (c > a);
    #elif between_method == 1
        return ((b <= a) && (c > a)) || ((b > a) && (c <= a));
    #elif between_method == 2
        return ((a - b) == 0.0f) || ((a - b) * (a - c) < 0.0f);
    #elif between_method == 3
        //Interestingly enough method 3 exactly the same as method 2, only in a different order.
        //the performance difference between method 2 and 3 can be huge depending on all the other optimization parameters.
        return ((a - b) * (a - c) < 0.0f) || (a - b == 0.0f);
    #endif
}



/*
 * The Point-in-Polygon kernel
 */
__global__ void cn_pnpoly(int* bitmap, float2* points, int n) {
    int i = blockIdx.x * block_size_x * tile_size + threadIdx.x;
    if (i < n) {

        int c[tile_size];
        float2 lpoints[tile_size];
        #pragma unroll
        for (int ti=0; ti<tile_size; ti++) {
            c[ti] = 0;
            if (i+block_size_x*ti < n) {
                lpoints[ti] = points[i+block_size_x*ti];
            }
        }

        int k = VERTICES-1;

        for (int j=0; j<VERTICES; k = j++) {    // edge from vj to vk
            float2 vj = d_vertices[j]; 
            float2 vk = d_vertices[k]; 

            #if use_precomputed_slopes == 0
            float slope = (vk.x-vj.x) / (vk.y-vj.y);
            #elif use_precomputed_slopes == 1
            float slope = d_slopes[j];
            #endif

            #pragma unroll
            for (int ti=0; ti<tile_size; ti++) {

                float2 p = lpoints[ti];

                #if use_method == 0
                if (  is_between(p.y, vj.y, vk.y) &&         //if p is between vj and vk vertically
                     (p.x < slope * (p.y-vj.y) + vj.x)
                        ) {  //if p.x crosses the line vj-vk when moved in positive x-direction
                    c[ti] = !c[ti];
                }

                #elif use_method == 1
                //Same as method 0, but attempts to reduce divergence by avoiding the use of an if-statement.
                //Whether this is more efficient is data dependent because there will be no divergence using method 0, when none
                //of the threads within a warp evaluate is_between as true
                int b = is_between(p.y, vj.y, vk.y);
                c[ti] += b && (p.x < vj.x + slope * (p.y - vj.y));

                #endif


            }

        }

        #pragma unroll
        for (int ti=0; ti<tile_size; ti++) {
            //could do an if statement here if 1s are expected to be rare
            if (i+block_size_x*ti < n) {
                #if use_method == 0
                bitmap[i+block_size_x*ti] = c[ti];
                #elif use_method == 1
                bitmap[i+block_size_x*ti] = c[ti] & 1;
                #endif
            }
        }
    }

}


/*
 * The naive implementation is used for verifying correctness of the optimized implementation
 * And in turn we have a unit test written in Python for this kernel
 */
__global__ void cn_pnpoly_naive(int* bitmap, float2* points, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        int c = 0;
        float2 p = points[i];

        int k = VERTICES-1;

        for (int j=0; j<VERTICES; k = j++) {    // edge from v to vp
            float2 vj = d_vertices[j]; 
            float2 vk = d_vertices[k]; 

            float slope = (vk.x-vj.x) / (vk.y-vj.y);

            if ( (  (vj.y>p.y) != (vk.y>p.y)) &&            //if p is between vj and vk vertically
                    (p.x < slope * (p.y-vj.y) + vj.x) ) {   //if p.x crosses the line vj-vk when moved in positive x-direction
                c = !c;
            }
        }

        bitmap[i] = c; // 0 if even (out), and 1 if odd (in)
    }


}


