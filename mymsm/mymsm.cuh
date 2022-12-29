
#ifndef __MYMSM_CUH__
#define __MYMSM_CUH__

#include <cuda.h>
#include <cassert>
#include <vector>

#include <util/exception.cuh>
#include <util/rusterror.h>
#include <util/gpu_t.cuh>

#include <stdio.h>

__global__ void msm_kernel(void){
    printf("hello from thread [%d,%d] from device.\n",
            threadIdx.x, blockIdx.x);
}

RustError invoke(
    point_t&         out,
    const affine_t*  points_,
    size_t           npoints,
    const scalar_t*  scalars,
    bool             mont =  true,
    size_t           ffi_affine_sz = sizeof(affine_t)
){
    printf("invoke()\n");
    printf("npoints: %ld\n", npoints);

    msm_kernel<<<2,2>>>();
    cudaDeviceSynchronize();

    return RustError{cudaSuccess};
}
RustError invoke(
    point_t&        out,
    vec_t<affine_t> points,
    const scalar_t* scalars,
    bool            mont = true,
    size_t          ffi_affine_sz = sizeof(affine_t))
{
    return invoke(out, points, points.size(), scalars, mont, ffi_affine_sz);
}

template<class bucket_t, class point_t, class affine_t, class scalar_t> 
static RustError mymsm_pippenger(
    point_t        *out,
    const affine_t points[],
    size_t         npoints,
    const scalar_t scalars[],
    bool           mont = true,
    size_t         ffi_affine_sz = sizeof(affine_t)   
)
{
    printf("mymsm_pippenger()\r\n");
    // *out.add( (point_t) points[0] );
    return invoke(*out, 
                vec_t<affine_t>{points, npoints},
                scalars,
                mont,
                ffi_affine_sz
            );
    // return RustError{cudaSuccess};
}


#endif

