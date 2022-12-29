
#ifndef __MYMSM_CUH__
#define __MYMSM_CUH__

#include <cuda.h>
#include <cassert>
#include <vector>

#include <util/exception.cuh>
#include <util/rusterror.h>
#include <util/gpu_t.cuh>

#include <stdio.h>

#define NTHREADS  256
#define WARP_SZ   32
#define WBITS     11 // Suitable for ~2^16
#define NWINS     23

__global__ void msm_kernel(void){
    printf("hello from thread [%d,%d] from device.\n",
            threadIdx.x, blockIdx.x);
}

cudaDeviceProp prop;
size_t N;

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

    // bucket methods
    bucket_t *d_buckets; // [NWINS][1<<WBITS];
    affine_t *d_points;

    size_t blob_sz = sizeof(d_buckets[0]);
    printf("blob_sz: %ld\n", blob_sz);
    size_t n       = (npoints + WARP_SZ -1) & ((size_t) 0 - WARP_SZ);
    printf("n :%ld\n", n);
    blob_sz  += n * sizeof(*d_points);
    printf("blob_sz: %ld\n", blob_sz);


    // 分配内存
    // d_buckets = 
    cudaMalloc( &d_buckets, sizeof(bucket_t) * NWINS * (1<<WBITS));
    // d_points = 
    cudaMalloc( &d_points, sizeof(affine_t) * npoints);

    msm_kernel<<<2,2>>>();
    cudaDeviceSynchronize();

    // 释放内存
    cudaFree( d_buckets);
    cudaFree( d_points );
    
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
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess &&
                prop.major >= 7) 
    {
        printf("read GPU property succeed\n");
        printf("sm count:%ld\n", prop.multiProcessorCount);
    }
    N = (32*256) / (NTHREADS*NWINS);
    printf("N :%ld\n", N); // N is 1,


    return invoke(*out, 
                vec_t<affine_t>{points, npoints},
                scalars,
                mont,
                ffi_affine_sz
            );
    // return RustError{cudaSuccess};
}


#endif

