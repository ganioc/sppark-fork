
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

__global__ void msm_kernel(bucket_t *d_buckets,affine_t *d_points,scalar_t *d_scalars ){
    printf("hello from thread [%d,%d] from device.\n",
            threadIdx.x, blockIdx.x);
    // if(threadIdx.x == 0){
        
    // }
    int idx = threadIdx.x % NWINS;
    int group_idx = threadIdx.x / NWINS;
    // bucket_t* bucket_row = d_bucket + group_idx * (1 << WBITS);
    // scalar_t *scalar = d_scalars[group_idx];
    // affine_t *point  = d_points[group_idx];

    printf("idx %d\n", idx);
    printf("group_idx %d\n", group_idx);
    scalar_t scalar = d_scalars[group_idx];
    affine_t point  = d_points[group_idx];
    // scalar_t 如何转换成长整数呢?
    // int row_idx = (scalar >> (idx * WBITS)) & (0U - 1<<WBITS)
    // bucket_t bucket = *(d_bucket + group_idx *(1<< WBITS) + row_idx);


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
    scalar_t *d_scalars;

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
    cudaMalloc( &d_scalars, sizeof(scalar_t) * npoints);

    cudaMemcpy( d_points, points_,sizeof(affine_t) * npoints , cudaMemcpyHostToDevice);
    cudaMemcpy(d_scalars, scalars ,sizeof(scalar_t) * npoints , cudaMemcpyHostToDevice);

    msm_kernel<<<1,NWINS>>>(d_buckets, d_points, d_scalars);
    cudaDeviceSynchronize();

    // 释放内存
    cudaFree( d_buckets);
    cudaFree( d_points );
    cudaFree( d_scalars);

    // 这些API都是用在GPU上运行的,
    // out.inf();
    // out.add(points_[0]);

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
    printf("sizeof size_t:%ld\n", sizeof(size_t));


    return invoke(*out, 
                vec_t<affine_t>{points, npoints},
                scalars,
                mont,
                ffi_affine_sz
            );
    // return RustError{cudaSuccess};
}


#endif

