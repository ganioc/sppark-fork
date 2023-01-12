// The file to call mymsm/

#include <cuda.h>

#if defined(FEATURE_BLS12_377)
#include <ff/bls12-377.hpp>
#endif

#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_inf_t affine_t;
typedef fr_t scalar_t;

// #include <mymsm/mymsm.cuh>
#include <mymsm/polynomial.cuh>


#ifndef __CUDA_ARCH__
extern "C" {
    // RustError mymsm_pippenger_inf(
    //     point_t*        out,
    //     const affine_t  points[],
    //     size_t          npoints,
    //     const scalar_t  scalars[],
    //     size_t          ffi_affine_sz
    // ){
    //     return mymsm_pippenger<bucket_t>(out, points, npoints, scalars, false, ffi_affine_sz);
    // }

    RustError mymsm_polynomial_inf(size_t degree){
        return mymsm_polynomial(degree);
    }

}


#endif

