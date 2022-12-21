
#ifndef __MYMSM_CUH__
#define __MYMSM_CUH__

#include <cuda.h>

#include <util/rusterror.h>

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
    return RustError{1};
}


#endif

