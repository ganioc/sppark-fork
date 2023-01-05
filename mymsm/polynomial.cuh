#ifndef __POLYNOMIAL_CUH__
#define __POLYNOMIAL_CUH__

#include <cuda.h>
#include <cassert>
#include <vector>

#include <util/exception.cuh>
#include <util/rusterror.h>
#include <util/gpu_t.cuh>

#include <stdio.h>

__global__ void polynomial_kernel(){
    printf("hello from thread [%d,%d] from device.\n",
            threadIdx.x, blockIdx.x);
}
RustError polynomial_invoke(size_t degree){

    uint8_t * d_data;
    cudaMalloc(&d_data , degree);

    polynomial_kernel<<<1,16>>>();

    cudaFree( d_data);

    return RustError{cudaSuccess};
}
static RustError mymsm_polynomial(size_t degree){
    printf("\r\nmymsm_polynomial()\r\n");
    printf("Start GPU computation\n");

    return polynomial_invoke(degree);
}


#endif

