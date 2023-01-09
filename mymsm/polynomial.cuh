#ifndef __POLYNOMIAL_CUH__
#define __POLYNOMIAL_CUH__

#include <cuda.h>
#include <cassert>
#include <vector>

#include <util/exception.cuh>
#include <util/rusterror.h>
#include <util/gpu_t.cuh>

#include <stdio.h>

#include "./blake2b.cuh"


__global__ void polynomial_kernel(uint16_t blake2_idx, uint8_t *out){
    printf("hello from thread [%d,%d] from device.\n",
            threadIdx.x, blockIdx.x);
    uint16_t idx =  threadIdx.x + blockIdx.x * blockDim.x;
    uint16_t counter =  idx + blake2_idx;
    uint8_t buf[4];
    uint8_t *out_buf = out + idx * 64;
    // little endian
    buf[3] = counter >> 24 & 0xFF;
    buf[2] = counter >> 16 & 0xFF;
    buf[1] = counter >> 8 & 0xFF;
    buf[0] = counter & 0xFF;
    
    int rtn = blake2b512(out_buf, BLAKE2B_OUTBYTES, buf, 4);

    __syncthreads();

    printf("%4d: %d\n", idx, rtn);
    for( int i=0; i< BLAKE2B_OUTBYTES; i++){
        printf("%02X", out_buf[i]);
    }
}
RustError polynomial_invoke(size_t degree){
    uint8_t * data;
    uint8_t * d_data;
    cudaMallocHost(&data , BLAKE2B_OUTBYTES * 16);
    cudaMalloc(&d_data , BLAKE2B_OUTBYTES * 16);

    polynomial_kernel<<<1,16>>>(0, d_data);

    cudaMemcpy(data , d_data, BLAKE2B_OUTBYTES * 16, cudaMemcpyDeviceToHost);

    printf("\nResult:\n");
    for(int i=0; i< 16; i++){
        printf("(%d)\n", i);
        for(int j = 0; j < BLAKE2B_OUTBYTES; j++){
            printf("%02X", data[ i* BLAKE2B_OUTBYTES + j]);
        }
        printf("\n");
    }


    cudaFree( d_data);
    cudaFreeHost( data);

    return RustError{cudaSuccess};
}
static RustError mymsm_polynomial(size_t degree){
    printf("\r\nmymsm_polynomial()\r\n");
    printf("Start GPU computation\n");

    return polynomial_invoke(degree);
}


#endif

