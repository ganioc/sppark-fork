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
#include "./field.cuh"

#define N  4


__global__ void polynomial_kernel(uint16_t blake2_idx, uint8_t* in, uint16_t in_len,  uint8_t *out, scalar_t *scalars);

#ifdef __CUDA_ARCH__




__global__ void polynomial_kernel(uint16_t blake2_idx, uint8_t* in, uint16_t in_len,  uint8_t *out, scalar_t *scalars){
    printf("hello from thread [%d,%d] from device.\n",
            threadIdx.x, blockIdx.x);
    uint16_t idx =  threadIdx.x + blockIdx.x * blockDim.x;
    uint16_t counter =  idx + blake2_idx;
    uint8_t *buf = (uint8_t *)malloc(in_len + 4);
    uint8_t *out_buf = out + idx * 64;
    scalar_t *scalar = scalars + idx;

    for(int i=0; i< in_len; i++){
        buf[i] = in[i];
    }
    // little endian
    buf[in_len + 3] = counter >> 24 & 0xFF;
    buf[in_len + 2] = counter >> 16 & 0xFF;
    buf[in_len + 1] = counter >> 8 & 0xFF;
    buf[in_len + 0] = counter & 0xFF;

    PRINT(buf, in_len + 4);

    printf("\n\n");
    
    int rtn = blake2b512(out_buf, BLAKE2B_OUTBYTES, buf, in_len + 4);

    PRINT(out_buf, BLAKE2B_OUTBYTES);

    free(buf);

    // hash out is in out_buf, BLAKE2B_OUTBYTES bytes [], *scalar has the final result
    from_bytes_le_mod_order(scalar, out_buf, BLAKE2B_OUTBYTES);


    __syncthreads();

    // printf("%4d: %d\n", idx, rtn);
    // for( int i=0; i< BLAKE2B_OUTBYTES; i++){
    //     printf("%02X", out_buf[i]);
    // }
}

#else


RustError polynomial_invoke(size_t degree){
    uint8_t * data;
    uint8_t * d_data;
    uint8_t hash_in[32] = {
        0xEB, 0x89, 0x62, 0x73, 0x02, 0x7B, 0xCF, 0xCA, 0xE1, 0x98, 
        0x20, 0x40, 0x8C, 0x62, 0x5C, 0x49, 0x19, 0xF6, 0x90, 0x97, 
        0x58, 0x5E, 0x3A, 0x14, 0x66, 0x42, 0x49, 0x08, 0xA4, 0xA6, 
        0x3C, 0x4E 
    };
    uint16_t hash_in_len = 32;
    uint8_t * d_hash_in;
    scalar_t * d_scalars;

    cudaMallocHost(&data , 64 * N);
    cudaMalloc(&d_data , 64 * N);
    cudaMalloc(&d_hash_in, 32);
    cudaMalloc(&d_scalars , sizeof(scalar_t) * N);

    for(int i =0; i< hash_in_len; i++){
        printf("%02x ", hash_in[i]);
    }
    printf("\n");

    cudaMemcpy(d_hash_in, hash_in, 32, cudaMemcpyHostToDevice);

    polynomial_kernel<<<1,N>>>(0, d_hash_in, hash_in_len ,d_data, d_scalars);

    cudaMemcpy(data , d_data, 64 * N, cudaMemcpyDeviceToHost);

    // printf("\nResult:\n");
    // for(int i=0; i< 16; i++){
    //     printf("(%d)\n", i);
    //     for(int j = 0; j < 64; j++){
    //         printf("%02X", data[ i* 64 + j]);
    //     }
    //     printf("\n");
    // }


    cudaFree( d_data);
    cudaFree( d_hash_in);
    cudaFree( d_scalars);
    cudaFreeHost( data);

    return RustError{cudaSuccess};
}
static RustError mymsm_polynomial(size_t degree){
    printf("\r\nmymsm_polynomial()\r\n");
    printf("Start GPU computation\n");

    return polynomial_invoke(degree);
}
#endif

#endif

