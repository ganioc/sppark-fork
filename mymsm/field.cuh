#ifndef __FIELD_CUH__
#define __FIELD_CUH__

#include <stdio.h>

#define SCALAR_T_MODULUS_BITS        253
#define NUM_MODULUS_BYTES            32
#define NUM_MODULUS_BYTES_MINUS_ONE  31
//#define u64_limbs
#define U64_LIMBS                    4


#ifdef __CUDA_ARCH__

__device__ void reverse(uint8_t * buf, uint16_t len){
    uint8_t temp;
    for(int i = 0; i<len/2; i++){
        temp = buf[i];
        buf[i] = buf[len-i-1];
        buf[len-i-1] = temp;
    } 
}

__device__ void from_bytes_le_mod_order(scalar_t *scalar, uint8_t *buf, uint16_t len){
    uint8_t * leading_bytes;
    uint8_t * remaining_bytes;
    // reverse array buf,
    reverse(buf, len);

    // from_bytes_be_mod_order()
    int num_bytes_to_directly_convert = min(NUM_MODULUS_BYTES_MINUS_ONE, len);
    leading_bytes = buf;
    remaining_bytes = buf + num_bytes_to_directly_convert;

    // Copy the leading big-endian bytes directly into a field element.
    // The number of bytes directly converted must be less than the
    // number of bytes needed to represent the modulus, as we must begin
    // modular reduction once the data is of the same number of bytes as the modulus.
    reverse(leading_bytes, num_bytes_to_directly_convert);

    // Guaranteed to not be None, as the input is less than the modulus size.
    // from_random_bytes
    fr_t a;
    printf("fr_t len(): %d\n", a.len());
    

}
#endif

#endif


