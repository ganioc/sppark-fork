#ifndef __FIELD_CUH__
#define __FIELD_CUH__

#include <stdio.h>

#define SCALAR_T_MODULUS_BITS        253
#define NUM_MODULUS_BYTES            32
#define NUM_MODULUS_BYTES_MINUS_ONE  31
//#define u64_limbs
#define U64_LIMBS                    4
#define RESULT_SIZE                  (U64_LIMBS * 8 + 1)

#ifdef __CUDA_ARCH__

__device__ void PRINT(uint8_t *buf, uint16_t len){
    char * hex = "0123456789ABCDEF";
    char * str_ptr = (char*)malloc(len*3 + 1);
    for(int i=0; i< len; i++){
        str_ptr[i*3 + 0] = hex[ buf[i] >> 4 & 0xF  ];
        str_ptr[i*3 + 1] = hex[ buf[i] & 0xF];
        str_ptr[i*3 + 2] = ' ';
    }
    str_ptr[len*3] = 0;
    printf("%s\n", str_ptr);

    free(str_ptr);
}

__device__ void reverse(uint8_t * buf, uint16_t len){
    uint8_t temp;
    for(int i = 0; i<len/2; i++){
        temp = buf[i];
        buf[i] = buf[len-i-1];
        buf[len-i-1] = temp;
    } 
}
__device__ fr_t byte_to_fr_t(uint32_t a){
    uint32_t p[8] = {0x00000000, 0x00000000, 0x00000000, 0x00000000, 
        0x00000000, 0x00000000, 0x00000000, 0x00000000};
    p[0] = a;
    return fr_t(p);
}

__device__ fr_t from_random_bytes(uint8_t * buf, uint16_t len){
    uint8_t result_bytes[RESULT_SIZE]={0};
    uint32_t fr_t_arr[8];

    for(int i=0; i< len; i++){
        result_bytes[i] = buf[i];
    }
    result_bytes[len] = 0;

    PRINT(result_bytes, 32);

    fr_t_arr[0] = (uint32_t)result_bytes[3]<<24 | (uint32_t)result_bytes[2] << 16 |
                (uint32_t)result_bytes[1] << 8  | (uint32_t)result_bytes[0];
    fr_t_arr[1] = (uint32_t)result_bytes[7]<<24 | (uint32_t)result_bytes[6] << 16 |
                (uint32_t)result_bytes[5] << 8  | (uint32_t)result_bytes[4];
    fr_t_arr[2] = (uint32_t)result_bytes[11]<<24 | (uint32_t)result_bytes[10] << 16 |
                (uint32_t)result_bytes[9] << 8  | (uint32_t)result_bytes[8];
    fr_t_arr[3] = (uint32_t)result_bytes[15]<<24 | (uint32_t)result_bytes[14] << 16 |
                (uint32_t)result_bytes[13] << 8  | (uint32_t)result_bytes[12];
    fr_t_arr[4] = (uint32_t)result_bytes[19]<<24 | (uint32_t)result_bytes[18] << 16 |
                (uint32_t)result_bytes[17] << 8  | (uint32_t)result_bytes[16];
    fr_t_arr[5] = (uint32_t)result_bytes[23]<<24 | (uint32_t)result_bytes[22] << 16 |
                (uint32_t)result_bytes[21] << 8  | (uint32_t)result_bytes[20];
    fr_t_arr[6] = (uint32_t)result_bytes[27]<<24 | (uint32_t)result_bytes[26] << 16 |
                (uint32_t)result_bytes[25] << 8  | (uint32_t)result_bytes[24];
    fr_t_arr[7] = (uint32_t)result_bytes[31]<<24 | (uint32_t)result_bytes[30] << 16 |
                (uint32_t)result_bytes[29] << 8  | (uint32_t)result_bytes[28];

    return fr_t(fr_t_arr);
}
__device__ void from_bytes_le_mod_order(scalar_t *scalar, uint8_t *buf, uint16_t len){
    uint8_t * leading_bytes, *bytes_to_directly_convert;
    uint8_t * remaining_bytes;
    int num_bytes_to_directly_convert, num_remaining_bytes;
    // reverse array buf,
    reverse(buf, len);

    // from_bytes_be_mod_order()
    // int num_bytes_to_directly_convert = min(NUM_MODULUS_BYTES_MINUS_ONE, len);
    num_bytes_to_directly_convert = NUM_MODULUS_BYTES_MINUS_ONE;
    num_remaining_bytes = len - num_bytes_to_directly_convert;

    bytes_to_directly_convert = leading_bytes = buf;
    remaining_bytes = buf + num_bytes_to_directly_convert;

    // Copy the leading big-endian bytes directly into a field element.
    // The number of bytes directly converted must be less than the
    // number of bytes needed to represent the modulus, as we must begin
    // modular reduction once the data is of the same number of bytes as the modulus.
    reverse(bytes_to_directly_convert, num_bytes_to_directly_convert);

    // Guaranteed to not be None, as the input is less than the modulus size.
    // from_random_bytes
    fr_t res = from_random_bytes(bytes_to_directly_convert, num_bytes_to_directly_convert);
    printf("fr_t len(): %d\n", res.len());
    printf("res %08X %08X %08X %08X %08X %08X %08X %08X\n",
        res[0], res[1],res[2],res[3],res[4],res[5],res[6],res[7]
    );

    PRINT(remaining_bytes, num_remaining_bytes);
    
    fr_t window_size = byte_to_fr_t(256);
    printf("window_size in fr_t: %08X %08X %08X %08X %08X %08X %08X %08X\n",
        window_size[0], window_size[1],window_size[2],window_size[3],
        window_size[4],window_size[5],window_size[6],window_size[7]
    );

    for(int i = 0; i < num_remaining_bytes; i++){
        res = res * window_size;
        res.to(); // must add res.to() after multiply, not for plus, 
        res = res + byte_to_fr_t(remaining_bytes[i]);
    }
    printf("res after: %08X %08X %08X %08X %08X %08X %08X %08X\n",
        res[0], res[1],res[2],res[3],res[4],res[5],res[6],res[7]);

    *scalar = res;
}
#endif

#endif


