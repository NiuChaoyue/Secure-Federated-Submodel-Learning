# -*- coding: utf-8 -*-

from hmac_drbg import HMAC_DRBG
import numpy as np

#############################################################################################
# Pseudo random number generator
# Input: DRBG instance, modulo_r_len, security strength in bits, and vector_len
# Output: random numbers array similar to np.random.randint(0, MODULO_R, VECTOR_LEN) but with cryptographic security
#############################################################################################

# If in python3: int.from_bytes(bytes, byteorder, *, signed=False), byteorder = 'big':
#     then in python 2: int(bytes.encode('hex'), 16)
# IF in python 3: int.from_bytes(bytes, byteorder='little'), but here python 2
#    then in python 2: int(''.join(reversed(bytes)).encode('hex'), 16)
# !!!!consistently use big here


def bytes2int(bytes):
    return int(bytes.encode('hex'), 16)


#python3: int.to_bytes(length, byteorder, *, signed=False)
def int2bytes(num_int, bytes_len):
    bytes = ""
    for i in range(bytes_len - 1, -1, -1):
        bytes += chr((num_int >> (i * 8)) & 0xFF)
    return bytes

def prng(DRBG, modulo_r_len, security_strength, vector_len):
    if modulo_r_len <= 8:
        rand_arr = np.zeros(vector_len, dtype='uint8')
    elif modulo_r_len <= 16:
        rand_arr = np.zeros(vector_len, dtype='uint16')
    elif modulo_r_len <= 32:
        rand_arr = np.zeros(vector_len, dtype='uint32')
    else:
        rand_arr = np.zeros(vector_len, dtype='uint64')
    bytes_num_per_element = modulo_r_len / 8
    total_bytes_num = bytes_num_per_element * vector_len
    # 7500 bits maximum per request
    # Please use multiplies of bytes_num_per_element
    max_bytes_per_request = int(7500 / 8)
    max_bytes_per_request = int(max_bytes_per_request / bytes_num_per_element) * bytes_num_per_element
    request_num = total_bytes_num / max_bytes_per_request
    cnt = 0
    for req in range(request_num):
        bytes_stream = DRBG.generate(max_bytes_per_request, security_strength)
        for i in range(0, max_bytes_per_request, bytes_num_per_element):
            bytes = bytes_stream[i:(i+bytes_num_per_element)]
            rand_arr[cnt] = bytes2int(bytes)
            cnt += 1
    remaining_bytes_num = total_bytes_num % max_bytes_per_request
    if remaining_bytes_num != 0:
        bytes_stream = DRBG.generate(remaining_bytes_num, security_strength)
        for i in range(0, remaining_bytes_num, bytes_num_per_element):
            bytes = bytes_stream[i:(i+bytes_num_per_element)]
            rand_arr[cnt] = bytes2int(bytes)
            cnt += 1
    if cnt != vector_len:
        print("Have not generated enough random numbers!")
    return rand_arr
