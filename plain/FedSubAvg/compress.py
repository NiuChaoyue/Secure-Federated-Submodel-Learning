# -*- coding: UTF-8 -*-
import numpy as np


def uint_para_compress(delta_model_para, hyperparameters):
    compress_k_levels = hyperparameters['compress_k_levels']
    compress_bound = hyperparameters['compress_bound']
    learning_rate = hyperparameters['learning_rate']
    if compress_k_levels < 2:
        return delta_model_para
    upper_bound = compress_bound * learning_rate
    lower_bound = -1.0 * compress_bound * learning_rate
    compressed_model = []
    k_level_delta = np.linspace(lower_bound, upper_bound, compress_k_levels)
    dist = (upper_bound - lower_bound) / (compress_k_levels - 1)
    for delta_model_index, para in enumerate(delta_model_para):
        para_shape = para.shape
        if para.ndim == 2:
            para = para.flatten()
        #print ((para+compress_bound * learning_rate)/dist)
        # Map all elements falling to [lower_bound, upper_bound]
        para[para > upper_bound] = upper_bound
        para[para < lower_bound] = lower_bound
        argmin_less = np.floor((para - lower_bound)/dist).astype(np.int32)
        argmin_larger = np.ceil((para - lower_bound)/dist).astype(np.int32)
        prop = (para-(k_level_delta[argmin_less]))/dist
        rannum = np.random.rand(len(para))
        int_array = np.where(rannum < prop, argmin_larger, argmin_less)
        if compress_k_levels <= 2**8:
            int_array = int_array.astype(np.uint8)
        elif compress_k_levels <= 2**16:
            int_array = int_array.astype(np.uint16)
        else:
            int_array = int_array.astype(np.uint32)
        int_array = int_array.reshape(para_shape)
        compressed_model.append(int_array)
    return compressed_model


def recover_compression(compressed_array, hyperparameters):
    compress_k_levels = hyperparameters['compress_k_levels']
    compress_bound = hyperparameters['compress_bound']
    learning_rate = hyperparameters['learning_rate']
    if compress_k_levels < 2:
        return compressed_array
    upper_bound = compress_bound * learning_rate
    lower_bound = -1.0 * compress_bound * learning_rate
    dist = (upper_bound - lower_bound)/(compress_k_levels - 1)
    recovered_array = compressed_array * dist + lower_bound
    return recovered_array
    