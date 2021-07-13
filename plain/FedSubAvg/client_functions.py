import sys
import os
import tensorflow as tf
import numpy as np
import pickle
import random
import math
from compress import uint_para_compress

# ============================== START: functions to extract real index set ======================================

def extract_real_index_set(client_index, max_len=100):
    """Extract client's real index set from her training data."""
    f_client_train = open('../taobao_data_process/taobao_datasets/user_%d' % client_index, 'r')
    real_train_set_size = 0
    userID = 0
    real_itemIDs = []
    for line in f_client_train:
        ss = line.strip("\n").split("\t")
        if line == "" or ss == '':
            break
        uid = int(ss[1])
        mid = int(ss[2])
        mid_list = []
        for fea in ss[4].split(""):
            mid_list.append(int(fea))
        if len(mid_list) > max_len:
            mid_list = mid_list[len(mid_list) - max_len:]
        userID = uid
        real_itemIDs.append(mid)
        real_itemIDs += mid_list
        real_train_set_size += 1
    userID = [userID]
    real_itemIDs = list( set(real_itemIDs) )
    real_itemIDs.sort()
    return userID, real_itemIDs, real_train_set_size

# ============================== START: functions to extract real index set ======================================


# ============================== START: functions to generate succinct dataset ======================================

def double_randomized_response(real_IDs, permanent_YES_IDs, permanent_NO_IDs, prob1, prob3, prob4):
    """Apply randomize response twice (Only) to real ids"""
    #real_IDs -> S
    once_perturbed_IDs = []  # S'
    double_perturbed_IDs = []  # S''
    # Permanent randomized response
    for j in real_IDs: # S
        if j not in permanent_YES_IDs and j not in permanent_NO_IDs:
            if random.uniform(0.0, 1.0) <= prob1:
                once_perturbed_IDs.append(j)  # add j to S' with probability prob1
                permanent_YES_IDs.append(j)
            else:
                permanent_NO_IDs.append(j)
    permanent_YES_IDs = list( set(permanent_YES_IDs) )
    permanent_NO_IDs = list( set(permanent_NO_IDs) )

    # Instantaneous randomized response
    for j in real_IDs: # S
        if j in permanent_YES_IDs:
            if random.uniform(0.0, 1.0) <= prob3:   # add j to S'' with probability prob3
                double_perturbed_IDs.append(j)
        else:
            if random.uniform(0.0, 1.0) <= prob4:   # add j to S'' with probability prob4
                double_perturbed_IDs.append(j)
    double_perturbed_IDs = list ( set(double_perturbed_IDs) )
    double_perturbed_IDs.sort()
    return double_perturbed_IDs


def generate_succinct_item_ids(client_index, real_itemIDs, prob1, prob3, prob4):
    """Generate succinct item ids by apply double randomized responses only to real item ids"""
    permanent_YES_itemIDs = []
    permanent_NO_itemIDs = []
    # load memoized permanent answers
    filename = './permanent_answers/user_%d.pkl' % client_index
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            permanent_YES_itemIDs = pickle.load(f)
            permanent_NO_itemIDs = pickle.load(f)

    # apply double randomize responses to real item ids, and here directly output succinct item ids
    succinct_itemIDs = double_randomized_response(real_itemIDs, permanent_YES_itemIDs, permanent_NO_itemIDs, prob1, prob3, prob4)

    # memoize (updated) permanent answers
    with open(filename, 'w') as f:
        pickle.dump(permanent_YES_itemIDs, f)
        pickle.dump(permanent_NO_itemIDs, f)

    return succinct_itemIDs


def generate_succinct_cate_ids(succinct_itemIDs, item_cate_dict):
    """Utilize item-cate map to generate succinct cate ids from succinct item ids"""
    succinct_cateIDs = []
    for itemID in succinct_itemIDs:
        cateID = int(item_cate_dict[str(itemID)])
        succinct_cateIDs.append(cateID)
    succinct_cateIDs = list( set(succinct_cateIDs) )
    succinct_cateIDs.sort()
    return succinct_cateIDs


def prepare_succinct_training_data(client_index, machine_index, succinct_itemIDs, succinct_cateIDs, item_cate_dict, max_len=100):
    """Utilize succinct item ids to prepare succinct training set, remove those irrelevant data"""
    # map ps's index to client's local index
    # succinct_userID_map = {userID: 0}
    succinct_itemIDs_map = dict(zip(succinct_itemIDs, range(len(succinct_itemIDs))))
    succinct_cateIDs_map = dict(zip(succinct_cateIDs, range(len(succinct_cateIDs))))

    # counting statistics over userID, succinct_itemIDs, succinct_cateIDs
    # using client's local indices [0, 1, ...]
    # succinct_userID_count: succinct_train_set_size
    succinct_itemIDs_count = np.zeros(len(succinct_itemIDs), dtype='uint32')
    succinct_cateIDs_count = np.zeros(len(succinct_cateIDs), dtype='uint32')
    succinct_train_set_size = 0

    f_client_train = open('../taobao_data_process/taobao_datasets/user_%d' % client_index, 'r')
    # Please use machine index to prepare the succinct data file to avoid the quicker machine in the next communication
    # round to choose the same client index as the slower machine in the previous communication round
    succinct_filename = './taobao_succinct_datasets/machine_%d' % machine_index
    with open(succinct_filename, 'w') as f_succinct:
        for line in f_client_train:
            ss = line.strip("\n").split("\t")
            if line == "" or ss == '':
                break

            clk = ss[0]
            uid = int(ss[1])   #mapped to 0 in fact
            mid = int(ss[2])
            cat = int(item_cate_dict[ss[2]])
            mid_list = []
            for fea in ss[4].split(""):
                mid_list.append(int(fea))

            if len(mid_list) > max_len:
                mid_list = mid_list[len(mid_list) - max_len :]

            if mid in succinct_itemIDs:
                # for counting usages
                all_new_mids = []
                all_new_cats = []

                new_uid = '0'
                new_mid = str(succinct_itemIDs_map[mid])
                new_cat = str(succinct_cateIDs_map[cat])

                all_new_mids.append(succinct_itemIDs_map[mid])
                all_new_cats.append(succinct_cateIDs_map[cat])

                # remove those historic item ids not in the succinct itemIDs
                # Attention: one cate id may correspond to multiple item ids!!!
                new_mid_str_list = []
                new_cat_str_list = []
                for m in mid_list:
                    if m in succinct_itemIDs:
                        new_mid_str_list.append(str(succinct_itemIDs_map[m]))
                        c = int(item_cate_dict[str(m)])
                        new_cat_str_list.append(str(succinct_cateIDs_map[c]))
                        all_new_mids.append(succinct_itemIDs_map[m])
                        all_new_cats.append(succinct_cateIDs_map[c])
                if len(new_mid_str_list) > 0:
                    new_mid_list = "".join(new_mid_str_list)
                    new_cat_list = "".join(new_cat_str_list)
                    newline = clk + "\t" + new_uid + "\t" + new_mid + "\t" + new_cat + "\t" + new_mid_list + "\t" + new_cat_list
                    print >> f_succinct, newline
                    # counting usage
                    all_new_mids = list( set(all_new_mids) )
                    all_new_cats = list( set(all_new_cats) )
                    for nm in all_new_mids:
                        succinct_itemIDs_count[nm] += 1
                    for nc in all_new_cats:
                        succinct_cateIDs_count[nc] += 1
                    succinct_train_set_size += 1

    return succinct_filename, succinct_itemIDs_count, succinct_cateIDs_count, succinct_train_set_size

# ============================== END: functions to generate succinct dataset ======================================

# ============================== END: functions to generate succinct weighted submodel update ======================================

def submodel_update_multiply_count_number(compressed_delta_submodel, userID_count, itemIDs_count, cateIDs_count, other_count):
    """Multiply the submodel update with its count/weight at the client's side"""
    weighted_delta_submodel = [np.zeros(weights.shape, dtype='uint32') for weights in compressed_delta_submodel]
    temp_compressed_delta_submodel = [weights.astype('uint32') for weights in compressed_delta_submodel]  # to avoid overflow
    for layer, delta_submodel_para in enumerate(temp_compressed_delta_submodel):
        if layer == 0: # embedding for user id
            weighted_delta_submodel[layer] = delta_submodel_para * userID_count
        elif layer == 1: # embedding for item ids
            for sidx in range(len(delta_submodel_para)):  # succinct item index (at local client)
                weighted_delta_submodel[layer][sidx] = delta_submodel_para[sidx] * itemIDs_count[sidx]
        elif layer == 2: # embedding for cate ids
            for cidx in range(len(delta_submodel_para)): # succinct cate index (at local client)
                weighted_delta_submodel[layer][cidx] = delta_submodel_para[cidx] * cateIDs_count[cidx]
        else:
            weighted_delta_submodel[layer] = delta_submodel_para * other_count
    return weighted_delta_submodel


def generate_succinct_weighted_submodel_update(old_succinct_submodel, new_succinct_submodel, succinct_itemIDs_count,\
                                               succinct_cateIDs_count, succinct_train_set_size, hyperparameters):
    """Generate the weighted submodel update to be uploaded using succinct index set
       According to training set size or just evenly."""
    # Generate pure submodel update using succinct index set
    #print('')
    delta_submodel = [np.zeros(weights.shape) for weights in old_succinct_submodel]
    for layer in range(len(old_succinct_submodel)):
        delta_submodel[layer] = new_succinct_submodel[layer] - old_succinct_submodel[layer]
        #print("layer %d: max element %f and min element %f"%(layer, np.max(delta_submodel[layer]), np.min(delta_submodel[layer])))
    #sys.stdout.flush()

    # Compress submodel update
    compressed_delta_submodel = uint_para_compress(delta_submodel, hyperparameters)

    if hyperparameters['size_agg_flag'] == 0:
        even_userID_count = False
        even_train_set_size = False
        if succinct_train_set_size > 0:
            even_userID_count = True
            even_train_set_size = True
        even_itemIDs_count = np.zeros(len(succinct_itemIDs_count), dtype='bool')
        for sidx, sic in enumerate(succinct_itemIDs_count):  # succinct item id, succinct item count
            if sic > 0:
                even_itemIDs_count[sidx] = True
        even_cateIDs_count = np.zeros(len(succinct_cateIDs_count), dtype='bool')
        for scdx, scc in enumerate(succinct_cateIDs_count):  # succinct cate id, succinct cate count
            if scc > 0:
                even_cateIDs_count[scdx] = True
        weighted_delta_submodel0 = submodel_update_multiply_count_number(compressed_delta_submodel, even_userID_count,\
                                                           even_itemIDs_count, even_cateIDs_count, even_train_set_size)
        return weighted_delta_submodel0, even_userID_count, even_itemIDs_count, even_cateIDs_count, even_train_set_size
    elif hyperparameters['size_agg_flag'] == 1:
        # multiply the other parameters except embedding in the compressed submodel update with the training set size
        even_userID_count = False
        if succinct_train_set_size > 0:
            even_userID_count = True
        even_itemIDs_count = np.zeros(len(succinct_itemIDs_count), dtype='bool')
        for sidx, sic in enumerate(succinct_itemIDs_count):  # succinct item id, succinct item count
            if sic > 0:
                even_itemIDs_count[sidx] = True
        even_cateIDs_count = np.zeros(len(succinct_cateIDs_count), dtype='bool')
        for scdx, scc in enumerate(succinct_cateIDs_count):  # succinct cate id, succinct cate count
            if scc > 0:
                even_cateIDs_count[scdx] = True
        weighted_delta_submodel1 = submodel_update_multiply_count_number(compressed_delta_submodel, even_userID_count, \
                                                    even_itemIDs_count, even_cateIDs_count, succinct_train_set_size)
        return weighted_delta_submodel1, even_userID_count, even_itemIDs_count, even_cateIDs_count, succinct_train_set_size
    elif hyperparameters['size_agg_flag'] == 2:
        # multiply the whole compressed submodel update with corresponding count numbers
        succinct_userID_count = succinct_train_set_size
        weighted_delta_submodel2 = submodel_update_multiply_count_number(compressed_delta_submodel, succinct_userID_count,\
                                                succinct_itemIDs_count, succinct_cateIDs_count, succinct_train_set_size)
        return weighted_delta_submodel2, succinct_userID_count, succinct_itemIDs_count, succinct_cateIDs_count, succinct_train_set_size
    elif hyperparameters['size_agg_flag'] == 3:
        # multiply all the parameters in the compressed submodel update with the training set size
        whole_succinct_userID_count = succinct_train_set_size
        whole_succinct_itemIDs_count = np.zeros(len(succinct_itemIDs_count), dtype='uint32')
        for sidx, sic in enumerate(succinct_itemIDs_count):  # succinct item id, succinct item count
            if sic > 0:
                whole_succinct_itemIDs_count[sidx] = succinct_train_set_size
        whole_succinct_cateIDs_count = np.zeros(len(succinct_cateIDs_count), dtype='uint32')
        for scdx, scc in enumerate(succinct_cateIDs_count):  # succinct cate id, succinct cate count
            if scc > 0:
                whole_succinct_cateIDs_count[scdx] = succinct_train_set_size
        weighted_delta_submodel3 = submodel_update_multiply_count_number(compressed_delta_submodel, whole_succinct_userID_count,\
                                    whole_succinct_itemIDs_count, whole_succinct_cateIDs_count, succinct_train_set_size)
        return weighted_delta_submodel3, whole_succinct_userID_count, whole_succinct_itemIDs_count, whole_succinct_cateIDs_count, succinct_train_set_size

# ============================== END: functions to generate succinct weighted submodel update ======================================

'''
def submodel_update_multiply_count_number1(compressed_delta_submodel, succinct_train_set_size):
    """Multiply the submodel update except the embedding parameters with the training set at the client's side"""
    weighted_delta_submodel = [np.zeros(weights.shape, dtype='uint32') for weights in compressed_delta_submodel]
    temp_compressed_delta_submodel = [weights.astype('uint32') for weights in compressed_delta_submodel]  # to avoid overflow
    for layer, delta_submodel_para in enumerate(temp_compressed_delta_submodel):
        if layer <= 2:
            weighted_delta_submodel[layer] = delta_submodel_para
        else: # except the embedding layers for userID, item IDs, and cate IDs
            weighted_delta_submodel[layer] = delta_submodel_para * succinct_train_set_size
    return weighted_delta_submodel

def submodel_update_multiply_count_number3(compressed_delta_submodel, succinct_train_set_size):
    """Multiply the submodel update with the training set at the client's side"""
    temp_compressed_delta_submodel = [weights.astype('uint32') for weights in compressed_delta_submodel]  # to avoid overflow
    weighted_delta_submodel = [weights * succinct_train_set_size for weights in temp_compressed_delta_submodel]
    return weighted_delta_submodel
'''