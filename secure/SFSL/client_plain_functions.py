import ssl
import sys
import os
import tensorflow as tf
import numpy as np
from config import SSL_CONF as SC
from config import SEND_RECEIVE_CONF as SRC
import pickle
import random
import math
from compress import uint_para_compress


# ============================== START: assistant functions ======================================

def set_intersection(a, b, sort_flag=False):
    """Intersection of two lists of elements"""
    c = list(set(a) & set(b))
    if sort_flag:
        c.sort()
    return c


def set_contain(a, b):
    a_unique = list ( set(a) )
    a_unique.sort()
    if len(set_intersection(a_unique, b)) == len(a_unique):
        return True
    else:
        return False


def convert_position(a, b):
    """Positions of list a's elements in list b"""
    return [b.index(aa) for aa in a]


def response_to_check_connection(client_socket, phase=0.0, close_flag=False):
    """Response to checking client's online or offline state"""
    client_socket.settimeout(1200)
    while True:
        signal_ = client_socket.recv(10)
        if signal_ == SRC.signal:
            client_socket.send(SRC.signal)
            if close_flag:
                client_socket.close()
            break
        else:
            print('Time out in online checking of Phase %.1f!'%phase)
            sys.stdout.flush()
            client_socket.close()
            exit(-1)

# ============================== END: assistant functions ======================================


# ============================== START: functions to support computing union ======================================

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


def client_side_set_union(communication, client_socket, client_index, real_itemIDs, union_u2_drop_flag):
    """
    This function is for debugging use only, by sending real itemIDs.
    Check the correctness of Private Set Union.
    """
    # Send real_itemIDs
    send_message = {'client_ID': client_index,
                    'real_itemIDs': real_itemIDs,
                    'union_u2_drop_flag': union_u2_drop_flag}
    communication.send_np_array(send_message, client_socket)
    print('Sent real item IDs.')
    sys.stdout.flush()

    # receive union
    real_itemIDs_union = communication.get_np_array(client_socket)
    print('Received union of real item ids (via plaintext protocol)')
    sys.stdout.flush()

    return real_itemIDs_union

# ============================== END: functions to support computing union ======================================


# ============================== START: functions to generate perturbed index set ======================================

def double_randomized_response(real_IDs, real_IDs_union, permanent_YES_IDs, permanent_NO_IDs,
                               prob1, prob2, prob3, prob4):
    """Apply randomize response twice to real ids"""
    #real_IDs -> S
    once_perturbed_IDs = []  # S'
    double_perturbed_IDs = []  # S''
    # Permanent randomized response
    for j in real_IDs_union:
        if j not in permanent_YES_IDs and j not in permanent_NO_IDs:
            if j in real_IDs:  # S
                if random.uniform(0.0, 1.0) <= prob1:
                    once_perturbed_IDs.append(j)  # add j to S' with probability prob1
                    permanent_YES_IDs.append(j)
                else:
                    permanent_NO_IDs.append(j)
            else:
                if random.uniform(0.0, 1.0) <= prob2:
                    once_perturbed_IDs.append(j)  # add j to S' with probability prob2
                    permanent_YES_IDs.append(j)
                else:
                    permanent_NO_IDs.append(j)
    permanent_YES_IDs = list( set(permanent_YES_IDs) )
    permanent_NO_IDs = list( set(permanent_NO_IDs) )

    # Instantaneous randomized response
    for j in real_IDs_union:
        if j in permanent_YES_IDs:
            if random.uniform(0.0, 1.0) <= prob3:   # add j to S'' with probability prob3
                double_perturbed_IDs.append(j)
        else:
            if random.uniform(0.0, 1.0) <= prob4:   # add j to S'' with probability prob4
                double_perturbed_IDs.append(j)
    double_perturbed_IDs = list ( set(double_perturbed_IDs) )
    double_perturbed_IDs.sort()
    return double_perturbed_IDs


def generate_perturbed_item_ids(client_index, real_itemIDs, real_itemIDs_union, prob1, prob2, prob3, prob4):
    """Generate perturbed item ids"""
    permanent_YES_itemIDs = []
    permanent_NO_itemIDs = []
    # load memoized permanent answers
    filename = './permanent_answers/user_%d.pkl' % client_index
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            permanent_YES_itemIDs = pickle.load(f)
            permanent_NO_itemIDs = pickle.load(f)

    # apply double randomize responses to real item ids
    perturbed_itemIDs = double_randomized_response(real_itemIDs, real_itemIDs_union, permanent_YES_itemIDs, permanent_NO_itemIDs,
                               prob1, prob2, prob3, prob4)

    # memoize (updated) permanent answers
    with open(filename, 'w') as f:
        pickle.dump(permanent_YES_itemIDs, f)
        pickle.dump(permanent_NO_itemIDs, f)

    return perturbed_itemIDs


def generate_perturbed_cate_ids(perturbed_itemIDs, item_cate_dict):
    """Utilize item-cate map to generate perturbed cate ids from perturbed item ids"""
    perturbed_cateIDs = []
    for itemID in perturbed_itemIDs:
        cateID = int(item_cate_dict[str(itemID)])
        perturbed_cateIDs.append(cateID)
    perturbed_cateIDs = list( set(perturbed_cateIDs) )
    perturbed_cateIDs.sort()
    return perturbed_cateIDs

# ============================== END: functions to generate perturbed index set ======================================


# ============================== START: functions to generate succinct index set, training data, and submodel ======================================

def generate_succinct_cate_ids(succinct_itemIDs, item_cate_dict):
    """Utilize item-cate map to generate succinct cate ids from succinct item ids"""
    succinct_cateIDs = []
    for itemID in succinct_itemIDs:
        cateID = int(item_cate_dict[str(itemID)])
        succinct_cateIDs.append(cateID)
    succinct_cateIDs = list( set(succinct_cateIDs) )
    succinct_cateIDs.sort()
    return succinct_cateIDs


def prepare_succinct_training_data(client_index, succinct_itemIDs, succinct_cateIDs, item_cate_dict, max_len=100):
    """Utilize succinct item ids to prepare succinct training set, remove those irrelevant data"""
    # map ps's index to client's local index
    # succinct_userID_map = {userID: 0}
    succinct_itemIDs_map = dict(zip(succinct_itemIDs, range(len(succinct_itemIDs))))
    succinct_cateIDs_map = dict(zip(succinct_cateIDs, range(len(succinct_cateIDs))))

    # counting statistics over userID, succinct_itemIDs, succinct_cateIDs
    # using client's local indices [0, 1, ...]
    # succinct_userID_count: succinct_train_set_size
    succinct_itemIDs_count = [0] * len(succinct_itemIDs)
    succinct_cateIDs_count = [0] * len(succinct_cateIDs)
    succinct_train_set_size = 0

    f_client_train = open('../taobao_data_process/taobao_datasets/user_%d' % client_index, 'r')
    succinct_filename = './taobao_succinct_datasets/user_%d' % client_index
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


def gather_succinct_submodel(succinct_submodel, pos_succinct_itemIDs_in_perturbed, pos_succinct_cateIDs_in_perturbed):
    """ Extract succinct local submodel from downloaded submodel
    from perturbed index set to succinct index set
    """
    for layer, perturbed_submodel_para in enumerate(succinct_submodel):
        if layer == 0:  # embedding for user id
            continue
        elif layer == 1:  # embedding for item ids
            succinct_submodel[layer] = perturbed_submodel_para[pos_succinct_itemIDs_in_perturbed]
        elif layer == 2:  # embedding for cate ids
            succinct_submodel[layer] = perturbed_submodel_para[pos_succinct_cateIDs_in_perturbed]
        else:
            break


# ============================== START: functions to generate weighted submodel update to be uploaded ======================================

def prepare_submodel_update_uploaded(dowloaded_submodel, old_succinct_submodel, new_succinct_submodel,\
                                   pos_succinct_itemIDs_in_perturbed, pos_succinct_cateIDs_in_perturbed):
    """
    Prepare pure submodel update to be uploaded, by padding zero rows
    from succinct index set to perturbed index set
    """
    delta_submodel = [np.zeros(weights.shape) for weights in dowloaded_submodel]
    for layer, submodel_para in enumerate(new_succinct_submodel):
        if layer == 0:  # embedding for user id
            delta_submodel[layer] = new_succinct_submodel[layer] - old_succinct_submodel[layer]
        elif layer == 1:  # embedding for item ids
            for sidx in range(len(submodel_para)):   # succinct item index -> perturbed
                delta_submodel[layer][pos_succinct_itemIDs_in_perturbed[sidx]] \
                    = new_succinct_submodel[layer][sidx] - old_succinct_submodel[layer][sidx]
        elif layer == 2:  # embedding for cate ids
            for scdx in range(len(submodel_para)):  # succinct cate index -> perturbed
                delta_submodel[layer][pos_succinct_cateIDs_in_perturbed[scdx]] \
                    = new_succinct_submodel[layer][scdx] - old_succinct_submodel[layer][scdx]
        else:
            delta_submodel[layer] = new_succinct_submodel[layer] - old_succinct_submodel[layer]
    return delta_submodel


def generate_count_numbers_uploaded(perturbed_itemIDs_size, perturbed_cateIDs_size, \
                         succinct_itemIDs_count, succinct_cateIDs_count, \
                        pos_succinct_itemIDs_in_perturbed, pos_succinct_cateIDs_in_perturbed):
    """
    Prepare count numbers to be uploaded, by padding zero counts
    from succinct index set to perturbed index set
    """
    perturbed_itemIDs_count = np.zeros(perturbed_itemIDs_size, dtype='uint16')
    perturbed_cateIDs_count = np.zeros(perturbed_cateIDs_size, dtype='uint16')
    for sidx, ic in enumerate(succinct_itemIDs_count):  # succinct item index -> perturbed
        pidx = pos_succinct_itemIDs_in_perturbed[sidx]
        perturbed_itemIDs_count[pidx] = ic
    for scdx, cc in enumerate(succinct_cateIDs_count):  # succinct cate index -> perturbed
        pcdx = pos_succinct_cateIDs_in_perturbed[scdx]
        perturbed_cateIDs_count[pcdx] = cc
    return perturbed_itemIDs_count, perturbed_cateIDs_count


def submodel_update_multiply_count_numbers(compressed_delta_submodel, userID_count, itemIDs_count, cateIDs_count, other_count, \
                                                  pos_succinct_itemIDs_in_perturbed, pos_succinct_cateIDs_in_perturbed):
    """Multiply the submodel update with its count/weight at the client's side"""
    weighted_delta_submodel = [np.zeros(weights.shape, dtype='uint32') for weights in compressed_delta_submodel]
    temp_compressed_delta_submodel = [weights.astype('uint32') for weights in compressed_delta_submodel]
    for layer, delta_submodel_para in enumerate(temp_compressed_delta_submodel):
        if layer == 0:  # embedding for user id
            weighted_delta_submodel[layer] = delta_submodel_para * userID_count
        elif layer == 1:  # embedding for item ids
            for pidx in pos_succinct_itemIDs_in_perturbed:  # perturbed item index
                weighted_delta_submodel[layer][pidx] \
                    = temp_compressed_delta_submodel[layer][pidx] * itemIDs_count[pidx]
        elif layer == 2:  # embedding for cate ids
            for pcdx in pos_succinct_cateIDs_in_perturbed:  # perturbed cate index
                weighted_delta_submodel[layer][pcdx] \
                    = temp_compressed_delta_submodel[layer][pcdx] * cateIDs_count[pcdx]
        else:
            weighted_delta_submodel[layer] = delta_submodel_para * other_count
    return weighted_delta_submodel


def generate_weighted_submodel_update(dowloaded_submodel, old_succinct_submodel, new_succinct_submodel,\
                   pos_succinct_itemIDs_in_perturbed, pos_succinct_cateIDs_in_perturbed, hyperparameters,\
                   perturbed_itemIDs_size, perturbed_cateIDs_size, \
                   succinct_itemIDs_count, succinct_cateIDs_count, succinct_train_set_size):
    """Generate the weighted submodel update to be uploaded using the perturbed index set.
       According to training set size or just evenly.
       # 0: indicate aggregate the whole submodel updates evenly
       # 1: indicate aggregate the embedding evenly, while the other network parameters according training set size
       # 2: indicate aggregate the whole submodel according to the involved training set size
       # 3: indicate aggregate the whole submodel according to the whole training set size
       # 4: original federated learning aggregating way (using size_agg_flag = 3 and fl_flag = True)
    """
    # Prepare pure submodel update using perturbed index set
    delta_submodel = prepare_submodel_update_uploaded(dowloaded_submodel, old_succinct_submodel, \
                   new_succinct_submodel, pos_succinct_itemIDs_in_perturbed, pos_succinct_cateIDs_in_perturbed)
    # Compress submodel update
    compressed_delta_submodel = uint_para_compress(delta_submodel, hyperparameters)
    # Generate count numbers
    perturbed_itemIDs_count, perturbed_cateIDs_count = generate_count_numbers_uploaded(perturbed_itemIDs_size, \
                   perturbed_cateIDs_size, succinct_itemIDs_count, succinct_cateIDs_count, \
                   pos_succinct_itemIDs_in_perturbed, pos_succinct_cateIDs_in_perturbed)
    if hyperparameters['size_agg_flag'] == 0:
        even_userID_count = False
        even_train_set_size = False
        if succinct_train_set_size > 0:
            even_userID_count = True
            even_train_set_size = True
        even_itemIDs_count = np.zeros(perturbed_itemIDs_size, dtype='bool')
        for pidx, pic in enumerate(perturbed_itemIDs_count):  # perturbed item id, perturbed item count
            if pic > 0:
                even_itemIDs_count[pidx] = True
        even_cateIDs_count = np.zeros(perturbed_cateIDs_size, dtype='bool')
        for pcdx, pcc in enumerate(perturbed_cateIDs_count): # perturbed cate id, perturbed cate count
            if pcc > 0:
                even_cateIDs_count[pcdx] = True
        weighted_delta_submodel0 = submodel_update_multiply_count_numbers(compressed_delta_submodel, even_userID_count,\
                                                          even_itemIDs_count, even_cateIDs_count, even_train_set_size, \
                                                   pos_succinct_itemIDs_in_perturbed, pos_succinct_cateIDs_in_perturbed)
        return weighted_delta_submodel0, even_userID_count, even_itemIDs_count, even_cateIDs_count, even_train_set_size
    elif hyperparameters['size_agg_flag'] == 1:
        even_userID_count = False
        if succinct_train_set_size > 0:
            even_userID_count = True
        even_itemIDs_count = np.zeros(perturbed_itemIDs_size, dtype='bool')
        for pidx, pic in enumerate(perturbed_itemIDs_count):  # perturbed item id, perturbed item count
            if pic > 0:
                even_itemIDs_count[pidx] = True
        even_cateIDs_count = np.zeros(perturbed_cateIDs_size, dtype='bool')
        for pcdx, pcc in enumerate(perturbed_cateIDs_count):  # perturbed cate id, perturbed cate count
            if pcc > 0:
                even_cateIDs_count[pcdx] = True
        weighted_delta_submodel1 = submodel_update_multiply_count_numbers(compressed_delta_submodel, even_userID_count, \
                                                    even_itemIDs_count, even_cateIDs_count, succinct_train_set_size, \
                                                 pos_succinct_itemIDs_in_perturbed, pos_succinct_cateIDs_in_perturbed)
        return weighted_delta_submodel1, even_userID_count, even_itemIDs_count, even_cateIDs_count, succinct_train_set_size
    elif hyperparameters['size_agg_flag'] == 2:
        perturbed_userID_count = succinct_train_set_size
        weighted_delta_submodel2 = submodel_update_multiply_count_numbers(compressed_delta_submodel, perturbed_userID_count, \
                                            perturbed_itemIDs_count, perturbed_cateIDs_count, succinct_train_set_size, \
                                                pos_succinct_itemIDs_in_perturbed, pos_succinct_cateIDs_in_perturbed)
        return weighted_delta_submodel2, perturbed_userID_count, perturbed_itemIDs_count, perturbed_cateIDs_count, succinct_train_set_size
    elif hyperparameters['size_agg_flag'] == 3:
        whole_perturbed_userID_count = succinct_train_set_size
        whole_perturbed_itemIDs_count = np.zeros(perturbed_itemIDs_size, dtype='uint16')
        for pidx, pic in enumerate(perturbed_itemIDs_count):  # perturbed item id, perturbed item count
            if pic > 0:
                whole_perturbed_itemIDs_count[pidx] = succinct_train_set_size
        whole_perturbed_cateIDs_count = np.zeros(perturbed_cateIDs_size, dtype='uint16')
        for pcdx, pcc in enumerate(perturbed_cateIDs_count):  # perturbed cate id, perturbed cate count
            if pcc > 0:
                whole_perturbed_cateIDs_count[pcdx] = succinct_train_set_size
        weighted_delta_submodel3 = submodel_update_multiply_count_numbers(compressed_delta_submodel, whole_perturbed_userID_count, \
                                whole_perturbed_itemIDs_count, whole_perturbed_cateIDs_count, succinct_train_set_size, \
                                pos_succinct_itemIDs_in_perturbed, pos_succinct_cateIDs_in_perturbed)
        return weighted_delta_submodel3, whole_perturbed_userID_count, whole_perturbed_itemIDs_count, whole_perturbed_cateIDs_count, succinct_train_set_size

# ============================== END: functions to generate weighted submodel update to be uploaded ======================================
