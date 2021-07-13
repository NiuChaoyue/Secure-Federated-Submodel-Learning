# -*- coding: UTF-8 -*-

import ssl
import sys
import tensorflow as tf
import numpy as np
from config import SSL_CONF as SC
from config import SEND_RECEIVE_CONF as SRC
from compress import recover_compression, fl_quantization_recover_compression
import evaluate as my_eval
import math
import random
import time
from general_functions import determine_data_type
from general_functions import write_csv
# Security module
import ps_secure_federated_submodel_averaging as ps_sfsa


class Client:
    def __init__(self, sock, address, ID=-1):
        self.connection_socket = sock
        self.address = address
        self.ID = ID    # type `int', consistent with client's side


# ============================== START: functions to check and close real-time live connections ======================================

def check_connection(clients, sleep_time=0.0):
    temp_clients = [client for client in clients]
    for client in temp_clients:
        try:
            client.connection_socket.settimeout(120)
            client.connection_socket.send(SRC.signal)
            recv_signal = client.connection_socket.recv(10)
            if recv_signal != SRC.signal:
                print('Fallen client ' + str(client.ID) + ' at: ' + client.address[0] + ':' + str(client.address[1]))
                clients.remove(client) #remove fallen client
                try:
                    client.connection_socket.close()
                except (OSError, OSError):
                    pass
            time.sleep(sleep_time)  # create asynchronous environments to avoid clients competing for CPU resource, if needed
        except (OSError, OSError):
            print('Fallen client ' + str(client.ID) + ' at: ' + client.address[0] + ':' + str(client.address[1]))
            clients.remove(client)
            try:
                client.connection_socket.close()
            except (OSError, OSError):
                pass
    del temp_clients


def close_connection(clients):
    for client in clients:
        try:
            client.connection_socket.close()
        except (OSError, OSError):
            pass


# ============================== END: functions to check and close real-time live connections ======================================


# ============================== START: functions to return hyperparameters ======================================

def send_hyperparameters(communication, clients, total_users_num, chosen_clients_num, hyperparameters,
                         union_security_para_dict, fedsubavg_security_para_dict):
    """Send all hyperparameters to chosen clients."""
    # determine chosen clients's indices, e.g., corresponding to Taobao users' data set indices
    user_id_set = random.sample(range(1, total_users_num + 1), chosen_clients_num)
    communication.ps_socket.listen(chosen_clients_num)
    for i in range(chosen_clients_num):
        try:
            communication.ps_socket.settimeout(1200)
            sock, address = communication.ps_socket.accept()
            connection_socket = ssl.wrap_socket(
                sock,
                server_side=True,
                certfile=SC.cert_path,
                keyfile=SC.key_path,
                ssl_version=ssl.PROTOCOL_TLSv1)

            print('Connected: ' + address[0] + ':' + str(address[1]))
        except socket.timeout:
            print('Some clients cannot be connected.')
        except Exception as e:
            print(e.message)

        try:
            client = Client(connection_socket, address)
            client.ID = user_id_set[i]
            send_message = {'client_index': client.ID,
                            'hyperparameters': hyperparameters,
                            'union_security_para_dict': union_security_para_dict,
                            'fedsubavg_security_para_dict': fedsubavg_security_para_dict}
            communication.send_np_array(send_message, client.connection_socket)
            clients.append(client)
            print('Sending hyperparameters to client at ' + address[0] + ':' + str(address[1]))
        except Exception as e:
            print(e.message)
            connection_socket.close()


# ============================== END: functions to return hyperparameters ======================================


# ============================== START: functions to obtain unions in plain text ======================================

def compute_union(union_id_dict):
    """compute unions of item and cate ids."""
    real_itemIDs_union = []
    for client_id, received_message in union_id_dict.items():
        if not received_message['union_u2_drop_flag']:   # Exclude itemIDs of the client who ``dropped" in round 2
            # Not append!
            real_itemIDs_union += received_message['real_itemIDs']
    real_itemIDs_union = list( set(real_itemIDs_union) )
    real_itemIDs_union.sort()
    return real_itemIDs_union

'''
def compute_union_from_perturbed_bloom_filters(union_id_dict, union_security_para_dict):
    """
    Compute union by summing the perturbed bloom filters
    """
    item_count = union_security_para_dict['item_count']
    modulo_r = union_security_para_dict['modulo_r']
    modulo_r_len = union_security_para_dict['modulo_r_len']
    clients_size_len = int(math.ceil(math.log(len(union_id_dict), 2)))
    sum_len = modulo_r_len + clients_size_len
    # for cache intermediate results only
    if sum_len <= 8:
        bf_perturbed_sum = np.zeros(item_count, dtype='uint8')
    elif sum_len <= 16:
        bf_perturbed_sum = np.zeros(item_count, dtype='uint16')
    elif sum_len <= 32:
        bf_perturbed_sum = np.zeros(item_count, dtype='uint32')
    else:
        bf_perturbed_sum = np.zeros(item_count, dtype='uint64')
    # To do: Use secure aggregation protocol
    for client_id, received_message in union_id_dict.items():
        bf_perturbed_sum += received_message['real_itemIDs_pbf']
    bf_perturbed_sum %= modulo_r
    if modulo_r_len <= 8:
        bf_perturbed_sum = bf_perturbed_sum.astype('uint8')
    elif modulo_r_len <= 16:
        bf_perturbed_sum = bf_perturbed_sum.astype('uint16')
    elif modulo_r_len <= 32:
        bf_perturbed_sum = bf_perturbed_sum.astype('uint32')
    else:
        bf_perturbed_sum = bf_perturbed_sum.astype('uint64')
    real_itemIDs_union = (np.nonzero(bf_perturbed_sum)[0]).tolist()
    return real_itemIDs_union
'''


def send_back_union(communication, clients, union_message):
    """Sending back union of user IDs, item IDs, and cate IDs"""
    temp_clients = [client for client in clients]
    for client in temp_clients:
        try:
            communication.send_np_array(union_message, client.connection_socket)
            print('Sending back unions (via plaintext protocol) to client ' + str(client.ID))
            sys.stdout.flush()
        except Exception:
            clients.remove(client)
            print('Fallen client: ' + str(client.ID) + ' at ' + client.address[0] + ':' + str(
                client.address[1]) + ' in the union sending back stage.')
            sys.stdout.flush()
            client.connection_socket.close()
    del temp_clients


def get_compute_and_return_union(communication, clients, round_num, ps_computation_time_path):
    """Get ids from live clients, compute union, and then return union"""
    # get ids from online clients and store them
    union_id_dict = {}
    temp_clients = [client for client in clients]
    for client in temp_clients:
        try:
            received_message = communication.get_np_array(client.connection_socket)
            assert client.ID == received_message['client_ID']
            union_id_dict[client.ID] = received_message
            print('Received real item ids from client ' + str(client.ID))
            sys.stdout.flush()
        except Exception:
            clients.remove(client)
            print('Fallen client: ' + str(client.ID) + ' at ' + client.address[0] + ':' + str(
                client.address[1]) + ' in the real item ids uploading stage.')
            sys.stdout.flush()
            client.connection_socket.close()
    del temp_clients

    # generate returned union message
    start_time = time.time()
    union_message = compute_union(union_id_dict)
    write_csv(ps_computation_time_path, [round_num, "plain set union of real item ids", time.time() - start_time])
    # send back union
    send_back_union(communication, clients, union_message)


# ============================== END: functions to obtain unions in plaintext ======================================

# ============================== START: functions to send back submodels ======================================

def gather_submodel(submodel, userID, perturbed_itemIDs, perturbed_cateIDs):
    for layer, model_para in enumerate(submodel):
        if layer == 0:
            submodel[layer] = model_para[userID]
        elif layer == 1:
            submodel[layer] = model_para[perturbed_itemIDs]
        elif layer == 2:
            submodel[layer] = model_para[perturbed_cateIDs]
        else:
            break


def choice_send_submodel(communication, client_message, connection_socket, sess):
    """Here should keep all the ID list be ascending before do the map to new ID list in build_dataset"""
    userID = client_message['userID']
    perturbed_itemIDs = client_message['perturbed_itemIDs']
    perturbed_cateIDs = client_message['perturbed_cateIDs']

    submodel = sess.run(tf.trainable_variables())
    gather_submodel(submodel, userID, perturbed_itemIDs, perturbed_cateIDs)

    communication.send_np_array(submodel, connection_socket)


def get_info_and_return_model(communication, client, batches_info_dict, sess):
    """
    Get information of involved ids from the client,
    send back a submodel to the client according to received message,
    then update batches_info_dict, this will be used when updating model.
    """
    try:
        received_message = communication.get_np_array(client.connection_socket)
        assert client.ID == received_message['client_ID']
        print('Received batches_info from client ' + str(client.ID))
        sys.stdout.flush()
        # get and send back the corresponding embedding and the model parameters
        choice_send_submodel(communication, received_message, client.connection_socket, sess)
        print('Sending back submodel parameters to client ' + str(client.ID))
        sys.stdout.flush()

        # update batches_info_dict
        batches_info_dict[client.ID] = received_message
    except Exception:
        print('Fallen client: ' + str(client.ID) + ' at ' + client.address[0] \
              + ':' + str(client.address[1]) + ' in the submodel downloading phase.')
        sys.stdout.flush()
        client.connection_socket.close()


def send_back_submodels(communication, clients, batches_info_dict, g1, sess):
    """
    Receive perturbed index sets and return submodels
    """
    temp_clients = [client for client in clients]
    for client in temp_clients:
        try:
            with g1.as_default():
                get_info_and_return_model(communication, client, batches_info_dict, sess)
                # print('Returned submodel to : ' + str(client.ID))
                sys.stdout.flush()
        except Exception:
            clients.remove(client)
            print('Fallen client: ' + str(client.ID) + ' at ' + client.address[0] + ':' + str(client.address[1])
                  + 'in the submodel returning stage.')
            sys.stdout.flush()
            client.connection_socket.close()


# ============================== END: functions to send back submodels ======================================


# ============================== START: functions to gather, aggregate, and apply submodel updates ======================================

def get_submodel_update(communication, clients, batches_info_dict, gathered_weights_dict):
    """
    Get submodel updates from live clients and save into gathered_weights_dict
    """
    temp_clients = [client for client in clients]
    for client in temp_clients:
        try:
            received_message = communication.get_np_array(client.connection_socket)
            assert client.ID == received_message['client_ID']
            gathered_weights_dict[client.ID] = received_message['weighted_delta_submodel']
            batches_info_dict[client.ID]['perturbed_userID_count'] = received_message['perturbed_userID_count']
            batches_info_dict[client.ID]['perturbed_itemIDs_count'] = received_message['perturbed_itemIDs_count']
            batches_info_dict[client.ID]['perturbed_cateIDs_count'] = received_message['perturbed_cateIDs_count']
            batches_info_dict[client.ID]['perturbed_other_count'] = received_message['perturbed_other_count']
            batches_info_dict[client.ID]['fedsubavg_u2_drop_flag'] = received_message['fedsubavg_u2_drop_flag']
            print('Received (via plaintext protocol) weighted submodel update and count numbers from client ' + str(client.ID))
            sys.stdout.flush()
        except Exception:
            clients.remove(client)
            print('Fallen client: ' + str(client.ID) + ' at ' + client.address[0] + ':' + str(
                client.address[1]) + 'in the submodel update uploading stage (plaintext protocol).')
            sys.stdout.flush()
            client.connection_socket.close()


def gather_weighted_submodel_updates(client_indices, dataset_info, gathered_weights_dict, batches_info_dict, global_model_shape, fedsubavg_security_para_dict):
    """Sum all weighted submodel updates and corresponding count numbers"""
    # Prepare (gathered) global weighted delta parameters
    # Model parameters' data type
    modulo_model_r_len = fedsubavg_security_para_dict['modulo_model_r_len']
    model_data_type = determine_data_type(modulo_model_r_len)
    gathered_weighted_delta_submodel = [np.zeros(para_shape, dtype=model_data_type) for para_shape in global_model_shape]

    # Prepare (gathered) global count numbers
    # Count numbers' data type (for embedding layer)
    modulo_count_r_len = fedsubavg_security_para_dict['modulo_count_r_len']
    count_data_type = determine_data_type(modulo_count_r_len)
    gathered_userIDs_count = np.zeros(dataset_info['user_count'], dtype=count_data_type)
    gathered_itemIDs_count = np.zeros(dataset_info['item_count'], dtype=count_data_type)
    gathered_cateIDs_count = np.zeros(dataset_info['cate_count'], dtype=count_data_type)
    gathered_other_count = 0

    for client_index in client_indices:
        client_info = batches_info_dict[client_index]
        fedsubavg_u2_drop_flag = client_info['fedsubavg_u2_drop_flag']
        if fedsubavg_u2_drop_flag:  # Exclude the client dropped in round 2 from submodel and count numbers aggregation
            continue                # Debugging usage only
        userID = client_info['userID'][0]
        client_perturbed_itemIDs = client_info['perturbed_itemIDs']
        client_perturbed_cateIDs = client_info['perturbed_cateIDs']
        client_perturbed_userID_count = client_info['perturbed_userID_count']
        client_perturbed_itemIDs_count = client_info['perturbed_itemIDs_count']
        client_perturbed_cateIDs_count = client_info['perturbed_cateIDs_count']
        client_perturbed_other_count = client_info['perturbed_other_count']
        gathered_other_count += client_perturbed_other_count

        client_weighted_delta_submodel = gathered_weights_dict[client_index]
        for layer, delta_submodel_para in enumerate(client_weighted_delta_submodel):
            if layer == 0:   # embedding for user id
                gathered_weighted_delta_submodel[layer][userID] += delta_submodel_para[0]   # one user id
                gathered_userIDs_count[userID] += client_perturbed_userID_count
            elif layer == 1:  # embedding for item ids
                for client_item_index in range(len(delta_submodel_para)):
                    ps_item_index = client_perturbed_itemIDs[client_item_index]
                    gathered_weighted_delta_submodel[layer][ps_item_index] += delta_submodel_para[client_item_index]
                    gathered_itemIDs_count[ps_item_index] += client_perturbed_itemIDs_count[client_item_index]
            elif layer == 2:  # embedding for cate ids
                for client_cate_index in range(len(delta_submodel_para)):
                    ps_cate_index = client_perturbed_cateIDs[client_cate_index]
                    gathered_weighted_delta_submodel[layer][ps_cate_index] += delta_submodel_para[client_cate_index]
                    gathered_cateIDs_count[ps_cate_index] += client_perturbed_cateIDs_count[client_cate_index]
            else:
                gathered_weighted_delta_submodel[layer] += delta_submodel_para
    return gathered_weighted_delta_submodel, gathered_userIDs_count, gathered_itemIDs_count, gathered_cateIDs_count, gathered_other_count


def average_submodel_updates_and_apply_global_update(gathered_weighted_delta_submodel, gathered_userIDs_count,
                        gathered_itemIDs_count, gathered_cateIDs_count, gathered_other_count, hyperparameters, sess):
    """Average (according to training set size of evenly) the gathered weighted submodel updates"""
    old_global_model = sess.run(tf.trainable_variables())
    new_global_model = [np.zeros(weights.shape) for weights in old_global_model]
    for layer, gathered_model_para in enumerate(gathered_weighted_delta_submodel):
        if layer == 0:  # embedding for user ids
            for ui in range(len(gathered_model_para)):
                # pay attention to divide zero error
                if gathered_userIDs_count[ui] > 0:
                    avg_userID_para = gathered_model_para[ui] * 1.0 / gathered_userIDs_count[ui]
                    avg_userID_para = recover_compression(avg_userID_para, hyperparameters)
                    new_global_model[layer][ui] = old_global_model[layer][ui] + avg_userID_para
                else:
                    new_global_model[layer][ui] = old_global_model[layer][ui]
        elif layer == 1:  # embedding for item ids
            for ii in range(len(gathered_model_para)):
                # pay attention to divide zero error
                if gathered_itemIDs_count[ii] > 0:
                    avg_itemID_para = gathered_model_para[ii] * 1.0 / gathered_itemIDs_count[ii]
                    avg_itemID_para = recover_compression(avg_itemID_para, hyperparameters)
                    new_global_model[layer][ii] = old_global_model[layer][ii] + avg_itemID_para
                else:
                    new_global_model[layer][ii] = old_global_model[layer][ii]
        elif layer == 2:  # embedding for cate ids
            for ci in range(len(gathered_model_para)):
                # pay attention to divide zero error
                if gathered_cateIDs_count[ci] > 0:
                    avg_cateID_para = gathered_model_para[ci] * 1.0 / gathered_cateIDs_count[ci]
                    avg_cateID_para = recover_compression(avg_cateID_para, hyperparameters)
                    new_global_model[layer][ci] = old_global_model[layer][ci] + avg_cateID_para
                else:
                    new_global_model[layer][ci] = old_global_model[layer][ci]
        else:
            # pay attention to divide zero error
            if gathered_other_count > 0:
                avg_other_para = gathered_model_para * 1.0 / gathered_other_count
                avg_other_para = recover_compression(avg_other_para, hyperparameters)
                new_global_model[layer] = old_global_model[layer] + avg_other_para
            else:
                new_global_model[layer] = old_global_model[layer]
    """
    For debug usage only
    """
    for layer, model_para in enumerate(new_global_model):
        print("Layer %d, max new element %f, min new element %f" % (layer, np.max(model_para), np.min(model_para)))
    sys.stdout.flush()
    return new_global_model


def fl_average_submodel_updates_and_apply_global_update(gathered_weighted_delta_submodel, gathered_userIDs_count,\
                        gathered_itemIDs_count, gathered_cateIDs_count, gathered_train_set_size, hyperparameters, sess):
    """Federated Learning's way: Average (according to training set size) the gathered weighted model updates"""
    old_global_model = sess.run(tf.trainable_variables())
    new_global_model = [np.zeros(weights.shape) for weights in old_global_model]
    for layer, gathered_model_para in enumerate(gathered_weighted_delta_submodel):
        if layer == 0:  # embedding for user ids
            for ui in range(len(gathered_model_para)):
                # pay attention to divide zero error
                if gathered_userIDs_count[ui] > 0:
                    avg_userID_para = gathered_model_para[ui] * 1.0 / gathered_train_set_size
                    compensation_ratio = 1.0 - gathered_userIDs_count[ui] * 1.0 / gathered_train_set_size
                    avg_userID_para = fl_quantization_recover_compression(avg_userID_para, hyperparameters, compensation_ratio)
                    new_global_model[layer][ui] = old_global_model[layer][ui] + avg_userID_para
                else:
                    new_global_model[layer][ui] = old_global_model[layer][ui]
        elif layer == 1:  # embedding for item ids
            for ii in range(len(gathered_model_para)):
                # pay attention to divide zero error
                if gathered_itemIDs_count[ii] > 0:
                    avg_itemID_para = gathered_model_para[ii] * 1.0 / gathered_train_set_size
                    compensation_ratio = 1.0 - gathered_itemIDs_count[ii] * 1.0 / gathered_train_set_size
                    avg_itemID_para = fl_quantization_recover_compression(avg_itemID_para, hyperparameters, compensation_ratio)
                    new_global_model[layer][ii] = old_global_model[layer][ii] + avg_itemID_para
                else:
                    new_global_model[layer][ii] = old_global_model[layer][ii]
        elif layer == 2:  # embedding for cate ids
            for ci in range(len(gathered_model_para)):
                # pay attention to divide zero error
                if gathered_cateIDs_count[ci] > 0:
                    avg_cateID_para = gathered_model_para[ci] * 1.0 / gathered_train_set_size
                    compensation_ratio = 1.0 - gathered_cateIDs_count[ci] * 1.0 / gathered_train_set_size
                    avg_cateID_para = fl_quantization_recover_compression(avg_cateID_para, hyperparameters, compensation_ratio)
                    new_global_model[layer][ci] = old_global_model[layer][ci] + avg_cateID_para
                else:
                    new_global_model[layer][ci] = old_global_model[layer][ci]
        else:
            # pay attention to divide zero error
            if gathered_train_set_size > 0:
                avg_other_para = gathered_model_para * 1.0 / gathered_train_set_size
                avg_other_para = recover_compression(avg_other_para, hyperparameters)
                new_global_model[layer] = old_global_model[layer] + avg_other_para
            else:
                new_global_model[layer] = old_global_model[layer]
    """
    For debug usage only
    """
    print("Federated learning ...")
    for layer, model_para in enumerate(new_global_model):
        print("Layer %d, max new element %f, min new element %f"%(layer, np.max(model_para), np.min(model_para)))
    sys.stdout.flush()
    return new_global_model


def do_update_weights(new_weights, placeholders, update_local_vars_op, sess):
    feed_dict = {}
    for place, para in zip(placeholders, new_weights):
        feed_dict[place] = para
    sess.run(update_local_vars_op, feed_dict=feed_dict)
    del feed_dict


def update_global_model(round_num, communication, clients, dataset_info, batches_info_dict, hyperparameters, \
        fedsubavg_security_para_dict, placeholders, update_local_vars_op, variables_pack_for_eval_and_save, \
                        global_model_auc_path, ps_computation_time_path, g1, sess):
    """
    Aggregate all submodel updates in this round and apply to the global model
    """
    #prepare some parameters in clear
    with g1.as_default():
        global_model_shape = [para.shape for para in sess.run(tf.trainable_variables())]

    ids_info_dict = dict()   #client_index -> userID, perturbed itemIDs, and perturbed cateIDs
    for client_index, client_ids_info in batches_info_dict.items():
        ids_info_dict[client_index] = {'userID': client_ids_info['userID'],\
            'perturbed_itemIDs': client_ids_info['perturbed_itemIDs'],\
            'perturbed_cateIDs': client_ids_info['perturbed_cateIDs']}

    # (Delete finally) Use plaintext protocol to aggregate submodel parameters and count numbers for debugging
    gathered_weights_dict = {}
    get_submodel_update(communication, clients, batches_info_dict, gathered_weights_dict)

    start_time = time.time()

    client_indices = [client.ID for client in clients]
    gathered_weighted_delta_submodel_plain, gathered_userIDs_count_plain, gathered_itemIDs_count_plain,\
        gathered_cateIDs_count_plain, gathered_other_count_plain = \
        gather_weighted_submodel_updates(client_indices, dataset_info, gathered_weights_dict, batches_info_dict, global_model_shape, fedsubavg_security_para_dict)

    write_csv(ps_computation_time_path, [round_num, "plain_sfsa", time.time() - start_time])

    # Secure federated submodel averaging works here!!!
    print("PS side secure federated submodel averaging starts")
    sys.stdout.flush()
    gathered_weighted_delta_submodel, gathered_userIDs_count, gathered_itemIDs_count, gathered_cateIDs_count, \
       gathered_other_count = ps_sfsa.server_side_secure_federated_submodel_averaging(communication, clients, \
                        dataset_info, ids_info_dict, global_model_shape, fedsubavg_security_para_dict, \
                                                                            round_num, ps_computation_time_path)

    assert (gathered_userIDs_count_plain == gathered_userIDs_count).all()
    assert (gathered_itemIDs_count_plain == gathered_itemIDs_count).all()
    assert (gathered_cateIDs_count_plain == gathered_cateIDs_count).all()
    assert gathered_other_count_plain == gathered_other_count
    assert_model_flag = False
    for layer, para_shape in enumerate(global_model_shape):
        if not (gathered_weighted_delta_submodel_plain[layer] == gathered_weighted_delta_submodel[layer]).all():
            print(layer, para_shape)
            assert_model_flag = True
    if assert_model_flag:
        print("Oh, no! Secure federated submodel averaging fails!")
        exit(-1)
    else:
        print("Yes! Secure federated submodel averaging successfully finishes!")
        sys.stdout.flush()

    with g1.as_default():
        start_time = time.time()
        if hyperparameters['fl_flag']:
            new_global_model = fl_average_submodel_updates_and_apply_global_update(gathered_weighted_delta_submodel, \
                    gathered_userIDs_count, gathered_itemIDs_count, gathered_cateIDs_count, gathered_other_count, \
                                                                            hyperparameters, sess)
        else:
            new_global_model = average_submodel_updates_and_apply_global_update(gathered_weighted_delta_submodel, \
                    gathered_userIDs_count, gathered_itemIDs_count, gathered_cateIDs_count, gathered_other_count,\
                                                                            hyperparameters, sess)
        # Update global model at parameter server
        do_update_weights(new_global_model, placeholders, update_local_vars_op, sess)

        write_csv(ps_computation_time_path, [round_num, "ps avg and then update global_model", time.time() - start_time])

        print('Round {}: Weights received, average applied '.format(round_num) +
              'among {} clients'.format(len(clients)) + ', model updated! Evaluating...')
        sys.stdout.flush()
        # Update learning rate
        hyperparameters['learning_rate'] *= hyperparameters['decay_rate']

        # Evaluate global model
        start_time = time.time()
        test_auc, loss_sum, accuracy_sum = my_eval.eval_and_save(variables_pack_for_eval_and_save,
                                                                 round_num, sess)
        write_csv(ps_computation_time_path, [round_num, "ps test global model", time.time() - start_time])
        write_csv(global_model_auc_path, [round_num, test_auc, loss_sum, accuracy_sum])
        print('Global Model performance: test_auc: %.4f ---- loss: %f ---- accuracy: %f' %
              (test_auc, loss_sum, accuracy_sum))
        print('Best round: ' + str(variables_pack_for_eval_and_save['best_round']) +
              ' Best test_auc: ' + str(variables_pack_for_eval_and_save['best_auc']))
        print('')
        sys.stdout.flush()

# ============================== END: functions to gather, aggregate, and apply submodel updates =====================================
