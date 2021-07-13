# -*- coding: UTF-8 -*-

import ssl
import sys
import time
import tensorflow as tf
import numpy as np
import csv
from copy import deepcopy
from config import SSL_CONF as SC
from config import SEND_RECEIVE_CONF as SRC
from compress import recover_compression
import evaluate as my_eval
import time
import math
import random


class Client:
    def __init__(self, sock, address, ID="-1"):
        self.connection_socket = sock
        self.address = address
        self.ID = ID


def judge_set_intersection(a, b):
    """Judge whether two lists contain common elements"""
    c = list(set(a) & set(b))
    return len(c) > 0


def gather_succinct_submodel(submodel, userID, succinct_itemIDs, succinct_cateIDs):
    """Extract succinct submodel"""
    for layer, model_para in enumerate(submodel):
        if layer == 0:
            submodel[layer] = model_para[userID]
        elif layer == 1:
            submodel[layer] = model_para[succinct_itemIDs]
        elif layer == 2:
            submodel[layer] = model_para[succinct_cateIDs]
        else:
            break


def choice_send_parameter(communication, client_message, connection_socket, random_user_index, sess):
    """Send back submodel to the client with her succinct index set"""
    userID = client_message['userID']
    succinct_itemIDs = client_message['succinct_itemIDs']
    succinct_cateIDs = client_message['succinct_cateIDs']

    # Load current global model
    submodel = sess.run(tf.trainable_variables())

    # Gather succinct submodel
    gather_succinct_submodel(submodel, userID, succinct_itemIDs, succinct_cateIDs)

    send_message = {'succinct_submodel': submodel,
                    'random_user_index': random_user_index}
    communication.send_np_array(send_message, connection_socket)


def get_info_and_return_model(communication, current_client, queues_and_locks, sess):
    """
    Get information of next batches from the client,
    send back subset of current model to the client according to received message,
    then update batches_info_dict, this will be used when updating model.
    """
    try:
        current_client.connection_socket.send(SRC.please_send_batches_info)
        received_message = communication.get_np_array(current_client.connection_socket)
        current_client.ID = str(received_message['client_ID'])
        print('Received succinct batches info from client ' + current_client.ID)
        sys.stdout.flush()
        random_user_index = queues_and_locks.next_random_user_index_set[received_message['machine_index'] - 1]
        # get and send back the corresponding embedding and the model parameters
        signal = current_client.connection_socket.recv(10)
        if signal == SRC.please_send_model:
            choice_send_parameter(communication, received_message, current_client.connection_socket, random_user_index, sess)
            print('Sent back succinct submodel parameters to client ' + current_client.ID)
            sys.stdout.flush()

        # Save userID, succinct itemIDs, and succinct cateIDs into batches_info_dict
        queues_and_locks.batches_info_dict[current_client.ID] = {'userID': received_message['userID'],
                                             'succinct_itemIDs': received_message['succinct_itemIDs'],
                                             'succinct_cateIDs': received_message['succinct_cateIDs']}
    except Exception:
        print('Fallen client: ' + current_client.ID + ' at ' + current_client.address[0] + ':' + str(current_client.address[1]))
        sys.stdout.flush()
        current_client.connection_socket.close()


#============================= START: assistant functions for classify_connections =====================================
def send_hyperparameters(communication, current_client, hyperparameters):
    try:
        communication.send_np_array(hyperparameters, current_client.connection_socket)
    except Exception as e:
        print(e.message)
        sys.stdout.flush()
        current_client.connection_socket.close()


def accept_new_connections(communication, queues_and_locks):
    """Called by main thread, keep accepting new connections and append to classification_queue."""
    classification_queue = queues_and_locks.classification_queue
    classification_queue_lock = queues_and_locks.classification_queue_lock

    while True:
        try:
            sock, address = communication.ps_socket.accept()
            connection_socket = ssl.wrap_socket(
                sock,
                server_side=True,
                certfile=SC.cert_path,
                keyfile=SC.key_path,
                ssl_version=ssl.PROTOCOL_TLSv1)

            if classification_queue_lock.acquire():
                classification_queue.append(Client(connection_socket, address))
                classification_queue_lock.release()

        except Exception as e:
            print(e.message)
            sys.stdout.flush()
            continue

#============================== END: assistant functions for classify_connections ======================================


def classify_connections(communication, queues_and_locks, hyperparameters, g1, sess):
    """
    Classify new connections (INIT/UPDATE),
    initialization takes place immediately, using hyperparameters, dataset_info, sess,
    clients willing to send updates will be appended to get_update_queue.
    """
    while True:
        if len(queues_and_locks.classification_queue) == 0:  # Check if there exists connection to be classified
            time.sleep(5)   # Sleep to avoid busy loop
            continue
        elif len(queues_and_locks.classification_queue) > 0:    # Begin classifying
            if queues_and_locks.classification_queue_lock.acquire():    # Get lock
                current_client = queues_and_locks.classification_queue.pop(0)
                queues_and_locks.classification_queue_lock.release()    # Release lock
            else:
                continue

            try:    # Communicate to know this connection's purpose
                current_client.connection_socket.settimeout(600)
                current_client.connection_socket.send(SRC.purpose)
                purpose = current_client.connection_socket.recv(10)

                if purpose == SRC.update:   # Append current_client to get_update_queue
                    if queues_and_locks.get_update_queue_lock.acquire():
                        queues_and_locks.get_update_queue.append(current_client)
                        queues_and_locks.get_update_queue_lock.release()
                elif purpose == SRC.init:   # Do initialization
                    send_hyperparameters(communication, current_client, hyperparameters)
                    if queues_and_locks.update_model_lock.acquire():
                        with g1.as_default():
                            get_info_and_return_model(communication, current_client, queues_and_locks, sess)
                        queues_and_locks.update_model_lock.release()
                        print('Initialized client: ' + current_client.ID)
                        sys.stdout.flush()
                        current_client.connection_socket.close()
                        del current_client

            except (OSError, OSError):  # Exception handler
                print('Fallen client: ' + current_client.address[0] + ':' + str(current_client.address[1]))
                sys.stdout.flush()
                try:
                    current_client.connection_socket.close()
                except (OSError, OSError):
                    pass


def get_update(communication, queues_and_locks):
    """
    Get updates from clients in get_update_queue and save updates into gathered_weights_dict,
    then pass clients to return_model_queue.
    """
    while True:
        if len(queues_and_locks.get_update_queue) == 0:  # Check if there exists updates to be gathered
            time.sleep(5)   # Sleep to avoid busy loop
            continue
        elif len(queues_and_locks.get_update_queue) > 0:
            if queues_and_locks.get_update_queue_lock.acquire():    # Get lock
                current_client = queues_and_locks.get_update_queue.pop(0)
                queues_and_locks.get_update_queue_lock.release()    # Release_lock
            else:
                continue

            try:    # Communicate to get updates
                current_client.connection_socket.settimeout(600)
                current_client.connection_socket.send(SRC.please_send_update)
                received_message = communication.get_np_array(current_client.connection_socket)
                current_client.ID = str(received_message['client_ID'])
                # Save count numbers into batches_info_dict
                queues_and_locks.batches_info_dict[current_client.ID]['userID_count'] = received_message['userID_count']
                queues_and_locks.batches_info_dict[current_client.ID]['itemIDs_count'] = received_message['itemIDs_count']
                queues_and_locks.batches_info_dict[current_client.ID]['cateIDs_count'] = received_message['cateIDs_count']
                queues_and_locks.batches_info_dict[current_client.ID]['train_set_size'] = received_message['train_set_size']

                if queues_and_locks.return_model_queue_lock.acquire():   # Get lock
                    queues_and_locks.return_model_queue.append(current_client)   # Append current_client to return_model_queue
                    queues_and_locks.gathered_weights_dict[current_client.ID] = received_message['weighted_delta_submodel'] # Add gathered_weights to gathered_weights_dict
                    queues_and_locks.return_model_queue_lock.release()   # Release lock
                print('Received weighted submodel update and count numbers from client ' + current_client.ID)
                sys.stdout.flush()

            except (OSError, OSError):  # Exception handler
                print('Fallen client: ' + current_client.address[0] + ':' + str(current_client.address[1]))
                sys.stdout.flush()
                try:
                    current_client.connection_socket.close()
                except (OSError, OSError):
                    pass


#================================== START: assistant functions for updating global model ====================================

def gather_weighted_submodel_updates(dataset_info, gathered_batches_info_list, gathered_weights_list, sess):
    """Sum all weighted submodel updates and corresponding count numbers"""
    # prepare (gathered) global weighted delta parameters
    gathered_weighted_delta_submodel = [np.zeros(weights.shape, dtype='uint32') for weights in sess.run(tf.trainable_variables())]
    # prepare (gathered) global count numbers
    gathered_userIDs_count = np.zeros(dataset_info['user_count'], dtype='uint32')
    gathered_itemIDs_count = np.zeros(dataset_info['item_count'], dtype='uint32')
    gathered_cateIDs_count = np.zeros(dataset_info['cate_count'], dtype='uint32')
    gathered_train_set_size = 0

    for client_info, client_weighted_delta_submodel in zip(gathered_batches_info_list, gathered_weights_list):
        userID = client_info['userID'][0]
        client_succinct_itemIDs = client_info['succinct_itemIDs']
        client_succinct_cateIDs = client_info['succinct_cateIDs']
        client_userID_count = client_info['userID_count']
        client_itemIDs_count = client_info['itemIDs_count']
        client_cateIDs_count = client_info['cateIDs_count']
        client_train_set_size = client_info['train_set_size']
        gathered_train_set_size += client_train_set_size

        for layer, delta_submodel_para in enumerate(client_weighted_delta_submodel):
            if layer == 0:   # embedding for user id
                gathered_weighted_delta_submodel[layer][userID] += delta_submodel_para[0]   # one user id
                gathered_userIDs_count[userID] += client_userID_count
            elif layer == 1:  # embedding for item ids
                for client_item_index in range(len(delta_submodel_para)):
                    ps_item_index = client_succinct_itemIDs[client_item_index]
                    gathered_weighted_delta_submodel[layer][ps_item_index] += delta_submodel_para[client_item_index]
                    gathered_itemIDs_count[ps_item_index] += client_itemIDs_count[client_item_index]
            elif layer == 2:  # embedding for cate ids
                for client_cate_index in range(len(delta_submodel_para)):
                    ps_cate_index = client_succinct_cateIDs[client_cate_index]
                    gathered_weighted_delta_submodel[layer][ps_cate_index] += delta_submodel_para[client_cate_index]
                    gathered_cateIDs_count[ps_cate_index] += client_cateIDs_count[client_cate_index]
            else:
                gathered_weighted_delta_submodel[layer] += delta_submodel_para
    return gathered_weighted_delta_submodel, gathered_userIDs_count, gathered_itemIDs_count, gathered_cateIDs_count, gathered_train_set_size


def average_submodel_updates_and_apply_global_update(gathered_weighted_delta_submodel, gathered_userIDs_count,\
                        gathered_itemIDs_count, gathered_cateIDs_count, gathered_train_set_size, hyperparameters, sess):
    """Average (according to training set size or evenly) the gathered weighted submodel updates"""
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
            if gathered_train_set_size > 0:
                avg_other_para = gathered_model_para * 1.0 / gathered_train_set_size
                avg_other_para = recover_compression(avg_other_para, hyperparameters)
                new_global_model[layer] = old_global_model[layer] + avg_other_para
            else:
                new_global_model[layer] = old_global_model[layer]
    """
    For debug usage only
    """
    for layer, model_para in enumerate(new_global_model):
        print("Layer %d, max new element %f, min new element %f"%(layer, np.max(model_para), np.min(model_para)))
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
                    avg_userID_para = recover_compression(avg_userID_para, hyperparameters)
                    new_global_model[layer][ui] = old_global_model[layer][ui] + avg_userID_para
                else:
                    new_global_model[layer][ui] = old_global_model[layer][ui]
        elif layer == 1:  # embedding for item ids
            for ii in range(len(gathered_model_para)):
                # pay attention to divide zero error
                if gathered_itemIDs_count[ii] > 0:
                    avg_itemID_para = gathered_model_para[ii] * 1.0 / gathered_train_set_size
                    avg_itemID_para = recover_compression(avg_itemID_para, hyperparameters)
                    new_global_model[layer][ii] = old_global_model[layer][ii] + avg_itemID_para
                else:
                    new_global_model[layer][ii] = old_global_model[layer][ii]
        elif layer == 2:  # embedding for cate ids
            for ci in range(len(gathered_model_para)):
                # pay attention to divide zero error
                if gathered_cateIDs_count[ci] > 0:
                    avg_cateID_para = gathered_model_para[ci] * 1.0 / gathered_train_set_size
                    avg_cateID_para = recover_compression(avg_cateID_para, hyperparameters)
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
    for layer, model_para in enumerate(new_global_model):
        print("Layer %d, max new element %f, min new element %f"%(layer, np.max(model_para), np.min(model_para)))
    sys.stdout.flush()
    return new_global_model


def do_update_weights(gathered_weights, placeholders, update_local_vars_op, sess):
    feed_dict = {}
    for place, para in zip(placeholders, gathered_weights):
        feed_dict[place] = para
    sess.run(update_local_vars_op, feed_dict=feed_dict)
    del feed_dict


def create_csv(path):
    with open(path, 'wb') as f:
        csv_writer = csv.writer(f)
        csv_head = ["round_num", "test_auc", "loss_sum", "accuracy_sum"]
        csv_writer.writerow(csv_head)


def write_csv(path, round_num, test_auc, loss_sum, accuracy_sum):
    with open(path, 'a+') as f:
        csv_writer = csv.writer(f)
        data_row = [round_num, test_auc, loss_sum, accuracy_sum]
        csv_writer.writerow(data_row)

#=================================== END: assistant functions for updating global model =====================================


def update_and_return_model(communication, queues_and_locks, hyperparameters, dataset_info, placeholders,\
                            update_local_vars_op, variables_pack_for_eval_and_save, g1, sess):
    """
    Aggregate all updates in this round and apply to the global model,
    then for each client in return_model_queue,
    get_info_and_return_model.
    """
    # create csv to store global model's aucs at ps side
    path = "./global_model_auc.csv"
    create_csv(path)

    round_num = 0
    # Test initial global model's auc
    test_auc, loss_sum, accuracy_sum = my_eval.eval_and_save(variables_pack_for_eval_and_save, round_num, sess)
    print('-----------------------------------------------------------------------')
    print('Initialized model: test_auc: %.4f ---- loss: %f ---- accuracy: %f' % (test_auc, loss_sum, accuracy_sum))
    print('-----------------------------------------------------------------------')
    sys.stdout.flush()
    write_csv(path, round_num, test_auc, loss_sum, accuracy_sum)


    round_num = 1
    while True:
        if len(queues_and_locks.return_model_queue) == 0:
            time.sleep(5)  # Sleep to avoid busy loop
            continue
        elif len(queues_and_locks.return_model_queue) > 0:    # Put return_model_queue to valid update queue
            queues_and_locks.return_model_queue_lock.acquire() # Get lock
            temp_return_model_queue = queues_and_locks.return_model_queue
            temp_gathered_weights_dict = queues_and_locks.gathered_weights_dict
            queues_and_locks.return_model_queue = []
            queues_and_locks.gathered_weights_dict = {}
            queues_and_locks.return_model_queue_lock.release() # Release_lock
            for client in temp_return_model_queue:
                queues_and_locks.valid_updates_dict[client.ID] = temp_gathered_weights_dict[client.ID]
                queues_and_locks.valid_updates_queue.append(client)
            del temp_return_model_queue
            del temp_gathered_weights_dict

        if len(queues_and_locks.valid_updates_queue) >= hyperparameters['sync_parameter']:
            gathered_batches_info_list = []
            gathered_weights_list = []
            for current_client in queues_and_locks.valid_updates_queue:
                gathered_batches_info_list.append(queues_and_locks.batches_info_dict[current_client.ID])
                gathered_weights_list.append(queues_and_locks.valid_updates_dict[current_client.ID])

            with g1.as_default():
                # Gather weighted submodel updates from clients and also gather count numbers
                gathered_weighted_delta_submodel, gathered_userIDs_count, gathered_itemIDs_count, gathered_cateIDs_count, \
                    gathered_train_set_size = gather_weighted_submodel_updates(dataset_info, gathered_batches_info_list, gathered_weights_list, sess)
                if hyperparameters['fl_flag']:
                    new_global_model = fl_average_submodel_updates_and_apply_global_update(gathered_weighted_delta_submodel, \
                    gathered_userIDs_count, gathered_itemIDs_count, gathered_cateIDs_count, gathered_train_set_size, hyperparameters, sess)
                else:
                    new_global_model = average_submodel_updates_and_apply_global_update(gathered_weighted_delta_submodel, \
                    gathered_userIDs_count, gathered_itemIDs_count, gathered_cateIDs_count, gathered_train_set_size, hyperparameters, sess)

                # Update global model at parameter server
                if queues_and_locks.update_model_lock.acquire():
                    do_update_weights(new_global_model, placeholders, update_local_vars_op, sess)
                    queues_and_locks.update_model_lock.release()
                print('Round {}: Weights received, average applied '.format(round_num) +
                      'among {} clients'.format(len(queues_and_locks.valid_updates_queue)) + ', model updated! Evaluating...')
                sys.stdout.flush()

                # Evaluate new global model
                test_auc, loss_sum, accuracy_sum = my_eval.eval_and_save(variables_pack_for_eval_and_save, round_num, sess)
                print('Model performance: test_auc: %.4f ---- loss: %f ---- accuracy: %f' % (test_auc, loss_sum, accuracy_sum))
                print('Best round: ' + str(variables_pack_for_eval_and_save['best_round']) + ' Best test_auc: ' + str(variables_pack_for_eval_and_save['best_auc']))
                print('-----------------------------------------------------------------------')
                print('')
                print('')
                sys.stdout.flush()
                write_csv(path, round_num, test_auc, loss_sum, accuracy_sum)

                # Choose client indices for next communication round
                # I.e., chosen Taobao users' file indices for next communication round
                # Attention for collusion
                total_users_num = hyperparameters['total_users_num']
                chosen_clients_num = hyperparameters['chosen_clients_num']
                current_random_user_index_set = queues_and_locks.next_random_user_index_set
                while True:
                    next_random_user_index_set = random.sample(range(1, total_users_num + 1), chosen_clients_num)
                    if judge_set_intersection(next_random_user_index_set, current_random_user_index_set):
                        continue
                    else:
                        queues_and_locks.next_random_user_index_set = next_random_user_index_set
                        break

                temp_valid_updates_queue = queues_and_locks.valid_updates_queue[:]
                # Attention: clear the stored information in the last round
                queues_and_locks.batches_info_dict = {}
                queues_and_locks.valid_updates_dict = {}
                queues_and_locks.valid_updates_queue = []
                # Get batch info and return submodel to clients in return_model_queue

                for current_client in temp_valid_updates_queue:
                    # Here is already congest to guarantee consistency between batches info, submodel update
                    # in a certain communication round, and also is consistent with the other chosen clients
                    get_info_and_return_model(communication, current_client, queues_and_locks, sess)
                    current_client.connection_socket.close()
                del temp_valid_updates_queue
            if round_num == hyperparameters['communication_rounds']:
                print('finished!')
                sys.stdout.flush()
                exit(-1)
            round_num += 1
            # update learning rate, only used for recover compression
            hyperparameters['learning_rate'] *= hyperparameters['decay_rate']

            del gathered_batches_info_list
            del gathered_weights_list


#######################################################################################################################
'''
def fl_average_submodel_updates_and_apply_global_update(gathered_weighted_delta_submodel, userID_union,\
                                       itemIDs_union, cateIDs_union, gathered_train_set_size, hyperparameters, sess):
    """Average (according to training set size) the gathered weighted submodel updates"""
    old_global_model = sess.run(tf.trainable_variables())
    new_global_model = [weights for weights in old_global_model]
    for layer, gathered_model_para in enumerate(gathered_weighted_delta_submodel):
        if layer == 0:  # embedding for user ids # 0 in compressed delta model is not 0.0 in fact
            for ui in userID_union:
                avg_userID_para = gathered_model_para[ui] * 1.0 / gathered_train_set_size
                avg_userID_para = recover_compression(avg_userID_para, hyperparameters)
                new_global_model[layer][ui] += avg_userID_para
        elif layer == 1:  # embedding for item ids
            for ii in itemIDs_union:
                avg_itemID_para = gathered_model_para[ii] * 1.0 / gathered_train_set_size
                avg_itemID_para = recover_compression(avg_itemID_para, hyperparameters)
                new_global_model[layer][ii] += avg_itemID_para
        elif layer == 2:  # embedding for cate ids
            for ci in cateIDs_union:
                avg_cateID_para = gathered_model_para[ci] * 1.0 / gathered_train_set_size
                avg_cateID_para = recover_compression(avg_cateID_para, hyperparameters)
                new_global_model[layer][ci] += avg_cateID_para
        else:
            # pay attention to divide zero error
            if gathered_train_set_size > 0:
                avg_other_para = gathered_model_para * 1.0 / gathered_train_set_size
                avg_other_para = recover_compression(avg_other_para, hyperparameters)
                new_global_model[layer] += avg_other_para
    return new_global_model

userID_union = []
itemIDs_union = []
cateIDs_union = []
for client_info in gathered_batches_info_list:
    userID = client_info['userID'][0]
    client_succinct_itemIDs = client_info['succinct_itemIDs']
    client_succinct_cateIDs = client_info['succinct_cateIDs']
    userID_union.append(userID)
    itemIDs_union += client_succinct_itemIDs
    cateIDs_union += client_succinct_cateIDs
userID_union = list( set(userID_union) )
userID_union.sort()
itemIDs_union = list( set(itemIDs_union) )
itemIDs_union.sort()
cateIDs_union = list( set(cateIDs_union) )
cateIDs_union.sort()
'''
