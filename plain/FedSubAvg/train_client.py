# -*- coding: UTF-8 -*-

import os
import pickle
import numpy as np
import tensorflow as tf
import sys
import time
from config import SEND_RECEIVE_CONF as SRC
from model import Model_DIN
from communication import Communication
from data_iterator import DataIterator
import general_functions as gn_fn
import client_functions as cl_fn
import math


################################################################
## Settings for This Machine
################################################################
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.machine_index % 2 + 1)

np.random.seed(1234)
tf.set_random_seed(1234)

################################################################
## Settings for Client Socket
################################################################
flags = tf.app.flags
flags.DEFINE_boolean('is_ps', False, 'True if it is parameter server')
flags.DEFINE_integer('machine_index', 1, 'Index of machine') # Index of simulation machine
FLAGS = flags.FLAGS

PS_PUBLIC_IP = 'localhost:9113'  # Public IP of the ps
PS_PRIVATE_IP = 'localhost:9113'  # Private IP of the ps

# Create the communication object and get the training hyperparameters
communication = Communication(FLAGS.is_ps, PS_PRIVATE_IP, PS_PUBLIC_IP)

################################################################
## Load item -> cate dict for generating succinct cate ids
################################################################
with open('../taobao_data_process/taobao_item_cate_dict.pkl', 'rb') as f:
    item_cate_dict = pickle.load(f) #str(itemID) -> str(cateID)


######################################################
# Phase 0: Initialization and prepare some parameters
######################################################
client_socket = communication.start_socket_client()
print('Waiting for PS\'s command...')
sys.stdout.flush()
client_socket.settimeout(600)
while True:
    signal = client_socket.recv(10)
    if signal == SRC.purpose:
        print('Sending init purpose...')
        sys.stdout.flush()
        client_socket.send(SRC.init)
        break
    else:
        client_socket.close()
        print('Server Error! Exit!')
        exit(-1)

hyperparameters = communication.get_np_array(client_socket)
communication_rounds = hyperparameters['communication_rounds']
local_epoch_num = hyperparameters['local_epoch_num']
train_batch_size = hyperparameters['train_batch_size']
embedding_dim = hyperparameters['embedding_dim']
opt_alg = hyperparameters['opt_alg']
clip_bound = hyperparameters['clip_bound']
init_user_id_set = hyperparameters['init_user_id_set']

# Some global parameters
machine_index = FLAGS.machine_index
random_user_index = init_user_id_set[machine_index - 1]  #To save next Taobao user file index

# Probability parameters customized by a client for double randomized response
# prob2 and prob6 are not needed here for testing model accuracy
prob1 = 15.0/16         # P(j in S' | j in S)
#prob2 = 1.0/16         # P(j in S' | j not in S)
prob3 = 15.0/16        # P(j in S'' | j in S')
prob4 = 1.0/16        # P(j in S'' | j not in S')
prob5 = prob1 * (prob3 - prob4) + prob4     # P(j in S'' | j in S)
#prob6 = prob2 * (prob3 - prob4) + prob4     # P(j in S'' | j not in S)

print("P(j in S'' | j in S) = %.3f"%prob5)
#print("P(j in S'' | j not in S) = %.3f"%prob6)


for round_num in range(1, communication_rounds + 1):

    global_start_time = time.time()

    ######################################################
    # Phase 1: Extract real index set, including real itemIDs, and user ID
    # Then, prepare succinct index set, succinct file
    # Finally, receive succinct model from parameter server
    ######################################################

    # Fetch client's training data: one Taobao user's dataset
    # It is the client's file index, but in fact contains the data of Taobao user with 'userID'
    client_index = random_user_index

    print("Client %d Communication round %d starts!" % (client_index, round_num))
    sys.stdout.flush()

    # real index set
    userID, real_itemIDs, real_train_set_size = cl_fn.extract_real_index_set(client_index)
    # succinct index set
    succinct_itemIDs = cl_fn.generate_succinct_item_ids(client_index, real_itemIDs, prob1, prob3, prob4)
    succinct_cateIDs = cl_fn.generate_succinct_cate_ids(succinct_itemIDs, item_cate_dict)
    # succinct dataset
    succinct_filename, succinct_itemIDs_count, succinct_cateIDs_count, succinct_train_set_size = \
        cl_fn.prepare_succinct_training_data(client_index, machine_index, succinct_itemIDs, succinct_cateIDs, item_cate_dict)
    succinct_train_set = DataIterator(succinct_filename, train_batch_size)

    print("Succinct train set size: %d" % succinct_train_set_size)
    print("Succinct / Real train set size: %f" % (succinct_train_set_size * 1.0 / real_train_set_size))
    sys.stdout.flush()

    # communicate with ps, send batches_info and receive succinct model
    print("Waiting for PS's send batches info command...")
    sys.stdout.flush()
    client_socket.settimeout(600)
    while True:
        signal = client_socket.recv(10)
        if signal == SRC.please_send_batches_info:
            print('Client %d sending real userID, succinct itemIDs, and succinct cateIDs...'%client_index)
            sys.stdout.flush()
            send_message = {'client_ID': client_index,
                            'userID': userID,
                            'succinct_itemIDs': succinct_itemIDs,
                            'succinct_cateIDs': succinct_cateIDs,
                            'machine_index': FLAGS.machine_index}  # Machine index to prefetch next client_index
            communication.send_np_array(send_message, client_socket)
            # print('Sending operation over. Receiving corresponding succinct submodel parameters...')
            # sys.stdout.flush()
            client_socket.send(SRC.please_send_model)
            received_message = communication.get_np_array(client_socket)
            old_succinct_submodel = received_message['succinct_submodel']
            random_user_index = received_message['random_user_index']
            client_socket.close()
            print('Received succinct submodel parameters from PS')
            sys.stdout.flush()
            break
        else:
            client_socket.close()
            print('Server Error! Exit!')
            exit(-1)

    ######################################################
    # Phase 2: Train succinct model over local succinct data
    ######################################################
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # create model and initialize it with the embedding and model parameters pulled from ps
    model = Model_DIN(1, len(succinct_itemIDs), len(succinct_cateIDs), embedding_dim, clip_bound, opt_alg)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    placeholders = gn_fn.create_placeholders()
    feed_dict = {}
    for place, para in zip(placeholders, old_succinct_submodel):
        feed_dict[place] = para
    update_local_vars_op = gn_fn.assign_vars(tf.trainable_variables(), placeholders)
    sess.run(update_local_vars_op, feed_dict=feed_dict)
    print('Weights succesfully initialized')
    sys.stdout.flush()

    # begin training process
    print('Begin training')
    sys.stdout.flush()
    temp_start_time = time.time()

    loss_sum = 0.0
    accuracy_sum = 0.0
    local_iter_cnt = 0
    for epoch in range(local_epoch_num):
        for src, tgt in succinct_train_set:
            uids, mids, cats, mid_his, cat_his, mid_mask, target, sl = gn_fn.prepare_data(src, tgt)
            loss, acc = model.train(sess, [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, hyperparameters['learning_rate']])
            loss_sum += loss
            accuracy_sum += acc
            local_iter_cnt += 1

    print('%d round training over' % round_num)
    if local_iter_cnt > 0:
        print('time: %d ----> iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f' %
             (time.time() - temp_start_time, local_iter_cnt, loss_sum / local_iter_cnt, accuracy_sum / local_iter_cnt))
    else:
        print('time: %d ----> iter: %d' % (time.time() - temp_start_time, local_iter_cnt))
    print('')
    sys.stdout.flush()


    ######################################################
    # Phase 3: Upload submodel update with the succinct index set
    ######################################################
    new_succinct_submodel = sess.run(tf.trainable_variables())
    sess.close()

    # Prepare weighted submodel update, count numbers of item ids, cate ids, and training set size
    # All using succinct index set
    # Also pay attention to weighted or even average flag in hyperparameters
    uploaded_weighted_delta_submodel, uploaded_userID_count, uploaded_itemIDs_count, uploaded_cateIDs_count, uploaded_train_set_size = \
        cl_fn.generate_succinct_weighted_submodel_update(old_succinct_submodel, new_succinct_submodel, succinct_itemIDs_count,\
                                                   succinct_cateIDs_count, succinct_train_set_size, hyperparameters)
    send_message = {'client_ID': client_index,
                    'weighted_delta_submodel': uploaded_weighted_delta_submodel,
                    'userID_count': uploaded_userID_count,
                    'itemIDs_count': uploaded_itemIDs_count,
                    'cateIDs_count': uploaded_cateIDs_count,
                    'train_set_size': uploaded_train_set_size}

    # connect to ps
    client_socket = communication.start_socket_client()
    client_socket.settimeout(600)
    while True:
        signal = client_socket.recv(10)
        if signal == SRC.purpose:
            client_socket.send(SRC.update)
            break
        else:
            client_socket.close()
            print('Server Error! Exit!')
            exit(-1)

    # send updates
    client_socket.settimeout(600)
    while True:
        signal = client_socket.recv(10)
        if signal == SRC.please_send_update:
            communication.send_np_array(send_message, client_socket)
            print('Sent succinct weighted submodel update and count numbers')
            sys.stdout.flush()
            break
        else:
            client_socket.close()
            print('Server Error! Exit!')
            exit(-1)


    # update learning rate at local client to keep consistent with parameter server
    hyperparameters['learning_rate'] *= hyperparameters['decay_rate']

    print("Client %d finishes in communication round %d, and takes %f second\n" % (client_index, round_num, time.time() - global_start_time))
    print('-----------------------------------------------------------------')
    print('')
    print('')
    sys.stdout.flush()

print('finished!')
sys.stdout.flush()
