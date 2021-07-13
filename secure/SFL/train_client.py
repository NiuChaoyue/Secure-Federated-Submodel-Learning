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
#from compress import uint_para_compress
import client_plain_functions as cl_pl_fn
import general_functions as gn_fn
import time
import math
from copy import deepcopy
import random

# security modules
# import client_private_set_union as cl_psu
import client_secure_federated_submodel_averaging as cl_sfsa

################################################################
## Settings for Client Socket
################################################################
flags = tf.app.flags
flags.DEFINE_boolean('is_ps', False, 'True if it is parameter server')
#Index of simulation machine
flags.DEFINE_integer('machine_index', 1, 'Index of machine')
FLAGS = flags.FLAGS

PS_PUBLIC_IP = 'localhost:4113'  # Public IP of the ps
PS_PRIVATE_IP = 'localhost:4113'  # Private IP of the ps

# Create the communication object and get the training hyperparameters
communication = Communication(FLAGS.is_ps, PS_PRIVATE_IP, PS_PUBLIC_IP)

communication_rounds = 10000

################################################################
## Settings for This Machine
################################################################

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.machine_index % 2 + 1)

np.random.seed(1234)
tf.set_random_seed(1234)

# create csvs to store client's results, including computation overhead of each stage in a communication round
# indexed by FLAGS.machine_index (consistent with log file)
client_computation_time_path = "./run_time/%d_client_computation_time.csv"%FLAGS.machine_index
gn_fn.create_csv(client_computation_time_path, ["round_num", "stage_name", "run_time(s)"])

################################################################
## Load item -> cate dict for generating perturbed cate ids
################################################################
with open('../taobao_data_process/taobao_item_cate_dict.pkl', 'rb') as f:
    item_cate_dict = pickle.load(f) #str(itemID) -> str(cateID)

################################################################
## Load global user count, item count, and cate count
################################################################
with open('../taobao_data_process/taobao_user_item_cate_count.pkl', 'rb') as f:
    user_count, item_count, cate_count = pickle.load(f)


for round_num in range(1, communication_rounds + 1):
    print('Round %d starts!'% round_num)
    sys.stdout.flush()

    global_start_time = time.time()
    ######################################################
    # Phase 0: Initialization and prepare some parameters
    ######################################################
    print('Phase 0: Initialization begins...')
    sys.stdout.flush()
    client_socket = communication.start_socket_client()
    received_message = communication.get_np_array(client_socket)
    # It is the client's file index, but in fact contains the data of Taobao user with 'userID'
    client_index = received_message['client_index']
    hyperparameters = received_message['hyperparameters']
    union_security_para_dict = received_message['union_security_para_dict']
    fedsubavg_security_para_dict = received_message['fedsubavg_security_para_dict']
    # reply to checking "online" state
    cl_pl_fn.response_to_check_connection(client_socket)

    local_epoch_num = hyperparameters['local_epoch_num']
    train_batch_size = hyperparameters['train_batch_size']
    learning_rate = hyperparameters['learning_rate']  # please pay attention to its update (already update by ps)
    embedding_dim = hyperparameters['embedding_dim']
    opt_alg = hyperparameters['opt_alg']
    clip_bound = hyperparameters['clip_bound']

    '''
    # Probability parameters for clients to drop in private set union (Simulation Usage Only)
    union_u2_drop_prop = 0.0    # Do not send masked input, and do not participate later procedures
    union_u3_drop_prop = 0.0    # Do not send secret shares about others for reconstruction, and do not participate later procedures
    union_u2_drop_flag = False
    union_u3_drop_flag = False
    if random.uniform(0.0, 1.0) < union_u2_drop_prop:
        union_u2_drop_flag = True
    if random.uniform(0.0, 1.0) < union_u3_drop_prop:
        union_u3_drop_flag = True
    '''

    # Probability parameters for clients to drop in secure federated submodel averaging (Simulation Usage Only)
    fedsubavg_u2_drop_prop = 0.025  # Do not send masked input dict, and do not participate later procedures
    fedsubavg_u3_drop_prop = 0.025  # Do not send secret shares about others for reconstruction, and do not participate later procedures
    fedsubavg_u2_drop_flag = False
    fedsubavg_u3_drop_flag = False
    if random.uniform(0.0, 1.0) < fedsubavg_u2_drop_prop:
        fedsubavg_u2_drop_flag = True
    if random.uniform(0.0, 1.0) < fedsubavg_u3_drop_prop:
        fedsubavg_u3_drop_flag = True

    '''
    # Probability parameters customized by a client for double randomized response
    cpp = 5        # choice of probability parameters, default is 3
    if cpp == 1:
        prob1 = 1.0  # P(j in S' | j in S)
        prob2 = 0.0  # P(j in S' | j not in S)
        prob3 = 1.0  # P(j in S'' | j in S')
        prob4 = 0.0  # P(j in S'' | j not in S')
    elif cpp == 2:
        prob1 = 15.0 / 16  # P(j in S' | j in S)
        prob2 = 1.0 / 16   # P(j in S' | j not in S)
        prob3 = 15.0 / 16  # P(j in S'' | j in S')
        prob4 = 1.0 / 16   # P(j in S'' | j not in S')
    elif cpp == 3:
        prob1 = 7.0 / 8  # P(j in S' | j in S)
        prob2 = 1.0 / 8  # P(j in S' | j not in S)
        prob3 = 7.0 / 8  # P(j in S'' | j in S')
        prob4 = 1.0 / 8  # P(j in S'' | j not in S')
    elif cpp == 4:
        prob1 = 3.0 / 4  # P(j in S' | j in S)
        prob2 = 1.0 / 4  # P(j in S' | j not in S)
        prob3 = 3.0 / 4  # P(j in S'' | j in S')
        prob4 = 1.0 / 4  # P(j in S'' | j not in S')
    elif cpp == 5:
        prob1 = 1.0  # P(j in S' | j in S)
        prob2 = 1.0  # P(j in S' | j not in S)
        prob3 = 1.0  # P(j in S'' | j in S')
        prob4 = 1.0  # P(j in S'' | j not in S')
    else:
        exit(-1)
    prob5 = prob1 * (prob3 - prob4) + prob4     # P(j in S'' | j in S)
    prob6 = prob2 * (prob3 - prob4) + prob4     # P(j in S'' | j not in S)

    print("P(j in S'' | j in S) = %.3f"%prob5)
    print("P(j in S'' | j not in S) = %.3f"%prob6)
    '''

    ######################################################
    # Phase 1: Prepare real index set
    #    Then, represent as perturbed Bloom Filter
    #    Finally, Participate in the union of real index sets (Through Secure Aggregation)
    ######################################################
    temp_start_time = time.time()

    userID, real_itemIDs, real_train_set_size = cl_pl_fn.extract_real_index_set(client_index)

    gn_fn.write_csv(client_computation_time_path, [round_num, "extract real itemIDs", time.time() - temp_start_time])

    print("Real train set size: %d" % real_train_set_size)

    '''
    # (Delete finally) Use plaintext protocol for debugging. Do not affect later procedures.
    real_itemIDs_union_plain = cl_pl_fn.client_side_set_union(communication, client_socket, client_index, real_itemIDs,\
                                                              union_u2_drop_flag)

    # reply to checking "online" state
    cl_pl_fn.response_to_check_connection(client_socket, 1.1)

    # Client Side Private Set Union
    print("Client %d side private set union starts!"%client_index)
    sys.stdout.flush()
    real_itemIDs_union = cl_psu.client_side_private_set_union(communication, client_socket, client_index, real_itemIDs,\
                                                    union_security_para_dict, union_u2_drop_flag, union_u3_drop_flag, \
                                                                               round_num, client_computation_time_path)
    # Skip all the following steps
    if union_u2_drop_flag or union_u3_drop_flag:
        time.sleep(600)     # wait for other clients to finish this communication round
        continue

    if real_itemIDs_union == real_itemIDs_union_plain:
        print("Yes! Private Set Union successfully finishes!")
    else:
        print("Oh, No! Private Set Union fails!")
        exit(-1)
    # reply to checking "online" state
    cl_pl_fn.response_to_check_connection(client_socket, 1.2)
    '''

    ######################################################
    # Phase 2: Generate perturbed index set, and use it to
    #          Download submodel
    ######################################################
    '''
    temp_start_time = time.time()

    perturbed_itemIDs = cl_pl_fn.generate_perturbed_item_ids(client_index, real_itemIDs, real_itemIDs_union,
                                                           prob1, prob2, prob3, prob4)
    perturbed_cateIDs = cl_pl_fn.generate_perturbed_cate_ids(perturbed_itemIDs, item_cate_dict)

    gn_fn.write_csv(client_computation_time_path, [round_num, "generate perturbed ids", time.time() - temp_start_time])
    '''
    # Federated Learning using all item IDs and cate IDs as perturbed IDs
    perturbed_itemIDs = range(item_count)
    perturbed_cateIDs = range(cate_count)

    send_message = {'client_ID': client_index,
                    'userID': userID,
                    'perturbed_itemIDs': perturbed_itemIDs,
                    'perturbed_cateIDs': perturbed_cateIDs}
    communication.send_np_array(send_message, client_socket)
    print('Sent perturbed user ids, item IDs, and cate IDs')
    sys.stdout.flush()
    dowloaded_submodel = communication.get_np_array(client_socket)
    print('Received submodel parameters.')
    sys.stdout.flush()

    # reply to checking "online" state
    cl_pl_fn.response_to_check_connection(client_socket, 2)

    ######################################################
    # Phase 3: Prepare succinct training data and local model
    ######################################################
    temp_start_time = time.time()

    '''
    succinct_itemIDs = cl_pl_fn.set_intersection(real_itemIDs, perturbed_itemIDs, True)
    succinct_cateIDs = cl_pl_fn.generate_succinct_cate_ids(succinct_itemIDs, item_cate_dict)
    '''
    succinct_itemIDs = perturbed_itemIDs
    succinct_cateIDs = perturbed_cateIDs

    # succinct data
    succinct_filename, succinct_itemIDs_count, succinct_cateIDs_count, succinct_train_set_size =\
        cl_pl_fn.prepare_succinct_training_data(client_index, succinct_itemIDs, succinct_cateIDs, item_cate_dict)
    succinct_train_set = DataIterator(succinct_filename, train_batch_size)

    # succinct model
    pos_succinct_itemIDs_in_perturbed = cl_pl_fn.convert_position(succinct_itemIDs, perturbed_itemIDs)
    pos_succinct_cateIDs_in_perturbed = cl_pl_fn.convert_position(succinct_cateIDs, perturbed_cateIDs)
    old_succinct_submodel = deepcopy(dowloaded_submodel)
    cl_pl_fn.gather_succinct_submodel(old_succinct_submodel, pos_succinct_itemIDs_in_perturbed, pos_succinct_cateIDs_in_perturbed)

    gn_fn.write_csv(client_computation_time_path, [round_num, "prepare succinct data and model", time.time() - temp_start_time])

    print("Succinct train set size: %d" % succinct_train_set_size)
    print("Succinct / Real train set size: %f" % (succinct_train_set_size * 1.0 / real_train_set_size))

    ######################################################
    # Phase 4: Train succinct model over local succinct data
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
    # heartbeat_keeper_flag = 1
    print('Begin training')
    sys.stdout.flush()

    temp_start_time = time.time()

    loss_sum = 0.0
    accuracy_sum = 0.
    local_iter_cnt = 0
    for epoch in range(local_epoch_num):
        for src, tgt in succinct_train_set:
            uids, mids, cats, mid_his, cat_his, mid_mask, target, sl = gn_fn.prepare_data(src, tgt)
            loss, acc = model.train(sess, [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, learning_rate])
            loss_sum += loss
            accuracy_sum += acc
            local_iter_cnt += 1

    gn_fn.write_csv(client_computation_time_path, [round_num, "training time", time.time() - temp_start_time])

    print('%d round training over' % round_num)
    if local_iter_cnt > 0:
        print('time: %d ----> iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f' %
              (time.time() - temp_start_time, local_iter_cnt, loss_sum / local_iter_cnt, accuracy_sum / local_iter_cnt))
    else:
        print('time: %d ----> iter: %d' % (time.time() - temp_start_time, local_iter_cnt))
    print('')
    sys.stdout.flush()

    ######################################################
    # Phase 5: Upload submodel update with the Perturbed index set
    ######################################################
    temp_start_time = time.time()

    new_succinct_submodel = sess.run(tf.trainable_variables())
    sess.close()

    uploaded_weighted_delta_submodel, uploaded_perturbed_userID_count, uploaded_perturbed_itemIDs_count, \
        uploaded_perturbed_cateIDs_count, uploaded_perturbed_other_count\
        = cl_pl_fn.generate_weighted_submodel_update(dowloaded_submodel, old_succinct_submodel, \
                new_succinct_submodel, pos_succinct_itemIDs_in_perturbed, pos_succinct_cateIDs_in_perturbed, \
                hyperparameters, len(perturbed_itemIDs), len(perturbed_cateIDs), \
                succinct_itemIDs_count, succinct_cateIDs_count, succinct_train_set_size)

    gn_fn.write_csv(client_computation_time_path, [round_num, "prepare weighted submodel update and count numbers to be uploaded", time.time() - temp_start_time])

    ## (Delete finally) Use plaintext protocol for debugging. Do not affect later procedures.
    send_message = {'client_ID': client_index,
                    'weighted_delta_submodel': uploaded_weighted_delta_submodel,
                    'perturbed_userID_count': uploaded_perturbed_userID_count,
                    'perturbed_itemIDs_count': uploaded_perturbed_itemIDs_count,
                    'perturbed_cateIDs_count': uploaded_perturbed_cateIDs_count,
                    'perturbed_other_count': uploaded_perturbed_other_count,
                    'fedsubavg_u2_drop_flag': fedsubavg_u2_drop_flag}
    # send to ps
    communication.send_np_array(send_message, client_socket)
    print('Sent weighted submodel update and count numbers w.r.t. the perturbed index set (via plaintext protocol).')
    sys.stdout.flush()

    # Participating in secure submodel averaging through secure aggregation protocol
    print('')
    print('Client %d side secure federated submodel averaging starts!' % client_index)
    sys.stdout.flush()
    fedsubavg_x_dict = {'weighted_delta_submodel': uploaded_weighted_delta_submodel,
                        'perturbed_userID_count': uploaded_perturbed_userID_count,
                        'perturbed_itemIDs_count': uploaded_perturbed_itemIDs_count,
                        'perturbed_cateIDs_count': uploaded_perturbed_cateIDs_count,
                        'perturbed_other_count': uploaded_perturbed_other_count}
    cl_sfsa.client_side_secure_federated_submodel_averaging(communication, client_socket, client_index, fedsubavg_x_dict,\
            fedsubavg_security_para_dict, fedsubavg_u2_drop_flag, fedsubavg_u3_drop_flag, round_num, client_computation_time_path)
    # Skip all the following steps
    if fedsubavg_u2_drop_flag or fedsubavg_u3_drop_flag:
        time.sleep(300)  # wait for other clients to finish this communication round
        continue

    # reply to checking "online" state
    cl_pl_fn.response_to_check_connection(client_socket, 5, True)

    print("Client %d finishes in communication round %d, and totally takes %f second\n" % (client_index, round_num, time.time() - global_start_time))
    print('-----------------------------------------------------------------------------------------------------')
    print('')
    print('')
    sys.stdout.flush()

print('finished!')
sys.stdout.flush()
