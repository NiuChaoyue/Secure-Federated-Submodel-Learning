# -*- coding: UTF-8 -*-

import os
import threading
import sys
import pickle
import tensorflow as tf
import numpy as np
from copy import deepcopy
from model import Model_DIN
from communication import Communication
from data_iterator import DataIterator
import ps_functions as ps_fn
import evaluate as my_eval
import general_functions as gn_fn
import time
import random

################################################################
# Settings for parameter server
################################################################
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
np.random.seed(1234)
tf.set_random_seed(1234)

################################################################
# Settings for Server Socket
################################################################
# set the connection information
PS_PUBLIC_IP = 'localhost:8441'         # Public IP of the ps
PS_PRIVATE_IP = 'localhost:8441'        # Private IP of the ps

# Create the communication object
communication = Communication(True, PS_PRIVATE_IP, PS_PUBLIC_IP)

################################################################
# Constants and Hyper-parameters
################################################################
with open('../taobao_data_process/taobao_user_item_cate_count.pkl', 'rb') as f:
    user_count, item_count, cate_count = pickle.load(f)

dataset_info = {'user_count': user_count,
                'item_count': item_count,
                'cate_count': cate_count}

# total number of Taobao users' datasets (dataset size >=32) for simulating chosen clients
total_users_num = 46336
chosen_clients_num = 100

# training parameters setup
communication_rounds = 10000       # number of communication rounds
local_epoch_num = 1                # number of local epochs before one round average/communication

embedding_dim = 18
train_batch_size = 2
test_batch_size = 1024
predict_batch_size = 1
predict_users_num = 1
predict_ads_num = 1
compress_k_levels = 2**15  # number of levels in compression, if it's less than 2, it means no compression
clip_bound = 0.1
compress_bound = 1.0       # {2**8: 0.95, 2**16: 1.0}
opt_alg = 'sgd'
if opt_alg == 'adam':
    learning_rate = 0.001
    decay_rate = 1.0  # decay rate of learning rate in each communication round
# default is sgd
elif opt_alg == 'sgd':
    learning_rate = 1.0 #* (0.999**1538) recover from the checkpoint
    decay_rate = 0.999   # decay rate of learning rate in each communication round
# 0: indicate aggregate the whole submodel updates evenly
# 1: indicate aggregate the embedding evenly, while the other network parameters according training set size
# 2: indicate aggregate the whole submodel according to the involved training set size
# 3: indicate aggregate the whole submodel according to the whole training set size
size_agg_flag = 3
# 4: original federated learning aggregating way (using size_agg_flag = 3 and fl_flag = True)
fl_flag = True

# Chosen Taobao users' files for federated submodel learning in first communication round
init_user_id_set = random.sample(range(1, total_users_num + 1), chosen_clients_num)

hyperparameters = {'total_users_num': total_users_num,
                   'chosen_clients_num': chosen_clients_num,
                   'communication_rounds': communication_rounds,
                   'local_epoch_num': local_epoch_num,
                   'train_batch_size': train_batch_size,
                   'learning_rate': learning_rate,
                   'decay_rate': decay_rate,
                   'embedding_dim': embedding_dim,
                   'opt_alg': opt_alg,
                   'sync_parameter': chosen_clients_num,
                   'compress_k_levels': compress_k_levels,
                   'clip_bound': clip_bound,
                   'compress_bound': compress_bound,
                   'init_user_id_set': init_user_id_set,
                   'size_agg_flag': size_agg_flag,
                   'fl_flag': fl_flag}

################################################################
# Preparations for Global Model and Test Phase
################################################################
test_data = DataIterator('../taobao_data_process/taobao_local_test_remap', test_batch_size)

CHECKPOINT_DIR_BEST = 'save_path_taobao/parameter_sever_best' #注意这个路径
if not os.path.exists(CHECKPOINT_DIR_BEST):
    os.makedirs(CHECKPOINT_DIR_BEST)
CHECKPOINT_DIR_BEST = CHECKPOINT_DIR_BEST + '/ckpt'

CHECKPOINT_DIR_CURRENT = 'save_path_taobao/parameter_sever_current' #注意这个路径
if not os.path.exists(CHECKPOINT_DIR_CURRENT):
    os.makedirs(CHECKPOINT_DIR_CURRENT)
CHECKPOINT_DIR_CURRENT = CHECKPOINT_DIR_CURRENT + '/ckpt'

# directories for clients actually, but put here can guarantee only create once
PERMANENT_ANSWERS_DIR = './permanent_answers/'
if not os.path.exists(PERMANENT_ANSWERS_DIR):
    os.makedirs(PERMANENT_ANSWERS_DIR)

SUCCINCT_DATA_DIR = './taobao_succinct_datasets/'
if not os.path.exists(SUCCINCT_DATA_DIR):
    os.makedirs(SUCCINCT_DATA_DIR)


gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

model = Model_DIN(user_count, item_count, cate_count, embedding_dim, clip_bound, opt_alg)
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
g1 = tf.get_default_graph()
saver = tf.train.Saver()
saver.restore(sess, './save_path_taobao_1538/parameter_sever_current/ckpt')
placeholders = gn_fn.create_placeholders()
update_local_vars_op = gn_fn.assign_vars(tf.trainable_variables(), placeholders)

variables_pack_for_eval_and_save = {
    'model': model,
    'saver': saver,
    'test_set': test_data,
    'test_batch_size': test_batch_size,
    'best_auc': 0.5431938313297316,
    'best_round': 867,
    'CHECKPOINT_DIR_BEST': CHECKPOINT_DIR_BEST,
    'CHECKPOINT_DIR_CURRENT': CHECKPOINT_DIR_CURRENT
}

g1.finalize()


#======================================= Shared variables among threads ================================================
class QueuesAndLocks:
    def __init__(self):
        self.classification_queue = []
        self.get_update_queue = []
        self.return_model_queue = []
        self.gathered_weights_dict = {} # key: client.ID; value: weights
        self.batches_info_dict = {} # key: client.ID; value: next_batches_info
        self.valid_updates_queue = []
        self.valid_updates_dict = {}
        self.classification_queue_lock = threading.Lock()
        self.get_update_queue_lock = threading.Lock()
        self.return_model_queue_lock = threading.Lock() # for return_model_queue
        self.update_model_lock = threading.Lock()
        self.next_random_user_index_set = []  # to avoid one client to be chosen in two continual communication rounds (which may cause collusion in filenames)

queues_and_locks = QueuesAndLocks()

# For chosen client index sets in current and next communication round, avoid collusion here!!!
while True:
    random_user_index_set_round2 = random.sample(range(1, total_users_num + 1), chosen_clients_num)
    if ps_fn.judge_set_intersection(random_user_index_set_round2, init_user_id_set):
        continue
    else:
        queues_and_locks.next_random_user_index_set = random_user_index_set_round2
        break

#======================================= Shared variables among threads ================================================


#========================================== Threads handler functions ==================================================
def connection_classifier():
    """
    communicate with new connected clients in classification_queue to know whether they need initialization or to send update,
    then make initialization or passing them to get_update_queue.
    """
    global communication, queues_and_locks, hyperparameters, g1, sess
    ps_fn.classify_connections(communication, queues_and_locks, hyperparameters, g1, sess)


def update_receiver():
    """
    gather updates from clients in get_update_queue and get their next batches' information,
    save updates in gathered_weights_queue, save batch information in batches_info_queue,
    then passing them to return_model_queue.
    """
    global communication, queues_and_locks
    ps_fn.get_update(communication, queues_and_locks)


def model_updater_and_returner():
    """
    check whether it's time to update the global model according to len(return_model_queue) >= n,
    if true: update the global model and return model to return_model_queue.
    """
    global communication, queues_and_locks, hyperparameters, dataset_info, placeholders, update_local_vars_op, variables_pack_for_eval_and_save, g1, sess
    ps_fn.update_and_return_model(communication, queues_and_locks, hyperparameters, dataset_info, placeholders, update_local_vars_op, variables_pack_for_eval_and_save, g1, sess)


thread_connection_classifier = threading.Thread(target=connection_classifier)
thread_get_update = threading.Thread(target=update_receiver)
thread_update_and_return_model = threading.Thread(target=model_updater_and_returner)

thread_connection_classifier.start()
thread_get_update.start()
thread_update_and_return_model.start()
#========================================== Threads handler functions ==================================================

# Main function
print('Server started.')
sys.stdout.flush()

# Server always accepting new clients from clients (Listening)
ps_fn.accept_new_connections(communication, queues_and_locks)
