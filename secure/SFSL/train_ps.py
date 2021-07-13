# -*- coding: UTF-8 -*-

import os
import sys
import pickle
import tensorflow as tf
import numpy as np
from model import Model_DIN
from communication import Communication
from data_iterator import DataIterator
import ps_plain_functions as ps_pl_fn
import evaluate as my_eval
import general_functions as gn_fn
import time
import random

# security modules
import ps_private_set_union as ps_psu

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
np.random.seed(1234)
tf.set_random_seed(1234)

################################################################
# Settings for Server Socket
################################################################

flags = tf.app.flags
flags.DEFINE_boolean('is_ps', True, 'True if it is ps')
FLAGS = flags.FLAGS

# set the connection information
PS_PUBLIC_IP = 'localhost:4113'         # Public IP of the ps
PS_PRIVATE_IP = 'localhost:4113'        # Private IP of the ps

# Create the communication object
communication = Communication(FLAGS.is_ps, PS_PRIVATE_IP, PS_PUBLIC_IP)

################################################################
# Constants and Hyperparameters
################################################################
with open('../taobao_data_process/taobao_user_item_cate_count.pkl', 'rb') as f:
    user_count, item_count, cate_count = pickle.load(f)

dataset_info = {'user_count': user_count,
                'item_count': item_count,
                'cate_count': cate_count}

# total number of Taobao users' datasets (dataset size >=32) for simulating clients
total_users_num = 46336
chosen_clients_num = 100

# training parameters setup
communication_rounds = 10000       # number of communication rounds
local_epoch_num = 1               # number of local epochs before one round average/communication

embedding_dim = 18
train_batch_size = 2
test_batch_size = 1024
predict_batch_size = 1
predict_users_num = 1
predict_ads_num = 1
compress_k_levels = 2**15 # number of levels in compression, if it's less than 2, it means no compression
clip_bound = 0.1
compress_bound = 1.0     # {2**8: 0.95, 2**16: 1.0}
opt_alg = 'sgd'
if opt_alg == 'adam':
    learning_rate = 0.001
    decay_rate = 1.0  # decay rate of learning rate in each communication round
# default is sgd
elif opt_alg == 'sgd':
    learning_rate = 1.0
    decay_rate = 0.999  # decay rate of learning rate in each communication round
# 0: indicate aggregate the whole submodel updates evenly
# 1: indicate aggregate the embedding evenly, while the other network parameters according training set size
# 2: indicate aggregate the whole submodel according to the involved training set size
# 3: indicate aggregate the whole submodel according to the whole training set size
size_agg_flag = 1
# 4: original federated learning aggregating way (using size_agg_flag = 3 and fl_flag = True)
fl_flag = False

# training hyperparameters shared with clients
hyperparameters = {'local_epoch_num': local_epoch_num,
                   'train_batch_size': train_batch_size,
                   'test_batch_size': test_batch_size,
                   'predict_batch_size': predict_batch_size,
                   'predict_users_num': predict_users_num,
                   'predict_ads_num': predict_ads_num,
                   'learning_rate': learning_rate,
                   'decay_rate': decay_rate,
                   'embedding_dim': embedding_dim,
                   'opt_alg': opt_alg,
                   'compress_k_levels': compress_k_levels,
                   'clip_bound': clip_bound,
                   'compress_bound': compress_bound,
                   'size_agg_flag': size_agg_flag,
                   'fl_flag': fl_flag}

# Security parameters for private set union
# Choosen modulo r, should consider ERROR '0' with probability 1/UNION_MODULO_R in perturbed Bloom filter
UNION_MODULO_R_LEN = 32
UNION_MODULO_R = 2 ** UNION_MODULO_R_LEN
# For HMAC_DRNG
UNION_PRNG_SECURITY_STRENGTH = 128  # bits
UNION_PRNG_SEED_LEN = 24 * 8  # bits 3/2 SECURITY_STRENGTH
union_security_para_dict = {'item_count': item_count,  # bloom filter's length
                            'modulo_r_len': UNION_MODULO_R_LEN,
                            'modulo_r': UNION_MODULO_R,
                            'seed_len': UNION_PRNG_SEED_LEN,
                            'security_strength': UNION_PRNG_SECURITY_STRENGTH}

# Security parameters for secure federated submodel averaging
# Choose modulo r, should incorporate maximum possible sum
FEDSUBAVG_MODEL_MODULO_R_LEN = 32
FEDSUBAVG_MODEL_MODULO_R = 2 ** FEDSUBAVG_MODEL_MODULO_R_LEN
if size_agg_flag == 0 or size_agg_flag == 1:
    FEDSUBAVG_COUNT_MODULO_R_LEN = 8 # log_2(chosen_clients_num)
else:
    FEDSUBAVG_COUNT_MODULO_R_LEN = 32 # log_2(the maximum sum of chosen clients' training set sizes)
FEDSUBAVG_COUNT_MODULO_R = 2 ** FEDSUBAVG_COUNT_MODULO_R_LEN
# For HMAC_DRNG
FEDSUBAVG_PRNG_SECURITY_STRENGTH = 128 # bits
FEDSUBAVG_PRNG_SEED_LEN = 24 * 8  # bits 3/2 SECURITY_STRENGTH
fedsubavg_security_para_dict = {'item_count': item_count,
                                'modulo_model_r_len': FEDSUBAVG_MODEL_MODULO_R_LEN,
                                'modulo_model_r': FEDSUBAVG_MODEL_MODULO_R,
                                'modulo_count_r_len': FEDSUBAVG_COUNT_MODULO_R_LEN,
                                'modulo_count_r': FEDSUBAVG_COUNT_MODULO_R,
                                'seed_len': FEDSUBAVG_PRNG_SEED_LEN,
                                'security_strength': FEDSUBAVG_PRNG_SECURITY_STRENGTH}

################################################################
# Preparations for Global Model and Test Phase
################################################################

test_data = DataIterator('../taobao_data_process/taobao_local_test_remap', test_batch_size)

CHECKPOINT_DIR = 'save_path_taobao/parameter_sever' #注意这个路径
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
CHECKPOINT_DIR = CHECKPOINT_DIR + '/ckpt'

# directories for clients
RUNTIME_DIR = './run_time/'   # Indexed by machine index, ps is 0
if not os.path.exists(RUNTIME_DIR):
    os.makedirs(RUNTIME_DIR)

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
# saver.restore(sess, './save_path_1/parameter_sever/ckpt')
placeholders = gn_fn.create_placeholders()
update_local_vars_op = gn_fn.assign_vars(tf.trainable_variables(), placeholders)

variables_pack_for_eval_and_save = {
    'model': model,
    'saver': saver,
    'test_set': test_data,
    'test_batch_size': test_batch_size,
    'best_auc': 0.0,
    'CHECKPOINT_DIR': CHECKPOINT_DIR,
    'best_round': 0
}

g1.finalize()

# create csvs to store PS's results, including global model's auc, computation overhead of each stage in a communication round
global_model_auc_path = "./global_model_auc.csv"
gn_fn.create_csv(global_model_auc_path, ["round_num", "test_auc", "loss_sum", "accuracy_sum"])
ps_computation_time_path = "./run_time/0_ps_computation_time.csv"
gn_fn.create_csv(ps_computation_time_path, ["round_num", "stage_name", "run_time(s)"])

################################################################
## Main Starts Here!
################################################################
round_num = 0
# test initialized model's auc
test_auc, loss_sum, accuracy_sum = my_eval.eval_and_save(variables_pack_for_eval_and_save, round_num, sess)
print('-----------------------------------------------------------------------')
print('Initialized model: test_auc: %.4f ---- loss: %f ---- accuracy: %f' % (test_auc, loss_sum, accuracy_sum))
print('-----------------------------------------------------------------------')
sys.stdout.flush()
gn_fn.write_csv(global_model_auc_path, [round_num, test_auc, loss_sum, accuracy_sum])


for round_num in range(1, communication_rounds + 1):
    print('Round %d starts!' % round_num)
    time_start = time.time()
    clients = []
    ps_pl_fn.send_hyperparameters(communication, clients, total_users_num, chosen_clients_num, hyperparameters, \
                               union_security_para_dict, fedsubavg_security_para_dict)
    # Update the communication socket set of online clients
    # In fact, for synchronization usage before your intended stage
    ps_pl_fn.check_connection(clients)

    # (Delete finally) Use plaintext protocol to compute and return union for debugging
    ps_pl_fn.get_compute_and_return_union(communication, clients, round_num, ps_computation_time_path)
    ps_pl_fn.check_connection(clients)

    # Server Side Private Set Union
    ps_psu.server_side_private_set_union(communication, clients, union_security_para_dict, round_num, ps_computation_time_path)
    ps_pl_fn.check_connection(clients, 5.0)

    batches_info_dict = {}
    ps_pl_fn.send_back_submodels(communication, clients, batches_info_dict, g1, sess)
    ps_pl_fn.check_connection(clients)

    # Server side Secure Federated Submodel Averaging
    ps_pl_fn.update_global_model(round_num, communication, clients, dataset_info, batches_info_dict, hyperparameters,
            fedsubavg_security_para_dict, placeholders, update_local_vars_op, variables_pack_for_eval_and_save, \
                                 global_model_auc_path, ps_computation_time_path, g1, sess)
    ps_pl_fn.check_connection(clients)
    ps_pl_fn.close_connection(clients)

    print ("The total time of this round at ps is " + str(time.time() - time_start) + "s")
    print('-----------------------------------------------------------------------')
    print('')
    sys.stdout.flush()
