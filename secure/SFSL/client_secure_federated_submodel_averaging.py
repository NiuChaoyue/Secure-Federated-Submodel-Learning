import os
import sys
import numpy as np
from PRNG import *
from diffie_hellman import DHKE
from shamir_secret_sharing import SecretSharer
from AES_CBC import AESCipher
import time
from general_functions import determine_data_type
from general_functions import write_csv


def client_side_sfsa_round0(communication, client_socket, FEDSUBAVG_SELF_STORAGE, FEDSUBAVG_OTHERS_STORAGE, FEDSUBAVG_DHKE):
    """
    Generate secret and public Keys, and Send public keys to server.
    Also, receive public keys of other clients from server.
    This part can be merged with that in private set union, but for clarify we separate them.
    """
    start_time_1 = time.time()
    # Generate the 2 pair of Diffie-Hellman keys
    # "s" to generate the seed for the shared mask, and "c" to generate the shared symmetric encryption key
    # my_csk, my_cpk can actually use those in private set union!!!
    my_ssk, my_spk = FEDSUBAVG_DHKE.generate_keys()
    my_csk, my_cpk = FEDSUBAVG_DHKE.generate_keys()

    # Store the previously generated keys
    FEDSUBAVG_SELF_STORAGE['my_ssk'] = my_ssk
    FEDSUBAVG_SELF_STORAGE['my_spk'] = my_spk
    FEDSUBAVG_SELF_STORAGE['my_csk'] = my_csk
    FEDSUBAVG_SELF_STORAGE['my_cpk'] = my_cpk
    fedsubavg_client_pubkeys = {'client_ID': FEDSUBAVG_SELF_STORAGE['my_index'],
                                'spk': my_spk,
                                'cpk': my_cpk}

    end_time_1 = time.time()

    communication.send_np_array(fedsubavg_client_pubkeys, client_socket)
    print('Client %d sent public keys to server in secure federated submodel averaging' % FEDSUBAVG_SELF_STORAGE['my_index'])
    sys.stdout.flush()

    fedsubavg_pubkeys_dict = communication.get_np_array(client_socket)
    print('Received public keys of all clients from server.')

    start_time_2 = time.time()

    for client_index, pubkeys in fedsubavg_pubkeys_dict.items():
        # Does not need to store my own keys (already in FEDSUBAVG_SELF_STORAGE)
        if client_index != FEDSUBAVG_SELF_STORAGE['my_index']:
            FEDSUBAVG_OTHERS_STORAGE.setdefault(client_index, {})['spk'] = pubkeys['spk']
            FEDSUBAVG_OTHERS_STORAGE.setdefault(client_index, {})['cpk'] = pubkeys['cpk']
    # Record number of live clients (including client self) and the required threshold
    FEDSUBAVG_SELF_STORAGE['n'] = len(fedsubavg_pubkeys_dict)
    FEDSUBAVG_SELF_STORAGE['t'] = int(FEDSUBAVG_SELF_STORAGE['n'] / 2) + 1

    end_time_2 = time.time()
    write_csv(FEDSUBAVG_SELF_STORAGE['client_computation_time_path'], [FEDSUBAVG_SELF_STORAGE['communication_round_number'],\
                                                    "sfsa_U0", end_time_1 - start_time_1 + end_time_2 - start_time_2])


def client_side_sfsa_round1(communication, client_socket, FEDSUBAVG_SELF_STORAGE, FEDSUBAVG_OTHERS_STORAGE, \
                           fedsubavg_security_para_dict, FEDSUBAVG_DHKE):
    """
    Generate and send encrypted secret shares for PRNG seed and ssk.
    This can be merged with that in private set union, but for clarity, we still do not do so.
    Different from private set union here is that, the client also receives the indices of other clients for
    mutual mask. Specifically, we need to handle mutual masks for the embedding layers of item ids and cate ids
    in a ``submodel" way.
    """
    start_time_1 = time.time()
    # Generate seed for PRNG
    seed_len = fedsubavg_security_para_dict['seed_len']
    fedsubavg_b_entropy = os.urandom(seed_len/8)  #bytes
    fedsubavg_b = bytes2int(fedsubavg_b_entropy)

    t = FEDSUBAVG_SELF_STORAGE['t']
    n = FEDSUBAVG_SELF_STORAGE['n']
    # Generate t-out-of-n shares for PRNG's seed b
    fedsubavg_shares_b = SecretSharer.split_secret(fedsubavg_b, t, n)
    # Generate t-out-of-n shares for client's ssk
    fedsubavg_shares_my_ssk = SecretSharer.split_secret(FEDSUBAVG_SELF_STORAGE['my_ssk'], t, n)

    # Store random seed, and secret shares into self dictionary
    FEDSUBAVG_SELF_STORAGE['b_entropy'] = fedsubavg_b_entropy
    '''
    FEDSUBAVG_SELF_STORAGE['b'] = fedsubavg_b
    FEDSUBAVG_SELF_STORAGE['shares_b'] = fedsubavg_shares_b
    FEDSUBAVG_SELF_STORAGE['shares_my_ssk'] = fedsubavg_shares_my_ssk
    '''

    # Store my share of b in isolation
    # No need to store my share of my ssk, since I am alive to myself!
    fedsubavg_my_share_b = fedsubavg_shares_b[0]
    fedsubavg_shares_b = list( set(fedsubavg_shares_b) - set([fedsubavg_my_share_b]))
    FEDSUBAVG_SELF_STORAGE['my_share_b'] = fedsubavg_my_share_b

    fedsubavg_ss_ciphers_dict = {}
    for idx, client_index in enumerate(FEDSUBAVG_OTHERS_STORAGE.keys()): # Already except myself
        # Derive symmetric encryption key "agreed" with other client (with client_index) (via Diffie-Hellman Agreement)
        sym_enc_key = FEDSUBAVG_DHKE.agree(FEDSUBAVG_SELF_STORAGE['my_csk'], FEDSUBAVG_OTHERS_STORAGE[client_index]['cpk'])
        # Send ciphertext to other client (with client_index), where PS works as a mediation
        msg = str(FEDSUBAVG_SELF_STORAGE['my_index']) + ' || ' + str(client_index) + ' || ' + str(fedsubavg_shares_b[idx]) \
              + ' || ' + str(fedsubavg_shares_my_ssk[idx])
        # Encrypt with AES_CBC
        enc_msg = AESCipher(str(sym_enc_key)).encrypt(msg)
        fedsubavg_ss_ciphers_dict[client_index] = enc_msg

        FEDSUBAVG_OTHERS_STORAGE[client_index]['sym_enc_key'] = sym_enc_key
        '''
        FEDSUBAVG_OTHERS_STORAGE[client_index]['msg'] = msg
        FEDSUBAVG_OTHERS_STORAGE[client_index]['enc_msg'] = enc_msg
        '''
    end_time_1 = time.time()

    # send encrypted shares to the server
    fedsubavg_ss_ciphers_send_message = {'client_ID': FEDSUBAVG_SELF_STORAGE['my_index'],
                                     'ss_ciphers': fedsubavg_ss_ciphers_dict}
    communication.send_np_array(fedsubavg_ss_ciphers_send_message, client_socket)
    print('Client %d sent encrypted secret shares to server in secure federated submodel averaging' % FEDSUBAVG_SELF_STORAGE['my_index'])
    sys.stdout.flush()

    # Receive other clients' encrypted shares and indices for mutual mask to me from the server
    round1_returned_message = communication.get_np_array(client_socket)
    print("Received other clients' encrypted secret shares and indices for mutual mask from server")
    sys.stdout.flush()

    start_time_2 = time.time()

    # Decrypt the secret shares and store them
    ss_ciphers_dict_received = round1_returned_message['ss_ciphers_dict']
    for client_index, enc_msg in ss_ciphers_dict_received.items():
        # Decrypt the encrypted message and parse it
        sym_enc_key = FEDSUBAVG_OTHERS_STORAGE[client_index]['sym_enc_key']
        msg = AESCipher(str(sym_enc_key)).decrypt(enc_msg)
        msg_parts = msg.split(' || ')
        # Sanity check
        from_client_index = int(msg_parts[0])
        my_index = int(msg_parts[1])
        assert from_client_index == client_index and my_index == FEDSUBAVG_SELF_STORAGE['my_index']
        # Store secret shares of other clients
        FEDSUBAVG_OTHERS_STORAGE[client_index]['share_b'] = msg_parts[2]
        FEDSUBAVG_OTHERS_STORAGE[client_index]['share_ssk'] = msg_parts[3]
    # Indices of other clients (except myself) for mutual mask
    FEDSUBAVG_SELF_STORAGE['mutual_mask_general_client_indices'] = round1_returned_message['mutual_mask_general_client_indices']
    FEDSUBAVG_SELF_STORAGE['mutual_mask_itemID_client_indices'] = round1_returned_message['mutual_mask_itemID_client_indices']
    FEDSUBAVG_SELF_STORAGE['mutual_mask_cateID_client_indices'] = round1_returned_message['mutual_mask_cateID_client_indices']

    end_time_2 = time.time()
    write_csv(FEDSUBAVG_SELF_STORAGE['client_computation_time_path'], [FEDSUBAVG_SELF_STORAGE['communication_round_number'], \
               "sfsa_U1", end_time_1 - start_time_1 + end_time_2 - start_time_2])


def client_side_sfsa_generate_self_mask(fedsubavg_b_entropy, fedsubavg_shapes_dict, fedsubavg_security_para_dict):
    """
    Assistant function to generate self masks to weighted_delta_submodel, perturbed_itemIDs_count, perturbed_cateIDs_count.
    """
    # Load shapes and sizes
    submodel_shape = fedsubavg_shapes_dict['submodel_shape']
    perturbed_itemIDs_size = fedsubavg_shapes_dict['perturbed_itemIDs_size']
    perturbed_cateIDs_size = fedsubavg_shapes_dict['perturbed_cateIDs_size']

    # Load parameters and seed for PRNG
    security_strength = fedsubavg_security_para_dict['security_strength']
    modulo_model_r_len = fedsubavg_security_para_dict['modulo_model_r_len']
    modulo_count_r_len = fedsubavg_security_para_dict['modulo_count_r_len']

    # PRNG for self mask
    fedsubavg_DRBG_b = HMAC_DRBG(fedsubavg_b_entropy, security_strength)

    weighted_delta_submodel_b_mask = [np.zeros(para_shape, dtype='int64') for para_shape in submodel_shape]
    for layer, para_shape in enumerate(submodel_shape):
        if layer == 0:  # Do not perform any mask for the embedding layer of user ID
            continue # TO keep submodel structure
        else:
            vector_len = 1
            for dim in para_shape:
                vector_len *= dim
            b_mask_one_layer = prng(fedsubavg_DRBG_b, modulo_model_r_len, security_strength, vector_len)
            b_mask_one_layer = b_mask_one_layer.astype('int64')
            b_mask_one_layer = b_mask_one_layer.reshape(para_shape)
            weighted_delta_submodel_b_mask[layer] = b_mask_one_layer

    perturbed_itemIDs_count_b_mask = prng(fedsubavg_DRBG_b, modulo_count_r_len, security_strength, perturbed_itemIDs_size)
    perturbed_cateIDs_count_b_mask = prng(fedsubavg_DRBG_b, modulo_count_r_len, security_strength, perturbed_cateIDs_size)

    # Return b_mask as dictionary
    fedsubavg_b_mask_dict = dict()
    fedsubavg_b_mask_dict['weighted_delta_submodel'] = weighted_delta_submodel_b_mask
    fedsubavg_b_mask_dict['perturbed_itemIDs_count'] = perturbed_itemIDs_count_b_mask.astype('int64')
    fedsubavg_b_mask_dict['perturbed_cateIDs_count'] = perturbed_cateIDs_count_b_mask.astype('int64')
    return fedsubavg_b_mask_dict


def client_side_sfsa_generate_mutual_mask(FEDSUBAVG_SELF_STORAGE, FEDSUBAVG_OTHERS_STORAGE,\
            fedsubavg_shapes_dict, perturbed_itemIDs, perturbed_cateIDs, fedsubavg_security_para_dict, FEDSUBAVG_DHKE):
    """
    Assistant function to generate mutual masks to weighted_delta_submodel, perturbed_itemIDs_count, perturbed_cateIDs_count.
    """
    # Load shapes and counts
    submodel_shape = fedsubavg_shapes_dict['submodel_shape']
    perturbed_itemIDs_size = fedsubavg_shapes_dict['perturbed_itemIDs_size']
    perturbed_cateIDs_size = fedsubavg_shapes_dict['perturbed_cateIDs_size']

    # Load parameters and seed for PRNG
    seed_len = fedsubavg_security_para_dict['seed_len']
    security_strength = fedsubavg_security_para_dict['security_strength']
    modulo_model_r_len = fedsubavg_security_para_dict['modulo_model_r_len']
    modulo_count_r_len = fedsubavg_security_para_dict['modulo_count_r_len']
    item_count = fedsubavg_security_para_dict['item_count']   # Used for index cate id uniquely and globally

    weighted_delta_submodel_mutual_mask = [np.zeros(para_shape, dtype='int64') for para_shape in submodel_shape]
    perturbed_itemIDs_count_mutual_mask = np.zeros(perturbed_itemIDs_size, dtype='int64')
    perturbed_cateIDs_count_mutual_mask = np.zeros(perturbed_cateIDs_size, dtype='int64')

    # PRNG dict for general mutual masks, also facilitate mutual masks for embedding layer
    fedsubavg_s_uv_entropy_dict = dict()
    fedsubavg_DRGB_s_general_dict = dict()
    for client_index in FEDSUBAVG_SELF_STORAGE['mutual_mask_general_client_indices']:
        # Derive seed for mutual mask, i.e., agreed key, with other client (u, v via Diffie-Hellman Agreement)
        s_uv = FEDSUBAVG_DHKE.agree(FEDSUBAVG_SELF_STORAGE['my_ssk'], FEDSUBAVG_OTHERS_STORAGE[client_index]['spk'])
        s_uv_modulo = s_uv % (2 ** seed_len)
        s_uv_entropy = int2bytes(s_uv_modulo, seed_len / 8)
        # No personalized string here
        fedsubavg_DRGB_s = HMAC_DRBG(s_uv_entropy, security_strength)
        fedsubavg_s_uv_entropy_dict[client_index] = s_uv_entropy
        fedsubavg_DRGB_s_general_dict[client_index] = fedsubavg_DRGB_s

    for layer, para_shape in enumerate(submodel_shape):
        if layer == 0: # Do not perform any mask for the embedding for user ID
            continue
        elif layer == 1:   # embedding for perturbed item IDs
            for item_idx in range(para_shape[0]):  # to fetch a certain item id using local index [0, perturbed_itemIDs_size - 1]
                for client_index in FEDSUBAVG_SELF_STORAGE['mutual_mask_itemID_client_indices'][item_idx]:
                    # Derive seed for mutual mask, i.e., agreed key, with other client (u, v via Diffie-Hellman Agreement)
                    s_uv_entropy = fedsubavg_s_uv_entropy_dict[client_index]
                    # Personalized string (Please use global index, i.e., at the ps)
                    personal_int = perturbed_itemIDs[item_idx]
                    personal_bytes = int2bytes(personal_int, 4) # bytes
                    fedsubavg_DRGB_s = HMAC_DRBG(s_uv_entropy, security_strength, personal_bytes)
                    # PRNG first for item embedding parameters with length embedding_dim actually
                    s_mask_one_row = prng(fedsubavg_DRGB_s, modulo_model_r_len, security_strength, para_shape[1])
                    # PRNG second for item count with length 1
                    s_mask_one_count = prng(fedsubavg_DRGB_s, modulo_count_r_len, security_strength, 1)
                    # Minus the mask when other client is with larger index,
                    # or add the mask when other client is with smaller index
                    sgn = np.sign(FEDSUBAVG_SELF_STORAGE['my_index'] - client_index)
                    weighted_delta_submodel_mutual_mask[layer][item_idx] += sgn * s_mask_one_row
                    perturbed_itemIDs_count_mutual_mask[item_idx] += sgn * s_mask_one_count[0]
        elif layer == 2:   # embedding for perturbed cate IDs
            for cate_idx in range(para_shape[0]):  # to fetch a certain cate id using local index [0, perturbed_cateIDs_size - 1]
                for client_index in FEDSUBAVG_SELF_STORAGE['mutual_mask_cateID_client_indices'][cate_idx]:
                    # Derive seed for mutual mask, i.e., agreed key, with other client (u, v via Diffie-Hellman Agreement)
                    s_uv_entropy = fedsubavg_s_uv_entropy_dict[client_index]
                    # Personalized string (Please use global index, i.e., at the ps)
                    personal_int = item_count + perturbed_cateIDs[cate_idx]
                    personal_bytes = int2bytes(personal_int, 4) # bytes
                    fedsubavg_DRGB_s = HMAC_DRBG(s_uv_entropy, security_strength, personal_bytes)
                    # PRNG first for cate embedding parameters with length embedding_dim actually
                    s_mask_one_row = prng(fedsubavg_DRGB_s, modulo_model_r_len, security_strength, para_shape[1])
                    # PRNG second for cate count with length 1
                    s_mask_one_count = prng(fedsubavg_DRGB_s, modulo_count_r_len, security_strength, 1)
                    # Minus the mask when other client is with larger index,
                    # or add the mask when other client is with smaller index
                    sgn = np.sign(FEDSUBAVG_SELF_STORAGE['my_index'] - client_index)
                    weighted_delta_submodel_mutual_mask[layer][cate_idx] += sgn * s_mask_one_row
                    perturbed_cateIDs_count_mutual_mask[cate_idx] += sgn * s_mask_one_count[0]
        else:
            vector_len = 1
            for dim in para_shape:
                vector_len *= dim
            for client_index in FEDSUBAVG_SELF_STORAGE['mutual_mask_general_client_indices']:
                fedsubavg_DRGB_s = fedsubavg_DRGB_s_general_dict[client_index]
                s_mask_one_layer = prng(fedsubavg_DRGB_s, modulo_model_r_len, security_strength, vector_len)
                s_mask_one_layer = s_mask_one_layer.reshape(para_shape)
                # Minus the mask when other client is with larger index,
                # or add the mask when other client is with smaller index
                sgn = np.sign(FEDSUBAVG_SELF_STORAGE['my_index'] - client_index)
                weighted_delta_submodel_mutual_mask[layer] += sgn * s_mask_one_layer

    # Return s_mask as dictionary
    fedsubavg_s_mask_dict = dict()
    fedsubavg_s_mask_dict['weighted_delta_submodel'] = weighted_delta_submodel_mutual_mask
    fedsubavg_s_mask_dict['perturbed_itemIDs_count'] = perturbed_itemIDs_count_mutual_mask
    fedsubavg_s_mask_dict['perturbed_cateIDs_count'] = perturbed_cateIDs_count_mutual_mask
    del fedsubavg_s_uv_entropy_dict
    del fedsubavg_DRGB_s_general_dict
    return fedsubavg_s_mask_dict


def client_side_sfsa_round2(communication, client_socket, FEDSUBAVG_SELF_STORAGE, FEDSUBAVG_OTHERS_STORAGE, \
                fedsubavg_x_dict, perturbed_itemIDs, perturbed_cateIDs, fedsubavg_security_para_dict, FEDSUBAVG_DHKE):
    """
    Doubly mask the input, including weighted_delta_submodel, perturbed_itemIDs_count, perturbed_cateIDs_count,
    and send it to server. Here no mask for the embedding layer for one userID and the train_set_size.
    """
    start_time_1 = time.time()
    # Load original "input" x_dict
    weighted_delta_submodel = fedsubavg_x_dict['weighted_delta_submodel']
    perturbed_userID_count = fedsubavg_x_dict['perturbed_userID_count']
    perturbed_itemIDs_count = fedsubavg_x_dict['perturbed_itemIDs_count']
    perturbed_cateIDs_count = fedsubavg_x_dict['perturbed_cateIDs_count']
    perturbed_other_count = fedsubavg_x_dict['perturbed_other_count']

    # Prepare the shape and sizes for facilitate generating masks
    fedsubavg_shapes_dict = dict()
    fedsubavg_shapes_dict['submodel_shape'] = [para.shape for para in weighted_delta_submodel] # list of tuples
    fedsubavg_shapes_dict['perturbed_itemIDs_size'] = len(perturbed_itemIDs_count)
    fedsubavg_shapes_dict['perturbed_cateIDs_size'] = len(perturbed_cateIDs_count)

    # First, generate self mask
    fedsubavg_b_entropy = FEDSUBAVG_SELF_STORAGE['b_entropy']
    fedsubavg_b_mask_dict = client_side_sfsa_generate_self_mask(fedsubavg_b_entropy, fedsubavg_shapes_dict, fedsubavg_security_para_dict)

    end_time_1 = time.time()

    write_csv(FEDSUBAVG_SELF_STORAGE['client_computation_time_path'], [FEDSUBAVG_SELF_STORAGE['communication_round_number'], \
               "sfsa_U2_b_dict", end_time_1 - start_time_1])

    sys.stdout.flush()
    print("Client %d side secure federated submodel learning self mask generation costs %f s" \
          %(FEDSUBAVG_SELF_STORAGE['my_index'], end_time_1 - start_time_1))

    start_time_2 = time.time()
    # Second, generate mutual mask
    fedsubavg_s_mask_dict = client_side_sfsa_generate_mutual_mask(FEDSUBAVG_SELF_STORAGE, FEDSUBAVG_OTHERS_STORAGE, \
             fedsubavg_shapes_dict, perturbed_itemIDs, perturbed_cateIDs, fedsubavg_security_para_dict, FEDSUBAVG_DHKE)

    end_time_2 = time.time()
    write_csv(FEDSUBAVG_SELF_STORAGE['client_computation_time_path'], [FEDSUBAVG_SELF_STORAGE['communication_round_number'], \
               "sfsa_U2_s_dict", end_time_2 - start_time_2])

    sys.stdout.flush()
    print("Client %d side secure federated submodel learning mutual mask generation costs %f s" \
          % (FEDSUBAVG_SELF_STORAGE['my_index'], end_time_2 - start_time_2))


    start_time_3 = time.time()
    # Third, Add self and mutual masks to the original input x_dict, and derive the final output "y_dict" to send to server
    # For submodel parameters
    weighted_delta_submodel_masked = [np.zeros(para_shape, dtype='int64') for para_shape in fedsubavg_shapes_dict['submodel_shape']]
    modulo_model_r = fedsubavg_security_para_dict['modulo_model_r']
    modulo_model_r_len = fedsubavg_security_para_dict['modulo_model_r_len']
    model_data_type = determine_data_type(modulo_model_r_len)
    for layer, para_shape in enumerate(fedsubavg_shapes_dict['submodel_shape']):
        if layer == 0:  # embedding for user id
            weighted_delta_submodel_masked[layer] = weighted_delta_submodel[layer] % modulo_model_r
        else:
            # Attention: Please do not forget to add original x !!!
            weighted_delta_submodel_masked[layer] += weighted_delta_submodel[layer]
            weighted_delta_submodel_masked[layer] += fedsubavg_b_mask_dict['weighted_delta_submodel'][layer]
            weighted_delta_submodel_masked[layer] += fedsubavg_s_mask_dict['weighted_delta_submodel'][layer]
            weighted_delta_submodel_masked[layer] %= modulo_model_r
    weighted_delta_submodel_masked = [weights.astype(model_data_type) for weights in weighted_delta_submodel_masked]


    # For count numbers
    modulo_count_r = fedsubavg_security_para_dict['modulo_count_r']
    modulo_count_r_len = fedsubavg_security_para_dict['modulo_count_r_len']
    count_data_type = determine_data_type(modulo_count_r_len)

    perturbed_itemIDs_count_masked = perturbed_itemIDs_count + fedsubavg_b_mask_dict['perturbed_itemIDs_count'] + \
                                     fedsubavg_s_mask_dict['perturbed_itemIDs_count']
    perturbed_itemIDs_count_masked %= modulo_count_r
    perturbed_itemIDs_count_masked = perturbed_itemIDs_count_masked.astype(count_data_type)

    perturbed_cateIDs_count_masked = perturbed_cateIDs_count + fedsubavg_b_mask_dict['perturbed_cateIDs_count'] + \
                                     fedsubavg_s_mask_dict['perturbed_cateIDs_count']
    perturbed_cateIDs_count_masked %= modulo_count_r
    perturbed_cateIDs_count_masked = perturbed_cateIDs_count_masked.astype(count_data_type)

    # All outputs
    y_dict = {'weighted_delta_submodel_masked': weighted_delta_submodel_masked,
              'perturbed_itemIDs_count_masked': perturbed_itemIDs_count_masked,
              'perturbed_cateIDs_count_masked': perturbed_cateIDs_count_masked,
              'perturbed_userID_count': perturbed_userID_count,
              'perturbed_other_count': perturbed_other_count}

    end_time_3 = time.time()
    write_csv(FEDSUBAVG_SELF_STORAGE['client_computation_time_path'], [FEDSUBAVG_SELF_STORAGE['communication_round_number'], \
               "sfsa_U2_add_b_s_to_y_dict", end_time_3 - start_time_3])
    write_csv(FEDSUBAVG_SELF_STORAGE['client_computation_time_path'], [FEDSUBAVG_SELF_STORAGE['communication_round_number'], \
               "sfsa_U2_y_dict_total", end_time_1 - start_time_1 + end_time_2 - start_time_2 + end_time_3 - start_time_3])

    # Send masked input to the server
    fedsubavg_client_y = {'client_ID': FEDSUBAVG_SELF_STORAGE['my_index'], 'y_dict': y_dict}
    communication.send_np_array(fedsubavg_client_y, client_socket)
    print('Client %d sent masked submodel parameters and count numbers to server in secure federated submodel learning'\
          % FEDSUBAVG_SELF_STORAGE['my_index'])
    sys.stdout.flush()

    # Receive Online and Offline sets of clients in round 2
    round2_clients_status = communication.get_np_array(client_socket)
    print("Received clients' status in round 2 from server")
    sys.stdout.flush()
    FEDSUBAVG_SELF_STORAGE['U2'] = round2_clients_status['U2']
    FEDSUBAVG_SELF_STORAGE['U1\U2'] = round2_clients_status['U1\U2']


def client_side_sfsa_round3(communication, client_socket, FEDSUBAVG_SELF_STORAGE, FEDSUBAVG_OTHERS_STORAGE):
    """
    Send shares of b (for self mask) and ssk (for mutual mask) for live and dropped clients in U2 and U1\U2, respectively.
    """
    start_time = time.time()
    # U2: Except myself
    fedsubavg_u2_live = list(set(FEDSUBAVG_SELF_STORAGE['U2']) - set([FEDSUBAVG_SELF_STORAGE['my_index']]))
    # U1/U2
    fedsubavg_u2_drop = FEDSUBAVG_SELF_STORAGE['U1\U2']

    # Shares of self mask's seed for live clients
    fedsubavg_live_b_shares = dict()
    for client_index_live in fedsubavg_u2_live:
        fedsubavg_live_b_shares[client_index_live] = FEDSUBAVG_OTHERS_STORAGE[client_index_live]['share_b']
    fedsubavg_live_b_shares[FEDSUBAVG_SELF_STORAGE['my_index']] = FEDSUBAVG_SELF_STORAGE['my_share_b']

    # Shares of mutual mask's secret key for dropped clients
    fedsubavg_drop_s_shares = dict()
    for client_index_drop in fedsubavg_u2_drop:
        fedsubavg_drop_s_shares[client_index_drop] = FEDSUBAVG_OTHERS_STORAGE[client_index_drop]['share_ssk']

    write_csv(FEDSUBAVG_SELF_STORAGE['client_computation_time_path'], [FEDSUBAVG_SELF_STORAGE['communication_round_number'], \
               "sfsa_U3", time.time() - start_time])

    # Send shares to the server
    fedsubavg_shares = {'client_ID': FEDSUBAVG_SELF_STORAGE['my_index'],
                    'live_b_shares': fedsubavg_live_b_shares,
                    'drop_s_shares': fedsubavg_drop_s_shares}
    communication.send_np_array(fedsubavg_shares, client_socket)
    print('Client %d sent secret shares of live and dropped clients in round 2 to server in secure federated submodel averaging'\
          % FEDSUBAVG_SELF_STORAGE['my_index'])
    sys.stdout.flush()

    del fedsubavg_live_b_shares
    del fedsubavg_drop_s_shares


def client_side_secure_federated_submodel_averaging(communication, client_socket, client_index, fedsubavg_x_dict,\
          perturbed_itemIDs, perturbed_cateIDs, fedsubavg_security_para_dict, fedsubavg_u2_drop_flag, fedsubavg_u3_drop_flag,\
                                                    round_num, client_computation_time_path):
    """
    Main function for client to join secure federated submodel averaging through secure aggregation
    """
    # This dictionary will contain all the values generated by this client herself
    FEDSUBAVG_SELF_STORAGE = {}
    # This dictionary will contain all the values about the OTHER clients. It is keyed by client_index
    FEDSUBAVG_OTHERS_STORAGE = {}

    FEDSUBAVG_SELF_STORAGE['my_index'] = client_index
    FEDSUBAVG_SELF_STORAGE['communication_round_number'] = round_num
    FEDSUBAVG_SELF_STORAGE['client_computation_time_path'] = client_computation_time_path

    # ID 14 - 2048-bit MODP group for Diffie-Hellman Key Exchange
    FEDSUBAVG_DHKE = DHKE(groupID=14)
    client_side_sfsa_round0(communication, client_socket, FEDSUBAVG_SELF_STORAGE, FEDSUBAVG_OTHERS_STORAGE, FEDSUBAVG_DHKE)

    client_side_sfsa_round1(communication, client_socket, FEDSUBAVG_SELF_STORAGE, FEDSUBAVG_OTHERS_STORAGE, \
                           fedsubavg_security_para_dict, FEDSUBAVG_DHKE)

    if fedsubavg_u2_drop_flag:  # drop from round 2
        client_socket.close()
        print("Client %d drops out from Secure Federated Submodel Averaging U2 in this communication round \n" % client_index)
        print('-----------------------------------------------------------------')
        print('')
        print('')
        sys.stdout.flush()
        return

    # client's original inputs fedsubavg_x_dict: 'weighted_delta_submodel': uploaded_weighted_delta_submodel,
    #                         'perturbed_userID_count': uploaded_perturbed_userID_count,
    #                         'perturbed_itemIDs_count': uploaded_itemIDs_count,
    #                         'perturbed_cateIDs_count': uploaded_cateIDs_count,
    #                         'perturbed_other_count': uploaded_perturbed_other_count
    client_side_sfsa_round2(communication, client_socket, FEDSUBAVG_SELF_STORAGE, FEDSUBAVG_OTHERS_STORAGE, \
                            fedsubavg_x_dict, perturbed_itemIDs, perturbed_cateIDs, fedsubavg_security_para_dict, FEDSUBAVG_DHKE)

    if fedsubavg_u3_drop_flag:   # drop from round 3
        client_socket.close()
        print("Client %d drops out from Secure Federated Submodel Averaging U3 in this communication round \n" % client_index)
        print('-----------------------------------------------------------------')
        print('')
        print('')
        return

    client_side_sfsa_round3(communication, client_socket, FEDSUBAVG_SELF_STORAGE, FEDSUBAVG_OTHERS_STORAGE)

    del FEDSUBAVG_SELF_STORAGE
    del FEDSUBAVG_OTHERS_STORAGE
