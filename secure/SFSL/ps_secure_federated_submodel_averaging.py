import os
import sys
import numpy as np
from PRNG import *
from diffie_hellman import DHKE
from shamir_secret_sharing import SecretSharer
import time
import multiprocessing as mp
from general_functions import determine_data_type
from general_functions import write_csv


def set_intersection_sfsa(a, b, sort_flag=True):
    """Union of two lists of elements"""
    c = list(set(a) & set(b))
    if sort_flag:
        c.sort()
    return c


def server_side_sfsa_round0(communication, clients, FEDSUBAVG_SERVER_STORAGE, FEDSUBAVG_ROUND_STORAGE):
    """
    Receive public keys from all live clients
    Then, Broadcast them.
    This part can be merged with that in private set union, but for clarify we separate them.
    """
    temp_clients = [client for client in clients]
    for client in temp_clients:
        try:
            received_message = communication.get_np_array(client.connection_socket)
            assert client.ID == received_message['client_ID']
            FEDSUBAVG_SERVER_STORAGE.setdefault(client.ID, {})['spk'] = received_message['spk']
            FEDSUBAVG_SERVER_STORAGE.setdefault(client.ID, {})['cpk'] = received_message['cpk']
            print('Received public keys from client ' + str(client.ID) + ' in secure federated submodel averaging round 0')
            sys.stdout.flush()
        except Exception:
            clients.remove(client)
            print('Fallen client: ' + str(client.ID) + ' at ' + client.address[0] + ':' + str(
                client.address[1]) + ' in secure federated submodel averaging round 0 (receive)')
            sys.stdout.flush()
            client.connection_socket.close()
    # Record update-to-date set of client indices in round 0
    FEDSUBAVG_ROUND_STORAGE['U0'] = [client.ID for client in clients]
    FEDSUBAVG_ROUND_STORAGE['n'] = len(FEDSUBAVG_ROUND_STORAGE['U0'])
    FEDSUBAVG_ROUND_STORAGE['t'] = int(FEDSUBAVG_ROUND_STORAGE['n'] / 2) + 1

    # At least 2 clients to participate (n = 2, t = 2)
    # Did not receive public keys from enough clients. Abort!
    assert FEDSUBAVG_ROUND_STORAGE['n'] >= 2

    start_time = time.time()
    # Gather public keys of all live clients in U0
    fedsubavg_pubkeys_dict = {}
    for client_index in FEDSUBAVG_ROUND_STORAGE['U0']:
        fedsubavg_pubkeys_dict[client_index] = {}
        fedsubavg_pubkeys_dict[client_index]['spk'] = FEDSUBAVG_SERVER_STORAGE[client_index]['spk']
        fedsubavg_pubkeys_dict[client_index]['cpk'] = FEDSUBAVG_SERVER_STORAGE[client_index]['cpk']

    write_csv(FEDSUBAVG_ROUND_STORAGE['ps_computation_time_path'], [FEDSUBAVG_ROUND_STORAGE['communication_round_num'], "sfsa_U0", time.time() - start_time])

    # Return gathered public keys to each live client
    temp_clients = [client for client in clients]
    for client in temp_clients:
        try:
            communication.send_np_array(fedsubavg_pubkeys_dict, client.connection_socket)
            print('Returned all public keys to client ' + str(client.ID) + ' in secure federated submodel averaging.')
            sys.stdout.flush()
        except Exception:
            clients.remove(client)
            print('Fallen client: ' + str(client.ID) + ' at ' + client.address[0] + ':' + str(
                client.address[1]) + ' in secure federated submodel averaging round 0 (return)')
            sys.stdout.flush()
            client.connection_socket.close()


def prepare_client_indices_for_mutual_mask(U1, ids_info_dict):
    """Prepare each client in U1 joins with which other clients for mutual mask,
       particularly specific to the embedding layers and counting numbers for item and cate ids,
       as well as general submodel parameters"""

    # First, generate dictionaries for id involving which clients
    itemID_client_indices_dict = {}
    cateID_client_indices_dict = {}
    for client_index in U1:
        for perturbed_itemID in ids_info_dict[client_index]['perturbed_itemIDs']:
            if perturbed_itemID not in itemID_client_indices_dict:
                itemID_client_indices_dict[perturbed_itemID] = []
            itemID_client_indices_dict[perturbed_itemID].append(client_index)
        for perturbed_cateID in ids_info_dict[client_index]['perturbed_cateIDs']:
            if perturbed_cateID not in cateID_client_indices_dict:
                cateID_client_indices_dict[perturbed_cateID] = []
            cateID_client_indices_dict[perturbed_cateID].append(client_index)
    # Second, generate each client's item and cate ids, and general parameters, involving which clients in U1
    # for mutual masks
    mutual_mask_general_client_indices_dict = {}  # for other layers as well as self mask
    mutual_mask_itemID_client_indices_dict = {}
    mutual_mask_cateID_client_indices_dict = {}
    for client_index in U1:
        # remove client herself at the server side
        mutual_mask_general_client_indices_dict[client_index] = list( set(U1) - set([client_index]) )
        mutual_mask_itemID_client_indices_dict[client_index] = [
            list( set(itemID_client_indices_dict[perturbed_itemID]) - set([client_index]) )
            for perturbed_itemID in ids_info_dict[client_index]['perturbed_itemIDs']
        ]
        mutual_mask_cateID_client_indices_dict[client_index] = [
            list( set(cateID_client_indices_dict[perturbed_cateID]) - set([client_index]) )
            for perturbed_cateID in ids_info_dict[client_index]['perturbed_cateIDs']
        ]

    return mutual_mask_general_client_indices_dict, mutual_mask_itemID_client_indices_dict, mutual_mask_cateID_client_indices_dict


def server_side_sfsa_round1(communication, clients, ids_info_dict, FEDSUBAVG_SERVER_STORAGE, FEDSUBAVG_ROUND_STORAGE):
    """
    Receive encrypted secret shares from clients and forward them
    """
    temp_clients = [client for client in clients]
    for client in temp_clients:
        try:
            received_message = communication.get_np_array(client.connection_socket)
            assert client.ID == received_message['client_ID']
            FEDSUBAVG_SERVER_STORAGE[client.ID]['ss_ciphers_dict'] = received_message['ss_ciphers']
            print('Received encrypted secret shares from client ' + str(client.ID) + ' in secure federated submodel averaging round 1')
            sys.stdout.flush()
        except Exception:
            clients.remove(client)
            print('Fallen client: ' + str(client.ID) + ' at ' + client.address[0] + ':' + str(
                client.address[1]) + ' in secure federated submodel averaging round 1 (receive)')
            sys.stdout.flush()
            client.connection_socket.close()
    # Record update-to-date set of client indices in round 1
    FEDSUBAVG_ROUND_STORAGE['U1'] = [client.ID for client in clients]
    # Record set of client indices live in round 0 but drop in round 1
    # I.e., Those clients who submit public keys but do not submit encrypted secret shares
    FEDSUBAVG_ROUND_STORAGE['U0\U1'] = list( set(FEDSUBAVG_ROUND_STORAGE['U1']) - set(FEDSUBAVG_ROUND_STORAGE['U0']) )

    # Did not receive encrypted secret shares from enough clients. Abort!
    assert len(FEDSUBAVG_ROUND_STORAGE['U1']) >= FEDSUBAVG_ROUND_STORAGE['t']

    start_time = time.time()

    # Instead of having a dictionary of messages FROM a given client, we want to construct
    # a dictionary of messages TO a given client.
    ss_ciphers_dicts_FROM = {}
    for client_index in FEDSUBAVG_ROUND_STORAGE['U1']:
        ss_ciphers_dicts_FROM[client_index] = FEDSUBAVG_SERVER_STORAGE[client_index]['ss_ciphers_dict']

    # This is here that we reverse the "FROM key TO value" dict to a "FROM value TO key" dict
    # e.g.: {1: {2:a, 3:b, 4:c}, 3: {1:d,2:e,4:f}, 4: {1:g,2:h,3:i}}  -->  {1: {3:d, 4:g}, 3:{1:b, 4:i}, 4: {1:c,3:f}}
    ss_ciphers_dicts_TO = {}
    # forward message "enc_msg_to_client_index" from "from_client_index" to "to_client_index"
    for from_client_index, enc_msg_from_client_index in ss_ciphers_dicts_FROM.items():
        for to_client_index, enc_msg_to_client_index in enc_msg_from_client_index.items():
            ss_ciphers_dicts_TO.setdefault(to_client_index, {})[from_client_index] = enc_msg_to_client_index

    # Special here: prepare each client involves which other clients for mutual mask
    mutual_mask_general_client_indices_dict, mutual_mask_itemID_client_indices_dict, mutual_mask_cateID_client_indices_dict, \
                          = prepare_client_indices_for_mutual_mask(FEDSUBAVG_ROUND_STORAGE['U1'], ids_info_dict)

    # Load client index -> userID, perturbed itemIDs, perturbed cate IDs into FEDSUBAVG_SERVER_STORAGE
    for client_index in FEDSUBAVG_ROUND_STORAGE['U1']:
        FEDSUBAVG_SERVER_STORAGE[client_index]['ids_info'] = ids_info_dict[client_index]

    write_csv(FEDSUBAVG_ROUND_STORAGE['ps_computation_time_path'], [FEDSUBAVG_ROUND_STORAGE['communication_round_num'], "sfsa_U1", time.time() - start_time])

    # Forward other clients'  encrypted secret share and indices for mutual mask to each live client
    temp_clients = [client for client in clients]
    for client in temp_clients:
        try:
            round1_return_message = {'ss_ciphers_dict': ss_ciphers_dicts_TO[client.ID],
                'mutual_mask_general_client_indices': mutual_mask_general_client_indices_dict[client.ID],
                'mutual_mask_itemID_client_indices': mutual_mask_itemID_client_indices_dict[client.ID],
                'mutual_mask_cateID_client_indices': mutual_mask_cateID_client_indices_dict[client.ID]}
            communication.send_np_array(round1_return_message, client.connection_socket)
            print('Forwarded encrypted secret shares and mutual mask related client indices to client ' + str(client.ID) + ' in secure federated submodel learning round 1')
            sys.stdout.flush()
            # Store in server's storage for later reconstruction usage
            # FEDSUBAVG_SERVER_STORAGE[client.ID]['mutual_mask_general_client_indices'] = mutual_mask_general_client_indices_dict[client.ID]
            FEDSUBAVG_SERVER_STORAGE[client.ID]['mutual_mask_itemID_client_indices'] = mutual_mask_itemID_client_indices_dict[client.ID]
            FEDSUBAVG_SERVER_STORAGE[client.ID]['mutual_mask_cateID_client_indices'] = mutual_mask_cateID_client_indices_dict[client.ID]
            time.sleep(5)  # create asynchronous environments to avoid clients competing for CPU resources
        except Exception:
            clients.remove(client)
            print('Fallen client: ' + str(client.ID) + ' at ' + client.address[0] + ':' + str(
                client.address[1]) + ' in secure federated submodel averaging round 1 (return)')
            sys.stdout.flush()
            client.connection_socket.close()


def server_side_sfsa_sum_y_dicts(fedsubavg_y_dicts, FEDSUBAVG_SERVER_STORAGE, FEDSUBAVG_ROUND_STORAGE):
    """
    Sum y_dicts contributed by clients in U2, and store in z_dict
    """
    # Load some parameters, and create and initialize z_dict
    global_model_shape = FEDSUBAVG_ROUND_STORAGE['global_model_shape']
    dataset_info = FEDSUBAVG_ROUND_STORAGE['dataset_info']

    # Prepare (gathered) global weighted delta parameters
    gathered_weighted_delta_submodel = [np.zeros(para_shape, dtype='int64') for para_shape in global_model_shape]
    # Prepare (gathered) global count numbers
    gathered_userIDs_count = np.zeros(dataset_info['user_count'], dtype='int64')
    gathered_itemIDs_count = np.zeros(dataset_info['item_count'], dtype='int64')
    gathered_cateIDs_count = np.zeros(dataset_info['cate_count'], dtype='int64')
    gathered_other_count = 0

    for client_index in FEDSUBAVG_ROUND_STORAGE['U2']:
        # Load y_dict of this client
        client_weighted_delta_submodel_masked = fedsubavg_y_dicts[client_index]['weighted_delta_submodel_masked']
        client_perturbed_itemIDs_count_masked = fedsubavg_y_dicts[client_index]['perturbed_itemIDs_count_masked']
        client_perturbed_cateIDs_count_masked = fedsubavg_y_dicts[client_index]['perturbed_cateIDs_count_masked']
        client_perturbed_userID_count = fedsubavg_y_dicts[client_index]['perturbed_userID_count']
        client_perturbed_other_count = fedsubavg_y_dicts[client_index]['perturbed_other_count']
        # Load client info (plaintext) of this client
        client_ids_info = FEDSUBAVG_SERVER_STORAGE[client_index]['ids_info']
        userID = client_ids_info['userID'][0]
        client_perturbed_itemIDs = client_ids_info['perturbed_itemIDs']
        client_perturbed_cateIDs = client_ids_info['perturbed_cateIDs']

        # Aggregate starts
        gathered_other_count += client_perturbed_other_count
        for layer, delta_submodel_para in enumerate(client_weighted_delta_submodel_masked):
            if layer == 0:   # embedding for user id
                gathered_weighted_delta_submodel[layer][userID] += delta_submodel_para[0]   # one user id
                gathered_userIDs_count[userID] += client_perturbed_userID_count
            elif layer == 1:  # embedding for item ids
                for client_item_index in range(len(delta_submodel_para)):
                    ps_item_index = client_perturbed_itemIDs[client_item_index]
                    gathered_weighted_delta_submodel[layer][ps_item_index] += delta_submodel_para[client_item_index]
                    gathered_itemIDs_count[ps_item_index] += client_perturbed_itemIDs_count_masked[client_item_index]
            elif layer == 2:  # embedding for cate ids
                for client_cate_index in range(len(delta_submodel_para)):
                    ps_cate_index = client_perturbed_cateIDs[client_cate_index]
                    gathered_weighted_delta_submodel[layer][ps_cate_index] += delta_submodel_para[client_cate_index]
                    gathered_cateIDs_count[ps_cate_index] += client_perturbed_cateIDs_count_masked[client_cate_index]
            else:
                gathered_weighted_delta_submodel[layer] += delta_submodel_para
        # Record submodel shapes and count numbers for later self mask reconstruction usage (live clients in U2)
        FEDSUBAVG_SERVER_STORAGE[client_index]['submodel_shape'] = [para.shape for para in client_weighted_delta_submodel_masked]
        FEDSUBAVG_SERVER_STORAGE[client_index]['perturbed_itemIDs_size'] = len(client_perturbed_itemIDs)
        FEDSUBAVG_SERVER_STORAGE[client_index]['perturbed_cateIDs_size'] = len(client_perturbed_cateIDs)

    z_dict = {'gathered_weighted_delta_submodel': gathered_weighted_delta_submodel,
              'gathered_userIDs_count': gathered_userIDs_count,
              'gathered_itemIDs_count': gathered_itemIDs_count,
              'gathered_cateIDs_count': gathered_cateIDs_count,
              'gathered_other_count': gathered_other_count}
    FEDSUBAVG_ROUND_STORAGE['z_dict'] = z_dict


def server_side_sfsa_round2(communication, clients, FEDSUBAVG_SERVER_STORAGE, FEDSUBAVG_ROUND_STORAGE):
    """
    Receive masked input from all live clients, and
    Send back the up-to-date set of client indices.
    """
    fedsubavg_y_dicts = dict()
    temp_clients = [client for client in clients]
    for client in temp_clients:
        try:
            received_message = communication.get_np_array(client.connection_socket)
            assert client.ID == received_message['client_ID']
            fedsubavg_y_dicts[client.ID] = received_message['y_dict']
            print('Received masked input from client ' + str(client.ID) + ' in secure federated submodel averaging round 2')
            sys.stdout.flush()
        except Exception:
            clients.remove(client)
            print('Fallen client: ' + str(client.ID) + ' at ' + client.address[0] + ':' + str(
                client.address[1]) + ' in secure federated submodel averaging round 2 (receive)')
            sys.stdout.flush()
            client.connection_socket.close()
    # Record update-to-date set of client indices in round 2
    FEDSUBAVG_ROUND_STORAGE['U2'] = [client.ID for client in clients]
    # Record set of client indices live in round 1 but dropped in round 2
    # Those clients who submit public keys, encrypted secret shares, but do not submit masked input
    FEDSUBAVG_ROUND_STORAGE['U1\U2'] = list( set(FEDSUBAVG_ROUND_STORAGE['U1']) - set(FEDSUBAVG_ROUND_STORAGE['U2']) )

    # Did not receive masked inputs from enough clients.
    assert len(FEDSUBAVG_ROUND_STORAGE['U2']) >= FEDSUBAVG_ROUND_STORAGE['t']

    # Forward statues of clients in round 2 to each live client
    round2_clients_status = {'U2': FEDSUBAVG_ROUND_STORAGE['U2'], 'U1\U2': FEDSUBAVG_ROUND_STORAGE['U1\U2']}
    temp_clients = [client for client in clients]
    for client in temp_clients:
        try:
            communication.send_np_array(round2_clients_status, client.connection_socket)
            print('Forwarded online and offline sets of clients to client ' + str(client.ID) + ' in secure federated submodel aggregating')
            sys.stdout.flush()
        except Exception:
            clients.remove(client)
            print('Fallen client: ' + str(client.ID) + ' at ' + client.address[0] + ':' + str(
                client.address[1]) + ' in secure federated submodel averaging round 2 (return)')
            sys.stdout.flush()
            client.connection_socket.close()

    start_time = time.time()
    # Avoid storing all y dicts into FEDSUBAVG_SERVER_STORAGE, and thus we here add them up, and store them in z_dict of
    # FEDSUBAVG_ROUND_STORAGE ahead of time,
    server_side_sfsa_sum_y_dicts(fedsubavg_y_dicts, FEDSUBAVG_SERVER_STORAGE, FEDSUBAVG_ROUND_STORAGE)

    write_csv(FEDSUBAVG_ROUND_STORAGE['ps_computation_time_path'], [FEDSUBAVG_ROUND_STORAGE['communication_round_num'], "sfsa_U2_add_y_dicts", time.time() - start_time])


def server_side_sfsa_reconstruct_single_self_mask(fedsubavg_b_shares, submodel_shape, perturbed_itemIDs_size, \
                                           perturbed_cateIDs_size, fedsubavg_security_para_dict):
    """
    Assistant function to reconstruct single self mask of weighted_delta_submodel, perturbed_itemIDs_count, perturbed_cateIDs_count.
    """
    # Load parameters and reconstruct seed for PRNG
    seed_len = fedsubavg_security_para_dict['seed_len']
    security_strength = fedsubavg_security_para_dict['security_strength']
    modulo_model_r_len = fedsubavg_security_para_dict['modulo_model_r_len']
    modulo_count_r_len = fedsubavg_security_para_dict['modulo_count_r_len']
    fedsubavg_b = SecretSharer.recover_secret(fedsubavg_b_shares)
    fedsubavg_b_entropy = int2bytes(fedsubavg_b, seed_len / 8)
    # PRNG for self mask
    fedsubavg_DRBG_b = HMAC_DRBG(fedsubavg_b_entropy, security_strength)

    # First, reconstruct
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


def server_side_sfsa_remove_single_self_mask(fedsubavg_b_mask_dict, client_ids_info, FEDSUBAVG_ROUND_STORAGE):
    """
    Remove single self mask from z_dict
    """
    # Load single self mask
    weighted_delta_submodel_b_mask = fedsubavg_b_mask_dict['weighted_delta_submodel']
    perturbed_itemIDs_count_b_mask = fedsubavg_b_mask_dict['perturbed_itemIDs_count']
    perturbed_cateIDs_count_b_mask = fedsubavg_b_mask_dict['perturbed_cateIDs_count']

    # Load ids for mapping to ps index system
    perturbed_itemIDs = client_ids_info['perturbed_itemIDs']
    perturbed_cateIDs = client_ids_info['perturbed_cateIDs']

    # Load global model and count numbers in z_dict
    z_dict_original = FEDSUBAVG_ROUND_STORAGE['z_dict']
    gathered_weighted_delta_submodel = z_dict_original['gathered_weighted_delta_submodel']
    gathered_itemIDs_count = z_dict_original['gathered_itemIDs_count']
    gathered_cateIDs_count = z_dict_original['gathered_cateIDs_count']

    # Remove starts, Please convert client's index system to ps's global index system
    for layer, para_b_mask in enumerate(weighted_delta_submodel_b_mask):
        if layer == 0:  # embedding for user id
            continue
        elif layer == 1:  # embedding for item ids
            for client_item_index in range(len(para_b_mask)):
                ps_item_index = perturbed_itemIDs[client_item_index]
                gathered_weighted_delta_submodel[layer][ps_item_index] -= para_b_mask[client_item_index]
                gathered_itemIDs_count[ps_item_index] -= perturbed_itemIDs_count_b_mask[client_item_index]
        elif layer == 2:  # embedding for cate ids
            for client_cate_index in range(len(para_b_mask)):
                ps_cate_index = perturbed_cateIDs[client_cate_index]
                gathered_weighted_delta_submodel[layer][ps_cate_index] -= para_b_mask[client_cate_index]
                gathered_cateIDs_count[ps_cate_index] -= perturbed_cateIDs_count_b_mask[client_cate_index]
        else:
            gathered_weighted_delta_submodel[layer] -= para_b_mask

    #Update global z_dict in FEDSUBAVG_ROUND_STORAGE
    FEDSUBAVG_ROUND_STORAGE['z_dict']['gathered_weighted_delta_submodel'] = gathered_weighted_delta_submodel
    FEDSUBAVG_ROUND_STORAGE['z_dict']['gathered_itemIDs_count'] = gathered_itemIDs_count
    FEDSUBAVG_ROUND_STORAGE['z_dict']['gathered_cateIDs_count'] = gathered_cateIDs_count


def server_side_sfsa_remove_self_mask(FEDSUBAVG_SERVER_STORAGE, FEDSUBAVG_ROUND_STORAGE, fedsubavg_security_para_dict):
    """
    Reconstruct the random seed b and thus remove self mask for each live client in U2 using the shares from U3.
    """
    # Gather b_shares for each live clients
    fedsubavg_all_b_shares = []
    for client_index in FEDSUBAVG_ROUND_STORAGE['U3']:
        fedsubavg_all_b_shares.append(FEDSUBAVG_SERVER_STORAGE[client_index]['live_b_shares'])
    fedsubavg_b_shares_dict = {
        k: [d.get(k) for d in fedsubavg_all_b_shares]
        for k in set().union(*fedsubavg_all_b_shares)     # U2
    }

    # Reconstruct and remove each self mask by PRNG expanding using the seed b
    # NOT U3!!!!!!! SHOULD BE U2, those clients who send masked input y
    # First, reconstruct self mask, parallel is easy
    fedsubavg_b_mask_dict_list = []
    for client_index in FEDSUBAVG_ROUND_STORAGE['U2']:
        fedsubavg_b_mask_dict = server_side_sfsa_reconstruct_single_self_mask(fedsubavg_b_shares_dict[client_index], \
            FEDSUBAVG_SERVER_STORAGE[client_index]['submodel_shape'], \
            FEDSUBAVG_SERVER_STORAGE[client_index]['perturbed_itemIDs_size'], \
            FEDSUBAVG_SERVER_STORAGE[client_index]['perturbed_cateIDs_size'], \
            fedsubavg_security_para_dict)
        fedsubavg_b_mask_dict_list.append(fedsubavg_b_mask_dict)
    # Second, remove self mask from global model, parallel is hard
    for idx, client_index in enumerate(FEDSUBAVG_ROUND_STORAGE['U2']):
        server_side_sfsa_remove_single_self_mask(fedsubavg_b_mask_dict_list[idx], \
                                    FEDSUBAVG_SERVER_STORAGE[client_index]['ids_info'], FEDSUBAVG_ROUND_STORAGE)


def server_side_sfsa_reconstruct_single_self_mask_indexed(client_index, fedsubavg_b_shares, submodel_shape, perturbed_itemIDs_size, \
                                           perturbed_cateIDs_size, fedsubavg_security_para_dict):
    """
    Assistant function to reconstruct single self mask of weighted_delta_submodel, perturbed_itemIDs_count, perturbed_cateIDs_count.
    To facilitate multiprocessing.
    """
    # Load parameters and reconstruct seed for PRNG
    seed_len = fedsubavg_security_para_dict['seed_len']
    security_strength = fedsubavg_security_para_dict['security_strength']
    modulo_model_r_len = fedsubavg_security_para_dict['modulo_model_r_len']
    modulo_count_r_len = fedsubavg_security_para_dict['modulo_count_r_len']
    fedsubavg_b = SecretSharer.recover_secret(fedsubavg_b_shares)
    fedsubavg_b_entropy = int2bytes(fedsubavg_b, seed_len / 8)
    # PRNG for self mask
    fedsubavg_DRBG_b = HMAC_DRBG(fedsubavg_b_entropy, security_strength)

    # First, reconstruct
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
    return client_index, fedsubavg_b_mask_dict


def server_side_sfsa_remove_self_mask_parallel(FEDSUBAVG_SERVER_STORAGE, FEDSUBAVG_ROUND_STORAGE, fedsubavg_security_para_dict):
    """
    Reconstruct the random seed b and thus remove self mask for each live client in U2 using the shares from U3.
    Optimized using multiprocessing the reconstruction stage.
    """
    # Gather b_shares for each live clients
    fedsubavg_all_b_shares = []
    for client_index in FEDSUBAVG_ROUND_STORAGE['U3']:
        fedsubavg_all_b_shares.append(FEDSUBAVG_SERVER_STORAGE[client_index]['live_b_shares'])
    fedsubavg_b_shares_dict = {
        k: [d.get(k) for d in fedsubavg_all_b_shares]
        for k in set().union(*fedsubavg_all_b_shares)     # U2
    }

    # Reconstruct and remove each self mask by PRNG expanding using the seed b
    # NOT U3!!!!!!! SHOULD BE U2, those clients who send masked input y
    # First, reconstruct self mask in parallel
    # Step 1: Initialize multiprocessing.Pool()
    pool = mp.Pool(mp.cpu_count())
    # Step 2: apply
    res_ojects = [ pool.apply_async(server_side_sfsa_reconstruct_single_self_mask_indexed, args=(client_index,\
                                               fedsubavg_b_shares_dict[client_index], \
                                               FEDSUBAVG_SERVER_STORAGE[client_index]['submodel_shape'], \
                                               FEDSUBAVG_SERVER_STORAGE[client_index]['perturbed_itemIDs_size'], \
                                               FEDSUBAVG_SERVER_STORAGE[client_index]['perturbed_cateIDs_size'], \
                                               fedsubavg_security_para_dict))\
                   for client_index in FEDSUBAVG_ROUND_STORAGE['U2'] ]
    fedsubavg_b_mask_dict_list = [ res.get() for res in res_ojects]
    # Step 3: close
    pool.close()
    # Second, remove self mask from global model, parallel is hard
    for client_index, fedsubavg_b_mask_dict in fedsubavg_b_mask_dict_list:
        server_side_sfsa_remove_single_self_mask(fedsubavg_b_mask_dict, \
                            FEDSUBAVG_SERVER_STORAGE[client_index]['ids_info'], FEDSUBAVG_ROUND_STORAGE)


def server_side_sfsa_reconstruct_single_mutual_mask(client_index_drop, fedsubavg_s_shares, global_model_shape, \
                client_ids_info, U2, client_mutual_mask_itemID_client_indices, client_mutual_mask_cateID_client_indices,\
                U2_fedsubavg_spk_dict, fedsubavg_security_para_dict, FEDSUBAVG_DHKE):
    """
    Assistant function to reconstruct mutual masks to weighted_delta_submodel, perturbed_itemIDs_count, perturbed_cateIDs_count.
    for one dropped client with its relevant live clients in U2.
    """
    # Load parameters and seed for PRNG
    seed_len = fedsubavg_security_para_dict['seed_len']
    security_strength = fedsubavg_security_para_dict['security_strength']
    modulo_model_r_len = fedsubavg_security_para_dict['modulo_model_r_len']
    modulo_count_r_len = fedsubavg_security_para_dict['modulo_count_r_len']
    item_count = fedsubavg_security_para_dict['item_count']   # Used for index cate id uniquely and globally

    # Load ids for mapping to ps index system, here used for personalized string in PRNG
    perturbed_itemIDs = client_ids_info['perturbed_itemIDs']
    perturbed_cateIDs = client_ids_info['perturbed_cateIDs']
    # The size can be used for reconstructing submodel shape and count number arrays
    perturbed_itemIDs_size = len(perturbed_itemIDs)
    perturbed_cateIDs_size = len(perturbed_cateIDs)

    # Attention: Since dropped client in U1/U2 do not provide her masked submodel, and thus PS needs to first reconstruct
    # her submodel shape using perturbed_itemIDs_size and perturbed_cateIDs_size
    submodel_shape = []
    for layer, para_shape in enumerate(global_model_shape):
        if layer == 0:
            submodel_shape.append((1, para_shape[1]))
        elif layer == 1:
            submodel_shape.append((perturbed_itemIDs_size, para_shape[1]))
        elif layer == 2:
            submodel_shape.append((perturbed_cateIDs_size, para_shape[1]))
        else:
            submodel_shape.append(para_shape)

    # Store mutual mask recover result
    weighted_delta_submodel_mutual_mask = [np.zeros(para_shape, dtype='int64') for para_shape in submodel_shape]
    perturbed_itemIDs_count_mutual_mask = np.zeros(perturbed_itemIDs_size, dtype='int64')
    perturbed_cateIDs_count_mutual_mask = np.zeros(perturbed_cateIDs_size, dtype='int64')

    # The dropped client's recovered ssk
    fedsubavg_drop_ssk = SecretSharer.recover_secret(fedsubavg_s_shares)

    # PRNG dict for general mutual mask
    U2_fedsubavg_DRGB_s_general_dict = dict()
    U2_fedsubavg_s_uv_entropy_dict = dict()
    for client_index_live in U2:  # Here client_mutual_mask_general_client_indices intersects with U2, in fact U2
        # Derive seed for mutual mask, i.e., agreed key, with other client (u, v via Diffie-Hellman Agreement)
        fedsubavg_live_spk = U2_fedsubavg_spk_dict[client_index_live]
        s_uv = FEDSUBAVG_DHKE.agree(fedsubavg_drop_ssk, fedsubavg_live_spk)
        s_uv_modulo = s_uv % (2 ** seed_len)
        s_uv_entropy = int2bytes(s_uv_modulo, seed_len / 8)
        # No personalized string here
        fedsubavg_DRGB_s = HMAC_DRBG(s_uv_entropy, security_strength)
        U2_fedsubavg_s_uv_entropy_dict[client_index_live] = s_uv_entropy
        U2_fedsubavg_DRGB_s_general_dict[client_index_live] = fedsubavg_DRGB_s

    for layer, para_shape in enumerate(submodel_shape):
        if layer == 0: # No mutual mask for the embedding for user ID
            continue
        elif layer == 1:   # embedding for perturbed item IDs
            for item_idx in range(para_shape[0]):  # to fetch a certain item id using local index [0, perturbed_itemIDs_size - 1]
                for client_index_live in set_intersection_sfsa(U2, client_mutual_mask_itemID_client_indices[item_idx]):  # Here already, intersection with U2
                    s_uv_entropy = U2_fedsubavg_s_uv_entropy_dict[client_index_live]
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
                    sgn = np.sign(client_index_live - client_index_drop)
                    weighted_delta_submodel_mutual_mask[layer][item_idx] += sgn * s_mask_one_row
                    perturbed_itemIDs_count_mutual_mask[item_idx] += sgn * s_mask_one_count[0]
        elif layer == 2:   # embedding for perturbed cate IDs
            for cate_idx in range(para_shape[0]):  # to fetch a certain cate id using local index [0, perturbed_cateIDs_size - 1]
                for client_index_live in set_intersection_sfsa(U2, client_mutual_mask_cateID_client_indices[cate_idx]): # Here Intersection with U2
                    s_uv_entropy = U2_fedsubavg_s_uv_entropy_dict[client_index_live]
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
                    sgn = np.sign(client_index_live - client_index_drop)
                    weighted_delta_submodel_mutual_mask[layer][cate_idx] += sgn * s_mask_one_row
                    perturbed_cateIDs_count_mutual_mask[cate_idx] += sgn * s_mask_one_count[0]
        else:
            vector_len = 1
            for dim in para_shape:
                vector_len *= dim
            for client_index_live in U2: # Analogous to Private set union
                fedsubavg_DRGB_s = U2_fedsubavg_DRGB_s_general_dict[client_index_live]
                s_mask_one_layer = prng(fedsubavg_DRGB_s, modulo_model_r_len, security_strength, vector_len)
                s_mask_one_layer = s_mask_one_layer.reshape(para_shape)
                # Minus the mask when other client is with larger index,
                # or add the mask when other client is with smaller index
                sgn = np.sign(client_index_live - client_index_drop)
                weighted_delta_submodel_mutual_mask[layer] += sgn * s_mask_one_layer

    # Return s_mask as dictionary
    fedsubavg_s_mask_dict = dict()
    fedsubavg_s_mask_dict['weighted_delta_submodel'] = weighted_delta_submodel_mutual_mask
    fedsubavg_s_mask_dict['perturbed_itemIDs_count'] = perturbed_itemIDs_count_mutual_mask
    fedsubavg_s_mask_dict['perturbed_cateIDs_count'] = perturbed_cateIDs_count_mutual_mask
    return fedsubavg_s_mask_dict


def server_side_sfsa_remove_single_mutual_mask(fedsubavg_s_mask_dict, client_ids_info, FEDSUBAVG_ROUND_STORAGE):
    """
    Remove single mutual mask from z_dict,
    Actually, just in the same way as server_side_sfsa_remove_single_self_mask
    """
    # Load single mutual mask
    weighted_delta_submodel_s_mask = fedsubavg_s_mask_dict['weighted_delta_submodel']
    perturbed_itemIDs_count_s_mask = fedsubavg_s_mask_dict['perturbed_itemIDs_count']
    perturbed_cateIDs_count_s_mask = fedsubavg_s_mask_dict['perturbed_cateIDs_count']

    # Load ids for mapping to ps index system
    perturbed_itemIDs = client_ids_info['perturbed_itemIDs']
    perturbed_cateIDs = client_ids_info['perturbed_cateIDs']

    # Load global model and count numbers in z_dict
    z_dict_original = FEDSUBAVG_ROUND_STORAGE['z_dict']
    gathered_weighted_delta_submodel = z_dict_original['gathered_weighted_delta_submodel']
    gathered_itemIDs_count = z_dict_original['gathered_itemIDs_count']
    gathered_cateIDs_count = z_dict_original['gathered_cateIDs_count']

    # Remove starts, Please convert client's index system to ps's global index system
    for layer, para_s_mask in enumerate(weighted_delta_submodel_s_mask):
        if layer == 0:  # embedding for user id
            continue
        elif layer == 1:  # embedding for item ids
            for client_item_index in range(len(para_s_mask)):
                ps_item_index = perturbed_itemIDs[client_item_index]
                gathered_weighted_delta_submodel[layer][ps_item_index] -= para_s_mask[client_item_index]
                gathered_itemIDs_count[ps_item_index] -= perturbed_itemIDs_count_s_mask[client_item_index]
        elif layer == 2:  # embedding for cate ids
            for client_cate_index in range(len(para_s_mask)):
                ps_cate_index = perturbed_cateIDs[client_cate_index]
                gathered_weighted_delta_submodel[layer][ps_cate_index] -= para_s_mask[client_cate_index]
                gathered_cateIDs_count[ps_cate_index] -= perturbed_cateIDs_count_s_mask[client_cate_index]
        else:
            gathered_weighted_delta_submodel[layer] -= para_s_mask

    #Update global z_dict in FEDSUBAVG_ROUND_STORAGE
    FEDSUBAVG_ROUND_STORAGE['z_dict']['gathered_weighted_delta_submodel'] = gathered_weighted_delta_submodel
    FEDSUBAVG_ROUND_STORAGE['z_dict']['gathered_itemIDs_count'] = gathered_itemIDs_count
    FEDSUBAVG_ROUND_STORAGE['z_dict']['gathered_cateIDs_count'] = gathered_cateIDs_count

def server_side_sfsa_remove_mutual_mask(FEDSUBAVG_SERVER_STORAGE, FEDSUBAVG_ROUND_STORAGE, fedsubavg_security_para_dict,\
                                        FEDSUBAVG_DHKE):
    """
    Reconstruct the secret key ssk and thus remove mutual mask for each dropped client in U1\U2 using the shares from U3.
    """
    # Deal with special case: no client drops in U2
    if len(FEDSUBAVG_ROUND_STORAGE['U1\U2']) == 0:
        return

    fedsubavg_all_s_shares = []
    for client_index in FEDSUBAVG_ROUND_STORAGE['U3']:
        fedsubavg_all_s_shares.append(FEDSUBAVG_SERVER_STORAGE[client_index]['drop_s_shares'])
    fedsubavg_s_shares_dict = {
        k: [d.get(k) for d in fedsubavg_all_s_shares]
        for k in set().union(*fedsubavg_all_s_shares)   # U1\U2
    }
    # Reconstruct and remover each mutual mask (pair of each dropped client in U1/U2 with each live client in U2)
    # First, reconstruct mutual mask for each dropped client, parallel is easy
    # NOT U3!!!!!!! SHOULD BE U2, those clients who send masked input y.
    # Cannot directly fetch this dropped client's submodel shape, use global model shape to reconstruct!
    U2 = FEDSUBAVG_ROUND_STORAGE['U2']
    U2_fedsubavg_spk_dict = {k: FEDSUBAVG_SERVER_STORAGE[k]['spk'] for k in U2}
    fedsubavg_s_mask_dict_list = []
    for client_index_drop in FEDSUBAVG_ROUND_STORAGE['U1\U2']:
        fedsubavg_s_mask_dict = server_side_sfsa_reconstruct_single_mutual_mask(client_index_drop,\
           fedsubavg_s_shares_dict[client_index_drop],\
           FEDSUBAVG_ROUND_STORAGE['global_model_shape'], \
           FEDSUBAVG_SERVER_STORAGE[client_index_drop]['ids_info'], \
           U2, FEDSUBAVG_SERVER_STORAGE[client_index_drop]['mutual_mask_itemID_client_indices'],\
           FEDSUBAVG_SERVER_STORAGE[client_index_drop]['mutual_mask_cateID_client_indices'],\
           U2_fedsubavg_spk_dict, fedsubavg_security_para_dict, FEDSUBAVG_DHKE)
        fedsubavg_s_mask_dict_list.append(fedsubavg_s_mask_dict)
    # Second, remove mutual mask for each dropped client, parallel is hard
    for idx, client_index_drop in enumerate(FEDSUBAVG_ROUND_STORAGE['U1\U2']):
        server_side_sfsa_remove_single_mutual_mask(fedsubavg_s_mask_dict_list[idx],\
                    FEDSUBAVG_SERVER_STORAGE[client_index_drop]['ids_info'], FEDSUBAVG_ROUND_STORAGE)


def server_side_sfsa_reconstruct_single_mutual_mask_indexed(client_index_drop, fedsubavg_s_shares, global_model_shape, \
                client_ids_info, U2, client_mutual_mask_itemID_client_indices, client_mutual_mask_cateID_client_indices,\
                U2_fedsubavg_spk_dict, fedsubavg_security_para_dict, FEDSUBAVG_DHKE):
    """
    Assistant function to reconstruct mutual masks to weighted_delta_submodel, perturbed_itemIDs_count, perturbed_cateIDs_count.
    for one dropped client with its relevant live clients in U2.
    To facilitate multiprocessing
    """
    # Load parameters and seed for PRNG
    seed_len = fedsubavg_security_para_dict['seed_len']
    security_strength = fedsubavg_security_para_dict['security_strength']
    modulo_model_r_len = fedsubavg_security_para_dict['modulo_model_r_len']
    modulo_count_r_len = fedsubavg_security_para_dict['modulo_count_r_len']
    item_count = fedsubavg_security_para_dict['item_count']   # Used for index cate id uniquely and globally

    # Load ids for mapping to ps index system, here used for personalized string in PRNG
    perturbed_itemIDs = client_ids_info['perturbed_itemIDs']
    perturbed_cateIDs = client_ids_info['perturbed_cateIDs']
    # The size can be used for reconstructing submodel shape and count number arrays
    perturbed_itemIDs_size = len(perturbed_itemIDs)
    perturbed_cateIDs_size = len(perturbed_cateIDs)

    # Attention: Since dropped client in U1/U2 do not provide her masked submodel, and thus PS needs to first reconstruct
    # her submodel shape using perturbed_itemIDs_size and perturbed_cateIDs_size
    submodel_shape = []
    for layer, para_shape in enumerate(global_model_shape):
        if layer == 0:
            submodel_shape.append((1, para_shape[1]))
        elif layer == 1:
            submodel_shape.append((perturbed_itemIDs_size, para_shape[1]))
        elif layer == 2:
            submodel_shape.append((perturbed_cateIDs_size, para_shape[1]))
        else:
            submodel_shape.append(para_shape)

    # Store mutual mask recover result
    weighted_delta_submodel_mutual_mask = [np.zeros(para_shape, dtype='int64') for para_shape in submodel_shape]
    perturbed_itemIDs_count_mutual_mask = np.zeros(perturbed_itemIDs_size, dtype='int64')
    perturbed_cateIDs_count_mutual_mask = np.zeros(perturbed_cateIDs_size, dtype='int64')

    # The dropped client's recovered ssk
    fedsubavg_drop_ssk = SecretSharer.recover_secret(fedsubavg_s_shares)

    # PRNG dict for general mutual mask
    U2_fedsubavg_DRGB_s_general_dict = dict()
    U2_fedsubavg_s_uv_entropy_dict = dict()
    for client_index_live in U2:  # Here client_mutual_mask_general_client_indices intersects with U2, in fact U2
        # Derive seed for mutual mask, i.e., agreed key, with other client (u, v via Diffie-Hellman Agreement)
        fedsubavg_live_spk = U2_fedsubavg_spk_dict[client_index_live]
        s_uv = FEDSUBAVG_DHKE.agree(fedsubavg_drop_ssk, fedsubavg_live_spk)
        s_uv_modulo = s_uv % (2 ** seed_len)
        s_uv_entropy = int2bytes(s_uv_modulo, seed_len / 8)
        # No personalized string here
        fedsubavg_DRGB_s = HMAC_DRBG(s_uv_entropy, security_strength)
        U2_fedsubavg_s_uv_entropy_dict[client_index_live] = s_uv_entropy
        U2_fedsubavg_DRGB_s_general_dict[client_index_live] = fedsubavg_DRGB_s

    for layer, para_shape in enumerate(submodel_shape):
        if layer == 0: # No mutual mask for the embedding for user ID
            continue
        elif layer == 1:   # embedding for perturbed item IDs
            for item_idx in range(para_shape[0]):  # to fetch a certain item id using local index [0, perturbed_itemIDs_size - 1]
                for client_index_live in set_intersection_sfsa(U2, client_mutual_mask_itemID_client_indices[item_idx]):  # Here already, intersection with U2
                    s_uv_entropy = U2_fedsubavg_s_uv_entropy_dict[client_index_live]
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
                    sgn = np.sign(client_index_live - client_index_drop)
                    weighted_delta_submodel_mutual_mask[layer][item_idx] += sgn * s_mask_one_row
                    perturbed_itemIDs_count_mutual_mask[item_idx] += sgn * s_mask_one_count[0]
        elif layer == 2:   # embedding for perturbed cate IDs
            for cate_idx in range(para_shape[0]):  # to fetch a certain cate id using local index [0, perturbed_cateIDs_size - 1]
                for client_index_live in set_intersection_sfsa(U2, client_mutual_mask_cateID_client_indices[cate_idx]): # Here Intersection with U2
                    s_uv_entropy = U2_fedsubavg_s_uv_entropy_dict[client_index_live]
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
                    sgn = np.sign(client_index_live - client_index_drop)
                    weighted_delta_submodel_mutual_mask[layer][cate_idx] += sgn * s_mask_one_row
                    perturbed_cateIDs_count_mutual_mask[cate_idx] += sgn * s_mask_one_count[0]
        else:
            vector_len = 1
            for dim in para_shape:
                vector_len *= dim
            for client_index_live in U2: # Analogous to Private set union
                fedsubavg_DRGB_s = U2_fedsubavg_DRGB_s_general_dict[client_index_live]
                s_mask_one_layer = prng(fedsubavg_DRGB_s, modulo_model_r_len, security_strength, vector_len)
                s_mask_one_layer = s_mask_one_layer.reshape(para_shape)
                # Minus the mask when other client is with larger index,
                # or add the mask when other client is with smaller index
                sgn = np.sign(client_index_live - client_index_drop)
                weighted_delta_submodel_mutual_mask[layer] += sgn * s_mask_one_layer

    # Return s_mask as dictionary
    fedsubavg_s_mask_dict = dict()
    fedsubavg_s_mask_dict['weighted_delta_submodel'] = weighted_delta_submodel_mutual_mask
    fedsubavg_s_mask_dict['perturbed_itemIDs_count'] = perturbed_itemIDs_count_mutual_mask
    fedsubavg_s_mask_dict['perturbed_cateIDs_count'] = perturbed_cateIDs_count_mutual_mask
    return client_index_drop, fedsubavg_s_mask_dict


def server_side_sfsa_remove_mutual_mask_parallel(FEDSUBAVG_SERVER_STORAGE, FEDSUBAVG_ROUND_STORAGE, fedsubavg_security_para_dict,\
                                                 FEDSUBAVG_DHKE):
    """
    Reconstruct the secret key ssk and thus remove mutual mask for each dropped client in U1\U2 using the shares from U3.
    Optimized using multiprocessing in the reconstruction stage.
    """
    # Deal with special case: no client drops in U2
    if len(FEDSUBAVG_ROUND_STORAGE['U1\U2']) == 0:
        return

    fedsubavg_all_s_shares = []
    for client_index in FEDSUBAVG_ROUND_STORAGE['U3']:
        fedsubavg_all_s_shares.append(FEDSUBAVG_SERVER_STORAGE[client_index]['drop_s_shares'])
    fedsubavg_s_shares_dict = {
        k: [d.get(k) for d in fedsubavg_all_s_shares]
        for k in set().union(*fedsubavg_all_s_shares)   # U1\U2
    }
    # Reconstruct and remover each mutual mask (pair of each dropped client in U1/U2 with each live client in U2)
    # First, reconstruct mutual mask for each dropped client in parallel
    # NOT U3!!!!!!! SHOULD BE U2, those clients who send masked input y.
    # Cannot directly fetch this dropped client's submodel shape, use global model shape to reconstruct!
    U2 = FEDSUBAVG_ROUND_STORAGE['U2']
    U2_fedsubavg_spk_dict = {k: FEDSUBAVG_SERVER_STORAGE[k]['spk'] for k in U2}
    # Step 1: Initialize multiprocessing.Pool()
    pool = mp.Pool(mp.cpu_count())
    # Step 2: apply
    res_ojects = [ pool.apply_async(server_side_sfsa_reconstruct_single_mutual_mask_indexed, args=(client_index_drop,\
                                  fedsubavg_s_shares_dict[client_index_drop],\
                                  FEDSUBAVG_ROUND_STORAGE['global_model_shape'], \
                                  FEDSUBAVG_SERVER_STORAGE[client_index_drop]['ids_info'], \
                                  U2, FEDSUBAVG_SERVER_STORAGE[client_index_drop]['mutual_mask_itemID_client_indices'],\
                                  FEDSUBAVG_SERVER_STORAGE[client_index_drop]['mutual_mask_cateID_client_indices'],\
                                  U2_fedsubavg_spk_dict, fedsubavg_security_para_dict, FEDSUBAVG_DHKE))\
                   for client_index_drop in FEDSUBAVG_ROUND_STORAGE['U1\U2'] ]
    fedsubavg_s_mask_dict_list = [ res.get() for res in res_ojects]
    # Step 3: close
    pool.close()

    # Second, remove mutual mask for each dropped client, parallel is hard
    for client_index_drop, fedsubavg_s_mask_dict in fedsubavg_s_mask_dict_list:
        server_side_sfsa_remove_single_mutual_mask(fedsubavg_s_mask_dict,\
                                 FEDSUBAVG_SERVER_STORAGE[client_index_drop]['ids_info'], FEDSUBAVG_ROUND_STORAGE)


def server_side_sfsa_round3(communication, clients, FEDSUBAVG_SERVER_STORAGE, FEDSUBAVG_ROUND_STORAGE,\
                            fedsubavg_security_para_dict, FEDSUBAVG_DHKE):
    """
    Receive mask related shares form live clients in U3, and perform unmasking.
    """
    temp_clients = [client for client in clients]
    for client in temp_clients:
        try:
            received_message = communication.get_np_array(client.connection_socket)
            assert client.ID == received_message['client_ID']
            FEDSUBAVG_SERVER_STORAGE[client.ID]['live_b_shares'] = received_message['live_b_shares']
            FEDSUBAVG_SERVER_STORAGE[client.ID]['drop_s_shares'] = received_message['drop_s_shares']
            print('Received mask related shares from client ' + str(client.ID) + ' in secure federated submodel averaging round 3')
            sys.stdout.flush()
        except Exception:
            clients.remove(client)
            print('Fallen client: ' + str(client.ID) + ' at ' + client.address[0] + ':' + str(client.address[1])
                  + ' in secure federated submodel averaging round 3 (receive)')
            sys.stdout.flush()
            client.connection_socket.close()
    # Record update-to-date set of client indices in round 3
    FEDSUBAVG_ROUND_STORAGE['U3'] = [client.ID for client in clients]
    # Record set of client indices live in round 2 but dropped in round 3
    # Those clients who submit public keys, encrypted secret shares, masked input, but do not submit mask related shares
    FEDSUBAVG_ROUND_STORAGE['U2\U3'] = list(set(FEDSUBAVG_ROUND_STORAGE['U2']) - set(FEDSUBAVG_ROUND_STORAGE['U3']))

    # Did not receive mask related shares from enough clients.
    assert len(FEDSUBAVG_ROUND_STORAGE['U3']) >= FEDSUBAVG_ROUND_STORAGE['t']

    start_time_1 = time.time()
    # Compute final output z_dict by removing self masks of live clients in U2 and mutual mask with dropped clients in U1/U2
    # First, remove self mask (optimized using multiprocessing in the reconstruction stage)
    server_side_sfsa_remove_self_mask_parallel(FEDSUBAVG_SERVER_STORAGE, FEDSUBAVG_ROUND_STORAGE, fedsubavg_security_para_dict)

    end_time_1 = time.time()
    write_csv(FEDSUBAVG_ROUND_STORAGE['ps_computation_time_path'], [FEDSUBAVG_ROUND_STORAGE['communication_round_num'], "sfsa_U3_sub_b_dicts_parallel", end_time_1 - start_time_1])

    start_time_2 = time.time()
    # Second, remove mutual mask (optimized using multiprocessing in the reconstruction stage)
    server_side_sfsa_remove_mutual_mask_parallel(FEDSUBAVG_SERVER_STORAGE, FEDSUBAVG_ROUND_STORAGE, fedsubavg_security_para_dict,\
                                        FEDSUBAVG_DHKE)

    end_time_2 = time.time()
    write_csv(FEDSUBAVG_ROUND_STORAGE['ps_computation_time_path'], [FEDSUBAVG_ROUND_STORAGE['communication_round_num'], "sfsa_U3_sub_s_dicts_parallel", end_time_2 - start_time_2])

    start_time_3 = time.time()
    # Finally, prepare final output
    z_dict = FEDSUBAVG_ROUND_STORAGE['z_dict']
    # Model parameters
    modulo_model_r = fedsubavg_security_para_dict['modulo_model_r']
    modulo_model_r_len = fedsubavg_security_para_dict['modulo_model_r_len']
    model_data_type = determine_data_type(modulo_model_r_len)
    gathered_weighted_delta_submodel = z_dict['gathered_weighted_delta_submodel']
    gathered_weighted_delta_submodel = [ weights % modulo_model_r for weights in gathered_weighted_delta_submodel]
    gathered_weighted_delta_submodel = [ weights.astype(model_data_type) for weights in gathered_weighted_delta_submodel]

    # Count numbers
    modulo_count_r = fedsubavg_security_para_dict['modulo_count_r']
    modulo_count_r_len = fedsubavg_security_para_dict['modulo_count_r_len']
    count_data_type = determine_data_type(modulo_count_r_len)
    gathered_userIDs_count = z_dict['gathered_userIDs_count']
    gathered_userIDs_count %= modulo_count_r
    gathered_userIDs_count = gathered_userIDs_count.astype(count_data_type)
    gathered_itemIDs_count = z_dict['gathered_itemIDs_count']
    gathered_itemIDs_count %= modulo_count_r
    gathered_itemIDs_count = gathered_itemIDs_count.astype(count_data_type)
    gathered_cateIDs_count = z_dict['gathered_cateIDs_count']
    gathered_cateIDs_count %= modulo_count_r
    gathered_cateIDs_count = gathered_cateIDs_count.astype(count_data_type)
    gathered_other_count = z_dict['gathered_other_count']

    end_time_3 = time.time()
    write_csv(FEDSUBAVG_ROUND_STORAGE['ps_computation_time_path'], [FEDSUBAVG_ROUND_STORAGE['communication_round_num'], "sfsa_U3_total",\
                                    end_time_1 - start_time_1 + end_time_2 - start_time_2 + end_time_3 - start_time_3])

    print("Secure federated submodel averaging round 3 costs %f s at ps"\
          %(end_time_1 - start_time_1 + end_time_2 - start_time_2 + end_time_3 - start_time_3))
    sys.stdout.flush()

    return gathered_weighted_delta_submodel, gathered_userIDs_count, gathered_itemIDs_count, gathered_cateIDs_count, \
           gathered_other_count


def server_side_secure_federated_submodel_averaging(communication, clients, dataset_info, ids_info_dict, global_model_shape,
                                                    fedsubavg_security_para_dict, round_num, ps_computation_time_path):
    """
    Main function for parameter server to dominate gathering submodel updates and count numbers through Secure Aggregation.
    Actually, implement the plaintext function "gather_weighted_submodel_updates" (ps_plain_functions.py) in a secure way.
    I.e., Obliviously sum all weighted submodel updates and corresponding count numbers (particularly itemIDs and cateIDs).
    """
    # This dictionary will contain the information that the server received from all clients.
    # It is keyed by client_index.
    FEDSUBAVG_SERVER_STORAGE = {}
    # This dictionary will mainly keep track of time, rounds, and set of (up-to-date) live clients
    FEDSUBAVG_ROUND_STORAGE = {}

    FEDSUBAVG_ROUND_STORAGE['communication_round_num'] = round_num
    FEDSUBAVG_ROUND_STORAGE['ps_computation_time_path'] = ps_computation_time_path

    # ID 14 - 2048-bit MODP group for Diffie-Hellman Key Exchange
    FEDSUBAVG_DHKE = DHKE(groupID=14)

    server_side_sfsa_round0(communication, clients, FEDSUBAVG_SERVER_STORAGE, FEDSUBAVG_ROUND_STORAGE)
    server_side_sfsa_round1(communication, clients, ids_info_dict, FEDSUBAVG_SERVER_STORAGE, FEDSUBAVG_ROUND_STORAGE)

    # Load some parameters into round storage before enter round 2
    FEDSUBAVG_ROUND_STORAGE['global_model_shape'] = global_model_shape
    FEDSUBAVG_ROUND_STORAGE['dataset_info'] = dataset_info
    server_side_sfsa_round2(communication, clients, FEDSUBAVG_SERVER_STORAGE, FEDSUBAVG_ROUND_STORAGE)

    gathered_weighted_delta_submodel, gathered_userIDs_count, gathered_itemIDs_count, gathered_cateIDs_count, \
           gathered_other_count = server_side_sfsa_round3(communication, clients, FEDSUBAVG_SERVER_STORAGE, \
                            FEDSUBAVG_ROUND_STORAGE, fedsubavg_security_para_dict, FEDSUBAVG_DHKE)
    return gathered_weighted_delta_submodel, gathered_userIDs_count, gathered_itemIDs_count, gathered_cateIDs_count,\
           gathered_other_count
