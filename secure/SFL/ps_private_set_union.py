import os
import sys
import numpy as np
from PRNG import *
from diffie_hellman import DHKE
from shamir_secret_sharing import SecretSharer
import time
import multiprocessing as mp
import itertools
from general_functions import write_csv


def server_side_psu_round0(communication, clients, UNION_SERVER_STORAGE, UNION_ROUND_STORAGE):
    """
    Receive public keys from all live clients
    Then, Broadcast them.
    """
    temp_clients = [client for client in clients]
    for client in temp_clients:
        try:
            received_message = communication.get_np_array(client.connection_socket)
            assert client.ID == received_message['client_ID']
            UNION_SERVER_STORAGE.setdefault(client.ID, {})['spk'] = received_message['spk']
            UNION_SERVER_STORAGE.setdefault(client.ID, {})['cpk'] = received_message['cpk']
            print('Received public keys from client ' + str(client.ID) + ' in private set union round 0')
            sys.stdout.flush()
        except Exception:
            clients.remove(client)
            print('Fallen client: ' + str(client.ID) + ' at ' + client.address[0] + ':' + str(
                client.address[1]) + ' in private set union round 0 (receive)')
            sys.stdout.flush()
            client.connection_socket.close()
    # Record update-to-date set of client indices in round 0
    UNION_ROUND_STORAGE['U0'] = [client.ID for client in clients]
    UNION_ROUND_STORAGE['n'] = len(UNION_ROUND_STORAGE['U0'])
    UNION_ROUND_STORAGE['t'] = int(UNION_ROUND_STORAGE['n'] / 2) + 1

    # At least 2 clients to participate (n = 2, t = 2)
    # Did not receive public keys from enough clients. Abort!
    assert UNION_ROUND_STORAGE['n'] >= 2

    start_time = time.time()
    # Gather public keys of all live clients in U0
    union_pubkeys_dict = {}
    for client_index in UNION_ROUND_STORAGE['U0']:
        union_pubkeys_dict[client_index] = {}
        union_pubkeys_dict[client_index]['spk'] = UNION_SERVER_STORAGE[client_index]['spk']
        union_pubkeys_dict[client_index]['cpk'] = UNION_SERVER_STORAGE[client_index]['cpk']

    write_csv(UNION_ROUND_STORAGE['ps_computation_time_path'], [UNION_ROUND_STORAGE['communication_round_num'], "psu_U0", time.time() - start_time])

    # Return gathered public keys to each live client
    temp_clients = [client for client in clients]
    for client in temp_clients:
        try:
            communication.send_np_array(union_pubkeys_dict, client.connection_socket)
            print('Returned all public keys to client ' + str(client.ID) + ' in private set union round 0')
            sys.stdout.flush()
        except Exception:
            clients.remove(client)
            print('Fallen client: ' + str(client.ID) + ' at ' + client.address[0] + ':' + str(
                client.address[1]) + ' in private set union round 0 (return)')
            sys.stdout.flush()
            client.connection_socket.close()
    del temp_clients
    del union_pubkeys_dict


def server_side_psu_round1(communication, clients, UNION_SERVER_STORAGE, UNION_ROUND_STORAGE):
    """
    Receive encrypted secret shares from clients and forward them
    """
    temp_clients = [client for client in clients]
    for client in temp_clients:
        try:
            received_message = communication.get_np_array(client.connection_socket)
            assert client.ID == received_message['client_ID']
            UNION_SERVER_STORAGE[client.ID]['ss_ciphers_dict'] = received_message['ss_ciphers']
            print('Received encrypted secret shares from client ' + str(client.ID) + ' in private set union round 1')
            sys.stdout.flush()
        except Exception:
            clients.remove(client)
            print('Fallen client: ' + str(client.ID) + ' at ' + client.address[0] + ':' + str(
                client.address[1]) + ' in private set union round 1 (receive)')
            sys.stdout.flush()
            client.connection_socket.close()
    # Record update-to-date set of client indices in round 1
    UNION_ROUND_STORAGE['U1'] = [client.ID for client in clients]
    # Record set of client indices live in round 0 but drop in round 1
    # I.e., Those clients who submit public keys but do not submit encrypted secret shares
    UNION_ROUND_STORAGE['U0\U1'] = list( set(UNION_ROUND_STORAGE['U1']) - set(UNION_ROUND_STORAGE['U0']) )

    # Did not receive encrypted secret shares from enough clients. Abort!
    assert len(UNION_ROUND_STORAGE['U1']) >= UNION_ROUND_STORAGE['t']

    start_time = time.time()

    # Instead of having a dictionary of messages FROM a given client, we want to construct
    # a dictionary of messages TO a given client.
    ss_ciphers_dicts_FROM = {}
    for client_index in UNION_ROUND_STORAGE['U1']:
        ss_ciphers_dicts_FROM[client_index] = UNION_SERVER_STORAGE[client_index]['ss_ciphers_dict']

    # This is here that we reverse the "FROM key TO value" dict to a "FROM value TO key" dict
    # e.g.: {1: {2:a, 3:b, 4:c}, 3: {1:d,2:e,4:f}, 4: {1:g,2:h,3:i}}  -->  {1: {3:d, 4:g}, 3:{1:b, 4:i}, 4: {1:c,3:f}}
    ss_ciphers_dicts_TO = {}
    # forward message "enc_msg_to_client_index" from "from_client_index" to "to_client_index"
    for from_client_index, enc_msg_from_client_index in ss_ciphers_dicts_FROM.items():
        for to_client_index, enc_msg_to_client_index in enc_msg_from_client_index.items():
            ss_ciphers_dicts_TO.setdefault(to_client_index, {})[from_client_index] = enc_msg_to_client_index

    write_csv(UNION_ROUND_STORAGE['ps_computation_time_path'], [UNION_ROUND_STORAGE['communication_round_num'], "psu_U1", time.time() - start_time])

    # Forward encrypted secret shares to each live client
    temp_clients = [client for client in clients]
    for client in temp_clients:
        try:
            communication.send_np_array(ss_ciphers_dicts_TO[client.ID], client.connection_socket)
            print('Forwarded encrypted secret shares to client ' + str(client.ID) + ' in private set union round 1')
            sys.stdout.flush()
            time.sleep(5)  #create asynchronous environments to avoid clients competing for CPU resources
        except Exception:
            clients.remove(client)
            print('Fallen client: ' + str(client.ID) + ' at ' + client.address[0] + ':' + str(
                client.address[1]) + ' in private set union round 1 (return)')
            sys.stdout.flush()
            client.connection_socket.close()
    del temp_clients
    del ss_ciphers_dicts_FROM
    del ss_ciphers_dicts_TO


def server_side_psu_round2(communication, clients, UNION_SERVER_STORAGE, UNION_ROUND_STORAGE):
    """
    Receive masked input from all live clients, and
    Send back the up-to-date set of client indices.
    """
    temp_clients = [client for client in clients]
    for client in temp_clients:
        try:
            received_message = communication.get_np_array(client.connection_socket)
            assert client.ID == received_message['client_ID']
            UNION_SERVER_STORAGE[client.ID]['y'] = received_message['y']
            print('Received masked input from client ' + str(client.ID) + ' in private set union round 2')
            sys.stdout.flush()
        except Exception:
            clients.remove(client)
            print('Fallen client: ' + str(client.ID) + ' at ' + client.address[0] + ':' + str(
                client.address[1]) + ' in private set union round 2 (receive)')
            sys.stdout.flush()
            client.connection_socket.close()
    # Record update-to-date set of client indices in round 2
    UNION_ROUND_STORAGE['U2'] = [client.ID for client in clients]
    # Record set of client indices live in round 1 but dropped in round 2
    # Those clients who submit public keys, encrypted secret shares, but do not submit masked input
    UNION_ROUND_STORAGE['U1\U2'] = list( set(UNION_ROUND_STORAGE['U1']) - set(UNION_ROUND_STORAGE['U2']) )

    # Did not receive masked inputs from enough clients.
    assert len(UNION_ROUND_STORAGE['U2']) >= UNION_ROUND_STORAGE['t']

    # Forward statues of clients in round 2 to each live client
    round2_clients_status = {'U2': UNION_ROUND_STORAGE['U2'], 'U1\U2': UNION_ROUND_STORAGE['U1\U2']}
    temp_clients = [client for client in clients]
    for client in temp_clients:
        try:
            communication.send_np_array(round2_clients_status, client.connection_socket)
            print('Forwarded online and offline sets of clients to client ' + str(client.ID) + ' in private set union round 2')
            sys.stdout.flush()
        except Exception:
            clients.remove(client)
            print('Fallen client: ' + str(client.ID) + ' at ' + client.address[0] + ':' + str(
                client.address[1]) + ' in private set union round 2 (return)')
            sys.stdout.flush()
            client.connection_socket.close()
    del temp_clients


def server_side_psu_reconstruct_self_mask(UNION_SERVER_STORAGE, UNION_ROUND_STORAGE, union_security_para_dict):
    """
    Reconstruct the random seed b and thus self mask for each live client in U2 using the shares from U3.
    """
    union_all_b_shares = []
    for client_index in UNION_ROUND_STORAGE['U3']:
        union_all_b_shares.append(UNION_SERVER_STORAGE[client_index]['live_b_shares'])
    union_b_shares_dict = {
        k: [d.get(k) for d in union_all_b_shares]
        for k in set().union(*union_all_b_shares)     # U2
    }
    # Reconstruct and add up each self mask by PRNG expanding using the seed b
    # Load parameters for PRNG
    seed_len = union_security_para_dict['seed_len']
    security_strength = union_security_para_dict['security_strength']
    modulo_r_len = union_security_para_dict['modulo_r_len']
    item_count = union_security_para_dict['item_count']  # length of perturbed Bloom filter (x)
    # Store sum result
    union_b_mask_sum = np.zeros(item_count, dtype='int64')

    # NOT U3!!!!!!! SHOULD BE U2, those clients who send masked input y
    for client_index in UNION_ROUND_STORAGE['U2']:
        union_b = SecretSharer.recover_secret(union_b_shares_dict[client_index])
        union_b_entropy = int2bytes(union_b, seed_len / 8)
        union_DRBG_b = HMAC_DRBG(union_b_entropy, security_strength)
        union_b_mask = prng(union_DRBG_b, modulo_r_len, security_strength, item_count)
        union_b_mask_sum += union_b_mask
    return union_b_mask_sum


def psu_reconstruct_single_self_mask(union_b_shares, union_security_para_dict):
    """
    Assistant function for reconstructing self masks in parallel. (Single process)
    """
    # Load parameters for PRNG
    seed_len = union_security_para_dict['seed_len']
    security_strength = union_security_para_dict['security_strength']
    modulo_r_len = union_security_para_dict['modulo_r_len']
    item_count = union_security_para_dict['item_count']  # length of perturbed Bloom filter (x)

    union_b = SecretSharer.recover_secret(union_b_shares)
    union_b_entropy = int2bytes(union_b, seed_len / 8)
    union_DRBG_b = HMAC_DRBG(union_b_entropy, security_strength)
    union_b_mask = prng(union_DRBG_b, modulo_r_len, security_strength, item_count)
    union_b_mask = union_b_mask.astype('int64')
    return union_b_mask


def server_side_psu_reconstruct_self_mask_parallel(UNION_SERVER_STORAGE, UNION_ROUND_STORAGE, union_security_para_dict):
    """
    Reconstruct the random seed b and thus self mask for each live client in U2 using the shares from U3.
    (Optimized using multiprocessing)
    """
    union_all_b_shares = []
    for client_index in UNION_ROUND_STORAGE['U3']:
        union_all_b_shares.append(UNION_SERVER_STORAGE[client_index]['live_b_shares'])
    union_b_shares_dict = {
        k: [d.get(k) for d in union_all_b_shares]
        for k in set().union(*union_all_b_shares)     # U2
    }
    # Reconstruct and add up each self mask by PRNG expanding using the seed b in parallel
    # Step 1: Initialize multiprocessing.Pool()
    pool = mp.Pool(mp.cpu_count())
    # Step 2: apply
    # NOT U3!!!!!!! SHOULD BE U2, those clients who send masked input y
    res_objects = [ pool.apply_async(psu_reconstruct_single_self_mask, args=(union_b_shares_dict[client_index], \
                                            union_security_para_dict)) for client_index in UNION_ROUND_STORAGE['U2'] ]
    union_b_mask_list = [ res.get() for res in res_objects]
    # Step 3: close
    pool.close()
    union_b_mask_sum = sum(union_b_mask_list)

    del res_objects
    del union_b_mask_list
    return union_b_mask_sum


def server_side_psu_reconstruct_mutual_mask(UNION_SERVER_STORAGE, UNION_ROUND_STORAGE, union_security_para_dict, UNION_DHKE):
    """
    Reconstruct the secret key ssk and thus mutual mask for each dropped client in U1\U2 using the shares from U3.
    """
    item_count = union_security_para_dict['item_count']  # length of perturbed Bloom filter (x)
    # Store sum result
    union_s_mask_sum = np.zeros(item_count, dtype='int64')
    if len(UNION_ROUND_STORAGE['U1\U2']) == 0: # Deal with special case no client drops in U2
        return union_s_mask_sum

    union_all_s_shares = []
    for client_index in UNION_ROUND_STORAGE['U3']:
        union_all_s_shares.append(UNION_SERVER_STORAGE[client_index]['drop_s_shares'])
    union_s_shares_dict = {
        k: [d.get(k) for d in union_all_s_shares]
        for k in set().union(*union_all_s_shares)   # U1\U2
    }
    # Reconstruct and add up each mutual mask (pair of each dropped client in U1/U2 with each live client in U2)
    # by PRNG expanding using the secret key ssk
    # Load parameters for PRNG
    seed_len = union_security_para_dict['seed_len']
    security_strength = union_security_para_dict['security_strength']
    modulo_r_len = union_security_para_dict['modulo_r_len']

    for client_index_drop in UNION_ROUND_STORAGE['U1\U2']:
        union_drop_ssk = SecretSharer.recover_secret(union_s_shares_dict[client_index_drop])
        # NOT U3!!!!!!! SHOULD BE U2, those clients who send masked input y.
        for client_index_live in UNION_ROUND_STORAGE['U2']:
            union_live_spk = UNION_SERVER_STORAGE[client_index_live]['spk']
            # Derive seed for mutual mask, i.e., agreed key, (u, v via Diffie-Hellman Agreement)
            s_uv = UNION_DHKE.agree(union_drop_ssk, union_live_spk)
            s_uv_modulo = s_uv % (2 ** seed_len)
            s_uv_entropy = int2bytes(s_uv_modulo, seed_len / 8)
            union_DRGB_s = HMAC_DRBG(s_uv_entropy, security_strength)
            union_s_mask = prng(union_DRGB_s, modulo_r_len, security_strength, item_count)
            sgn = np.sign(client_index_live - client_index_drop)
            union_s_mask_sum += sgn * union_s_mask
    return union_s_mask_sum


def psu_reconstruct_single_mutual_mask(client_index_drop, client_index_live, union_s_shares_drop, union_live_spk, \
                                       UNION_DHKE, union_security_para_dict):
    """
    Assistant function for reconstructing mutual masks in parallel. (Single process)
    """
    # Load parameters for PRNG
    seed_len = union_security_para_dict['seed_len']
    security_strength = union_security_para_dict['security_strength']
    modulo_r_len = union_security_para_dict['modulo_r_len']
    item_count = union_security_para_dict['item_count']  # length of perturbed Bloom filter (x)

    union_drop_ssk = SecretSharer.recover_secret(union_s_shares_drop)

    # Derive seed for mutual mask, i.e., agreed key, (u, v via Diffie-Hellman Agreement)
    s_uv = UNION_DHKE.agree(union_drop_ssk, union_live_spk)
    s_uv_modulo = s_uv % (2 ** seed_len)
    s_uv_entropy = int2bytes(s_uv_modulo, seed_len / 8)
    union_DRGB_s = HMAC_DRBG(s_uv_entropy, security_strength)
    union_s_mask = prng(union_DRGB_s, modulo_r_len, security_strength, item_count)
    sgn = np.sign(client_index_live - client_index_drop)
    union_sgn_s_mask = sgn * union_s_mask
    union_sgn_s_mask = union_sgn_s_mask.astype('int64')
    return union_sgn_s_mask


def server_side_psu_reconstruct_mutual_mask_parallel(UNION_SERVER_STORAGE, UNION_ROUND_STORAGE, union_security_para_dict, UNION_DHKE):
    """
    Reconstruct the secret key ssk and thus mutual mask for each dropped client in U1\U2 using the shares from U3.
    (Optimized using multiprocessing)
    """
    item_count = union_security_para_dict['item_count']  # length of perturbed Bloom filter (x)
    # Store sum result
    union_s_mask_sum = np.zeros(item_count, dtype='int64')
    if len(UNION_ROUND_STORAGE['U1\U2']) == 0: # Deal with special case no client drops in U2
        return union_s_mask_sum

    union_all_s_shares = []
    for client_index in UNION_ROUND_STORAGE['U3']:
        union_all_s_shares.append(UNION_SERVER_STORAGE[client_index]['drop_s_shares'])
    union_s_shares_dict = {
        k: [d.get(k) for d in union_all_s_shares]
        for k in set().union(*union_all_s_shares)   # U1\U2
    }
    # Reconstruct and add up each mutual mask (pair of each dropped client in U1/U2 with each live client in U2)
    # by PRNG expanding using the secret key ssk in parallel
    # Step 1: Initialize multiprocessing.Pool()
    pool = mp.Pool(mp.cpu_count())
    # Step 2: apply
    # NOT U3!!!!!!! SHOULD BE U2, those clients who send masked input y
    clients_drop_live_pairs = list( itertools.product(UNION_ROUND_STORAGE['U1\U2'], UNION_ROUND_STORAGE['U2']) )
    res_objects = [ pool.apply_async(psu_reconstruct_single_mutual_mask, args=(client_index_drop, client_index_live,\
                               union_s_shares_dict[client_index_drop], UNION_SERVER_STORAGE[client_index_live]['spk'], \
                                       UNION_DHKE, union_security_para_dict)) \
                                       for client_index_drop, client_index_live in clients_drop_live_pairs ]


    union_sgn_s_mask_list = [res.get() for res in res_objects]
    # Step 3: close
    pool.close()
    union_s_mask_sum = sum(union_sgn_s_mask_list)

    del res_objects
    del union_sgn_s_mask_list
    return union_s_mask_sum


def server_side_psu_round3(communication, clients, UNION_SERVER_STORAGE, UNION_ROUND_STORAGE, union_security_para_dict, UNION_DHKE):
    """
    Receive mask related shares form live clients, and perform unmasking.
    """
    temp_clients = [client for client in clients]
    for client in temp_clients:
        try:
            received_message = communication.get_np_array(client.connection_socket)
            assert client.ID == received_message['client_ID']
            UNION_SERVER_STORAGE[client.ID]['live_b_shares'] = received_message['live_b_shares']
            UNION_SERVER_STORAGE[client.ID]['drop_s_shares'] = received_message['drop_s_shares']
            print('Received mask related shares from client ' + str(client.ID) + ' in private set union round 3')
            sys.stdout.flush()
        except Exception:
            clients.remove(client)
            print('Fallen client: ' + str(client.ID) + ' at ' + client.address[0] + ':' + str(client.address[1])
                  + ' in private set union round 3 (receive)')
            sys.stdout.flush()
            client.connection_socket.close()
    # Record update-to-date set of client indices in round 3
    UNION_ROUND_STORAGE['U3'] = [client.ID for client in clients]
    # Record set of client indices live in round 2 but dropped in round 3
    # Those clients who submit public keys, encrypted secret shares, masked input, but do not submit mask related shares
    UNION_ROUND_STORAGE['U2\U3'] = list(set(UNION_ROUND_STORAGE['U2']) - set(UNION_ROUND_STORAGE['U3']))

    # Did not receive mask related shares from enough clients.
    assert len(UNION_ROUND_STORAGE['U3']) >= UNION_ROUND_STORAGE['t']

    start_time_1 = time.time()
    # Compute final output z by removing self masks of live clients in U2 and mutual mask with dropped clients in U1/U2
    item_count = union_security_para_dict['item_count']
    modulo_r = union_security_para_dict['modulo_r']
    z = np.zeros(item_count, dtype='int64')

    # First add each 'y'
    for client_index in UNION_ROUND_STORAGE['U2']:
        z += UNION_SERVER_STORAGE[client_index]['y']
    end_time_1 = time.time()

    write_csv(UNION_ROUND_STORAGE['ps_computation_time_path'], [UNION_ROUND_STORAGE['communication_round_num'], "psu_U3_add_y", end_time_1 - start_time_1])

    start_time_2 = time.time()
    # Second, reconstruct and then remove self mask of U2, using asynchronous multiprocessing
    union_b_mask_sum = server_side_psu_reconstruct_self_mask_parallel(UNION_SERVER_STORAGE, UNION_ROUND_STORAGE,\
                                                             union_security_para_dict)
    z -= union_b_mask_sum

    end_time_2 = time.time()

    write_csv(UNION_ROUND_STORAGE['ps_computation_time_path'], [UNION_ROUND_STORAGE['communication_round_num'], "psu_U3_sub_b_parallel", end_time_2 - start_time_2])

    start_time_3 = time.time()
    # Third, reconstruct and then remove mutual mask of U1/U2, using asynchronous multiprocessing
    union_s_mask_sum = server_side_psu_reconstruct_mutual_mask_parallel(UNION_SERVER_STORAGE, UNION_ROUND_STORAGE, \
                                                               union_security_para_dict, UNION_DHKE)
    z -= union_s_mask_sum
    z %= modulo_r

    end_time_3 = time.time()

    write_csv(UNION_ROUND_STORAGE['ps_computation_time_path'], [UNION_ROUND_STORAGE['communication_round_num'], "psu_U3_sub_s_parallel", end_time_3 - start_time_3])
    write_csv(UNION_ROUND_STORAGE['ps_computation_time_path'], [UNION_ROUND_STORAGE['communication_round_num'], "psu_U3_total", \
                                    end_time_1 - start_time_1 + end_time_2 - start_time_2 + end_time_3 - start_time_3])

    print("Private set union round 3 in parallel costs %f s at ps"%(end_time_1 - start_time_1 + end_time_2 - start_time_2 + end_time_3 - start_time_3))
    sys.stdout.flush()

    return z


def send_back_union_psu(communication, clients, union_message):
    """Sending back union of user IDs, item IDs, and cate IDs"""
    temp_clients = [client for client in clients]
    for client in temp_clients:
        try:
            communication.send_np_array(union_message, client.connection_socket)
            print('Sending back unions (via private set union) to client ' + str(client.ID))
            sys.stdout.flush()
        except Exception:
            clients.remove(client)
            print('Fallen client: ' + str(client.ID) + ' at ' + client.address[0] + ':' + str(
                client.address[1]) + ' in the private set union sending back stage')
            sys.stdout.flush()
            client.connection_socket.close()
    del temp_clients


def server_side_private_set_union(communication, clients, union_security_para_dict, round_num, ps_computation_time_path):
    """
    Main function for parameter server to dominate private set union through perturbed Bloom filter and Secure Aggregation.
    """
    # This dictionary will contain the information that the server received from all clients.
    # It is keyed by client_index.
    UNION_SERVER_STORAGE = {}
    # This dictionary will mainly keep track of time, rounds, and set of (up-to-date) live clients
    UNION_ROUND_STORAGE = {}

    UNION_ROUND_STORAGE['communication_round_num'] = round_num
    UNION_ROUND_STORAGE['ps_computation_time_path'] = ps_computation_time_path

    # ID 14 - 2048-bit MODP group for Diffie-Hellman Key Exchange
    UNION_DHKE = DHKE(groupID=14)

    server_side_psu_round0(communication, clients, UNION_SERVER_STORAGE, UNION_ROUND_STORAGE)
    server_side_psu_round1(communication, clients, UNION_SERVER_STORAGE, UNION_ROUND_STORAGE)
    server_side_psu_round2(communication, clients, UNION_SERVER_STORAGE, UNION_ROUND_STORAGE)
    bf_perturbed_sum = server_side_psu_round3(communication, clients, UNION_SERVER_STORAGE, UNION_ROUND_STORAGE, \
                                              union_security_para_dict, UNION_DHKE)
    real_itemIDs_union = (np.nonzero(bf_perturbed_sum)[0]).tolist()
    # send back union
    send_back_union_psu(communication, clients, real_itemIDs_union)
