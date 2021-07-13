import os
import sys
import numpy as np
import time
from PRNG import *
from diffie_hellman import DHKE
from shamir_secret_sharing import SecretSharer
from AES_CBC import AESCipher
from general_functions import determine_data_type
from general_functions import write_csv


def represent_set_as_perturbed_bloom_filter(original_set, union_security_para_dict):
    """
    Use bit array (with length item_count), namely Bloom filter, to represent real index set
    Then, replace those bit 1's with a random number (0 to modulo_r - 1).
    We call this perturbed bloom filter
    """
    # prepare security parameters for PRNG
    item_count = union_security_para_dict['item_count']
    seed_len = union_security_para_dict['seed_len']
    security_strength = union_security_para_dict['security_strength']
    modulo_r_len = union_security_para_dict['modulo_r_len']
    data_type = determine_data_type(modulo_r_len)
    bf_perturbed = np.zeros(item_count, dtype=data_type)
    # generate random numbers to replace bit '1's in bloom filter
    entropy = os.urandom(seed_len/8)
    DRBG = HMAC_DRBG(entropy, security_strength)
    mask = prng(DRBG, modulo_r_len, security_strength, len(original_set))
    bf_perturbed[original_set] = mask
    return bf_perturbed


def client_side_psu_round0(communication, client_socket, UNION_SELF_STORAGE, UNION_OTHERS_STORAGE, UNION_DHKE):
    """
    Generate secret and public Keys, and Send public keys to server.
    Also, receive public keys of other clients from server.
    """
    start_time_1 = time.time()
    # Generate the 2 pair of Diffie-Hellman keys
    # "s" to generate the seed for the shared mask, and "c" to generate the shared symmetric encryption key
    my_ssk, my_spk = UNION_DHKE.generate_keys()
    my_csk, my_cpk = UNION_DHKE.generate_keys()

    # Store the previously generated keys
    UNION_SELF_STORAGE['my_ssk'] = my_ssk
    UNION_SELF_STORAGE['my_spk'] = my_spk
    UNION_SELF_STORAGE['my_csk'] = my_csk
    UNION_SELF_STORAGE['my_cpk'] = my_cpk
    union_client_pubkeys = {'client_ID': UNION_SELF_STORAGE['my_index'],
                            'spk': my_spk,
                            'cpk': my_cpk}
    end_time_1 = time.time()

    communication.send_np_array(union_client_pubkeys, client_socket)
    print('Client %d sent public keys to server in private set union' % UNION_SELF_STORAGE['my_index'])
    sys.stdout.flush()

    union_pubkeys_dict = communication.get_np_array(client_socket)
    print('Received public keys of all clients from server.')

    start_time_2 = time.time()
    for client_index, pubkeys in union_pubkeys_dict.items():
        # Does not need to store my own keys (already in UNION_SELF_STORAGE)
        if client_index != UNION_SELF_STORAGE['my_index']:
            UNION_OTHERS_STORAGE.setdefault(client_index, {})['spk'] = pubkeys['spk']
            UNION_OTHERS_STORAGE.setdefault(client_index, {})['cpk'] = pubkeys['cpk']
    # Record number of live clients (including client self) and the required threshold
    UNION_SELF_STORAGE['n'] = len(union_pubkeys_dict)
    UNION_SELF_STORAGE['t'] = int(UNION_SELF_STORAGE['n'] / 2) + 1

    end_time_2 = time.time()
    write_csv(UNION_SELF_STORAGE['client_computation_time_path'], [UNION_SELF_STORAGE['communication_round_number'], \
                                        "psu_U0", end_time_1 - start_time_1 + end_time_2 - start_time_2])


def client_side_psu_round1(communication, client_socket, UNION_SELF_STORAGE, UNION_OTHERS_STORAGE, \
                           union_security_para_dict, UNION_DHKE):
    """
    Generate and send encrypted secret shares for PRNG seed and ssk
    """
    start_time_1 = time.time()
    # Generate seed for PRNG
    seed_len = union_security_para_dict['seed_len']
    union_b_entropy = os.urandom(seed_len/8)  #bytes
    union_b = bytes2int(union_b_entropy)

    t = UNION_SELF_STORAGE['t']
    n = UNION_SELF_STORAGE['n']
    # Generate t-out-of-n shares for PRNG's seed b
    union_shares_b = SecretSharer.split_secret(union_b, t, n)
    # Generate t-out-of-n shares for client's ssk
    union_shares_my_ssk = SecretSharer.split_secret(UNION_SELF_STORAGE['my_ssk'], t, n)

    # Store random seed, and secret shares into self dictionary
    UNION_SELF_STORAGE['b_entropy'] = union_b_entropy
    '''
    UNION_SELF_STORAGE['b'] = union_b
    UNION_SELF_STORAGE['shares_b'] = union_shares_b
    UNION_SELF_STORAGE['shares_my_ssk'] = union_shares_my_ssk
    '''

    # Store my share of b in isolation
    # No need to store my share of my ssk, since I am alive to myself!
    union_my_share_b = union_shares_b[0]
    union_shares_b = list( set(union_shares_b) - set([union_my_share_b]))
    UNION_SELF_STORAGE['my_share_b'] = union_my_share_b

    union_ss_ciphers_dict = {}
    for idx, client_index in enumerate(UNION_OTHERS_STORAGE.keys()):
        # Derive symmetric encryption key "agreed" with other client (with client_index) (via Diffie-Hellman Agreement)
        sym_enc_key = UNION_DHKE.agree(UNION_SELF_STORAGE['my_csk'], UNION_OTHERS_STORAGE[client_index]['cpk'])
        # Send ciphertext to other client (with client_index), where PS works as a mediation
        msg = str(UNION_SELF_STORAGE['my_index']) + ' || ' + str(client_index) + ' || ' + str(union_shares_b[idx]) \
              + ' || ' + str(union_shares_my_ssk[idx])
        # Encrypt with AES_CBC
        enc_msg = AESCipher(str(sym_enc_key)).encrypt(msg)
        union_ss_ciphers_dict[client_index] = enc_msg

        UNION_OTHERS_STORAGE[client_index]['sym_enc_key'] = sym_enc_key
        '''
        UNION_OTHERS_STORAGE[client_index]['msg'] = msg
        UNION_OTHERS_STORAGE[client_index]['enc_msg'] = enc_msg
        '''
    end_time_1 = time.time()

    # send encrypted shares to the server
    union_ss_ciphers_send_message = {'client_ID': UNION_SELF_STORAGE['my_index'],
                                     'ss_ciphers': union_ss_ciphers_dict}
    communication.send_np_array(union_ss_ciphers_send_message, client_socket)
    print('Client %d sent encrypted secret shares to server in private set union' % UNION_SELF_STORAGE['my_index'])
    sys.stdout.flush()

    # receive other clients' encrypted shares to me from the server
    ss_ciphers_dict_received = communication.get_np_array(client_socket)
    print("Received other clients' encrypted secret shares from server.")
    sys.stdout.flush()

    start_time_2 = time.time()

    for client_index, enc_msg in ss_ciphers_dict_received.items():
        # Decrypt the encrypted message and parse it
        sym_enc_key = UNION_OTHERS_STORAGE[client_index]['sym_enc_key']
        msg = AESCipher(str(sym_enc_key)).decrypt(enc_msg)
        msg_parts = msg.split(' || ')
        # Sanity check
        from_client_index = int(msg_parts[0])
        my_index = int(msg_parts[1])
        assert from_client_index == client_index and my_index == UNION_SELF_STORAGE['my_index']
        # Store secret shares of other clients
        UNION_OTHERS_STORAGE[client_index]['share_b'] = msg_parts[2]
        UNION_OTHERS_STORAGE[client_index]['share_ssk'] = msg_parts[3]
    # clients in U1 (except myself) for mutual masks
    UNION_SELF_STORAGE['mutual_mask_client_indices'] = ss_ciphers_dict_received.keys()

    end_time_2 = time.time()
    write_csv(UNION_SELF_STORAGE['client_computation_time_path'], [UNION_SELF_STORAGE['communication_round_number'], \
                                        "psu_U1", end_time_1 - start_time_1 + end_time_2 - start_time_2])


def client_side_psu_round2(communication, client_socket, UNION_SELF_STORAGE, UNION_OTHERS_STORAGE, \
                           union_security_para_dict, UNION_DHKE):
    """
    Doubly mask the input vector, i.e., perturbed Bloom filter, and send it to server.
    """
    start_time = time.time()

    # Load parameters for PRNG
    seed_len = union_security_para_dict['seed_len']
    security_strength = union_security_para_dict['security_strength']
    modulo_r_len = union_security_para_dict['modulo_r_len']
    modulo_r = union_security_para_dict['modulo_r']
    item_count = union_security_para_dict['item_count']  # length of perturbed Bloom filter (x)

    # Generate self mask
    union_b_entropy = UNION_SELF_STORAGE['b_entropy']
    union_DRBG_b = HMAC_DRBG(union_b_entropy, security_strength)
    union_b_mask = prng(union_DRBG_b, modulo_r_len, security_strength, item_count)
    '''
    UNION_SELF_STORAGE['b_mask'] = union_b_mask
    '''

    # Generate mutual mask
    union_mutual_mask = np.zeros(item_count, dtype='int64')
    for client_index in UNION_SELF_STORAGE['mutual_mask_client_indices']:   #U1 except myself
        # Derive seed for mutual mask, i.e., agreed key, with other client (u, v via Diffie-Hellman Agreement)
        s_uv = UNION_DHKE.agree(UNION_SELF_STORAGE['my_ssk'], UNION_OTHERS_STORAGE[client_index]['spk'])
        s_uv_modulo = s_uv % (2**seed_len)
        s_uv_entropy = int2bytes(s_uv_modulo, seed_len / 8)
        union_DRGB_s = HMAC_DRBG(s_uv_entropy, security_strength)
        union_s_mask = prng(union_DRGB_s, modulo_r_len, security_strength, item_count)
        # Minus the mask when other client is with larger index,
        # or add the mask when other client is with smaller index
        sgn = np.sign(UNION_SELF_STORAGE['my_index'] - client_index)
        union_mutual_mask += sgn * union_s_mask
        '''
        # Store mutual mask related info
        UNION_OTHERS_STORAGE[client_index]['s'] = s_uv
        UNION_OTHERS_STORAGE[client_index]['s_mask'] = union_s_mask
        '''
    # Add self and mutual masks
    # Here is the final output "y" to send to server
    y = (UNION_SELF_STORAGE['x'] + union_b_mask + union_mutual_mask) % modulo_r
    data_type = determine_data_type(modulo_r_len)
    y = y.astype(data_type)

    '''
    UNION_SELF_STORAGE['y'] = y
    '''

    write_csv(UNION_SELF_STORAGE['client_computation_time_path'], [UNION_SELF_STORAGE['communication_round_number'], \
                                        "psu_U2_y", time.time() - start_time])

    # Send masked input to the server
    union_client_y = {'client_ID': UNION_SELF_STORAGE['my_index'], 'y': y}
    communication.send_np_array(union_client_y, client_socket)
    print('Client %d sent masked perturbed Bloom filter to server in private set union' % UNION_SELF_STORAGE['my_index'])
    sys.stdout.flush()

    # Receive Online and Offline sets of clients in round 2
    round2_clients_status = communication.get_np_array(client_socket)
    print("Received clients' status in round 2 from server")
    sys.stdout.flush()
    UNION_SELF_STORAGE['U2'] = round2_clients_status['U2']
    UNION_SELF_STORAGE['U1\U2'] = round2_clients_status['U1\U2']


def client_side_psu_round3(communication, client_socket, UNION_SELF_STORAGE, UNION_OTHERS_STORAGE):
    """
    Send shares of b (for self mask) and ssk (for mutual mask) for live and dropped clients in U2 and U1\U2, respectively.
    """
    start_time = time.time()

    # U2: Except myself
    union_u2_live = list(set(UNION_SELF_STORAGE['U2']) - set([UNION_SELF_STORAGE['my_index']]))
    # U1/U2
    union_u2_drop = UNION_SELF_STORAGE['U1\U2']

    # Shares of self mask's seed for live clients
    union_live_b_shares = {}
    for client_index_live in union_u2_live:
        union_live_b_shares[client_index_live] = UNION_OTHERS_STORAGE[client_index_live]['share_b']
    union_live_b_shares[UNION_SELF_STORAGE['my_index']] = UNION_SELF_STORAGE['my_share_b']

    # Shares of mutual mask's secret key for dropped clients
    union_drop_s_shares = {}
    for client_index_drop in union_u2_drop:
        union_drop_s_shares[client_index_drop] = UNION_OTHERS_STORAGE[client_index_drop]['share_ssk']

    write_csv(UNION_SELF_STORAGE['client_computation_time_path'], [UNION_SELF_STORAGE['communication_round_number'], \
                                                                   "psu_U3", time.time() - start_time])

    # Send shares to the server
    union_shares = {'client_ID': UNION_SELF_STORAGE['my_index'],
                    'live_b_shares': union_live_b_shares,
                    'drop_s_shares': union_drop_s_shares}
    communication.send_np_array(union_shares, client_socket)
    print('Client %d sent secret shares of live and dropped clients in round 2 to server in private set union'\
          % UNION_SELF_STORAGE['my_index'])
    sys.stdout.flush()


def client_side_private_set_union(communication, client_socket, client_index, real_itemIDs, union_security_para_dict,
                                  union_u2_drop_flag, union_u3_drop_flag, round_num, client_computation_time_path):
    """
    Main function for client to join Private Set Union (PSU) through perturbed Bloom filter and Secure Aggregation
    """
    # This dictionary will contain all the values generated by this client herself
    UNION_SELF_STORAGE = {}
    # This dictionary will contain all the values about the OTHER clients. It is keyed by client_index
    UNION_OTHERS_STORAGE = {}
    UNION_SELF_STORAGE['my_index'] = client_index
    UNION_SELF_STORAGE['communication_round_number'] = round_num
    UNION_SELF_STORAGE['client_computation_time_path'] = client_computation_time_path

    # ID 14 - 2048-bit MODP group for Diffie-Hellman Key Exchange
    UNION_DHKE = DHKE(groupID=14)
    client_side_psu_round0(communication, client_socket, UNION_SELF_STORAGE, UNION_OTHERS_STORAGE, UNION_DHKE)
    client_side_psu_round1(communication, client_socket, UNION_SELF_STORAGE, UNION_OTHERS_STORAGE, \
                           union_security_para_dict, UNION_DHKE)

    if union_u2_drop_flag:  # drop from round 2
        client_socket.close()
        print("Client %d drops out from Private Set Union U2 in this communication round \n" % client_index)
        print('-----------------------------------------------------------------')
        print('')
        print('')
        sys.stdout.flush()
        return []

    # Represent real_itemIDs as a perturbed Bloom filter
    start_time = time.time()
    real_itemIDs_pbf = represent_set_as_perturbed_bloom_filter(real_itemIDs, union_security_para_dict)

    write_csv(client_computation_time_path, [round_num, "psu_generated_perturbed_bf", time.time() - start_time])

    UNION_SELF_STORAGE['x'] = real_itemIDs_pbf
    client_side_psu_round2(communication, client_socket, UNION_SELF_STORAGE, UNION_OTHERS_STORAGE, \
                           union_security_para_dict, UNION_DHKE)

    if union_u3_drop_flag:   # drop from round 3
        client_socket.close()
        print("Client %d drops out from Private Set Union U3 in this communication round \n" % client_index)
        print('-----------------------------------------------------------------')
        print('')
        print('')
        return []

    client_side_psu_round3(communication, client_socket, UNION_SELF_STORAGE, UNION_OTHERS_STORAGE)

    # receive union
    real_itemIDs_union = communication.get_np_array(client_socket)
    print('Received union of real item ids (via Private Set Union).')
    sys.stdout.flush()

    del UNION_SELF_STORAGE
    del UNION_OTHERS_STORAGE

    return real_itemIDs_union
