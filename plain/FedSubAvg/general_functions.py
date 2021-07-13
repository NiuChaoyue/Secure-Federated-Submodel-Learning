import tensorflow as tf
import numpy as np


def create_placeholders():
    """Creates the placeholders that we will use to inject the weights into the graph"""
    placeholders = []
    for var in tf.trainable_variables():
        placeholders.append(tf.placeholder_with_default(var, var.shape, \
                                                             name="%s/%s" % ("FedAvg", var.op.name)))
    return placeholders


def assign_vars(local_vars, placeholders):
    """Utility to refresh local variables.

    Args:
        local_vars: List of local variables.

    Returns:
        refresh_ops: The ops to assign value of global vars to local vars.
    """
    reassign_ops = []
    for var, fvar in zip(local_vars, placeholders):
        reassign_ops.append(tf.assign(var, fvar))
    return tf.group(*(reassign_ops))


def prepare_data(input, target, max_len=100):
    # x: a list of sentences
    lengths_x = [len(s[4]) for s in input]
    seqs_mid = [inp[3] for inp in input]
    seqs_cat = [inp[4] for inp in input]

    if max_len is not None:
        new_seqs_mid = []
        new_seqs_cat = []
        new_lengths_x = []
        for l_x, inp in zip(lengths_x, input):
            if l_x > max_len:
                new_seqs_mid.append(inp[3][l_x - max_len:])
                new_seqs_cat.append(inp[4][l_x - max_len:])
                new_lengths_x.append(max_len)
            else:
                new_seqs_mid.append(inp[3])
                new_seqs_cat.append(inp[4])
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_mid = new_seqs_mid
        seqs_cat = new_seqs_cat

        if len(lengths_x) < 1:
            return None, None, None, None

    n_samples = len(seqs_mid)
    max_len_x = np.max(lengths_x)

    mid_his = np.zeros((n_samples, max_len_x)).astype('int64')
    cat_his = np.zeros((n_samples, max_len_x)).astype('int64')
    mid_mask = np.zeros((n_samples, max_len_x)).astype('float32')
    for idx, [s_x, s_y] in enumerate(zip(seqs_mid, seqs_cat)):
        mid_mask[idx, :lengths_x[idx]] = 1.
        mid_his[idx, :lengths_x[idx]] = s_x
        cat_his[idx, :lengths_x[idx]] = s_y

    uids = np.array([inp[0] for inp in input])
    mids = np.array([inp[1] for inp in input])
    cats = np.array([inp[2] for inp in input])

    return uids, mids, cats, mid_his, cat_his, mid_mask, np.array(target), np.array(lengths_x)


def din_attention(query, facts, mask, mode='SUM', softmax_stag=1):
    mask = tf.equal(mask, tf.ones_like(mask))   # [B, L]
    queries = tf.tile(query, [1, tf.shape(facts)[1]])   # [B, L*2*H]
    queries = tf.reshape(queries, tf.shape(facts))      # [B, L, 2*H]
    din_all = tf.concat([queries, facts, queries-facts, queries*facts], axis=-1)    # [B, L, 4*2*H]
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att')
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att')
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att')    # [B, L, 1]
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])  # [B, 1, L]
    scores = d_layer_3_all  # [B, 1, L]

    key_masks = tf.expand_dims(mask, 1)     # [B, 1, L]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks, scores, paddings)  # [B, 1, L]

    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, L]

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [B, 1, L]*[B, L, 2*H] = [B, 1, 2*H]
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])   # [B, L]
        output = facts * tf.expand_dims(scores, -1)     # [B, L, 2*H]*[B, L, 1]
        output = tf.reshape(output, tf.shape(facts))    # [B, L, 2*H]
    return output
