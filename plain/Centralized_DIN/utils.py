import tensorflow as tf


def calc_auc(raw_arr):
    """Summary

    Args:
        raw_arr (TYPE): Description

    Returns:
        TYPE: Description
    """

    arr = sorted(raw_arr, key=lambda d:d[0], reverse=True)
    pos, neg = 0., 0.
    for record in arr:
        if record[1] == 1.:
            pos += 1
        else:
            neg += 1

    fp, tp = 0., 0.
    xy_arr = []
    for record in arr:
        if record[1] == 1.:
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp/neg, tp/pos])

    auc = 0.
    prev_x = 0.
    prev_y = 0.
    for x, y in xy_arr:
        if x != prev_x:
            auc += ((x - prev_x) * (y + prev_y) / 2.)
            prev_x = x
            prev_y = y

    return auc


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
