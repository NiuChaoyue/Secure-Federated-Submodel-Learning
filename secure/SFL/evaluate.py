# -*- coding: UTF-8 -*-

import tensorflow as tf
import general_functions as gn_fn
from math import exp


def sigmoid(logits):
    return [1/(1+exp(-i)) for i in logits]


def calc_auc(raw_arr):
    """Summary

    Args:
        raw_arr (TYPE): Description

    Returns:
        TYPE: Description
    """

    arr = sorted(raw_arr, key=lambda d: d[0], reverse=True)
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


def eval_and_save(variables_pack_for_eval_and_save, round_num, sess):
    loss_sum = 0.
    accuracy_sum = 0.
    nums = 0
    stored_arr = []
    model = variables_pack_for_eval_and_save['model']
    for src, tgt in variables_pack_for_eval_and_save['test_set']:
        nums += 1
        uids, mids, cats, mid_his, cat_his, mid_mask, target, sl = gn_fn.prepare_data(src, tgt)
        prob, loss, acc = model.calculate(sess, [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl])
        loss_sum += loss
        accuracy_sum += acc
        prob_1 = prob[:, 0].tolist()
        target_1 = target[:, 0].tolist()
        for p, t in zip(prob_1, target_1):
            stored_arr.append([p, t])
    test_auc = calc_auc(stored_arr)
    accuracy_sum = accuracy_sum / nums
    loss_sum = loss_sum / nums
    if variables_pack_for_eval_and_save['best_auc'] < test_auc:
        variables_pack_for_eval_and_save['best_auc'] = test_auc
        variables_pack_for_eval_and_save['best_round'] = round_num
        variables_pack_for_eval_and_save['saver'].save(sess, variables_pack_for_eval_and_save['CHECKPOINT_DIR'])
    return test_auc, loss_sum, accuracy_sum
