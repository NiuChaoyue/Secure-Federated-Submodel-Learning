import numpy
from data_iterator import DataIterator
import tensorflow as tf
from model import *
import time
import random
import sys
from utils import *
import csv

EMBEDDING_DIM = 18
#HIDDEN_SIZE = 18 * 2
#ATTENTION_SIZE = 18 * 2
best_auc = 0.0

def create_csv(path):
    with open(path, 'wb') as f:
        csv_writer = csv.writer(f)
        csv_head = ["round_num", "test_auc", "test_loss", "test_acc"]
        csv_writer.writerow(csv_head)


def write_csv(path, round_num, test_auc, test_loss, test_acc):
    with open(path, 'a+') as f:
        csv_writer = csv.writer(f)
        data_row = [round_num, test_auc, test_loss, test_acc]
        csv_writer.writerow(data_row)

#input: uid, mid, cat, mid_list, cat_list
def prepare_data(input, target, max_len=None):
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
    max_len_x = numpy.max(lengths_x)

    mid_his = numpy.zeros((n_samples, max_len_x)).astype('int64')
    cat_his = numpy.zeros((n_samples, max_len_x)).astype('int64')
    mid_mask = numpy.zeros((n_samples, max_len_x)).astype('float32')
    for idx, [s_x, s_y] in enumerate(zip(seqs_mid, seqs_cat)):
        mid_mask[idx, :lengths_x[idx]] = 1.
        mid_his[idx, :lengths_x[idx]] = s_x
        cat_his[idx, :lengths_x[idx]] = s_y

    uids = numpy.array([inp[0] for inp in input])
    mids = numpy.array([inp[1] for inp in input])
    cats = numpy.array([inp[2] for inp in input])

    return uids, mids, cats, mid_his, cat_his, mid_mask, numpy.array(target), numpy.array(lengths_x)


def eval(sess, test_data, model, model_path):

    loss_sum = 0.
    accuracy_sum = 0.
    nums = 0
    stored_arr = []
    for src, tgt in test_data:
        nums += 1
        uids, mids, cats, mid_his, cat_his, mid_mask, target, sl = prepare_data(src, tgt)
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
    global best_auc
    if best_auc < test_auc:
        best_auc = test_auc
        model.save(sess, model_path)
    return test_auc, loss_sum, accuracy_sum


def train(
        files,
        batch_size=1024,
        max_len=100,
        test_iter=30,
        seed=2,
        shuffle_each_epoch=False
):
    # create csv to record auc within "similar" communication round numbers
    round_num_simulated = 0
    auc_path = "./central_model_auc.csv"
    create_csv(auc_path)

    train_file, test_file, uid_voc, mid_voc, cat_voc = files[0], files[1], files[2], files[3], files[4]
    if shuffle_each_epoch:
        best_model_path = "best_model_SGD/ckpt_shuffle" + str(seed)
    else:
        best_model_path = "best_model_SGD/ckpt_noshuffle" + str(seed)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, batch_size, max_len, shuffle_each_epoch)
        test_data = DataIterator(test_file, uid_voc, mid_voc, cat_voc, batch_size, max_len)
        n_uid, n_mid, n_cat = train_data.get_n()

        model = Model_DIN(n_uid, n_mid, n_cat, EMBEDDING_DIM)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sys.stdout.flush()
        test_auc, test_loss, test_accuracy = eval(sess, test_data, model, best_model_path)
        print('Initial test_auc: %.4f ---- test_loss: %.4f ---- test_accuracy: %.4f' %
              (test_auc, test_loss, test_accuracy))
        write_csv(auc_path, round_num_simulated, test_auc, test_loss, test_accuracy)
        round_num_simulated += 1
        sys.stdout.flush()

        iter = 0
        lr = 1.0
        decay_rate = 0.999
        loss_sum = 0.0
        accuracy_sum = 0.0
        for epoch in range(50):
            start_time = time.time()
            for src, tgt in train_data:
                uids, mids, cats, mid_his, cat_his, mid_mask, target, sl = prepare_data(src, tgt, max_len)
                loss, acc = model.train(sess, [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, lr])
                loss_sum += loss
                accuracy_sum += acc
                iter += 1
                sys.stdout.flush()
                if (iter % test_iter) == 0:
                    print('Epoch: %d ----> iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f' %
                          (epoch, iter, loss_sum / test_iter, accuracy_sum / test_iter))
                    test_auc, test_loss, test_accuracy = eval(sess, test_data, model, best_model_path)
                    print('                         test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f'%
                          (test_auc, test_loss, test_accuracy))
                    write_csv(auc_path, round_num_simulated, test_auc, test_loss, test_accuracy)
                    round_num_simulated += 1
                    lr *= decay_rate
                    loss_sum = 0.0
                    accuracy_sum = 0.0
            '''
            if epoch == 1:
                lr *= 0.1
            '''
            print('Epoch %d finished: Used Time: %d s, Best auc: %.4f'%(epoch, time.time() - start_time, best_auc))
            print('')
            sys.stdout.flush()


def test(
        files,
        batch_size=1024,
        max_len=100,
        seed=2,
        shuffle_each_epoch=False,
):
    train_file, test_file, uid_voc, mid_voc, cat_voc = files[0], files[1], files[2], files[3], files[4]
    if shuffle_each_epoch:
        model_path = "best_model_SGD/ckpt_shuffle" + str(seed)
    else:
        model_path = "best_model_SGD/ckpt_noshuffle" + str(seed)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, batch_size, max_len)
        test_data = DataIterator(test_file, uid_voc, mid_voc, cat_voc, batch_size, max_len)
        n_uid, n_mid, n_cat = train_data.get_n()

        model = Model_DIN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        model.restore(sess, model_path)
        print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f ' %
              eval(sess, test_data, model, model_path))


if __name__ == '__main__':
    SEED = 3
    tf.set_random_seed(SEED)
    numpy.random.seed(SEED)
    random.seed(SEED)
    source_path = "../taobao_data_process/central_taobao_data/"
    train_file = source_path + "taobao_local_train"
    test_file = source_path + "taobao_local_test"
    uid_voc = source_path + "taobao_uid_voc.pkl"
    mid_voc = source_path + "taobao_mid_voc.pkl"
    cat_voc = source_path + "taobao_cat_voc.pkl"
    files = [train_file, test_file, uid_voc, mid_voc, cat_voc]
    train(files=files, seed=SEED, shuffle_each_epoch=False)
    print('finished!')
