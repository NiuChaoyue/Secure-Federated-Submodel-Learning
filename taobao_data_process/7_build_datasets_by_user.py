# -*- coding: UTF-8 -*-
"""
build datasets: one Taobao user one dataset
Filtering out data set sizes >= 32
Randomly Shuffle to simulate randomly chosen clients in each communication round
"""

import os
import numpy as np
import cPickle as pkl
import random

def chunk_by_user(train_seq):
    userset = set()
    train_seq_split = []
    for line in train_seq:
        line_split = line.strip("\n").split("\t")
        train_seq_split.append(line_split)
        userset.add(eval(line_split[1]))
    print("#users: %d" % len(userset))
    print("#samples: %d" % len(train_seq_split))
    print("#samples per user: %d" % (len(train_seq_split) / len(userset)))
    user_num = len(userset)

    start_index = []
    end_index = []
    start_index.append(0)
    current_user = train_seq_split[0][1]
    for index in range(len(train_seq_split)):
        if current_user != train_seq_split[index][1]:
            current_user = train_seq_split[index][1]
            end_index.append(index)
            start_index.append(index)
    end_index.append(len(train_seq_split))
    if len(start_index) != len(end_index) or len(start_index) != user_num:
        print("Oh, No! Mismatched User Numbers.")
    out_train = []
    for start, end in zip(start_index, end_index):
        if end - start >= 32:
            out_train.append(train_seq[start:end])
    return out_train

print('7. build_datasets_by_user is running')

with open('taobao_local_train_remap', 'rb') as f:
    train_set = f.readlines()

train_sets = chunk_by_user(train_set)

SAVE_DIR = './taobao_datasets/'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

random.shuffle(train_sets)
print("Actual #users (data size >= 32): %d"%(len(train_sets)))

for i in range(len(train_sets)):
    filename = './taobao_datasets/user_%s' % (str(i + 1))
    with open(filename, 'wb') as f:
        for line in train_sets[i]:
            print >> f, line.strip('\n')
print('Finished!')
