# -*- coding: UTF-8 -*-
import numpy
import json
import cPickle as pkl
import random
import gzip
import shuffle


def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key,value) in d.items())


def load_dict(filename):
    try:
        with open(filename, 'rb') as f:
            return unicode_to_utf8(json.load(f))
    except:
        with open(filename, 'rb') as f:
            return unicode_to_utf8(pkl.load(f))


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


class DataIterator:

    def __init__(self, source,
                 batch_size=128,
                 max_len=100,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=False,
                 max_batch_size=20,
                 min_len=None):
        if shuffle_each_epoch:
            self.source_orig = source
            self.source = shuffle.main(self.source_orig, temporary=True)
        else:
            self.source = fopen(source, 'r')
        self.source_dicts = []
        self.batch_size = batch_size
        self.max_len = max_len
        self.min_len = min_len
        self.skip_empty = skip_empty

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.k = batch_size * max_batch_size

        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        if self.shuffle:
            self.source = shuffle.main(self.source_orig, temporary=True)
        else:
            self.source.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        if len(self.source_buffer) == 0:
            for k_ in range(self.k):
                ss = self.source.readline()
                if ss == "" or ss.strip("\n").split("\t") == '':
                    break
                self.source_buffer.append(ss.strip("\n").split("\t"))

            # sort by  history behavior length
            if self.sort_by_length:
                his_length = numpy.array([len(s[4].split("")) for s in self.source_buffer])
                tidx = his_length.argsort()

                _sbuf = [self.source_buffer[i] for i in tidx]
                self.source_buffer = _sbuf
            else:
                self.source_buffer.reverse()

        if len(self.source_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break

                uid = int(ss[1])
                mid = int(ss[2])
                cat = int(ss[3])
                tmp = []
                for fea in ss[4].split(""):
                    tmp.append(int(fea))
                mid_list = tmp  # item list

                tmp1 = []
                for fea in ss[5].split(""):
                    tmp1.append(int(fea))
                cat_list = tmp1     # cate list

                # read from source file and map to word index
                # if len(mid_list) > self.max_len:
                #    continue
                if self.min_len is not None:
                    if len(mid_list) <= self.min_len:
                        continue
                if self.skip_empty and (not mid_list):
                    continue

                source.append([uid, mid, cat, mid_list, cat_list])
                target.append([float(ss[0]), 1-float(ss[0])])

                if len(source) >= self.batch_size or len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        # all sentence pairs in maxibatch filtered out because of length
        if len(source) == 0 or len(target) == 0:
            source, target = self.next()

        return source, target


