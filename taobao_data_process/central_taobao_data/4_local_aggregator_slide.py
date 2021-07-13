import sys
import hashlib
import random

fin = open("./taobao-jointed-new", "r")
ftrain = open("taobao_local_train", "w")
ftest = open("taobao_local_test", "w")
print('local_aggregator_slide is running')


def index_date(ds):
    if ds < '2019061600:00:00':
        return 1
    elif ds < '2019061700:00:00':
        return 2
    elif ds < '2019061800:00:00':
        return 3
    elif ds < '2019061900:00:00':
        return 4
    elif ds < '2019062000:00:00':
        return 5
    elif ds < '2019062100:00:00':
        return 6
    elif ds < '2019062200:00:00':
        return 7
    elif ds < '2019062300:00:00':
        return 8
    elif ds < '2019062400:00:00':
        return 9
    elif ds < '2019062500:00:00':
        return 10
    elif ds < '2019062600:00:00':
        return 11
    elif ds < '2019062700:00:00':
        return 12
    elif ds < '2019062800:00:00':
        return 13
    elif ds < '2019062900:00:00':
        return 14
    elif ds < '2019063000:00:00':
        return 15
    elif ds < '2019070100:00:00':
        return 16
    elif ds < '2019070200:00:00':
        return 17
    elif ds < '2019070300:00:00':
        return 18
    elif ds < '2019070400:00:00':
        return 19
    elif ds < '2019070500:00:00':
        return 20
    elif ds < '2019070600:00:00':
        return 21
    elif ds < '2019070700:00:00':
        return 22
    elif ds < '2019070800:00:00':
        return 23
    elif ds < '2019070900:00:00':
        return 24
    elif ds < '2019071000:00:00':
        return 25
    elif ds < '2019071100:00:00':
        return 26
    elif ds < '2019071200:00:00':
        return 27
    elif ds < '2019071300:00:00':
        return 28
    elif ds < '2019071400:00:00':
        return 29
    else:
        return 30

last_user = ""
item_id_list = None
cate_id_list = None
line_idx = 0
len_train = 0
len_test = 0
for line in fin:
    items = line.strip().split("\t")
    ds = items[0]
    day_index = index_date(ds)
    clk = int(items[1])
    user_id = items[2]
    item_id = items[3]
    cate_id = items[4]

    #split train/test by date
    #use previous 14 days to predict next 1 day
    if day_index >= 15 and day_index <= 29:
        fo = ftrain
    if day_index == 30:
        fo = ftest

    if user_id != last_user:
        item_id_list = []
        cate_id_list = []
    else:
        if day_index >= 15:
            history_clk_num = 0
            cate_str = ""
            item_str = ""
            for c1 in cate_id_list:
                if day_index - c1[0] <= 14 and day_index - c1[0] > 0:
                    cate_str += c1[1] + ""
                    history_clk_num += 1
            for mid in item_id_list:
                if day_index - mid[0] <= 14 and day_index - mid[0] > 0:
                    item_str += mid[1] + ""
            if len(cate_str) > 0:
                cate_str = cate_str[:-1]
            if len(item_str) > 0:
                item_str = item_str[:-1]
            if history_clk_num >= 1:    # the length of user click behavior in previous 14 days
                print >> fo, items[1] + "\t" + user_id + "\t" + item_id + "\t" + cate_id + "\t" + item_str + "\t" + cate_str
                if day_index >= 15 and day_index <= 29:
                    len_train += 1
                if day_index == 30:
                    len_test += 1
                line_idx += 1
    last_user = user_id
    if clk:
        item_id_list.append((day_index, item_id))
        cate_id_list.append((day_index, cate_id))

print('All size: %d; Train set size: %d; Test set size: %d' % (line_idx, len_train, len_test))
print('Train/Test:%.4f' % (len_train * 1.0 / len_test))

print "finished"
