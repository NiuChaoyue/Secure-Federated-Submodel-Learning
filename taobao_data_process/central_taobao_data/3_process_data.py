import sys
import random
import time
import pickle


def process_meta(file):
    with open(file, 'rb') as f:
        review_df = pickle.load(f)
        review_df = review_df[['auction_id', 'cate_id']]
    fo = open("taobao-item-info", "w")
    for index, item_id in enumerate(review_df['auction_id']):
        cat = review_df["cate_id"][index]
        print>>fo, item_id + "\t" + cat


def process_reviews(file):
    with open(file, 'rb') as f:
        review_df = pickle.load(f)
        review_df = review_df[['user_id', 'auction_id', 'cate_id', 'event_id', 'event_time']]
    fo = open("taobao-reviews-info", "w")
    for index, user_id in enumerate(review_df['user_id']):
        item_id = review_df["auction_id"][index]
        event_id = review_df["event_id"][index]
        time = review_df["event_time"][index]
        print>>fo, user_id + "\t" + item_id + "\t" + event_id + "\t" + str(time)


def manual_join():
    f_rev = open("taobao-reviews-info", "r")
    user_map = {}
    item_list = []
    for line in f_rev:
        line = line.strip()
        items = line.split("\t")
        if items[0] not in user_map:
            user_map[items[0]] = []
        user_map[items[0]].append(("\t".join(items), str(items[-1])))
        item_list.append(items[1])
    f_meta = open("taobao-item-info", "r")
    meta_map = {}
    for line in f_meta:
        arr = line.strip().split("\t")
        if arr[0] not in meta_map:
            meta_map[arr[0]] = arr[1]
    fo = open("taobao-jointed-new", "w")
    for key in user_map:
        sorted_user_bh = sorted(user_map[key], key=lambda x: x[1])
        for line, t in sorted_user_bh:
            items = line.split("\t")
            if items[2] == '2101':
                if items[1] in meta_map:
                    # including event_time + tag + user_id + item_id + cate_id
                    print>> fo, items[3] + '\t' + "1" + "\t" + items[0] + "\t" + items[1] + "\t" + meta_map[items[1]]
                else:
                    print>> fo, items[3] + '\t' + "1" + "\t" + items[0] + "\t" + items[1] + "\t" + "default_cat"
            else:
                if items[1] in meta_map:
                    print>> fo, items[3] + '\t' + "0" + "\t" + items[0] + "\t" + items[1] + "\t" + meta_map[items[1]]
                else:
                    print>> fo, items[3] + '\t' + "0" + "\t" + items[0] + "\t" + items[1] + "\t" + "default_cat"


print('3_process_data is running')
process_meta('./taobao_reviews.pkl')
process_reviews('./taobao_reviews.pkl')
manual_join()
print('finished!')
