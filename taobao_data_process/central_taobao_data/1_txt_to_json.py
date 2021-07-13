# -*- coding: UTF-8 -*-

import pickle
import json
import sys
import pandas as pd


def to_json(file):
    dict_list = []
    user_item_dict = dict()
    with open(file, 'r') as fin:
        for line in fin:
            line_dict = dict()
            line_list = line.split()
            line_dict['user_id'] = line_list[0]
            line_dict['auction_id'] = line_list[1]
            line_dict['event_id'] = line_list[2]
            line_dict['event_time'] = line_list[5]+line_list[4]
            line_dict['cate_id'] = line_list[6]
            dict_list.append(line_dict)
            if line_list[2] == '2101':
                key = (line_list[0], line_list[1])
                user_item_dict[key] = 1

    length = len(dict_list)
    print(length)
    for index in range(length-1, -1, -1):
        item = dict_list[index]
        print(index)
        if item['event_id'] == '2201':
            key = (item['user_id'], item['auction_id'])
            if user_item_dict.get(key) is not None:
                dict_list.pop(index)
    print(len(dict_list))

    with open('./taobao_data.json', 'a') as fout:
        i = 0
        for line_dict in dict_list:
            fout.write(json.dumps(line_dict)+'\n')
            i += 1
            if i % 1000 == 0:
                print(str(i)+'/'+str(len(dict_list)))


print('1_txt_to_json is running')
to_json(sys.argv[1])