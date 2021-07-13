import cPickle as pkl


def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key, value) in d.items())


def load_dict(filename):
    try:
        with open(filename, 'rb') as f:
            return unicode_to_utf8(json.load(f))
    except:
        with open(filename, 'rb') as f:
            return unicode_to_utf8(pkl.load(f))

source_path = "./central_taobao_data/"
f_train = open(source_path + "taobao_local_train", "r")
f_test = open(source_path + "taobao_local_test", "r")
uid_dict = load_dict(source_path + "taobao_uid_voc.pkl")
mid_dict = load_dict(source_path + "taobao_mid_voc.pkl")
cat_dict = load_dict(source_path + "taobao_cat_voc.pkl")
new_f_train = open("taobao_local_train_remap", "w")
new_f_test = open("taobao_local_test_remap", "w")


def convert_to_new_id(fi, fo, item_cate_dict):
    for line in fi:
        arr = line.strip("\n").split("\t")
        clk = arr[0]
        uid = arr[1]
        mid = arr[2]
        cat = arr[3]
        mid_list = arr[4]
        cat_list = arr[5]

        new_uid = str(uid_dict[uid]) if uid in uid_dict else '0'
        new_mid = str(mid_dict[mid]) if mid in mid_dict else '0'
        new_cat = str(cat_dict[cat]) if cat in cat_dict else '0'
        item_cate_dict[new_mid] = new_cat

        if len(mid_list) > 0:
            new_mid_list = mid_list.split("")
            new_cat_list = cat_list.split("")
            for i, m in enumerate(new_mid_list):
                new_mid_list[i] = str(mid_dict[m]) if m in mid_dict else '0'
            for i, c in enumerate(new_cat_list):
                new_cat_list[i] = str(cat_dict[c]) if c in cat_dict else '0'
                item_cate_dict[new_mid_list[i]] = new_cat_list[i]
            mid_list = "".join(new_mid_list)
            cat_list = "".join(new_cat_list)
            print >> fo, clk + "\t" + new_uid + "\t" + new_mid + "\t" + new_cat + "\t" + mid_list + "\t" + cat_list


print('6. remap_to_new_id is running')
item_cate_dict = dict()   #str -> str
convert_to_new_id(f_train, new_f_train, item_cate_dict)
convert_to_new_id(f_test, new_f_test, item_cate_dict)

user_number = len(uid_dict)
item_number = len(mid_dict)
cate_number = len(cat_dict)
print(user_number, item_number, cate_number)
with open("taobao_item_cate_dict.pkl", "w") as f:
    pkl.dump(item_cate_dict, f)

with open("taobao_user_item_cate_count.pkl", "w") as f:
    pkl.dump((user_number, item_number, cate_number), f)

f_train.close()
f_test.close()
new_f_train.close()
new_f_test.close()
print "finished!"