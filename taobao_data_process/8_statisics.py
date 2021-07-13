import os

total_users_num = 46336
chosen_clients_num = 100

def extract_real_index_set(client_index, max_len=100):
    """Extract client's real index set from her training data."""
    f_client_train = open('./taobao_datasets/user_%d' % client_index, 'r')
    real_train_set_size = 0
    userID = 0
    real_itemIDs = []
    for line in f_client_train:
        ss = line.strip("\n").split("\t")
        if line == "" or ss == '':
            break
        uid = int(ss[1])
        mid = int(ss[2])
        mid_list = []
        for fea in ss[4].split(""):
            mid_list.append(int(fea))
        if len(mid_list) > max_len:
            mid_list = mid_list[len(mid_list) - max_len:]
        userID = uid
        real_itemIDs.append(mid)
        real_itemIDs += mid_list
        real_train_set_size += 1
    userID = [userID]
    real_itemIDs = list( set(real_itemIDs) )
    real_itemIDs.sort()
    return userID, real_itemIDs, real_train_set_size

real_itemIDs_len_list = []
total_train_set_size = 0
for client_index in range(1, total_users_num + 1):
    userID, real_itemIDs, real_train_set_size = extract_real_index_set(client_index)
    real_itemIDs_len_list.append(len(real_itemIDs))
    total_train_set_size += real_train_set_size
real_itemIDs_len_list.sort(reverse=True)
top_ones = real_itemIDs_len_list[0:chosen_clients_num]
print("Sum of Top %d clients' item ids numbers: %d" % (chosen_clients_num, sum(top_ones)))
print("Average number of one client's item ids: %f" % (sum(real_itemIDs_len_list) * 1.0 / total_users_num))
print("Average number of one client's train set: %f" % (total_train_set_size * 1.0 / total_users_num))
