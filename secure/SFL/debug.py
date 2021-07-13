import client_functions as cl_fn
import os

PERMANENT_ANSWERS_DIR = './permanent_answers/'
if not os.path.exists(PERMANENT_ANSWERS_DIR):
    os.makedirs(PERMANENT_ANSWERS_DIR)

client_index = 0
real_itemIDs = range(0, 4)
real_cateIDs = range(0, 4)
real_itemIDs_union = range(0, 1004)
real_cateIDs_union = range(0, 1004)
prob1 = 0.75
prob2 = 0.25
prob3 = 0.75
prob4 = 0.25
perturbed_itemIDs, perturbed_cateIDs = cl_fn.generate_perturbed_index_set(client_index, real_itemIDs, real_itemIDs_union,
                                            real_cateIDs, real_cateIDs_union, prob1, prob2, prob3, prob4)

print(perturbed_itemIDs, len(perturbed_itemIDs))
print(perturbed_cateIDs, len(perturbed_cateIDs))

prob5 = prob1 * (prob3 - prob4) + prob4
prob6 = prob2 * (prob3 - prob4) + prob4
print(prob5, prob6)
print(4 * prob5 + 1000 * prob6)