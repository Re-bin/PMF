# -*- coding: utf-8 -*-
"""
Created on Tue May  4 14:58:48 2021

@author: Rbin
"""

import numpy as np
from sklearn.model_selection import train_test_split

def load_data(file_dir):
    user_id_dict, rated_item_ids_dict = {}, {}
    N, M, u_idx, i_idx = 0, 0, 0, 0
    data = []
    f = open(file_dir)
    for line in f.readlines():
        if '::' in line:
            u, i, r, _ = line.split('::')
        else:
            u, i, r, _ = line.split()
        if int(u) not in user_id_dict:
            user_id_dict[int(u)] = u_idx
            u_idx += 1
        if int(i) not in rated_item_ids_dict:
            rated_item_ids_dict[int(i)] = i_idx
            i_idx += 1
        data.append([user_id_dict[int(u)], rated_item_ids_dict[int(i)], float(r)])
    f.close()
    N = u_idx
    M = i_idx
    return N, M, data, rated_item_ids_dict

def sequence_to_mat(sequence, N, M):
    records_array = np.array(sequence)
    mat = np.zeros([N, M])
    row = records_array[:, 0].astype(int)
    col = records_array[:, 1].astype(int)
    values = records_array[:, 2].astype(np.float32)
    mat[row, col] = values
    return mat
    

if __name__ == '__main__':
    N, M, data_list, _ = load_data('datasets/ml-100k/u.data')
    train_list, test_list = train_test_split(data_list,test_size=0.2)

