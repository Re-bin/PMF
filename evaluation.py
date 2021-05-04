# -*- coding: utf-8 -*-
"""
Created on Tue May  4 15:13:21 2021

@author: Rbin
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pylab import *
import matplotlib
import matplotlib.pyplot as plt

def get_topn(pred_mat, train_mat, n=10):
    unrated_items = pred_mat * (train_mat == 0)
    idx = np.argsort(-unrated_items)    # only sort products that users have not scored(rated)
    return idx[:, :n]

def recall_precision(topn, test_mat):
    n, m = test_mat.shape
    hits, total_pred, total_true = 0., 0., 0.
    for u in range(n): # u for users
        hits += len([i for i in topn[u, :] if test_mat[u, i] > 0]) # nums of hits
        size_pred = len(topn[u, :]) # nums of pred (n)
        size_true = np.sum(test_mat[u, :] > 0, axis=0) # nums of total
        total_pred += size_pred
        total_true += size_true
    
    recall = hits/total_true
    precision = hits/total_pred
    
    return recall, precision

def evaluation(pred_mat, train_mat, test_mat):
    topn = get_topn(pred_mat, train_mat, n = 10)
    recall, precision = recall_precision(topn, test_mat)
    return recall, precision