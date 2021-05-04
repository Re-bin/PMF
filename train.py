# -*- coding: utf-8 -*-
"""
Created on Tue May  4 16:37:54 2021

@author: Rbin
"""



import numpy as np
from sklearn.model_selection import train_test_split
from data import *
from evaluation import *
from pmf import *

if __name__ == '__main__':
    N, M, data_list, _ = load_data('datasets/ml-100k/u.data')
    train_list, test_list = train_test_split(data_list,test_size=0.2)
    pmf = PMF(train_list, test_list, N, M)
    record_list = pmf.train()
    
    # draw loss curve
    plot(x, record_list[:, 0], color='g',linewidth=3)
    plt.title('loss curve')
    plt.xlabel('Iterations')
    plt.ylabel('loss')
    plt.show()
    
    # draw recall curve
    plot(x, record_list[:, 1], color='r',linewidth=3)
    plt.title('recall curve')
    plt.xlabel('recall')
    plt.ylabel('loss')
    plt.show()
    
    # draw precision curve
    plot(x, record_list[:, 2], color='b',linewidth=3)
    plt.title('precision curve')
    plt.xlabel('Iterations')
    plt.ylabel('precision')
    plt.show()
    