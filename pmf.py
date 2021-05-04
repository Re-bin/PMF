# -*- coding: utf-8 -*-
"""
Created on Tue May  4 16:16:01 2021

@author: Rbin
"""

import numpy as np
import random
from data import *
from evaluation import *

class PMF():
    def __init__(self,
                 train_list,                # train data
                 test_list,                 # test data
                 N,                         # the number of user
                 M,                         # the number of item
                 K=10,                      # the number of latent factor
                 learning_rate = 0.001,     # learning rate
                 lamda_regularizer = 0.1,   # regularization parameter
                 max_iteration = 50         # the max iteration
                 ):
        self.train_list = train_list
        self.test_list = test_list
        self.N = N
        self.M = M
        self.K = K
        self.learning_rate = learning_rate
        self.lamda_regularizer = lamda_regularizer
        self.max_iteration = max_iteration
    
    def train(self):
        self.P = np.random.normal(0, 0.1, (self.N, self.K))
        self.Q = np.random.normal(0, 0.1, (self.M, self.K))
        
        train_mat = sequence_to_mat(sequence = self.train_list, N = self.N, M = self.M)
        test_mat = sequence_to_mat(sequence = self.test_list, N = self.N, M = self.M)
        
        record_list = []
        for step in range(self.max_iteration):
            loss = 0.0
            for data in self.train_list:
                u, i, r = data
                self.P[u], self.Q[i], ls = self.update(self.P[u], self.Q[i], r)
                loss += ls
            pred_mat = self.prediction()
            recall, precision = evaluation(pred_mat, train_mat, test_mat)
            record_list.append(np.array([loss, recall, precision]))
            
            if(step % 10 == 0):
                print('step:%d \n loss:%.4f, recall:%.4f, precision:%.4f'%(step, loss, recall, precision))
        
        return np.array(record_list)
    
    def update(self, p, q, r):
        error = r - np.dot(p, q.T)
        p = p + self.learning_rate * (error * q - self.lamda_regularizer * p)
        q = q + self.learning_rate * (error * p - self.lamda_regularizer * q)
        loss = 0.5 * (error**2 + self.lamda_regularizer * (np.square(p).sum() + np.square(q).sum()))
        return p, q, loss
    
    def prediction(self):
        rating_list = []
        for u in range(self.N): # u for users
            u_rating = np.sum(self.P[u, :] * self.Q, axis = 1)
            rating_list.append(u_rating)
        pred_mat = np.array(rating_list)
        return pred_mat
