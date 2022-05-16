# -*- coding: utf-8 -*-
#version 2

#1.16更改：
#1.生成随机图，所有edge-probility 均为0.8
#(这样做似乎不是很可行，因为我们结点编号不是从0~n,如果构建完再删边的话有额外的时间复杂度
import networkx as nx
import numpy as np

import Base
import Environment as Envi
import copy
import os
from Environment import  delt, epsi, sigma, alpha2

S = 1

alpha = 20

class Global_server:  # 最开始每个local_server中的user数目是已知的
    def __init__(self, n, d, T):  # 这里n指的是user的总数量，假设最开始是已知的,每个cluster的user的数量已知，存储在numlist中
        self.l_server_list = []
        self.usernum = n
        self.rounds = T
        self.d = d
        self.regret = np.zeros(self.rounds)
        self.reward = np.zeros(self.rounds)
        self.best_reward = np.zeros(self.rounds)
        self.users = dict()
        for i in range(self.usernum):
            self.users[i] = Base.User(self.d, i, self.rounds)

    def recommend(self, user_index, items):
        user = self.users[user_index]
        V_t, b_t, T_t = user.get_info()
        # print('b_t:',b_t)
        gamma_t = Envi.gamma(T_t, self.d, alpha, sigma)
        lambda_t = gamma_t * 2
        M_t = np.eye(self.d) * np.float_(lambda_t) + V_t
        # 此处S是一个常数，取了1，用于计算β
        # 参数调整
        beta_t = Envi.beta(sigma, alpha, gamma_t, S, self.d, T_t)
        print("beta LinUCB :", beta_t)
        Minv = np.linalg.inv(M_t)
        theta = np.dot(Minv, b_t)
        # 将global server上算得的M，b，β传给当前的local server
        r_item_index = np.argmax(np.dot(items, theta) + beta_t * (np.matmul(items, Minv) * items).sum(axis=1))
        # np.argmax(np.dot(items, theta) + self._beta(N, t) * (np.matmul(items, Sinv) * items).sum(axis=1))
        return r_item_index

    def run(self, envir, T, number):
        theta_exp = dict()
        theta_one_user = list()
        y_list = list()
        x_list = list()
        result = dict()
        for i in range(1, T+1):
            if i % 5000 == 0:
                print(i)
            user_all = envir.generate_users()
            user_index = user_all[0]
            # the context set
            items = envir.get_items()
            r_item_index = self.recommend(user_index, items)
            x = items[r_item_index]
            x_list.append(x)
            self.reward[i - 1], y, self.best_reward[i - 1], ksi_noise, B_noise = envir.feedback_Local(items= items,i= user_index,k= r_item_index,d= self.d)
            y_list.append(y)
            # print("ksi_noise in ",i,"round is",ksi_noise)
            # print("B_noise in ", i, "round is", B_noise)
            self.users[user_index].store_info(x, y, i - 1, self.reward[i - 1],self.best_reward[i -1], ksi_noise[0], B_noise)
            self.regret[i - 1] = self.best_reward[i - 1] - self.reward[i - 1]


            if i % 100000 ==  0:
                for i in range(self.usernum):
                    result[i] = self.users[i].theta
                #12_18_100_user_alpha_4.5是30user,alpha=1.5
                npzname = "LinUCB" + "no_"+str(number)+"_1_26"+ str(self.usernum) + "_user_" + str(i)
                np.savez(npzname, nu=self.usernum, d=self.d, T=i, G_server_regret=self.regret,
                    theta_exp= result, theta_theo=envir.theta, reward= self.reward)


        return self.regret,result, self.reward, x_list, y_list







































