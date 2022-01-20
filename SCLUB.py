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
from Environment import alpha, delt, epsi, sigma, alpha2

S = 1
phase_cardinality = 2
alpha_p = np.sqrt(2)
alpha = 1
#class Global_server

#Global_server G_server



class Local_server:
    def __init__(self, nl, d, begin_num, T, edge_probability=0.8):
        self.nl = nl
        self.d = d
        self.rounds = T  # the number of all rounds
        user_index_list = list(range(begin_num,begin_num + nl))
        self.clusters = {0: Base.sclub_Cluster(b=np.zeros(d), T=0, V=np.zeros((d,d)), users_begin = begin_num, d=d, user_num = nl, t= self.rounds, rewards= np.zeros(self.rounds), T_phase= 0,theta_phase= np.zeros(self.d), best_rewards= np.zeros(self.rounds))}  # users的初始化方法直接抄了老师里面的，不知道对不对
        self.T_phase = 0
        self.theta = np.zeros(self.d)
        self.init_each_stage()
        self.cluster_inds = dict()
        self.begin_num = begin_num
        for i in range(begin_num, begin_num + nl):
            self.cluster_inds[i] = 0    # 每个user所属于的cluster的index, key:user_index ,value:cluster_index
        #self.cluster_inds = np.zeros(nl)  # 每个user所属于的cluster的index,这种方法貌似不可行，得用字典，因为user index是从global的维度而言的
        self.num_clusters = np.zeros(self.rounds,np.int64)  # 每一轮中cluster的数量，总共记录round T 次
        self.num_clusters[0] = 1
        self.V = np.zeros((d,d)) # 对于server而言V，b，T是否是必要的？
        self.b = np.zeros(d)
        self.T = 0


    def init_each_stage(self):
        for i in self.clusters:
            cluster = self.clusters[i]
            cluster.checks = {j: False for j in cluster.users}
            cluster.checked = False
            cluster.phase_update()


    def cluster_aver_freq(self, c, t):
        if len(self.clusters[c].users) == 0:
            print('c:',c)
            return 0
        return self.clusters[c].T / (len(self.clusters[c].users) * t)


    def locate_user_index(self, user_index):
        # 如果这种方法不可行的话，只能强行遍历
        # 确定user属于哪个local cluster
        l_cluster_index = self.cluster_inds[user_index]
        #确定user属于哪个global cluster
        return l_cluster_index

    def recommend(self, l_cluster_index, items):
        cluster = self.clusters[l_cluster_index]
        V_t, b_t , T_t = cluster.get_info()
        #print('b_t:',b_t)
        gamma_t = Envi.gamma(T_t, self.d, alpha, sigma)
        lambda_t = gamma_t * 2
        M_t = np.eye(self.d) * np.float_(lambda_t) + V_t
        #此处S是一个常数，取了1，用于计算β
        #参数调整
        beta_t = Envi.beta(sigma, alpha, gamma_t, S, self.d, T_t)
        Minv = np.linalg.inv(M_t)
        theta = np.dot(Minv,b_t)
        # 将global server上算得的M，b，β传给当前的local server
        r_item_index = np.argmax(np.dot(items,theta) + beta_t * (np.matmul(items, Minv) * items).sum(axis=1))
        #np.argmax(np.dot(items, theta) + self._beta(N, t) * (np.matmul(items, Sinv) * items).sum(axis=1))
        return r_item_index

    #这里t是目前的round
    def if_split(self, user_index1, cluster, t):
        T1 = cluster.users[user_index1].T
        T2 = cluster.T_phase
        fact_T1 = np.sqrt((1 + np.log(1 + T1)) / (1 + T1))
        fact_T2 = np.sqrt((1 + np.log(1 + T2)) / (1 + T2))
        fact_t = np.sqrt((1 + np.log(1 + t)) / (1 + t))
        theta1 = cluster.users[user_index1].theta
        theta2 = cluster.theta_phase
        if np.linalg.norm(theta1 - theta2) > alpha * (fact_T1 + fact_T2):
            return True

        p1 = T1 / t
        for user_index2 in cluster.users:
            if user_index2 == user_index1:
                continue
            p2 = cluster.users[user_index2].T / t
            if np.abs(p1 - p2) > alpha_p * 2 * fact_t:
                return True

        return False



    #t 是目前的round
    def if_merge(self, c1, c2, t):
        fact_t = np.sqrt((1 + np.log(1 + t)) / (1 + t))
        T1 = self.clusters[c1].T
        T2 = self.clusters[c2].T
        fact_T1 = np.sqrt((1 + np.log(1 + T1)) / (1 + T1))
        fact_T2 = np.sqrt((1 + np.log(1 + T2)) / (1 + T2))
        fact_t = np.sqrt((1 + np.log(1 + t)) / (1 + t))
        theta1 = self.clusters[c1].theta
        theta2 = self.clusters[c2].theta
        p1 = self.cluster_aver_freq(c1, t)
        p2 = self.cluster_aver_freq(c2, t)

        #参数回头调整
        if np.linalg.norm(theta1 - theta2) >= (alpha / 2) * (fact_T1 + fact_T2):
            return False
        if np.abs(p1 - p2) >= alpha_p * fact_t:
            return False

        return True

    def find_available_index(self):
        cmax = max(self.clusters)
        for c1 in range(cmax + 1):
            if c1 not in self.clusters:
                return c1
        return cmax + 1


    def update(self, user_index, t): # 这里的user_index是从0到n的一个自然数
        update_cluster = False
        c = self.cluster_inds[user_index] # 找到被更新的user所属于的local cluster
        cluster = self.clusters[c]
        cluster.update_check(user_index)
        now_user = cluster.users[user_index]

        if self.if_split(user_index, cluster, t):
            cnew = self.find_available_index()
            tmp_cluster = Base.sclub_Cluster(b= now_user.b, T= now_user.T, V= now_user.V, users_begin= user_index, d=self.d, user_num= 1,
                                             t=self.rounds, users= {user_index: now_user}, rewards= now_user.rewards,
                                             best_rewards= now_user.best_rewards, T_phase= cluster.T_phase, theta_phase= cluster.theta_phase)
            self.clusters[cnew] = tmp_cluster
            self.cluster_inds[user_index] = cnew

            del cluster.users[user_index]
            cluster.V = cluster.V - now_user.V
            cluster.b = cluster.b - now_user.b
            cluster.T = cluster.T - now_user.T
            del cluster.checks[user_index]

        self.num_clusters[t - 1] = len(self.clusters)


    def merge(self, t):
        cmax = max(self.clusters)
        for c1 in range(cmax-1):
            if c1 not in self.clusters or self.clusters[c1].checked == False:
                continue
            for c2 in range(c1 + 1,cmax) or self.clusters[c2].checked == False:
                if c2 not in self.clusters:
                    continue
                if not self.if_merge(c1, c2, t):
                    continue
                else:
                    for i in self.clusters[c2].users:
                            self.cluster_inds[i] = c1

                    self.clusters[c1].V = self.clusters[c1].V + self.clusters[c2].V
                    self.clusters[c1].b = self.clusters[c1].b + self.clusters[c2].b
                    self.clusters[c1].T = self.clusters[c1].T + self.clusters[c2].T
                    self.clusters[c1].theta =  np.matmul(np.linalg.inv(np.eye(self.d) + self.clusters[c1].V), self.clusters[c1].b)
                    for user in self.clusters[c2].users:
                        self.clusters[c1].users.setdefault(user, self.clusters[c2].users[user] )
                    self.clusters[c1].checks = {**self.clusters[c1].checks, **self.clusters[c2].checks}
                    del self.clusters[c2]

        self.num_clusters[t - 1] = len(self.clusters)




class Global_server:  # 最开始每个local_server中的user数目是已知的
    def __init__(self, L, n, userList, d, T):  # 这里n指的是user的总数量，假设最开始是已知的,每个cluster的user的数量已知，存储在numlist中
        self.l_server_list = []
        self.usernum = n
        self.rounds = T
        self.l_server_num = L
        self.d = d
        self.regret = np.zeros(self.rounds)
        self.reward = np.zeros(self.rounds)
        self.best_reward = np.zeros(self.rounds)
        user_begin = 0
        self.l_server_inds = np.zeros(n,np.int64)  # 存储每个user所对应的local server的index,现在这种方法是否可行的关键在于user属local server index 的信息会不会传回global server
        user_index = 0
        j = 0
        for i in userList: # userlist中记录的的是每个local_server中的user的数目
            self.l_server_list.append(Local_server(i, d, user_index, self.rounds))
            self.l_server_inds[user_index:user_index + i] = j
            user_index = user_index + i
            j = j + 1

    def locate_user_index(self, user_index):
        # 确定user属于哪个local server
        l_server_index = self.l_server_inds[user_index]
        return l_server_index


    def run(self, envir, phase, number):
        theta_exp = dict()
        result_tmp = list()
        theta_exp = dict()
        theta_one_user = list()
        result = dict()
        for s in range(1, phase + 1):
            for l_server in self.l_server_list:
                l_server.init_each_stage()

            for i in range(1, phase_cardinality ** s + 1):
                t = (phase_cardinality ** s - 1) // (phase_cardinality - 1) + i - 1
                print(t)
                if t % 5000 == 0:
                    print(t)
                user_all = envir.generate_users()
                user_index = user_all[0]
                l_server_index = self.locate_user_index(user_index)
                l_server = self.l_server_list[l_server_index]
                l_cluster_index = l_server.locate_user_index(user_index)
                l_cluster = l_server.clusters[l_cluster_index]
                # the context set
                items = envir.get_items()
                r_item_index = l_server.recommend(l_cluster_index=l_cluster_index, items=items)
                x = items[r_item_index]
                self.reward[t - 1], y, self.best_reward[t - 1], ksi_noise, B_noise = envir.feedback_Local(items=items,i=user_index,
                                                                                                          k=r_item_index, d=self.d)
                l_cluster.users[user_index].store_info(x, y, t - 1, self.reward[t - 1], self.best_reward[t - 1],
                                                       ksi_noise[0], B_noise)
                l_cluster.store_info(x, y, t - 1, self.reward[t - 1], self.best_reward[t - 1], ksi_noise[0], B_noise)
                l_server.update(user_index, t)
                l_server.merge(t)
                self.regret[t - 1] = self.best_reward[t - 1] - self.reward[t - 1]

                cluster_num = 0
                if t % 100000 == 0:
                    for server in self.l_server_list:
                        print("type:", type(server))
                        cluster_num += len(server.clusters)
                        for clst in server.clusters:
                            now_clus = server.clusters[clst]
                            for user in now_clus.users:
                                theta_exp[now_clus.users[user].index] = now_clus.users[user].theta
                        result = dict(sorted(theta_exp.items(), key=lambda k: k[0]))

                if t % 100000 == 0:
                    # 12_18_100_user_alpha_4.5是30user,alpha=1.5
                    npzname = "no_" + str(number) + "_SCLUB_1_19" + str(self.usernum) + "_user_" + str(i)
                    np.savez(npzname, nu=self.usernum, d=self.d, L=len(self.l_server_list), T=t,
                             G_server_regret=self.regret,
                             cluster_num=cluster_num, theta_exp=result, theta_theo=envir.theta, reward=self.reward)

        return self.regret, result, self.reward, cluster_num








































