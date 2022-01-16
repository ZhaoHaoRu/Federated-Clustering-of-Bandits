# -*- coding: utf-8 -*-
#version 2
import networkx as nx
import numpy as np

import Base
import Environment as Envi
import copy
import os
from Environment import alpha, delt, epsi, sigma, alpha2

S = 1


class Local_server:
    def __init__(self, nl, d, begin_num, T, edge_probability=0.8):
        self.nl = nl
        self.d = d
        self.rounds = T  # the number of all rounds
        #self.G = nx.gnp_random_graph(nl, edge_probability)
        # 生成无向完全图，此时index是真正的user index
        user_index_list = list(range(begin_num,begin_num + nl))
        self.G = nx.generators.classic.complete_graph(user_index_list)
        #self.G = nx.gnp_random_graph(user_index_list, edge_probability)
        #print('图中所有的节点', self.G.nodes())
        self.clusters = {0: Base.Cluster(b=np.zeros(d), T=0, V=np.zeros((d,d)), users_begin = begin_num, d=d, user_num = nl, t= self.rounds, rewards= np.zeros(self.rounds), best_rewards= np.zeros(self.rounds))}  # users的初始化方法直接抄了老师里面的，不知道对不对
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

    def recommend(self, M, b , beta, user_index, items):
        Minv = np.linalg.inv(M)
        theta = np.dot(Minv,b)
        # print(type(theta))
        # print(type(items))
        # 将global server上算得的M，b，β传给当前的local server
        r_item_index = np.argmax(np.dot(items,theta) + beta * (np.matmul(items, Minv) * items).sum(axis=1))
        #np.argmax(np.dot(items, theta) + self._beta(N, t) * (np.matmul(items, Sinv) * items).sum(axis=1))
        return r_item_index

    def if_delete(self, user_index1, user_index2, cluster): # 这个cluster是要进行删边操作的cluster
        T1 = cluster.users[user_index1].T
        T2 = cluster.users[user_index2].T
        fact_T1 = np.sqrt((1 + np.log(1 + T1)) / (1 + T1))
        fact_T2 = np.sqrt((1 + np.log(1 + T2)) / (1 + T2))
        theta1 = cluster.users[user_index1].theta
        theta2 = cluster.users[user_index2].theta
        return np.linalg.norm(theta1 - theta2) > alpha * (fact_T1 + fact_T2)




class Global_server:
    #在homogeneous的版本中，自始至终只有一个global cluster
    #有多个local server,在merge的时候一定会merge到一起
    def __init__(self, L, n, userList, d, T):  # 这里n指的是user的总数量，假设最开始是已知的,每个cluster的user的数量已知，存储在numlist中
        self.l_server_list = []
        self.usernum = n
        self.rounds = T
        self.l_server_num = L
        self.d = d
        self.cluster_usernum = np.zeros(L*n,np.int64)  # 这个是记录每个global cluster的user数量的
        self.clusters = dict()
        self.regret = np.zeros(self.rounds)
        self.reward = np.zeros(self.rounds)
        self.best_reward = np.zeros(self.rounds)
        user_begin = 0
        #global上面的cluster，最开始只有一个
        self.clusters[0]= Base.Cluster(b=np.zeros(self.d), T=0, V=np.zeros((d,d)), users_begin=user_begin,d = self.d, user_num=self.usernum, t=self.rounds,users = {},rewards= np.zeros(self.rounds), best_rewards= np.zeros(self.rounds))
        self.cluster_inds = np.zeros(n,np.int64)   # 存储每个user对应的global cluster的index,下标索引值代表了user index
        self.l_server_inds = np.zeros(n,np.int64)  # 存储每个user所对应的local server的index,现在这种方法是否可行的关键在于user属local server index 的信息会不会传回global server
        user_index = 0
        j = 0
        for i in userList:  # userlist中记录的的是每个local_server中的user的数目
            self.l_server_list.append(Local_server(i, d, user_index, self.rounds))
            self.cluster_usernum[j] = i
            self.l_server_inds[user_index:user_index + i] = j
            user_index = user_index + i
            j = j + 1


    def locate_user_index(self, user_index):
        # 如果这种方法不可行的话，只能强行遍历
        # 确定user属于哪个local server
        l_server_index = self.l_server_inds[user_index]
        #确定user属于哪个global cluster
        g_cluster_index = self.cluster_inds[user_index]
        return l_server_index, g_cluster_index

    def global_info(self,user_index, g_cluster_index):
       #l_server = self.l_server_list[l_server_index]
       g_cluster = self.clusters[g_cluster_index]
       V = g_cluster.V
       b = g_cluster.b
       T = g_cluster.T
       gamma_t = Envi.gamma(T+1, self.d, alpha, sigma)
       lambda_t = gamma_t * 2
       # 这里算β的S和L还不是很确定，S=L=1
       beta_t = Envi.beta(sigma, alpha, gamma_t, S, self.d, T+1, self.l_server_num)
       M_t = np.eye(self.d)* np.float_(lambda_t) + V
       # 接下来将M_t, b, beta传给local server，先直接调local server类中的函数
       #l_server.recommend(M_t, b, beta_t, user_index)
       return M_t, b, beta_t

    def communicate(self):
        g_cluster_index = 0
        for i in range(self.l_server_num):
            l_server = self.l_server_list[i]
            for cluster_index in l_server.clusters:
                # self.clusters[g_cluster_index] = Base.Cluster(b=l_server.clusters[cluster_index].b, T =l_server.clusters[cluster_index].T,
                #                            V = l_server.clusters[cluster_index].V, users_begin = l_server.clusters[cluster_index].users_begin, d = self.d, user_num = l_server.clusters[cluster_index].user_num, t=self.rounds, theta= l_server.clusters[cluster_index].theta, users = l_server.clusters[cluster_index].users)
                self.clusters[g_cluster_index] = copy.deepcopy(l_server.clusters[cluster_index]);

                for user in l_server.cluster_inds:
                    if l_server.cluster_inds[user] == cluster_index:
                        self.cluster_inds[user] = g_cluster_index
                self.cluster_usernum[g_cluster_index] = l_server.clusters[cluster_index].user_num
                g_cluster_index += 1

        #print("gcluster number:" ,len(self.clusters))

    def merge(self):
        cmax = max(self.clusters)
        cmin = min(self.clusters)
        for c1 in range(cmin + 1,cmax + 1):
            if c1 not in self.clusters:
                continue
            else:
                for j in range(self.usernum):
                    if self.cluster_inds[j] == c1:
                        self.cluster_inds[j] = cmin

                self.cluster_usernum[cmin] += self.cluster_usernum[c1]
                self.cluster_usernum[c1] = 0

                self.clusters[cmin].V += self.clusters[c1].V
                self.clusters[cmin].b += self.clusters[c1].b
                self.clusters[cmin].T += self.clusters[c1].T
                self.clusters[cmin].user_num += self.clusters[c1].user_num
                for user in self.clusters[c1].users:
                    self.clusters[cmin].users.setdefault(user, self.clusters[c1].users[user] )
                del self.clusters[c1]

        self.clusters[cmin].theta = np.matmul(np.linalg.inv(np.eye(self.d) + self.clusters[cmin].V), self.clusters[cmin].b)


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
            l_server_index, g_cluster_index = self.locate_user_index(user_index)
            M_t, b, beta_t = self.global_info(user_index, g_cluster_index)
            l_server = self.l_server_list[l_server_index]
            g_cluster = self.clusters[g_cluster_index]
            l_cluster = l_server.clusters[l_server.cluster_inds[user_index]]
            # the context set
            items = envir.get_items()
            r_item_index = l_server.recommend(M_t, b, beta_t, user_index, items)
            x = items[r_item_index]
            x_list.append(x)
            self.reward[i - 1], y, self.best_reward[i - 1], ksi_noise, B_noise = envir.feedback(items,user_index, b, M_t, r_item_index, self.d)
            y_list.append(y)
            l_cluster.users[user_index].store_info(x, y, i - 1, self.reward[i - 1],self.best_reward[i -1], ksi_noise[0], B_noise)
            l_cluster.store_info(x, y, i - 1, self.reward[i - 1], self.best_reward[i - 1], ksi_noise[0], B_noise)
            # 这一步相当于delete edge 并计算 aggregated information，但是没有send to global server 这一步
            g_cluster.store_info(x, y, i - 1, self.reward[i - 1], self.best_reward[i - 1], ksi_noise[0], B_noise)
            self.communicate()
            self.merge()
            self.regret[i - 1] = self.best_reward[i - 1] - self.reward[i - 1]


            if i % 100000 ==  0:
                for clst in self.clusters:
                    now_clus = self.clusters[clst]
                    for user in now_clus.users:
                        theta_exp[now_clus.users[user].index] = now_clus.users[user].theta
                result = dict(sorted(theta_exp.items(), key=lambda k: k[0]))


            if i % 100000 ==  0:
                #12_18_100_user_alpha_4.5是30user,alpha=1.5
                npzname = "no_"+str(number)+"homogeneous_FCLUB_1_5_20_user_" + str(i)
                np.savez(npzname, nu=self.usernum, d=self.d, L=len(self.clusters), T=i, G_server_regret=self.regret,
                         cluster_num=len(self.clusters), theta_exp= result, theta_theo=envir.theta, reward= self.reward)


        return self.regret,result, self.reward, x_list, y_list




































