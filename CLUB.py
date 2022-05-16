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
alpha = 0.25
#class Global_server

#Global_server G_server



class Local_server:
    def __init__(self, nl, d, begin_num, T, edge_probability=0.8):
        self.nl = nl
        self.d = d
        self.rounds = T  # the number of all rounds
        # self.G = nx.gnp_random_graph(nl, edge_probability)
        # 生成无向完全图，此时index是真正的user index
        user_index_list = list(range(begin_num,begin_num + nl))
        self.G = nx.generators.classic.complete_graph(user_index_list)
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


    def if_delete(self, user_index1, user_index2, cluster): # 这个cluster是要进行删边操作的cluster
        T1 = cluster.users[user_index1].T
        T2 = cluster.users[user_index2].T
        fact_T1 = np.sqrt((1 + np.log(1 + T1)) / (1 + T1))
        fact_T2 = np.sqrt((1 + np.log(1 + T2)) / (1 + T2))
        theta1 = cluster.users[user_index1].theta
        theta2 = cluster.users[user_index2].theta
        return np.linalg.norm(theta1 - theta2) > alpha * (fact_T1 + fact_T2)

    def update(self, user_index, t): # 这里的user_index是从0到n的一个自然数
        update_cluster = False
        c = self.cluster_inds[user_index] # 找到被更新的user所属于的local cluster
        #i = user_index-self.begin_num # 感觉图中的node是从0开始排序的，但是user index其实并不是
        i = user_index
        #print("user_index in line 57:" ,user_index)
        # 存储最开始的cluster，以便下一步使用
        origin_cluster = self.clusters[c]
        A = [a for a in self.G.neighbors(i)]
        #print("user_index in line 61:", user_index)
        for j in A:
            #user2_index = j + self.begin_num
            user2_index = j
            c2 = self.cluster_inds[user2_index] # c和c2应该是同一个cluster
            user1 = self.clusters[c].users[i]
            user2 = self.clusters[c2].users[user2_index]
            if user1.T != 0 and user2.T != 0 and self.if_delete(i, user2_index, self.clusters[c]):
                self.G.remove_edge(i,j)
                update_cluster = True

        if update_cluster:
            C =set()
            #print("user_index in line 74:", i)
            C = nx.node_connected_component(self.G, i) #对应的user index,这是更新后的cluster应该有的user
            #print("the set:", C)
            remain_users = dict()
            for m in C:
                remain_users[m] = self.clusters[c].get_user(m)


            if len(C) < len(self.clusters[c].users):
                print(t)
                all_users_index = set(self.clusters[c].users) # 这是原始的cluster中的所有user index
                all_users = dict()
                for user_index2 in all_users_index:
                    all_users[user_index2] = self.clusters[c].get_user(user_index2)
                # 将当前cluster应有的user放到这个cluster中
                tmp_cluster = Base.Cluster(b=sum([remain_users[k].b for k in remain_users]), T =sum([remain_users[k].T for k in remain_users]),
                                           V = sum([remain_users[k].V for k in remain_users]), users_begin = min(remain_users), d = self.d, user_num = len(remain_users), t=self.rounds,
                                           users = copy.deepcopy(remain_users),
                                           rewards= sum([remain_users[k].rewards for k in remain_users]), best_rewards= sum([remain_users[k].best_rewards for k in remain_users]))
                self.clusters[c] = tmp_cluster

                # 将新的cluster中包含的user从原始的cluster中删除
                for user_index3 in all_users_index:
                    if remain_users.__contains__(user_index3):
                        all_users.pop(user_index3)

                c = max(self.clusters) + 1
                while len(all_users) > 0:
                    #print("remain users:", list(all_users))
                    j = np.random.choice(list(all_users))
                    C = nx.node_connected_component(self.G, j)
                    #print(type(C))
                    #print("j:",j)
                    new_cluster_users = dict()
                    for k in C:
                        new_cluster_users[k] = origin_cluster.get_user(k)
                    self.clusters[c] = Base.Cluster(b=sum([new_cluster_users[n].b for n in new_cluster_users]), T=sum([new_cluster_users[n].T for n in new_cluster_users]),
                                          V=sum([new_cluster_users[n].V for n in new_cluster_users]), users_begin = min(new_cluster_users), d = self.d, user_num = len(new_cluster_users),
                                                    t=self.rounds, users = copy.deepcopy(new_cluster_users),
                                                    rewards= sum([new_cluster_users[k].rewards for k in new_cluster_users]), best_rewards= sum([new_cluster_users[k].best_rewards for k in new_cluster_users]))
                    for k in C:
                        self.cluster_inds[k] = c

                    c += 1
                    for user_index in all_users_index:
                        if new_cluster_users.__contains__(user_index):
                            all_users.pop(user_index)

        self.num_clusters[t] = len(self.clusters)



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
            l_server_index = self.locate_user_index(user_index)
            l_server = self.l_server_list[l_server_index]
            cluster_index = l_server.locate_user_index(user_index)
            cluster = l_server.clusters[cluster_index]
            # the context set
            items = envir.get_items()
            r_item_index = l_server.recommend(cluster_index, items)
            x = items[r_item_index]
            x_list.append(x)
            self.reward[i - 1], y, self.best_reward[i - 1], ksi_noise, B_noise = envir.feedback_Local(items= items,i= user_index,k= r_item_index,d= self.d)
            y_list.append(y)
            # print("ksi_noise in ",i,"round is",ksi_noise)
            # print("B_noise in ", i, "round is", B_noise)
            cluster.users[user_index].store_info(x, y, i - 1, self.reward[i - 1],self.best_reward[i -1], ksi_noise[0], B_noise)
            cluster.store_info(x, y, i - 1, self.reward[i - 1], self.best_reward[i - 1], ksi_noise[0], B_noise)
            # 这一步相当于delete edge 并计算 aggregated information，但是没有send to global server 这一步
            l_server.update(user_index,i - 1)
            self.regret[i - 1] = self.best_reward[i - 1] - self.reward[i - 1]


            cluster_num = 0
            if i % 100000 ==  0:
                for server in self.l_server_list:
                    print("type:",type(server))
                    cluster_num += len(server.clusters)
                    for clst in server.clusters:
                        now_clus = server.clusters[clst]
                        for user in now_clus.users:
                            theta_exp[now_clus.users[user].index] = now_clus.users[user].theta
                    result = dict(sorted(theta_exp.items(), key=lambda k: k[0]))

            if i % 100000 ==  0:
                #12_18_100_user_alpha_4.5是30user,alpha=1.5
                npzname = "CLUB" + "no_"+str(number)+"_1_26" + str(self.usernum) + "_user_" + str(i)
                np.savez(npzname, nu=self.usernum, d=self.d, L=len(self.l_server_list), T=i, G_server_regret=self.regret,
                         cluster_num= cluster_num, theta_exp= result, theta_theo=envir.theta, reward= self.reward)


        return self.regret,result, self.reward, x_list, y_list, cluster_num







































