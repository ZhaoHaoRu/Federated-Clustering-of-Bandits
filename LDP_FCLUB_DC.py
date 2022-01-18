# -*- coding: utf-8 -*-
#version 3


import networkx as nx
import numpy as np

import Base
import Environment as Envi
import copy
import os
from Environment import alpha, delt, epsi, sigma, alpha2,U, D

S = 1
phase_cardinality = 2

#class Global_server

#Global_server G_server


class Local_server:
    def __init__(self, nl, d, begin_num, T, server_index, edge_probability=0.8):
        self.nl = nl
        self.d = d
        self.rounds = T  # the number of all rounds
        #self.G = nx.gnp_random_graph(nl, edge_probability)
        # 生成无向完全图，此时index是真正的user index
        user_index_list = list(range(begin_num,begin_num + nl))
        self.G = nx.generators.classic.complete_graph(user_index_list)
        #self.G = nx.gnp_random_graph(user_index_list, edge_probability)
        #print('图中所有的节点', self.G.nodes())
        self.clusters = {0: Base.DC_Cluster(b=np.zeros(d), T=0, V=np.zeros((d,d)), users_begin = begin_num, d=d, user_num = nl, t= self.rounds, rewards= np.zeros(self.rounds), best_rewards= np.zeros(self.rounds), l_server_index=server_index, index= 0)}  # users的初始化方法直接抄了老师里面的，不知道对不对
        self.index = server_index
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

    #这里的T是当前的round,首先计算了当前的信息
    def recommend(self, l_cluster_index, T, items):
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
        t1 = cluster.users[user_index1].T
        gamma_1 = Envi.gamma(t1,self.d,alpha,sigma)
        theta1 = np.matmul(np.linalg.inv(gamma_1*2*np.eye(self.d) + cluster.users[user_index1].V), cluster.users[user_index1].b)
        cluster.users[user_index1].theta = theta1
        # theta1 = cluster.users[user_index1].theta
        t2 = cluster.users[user_index2].T
        gamma_2 = Envi.gamma(t2, self.d, alpha, sigma)
        theta2 = np.matmul(np.linalg.inv(gamma_2 * 2 * np.eye(self.d) + cluster.users[user_index2].V),
                           cluster.users[user_index2].b)
        cluster.users[user_index2].theta = theta2
        # theta2 = cluster.users[user_index2].theta
        return np.linalg.norm(theta1 - theta2) > alpha * (fact_T1 + fact_T2)

    def check_update(self, user_index, t): # 这里的user_index是从0到n的一个自然数
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



    def update_cluster(self):
        user_list = list(self.cluster_inds.keys())
        user_dict = dict()

        #删除原先所有的cluster
        for j in list(self.clusters.keys()):
            for i in self.clusters[j].users:
                user_dict[i] = copy.deepcopy(self.clusters[j].users[i])
            del self.clusters[j]

        c = 0
        #根据现在的connected_components重新分
        #print("components num:", nx.number_connected_components(self.G))
        for cluster_set in nx.connected_components(self.G):
            all_user = list(cluster_set)
            remain_users = dict()
            for k in all_user:
                remain_users[k] = user_dict[k]
            tmp_cluster = Base.DC_Cluster(b=sum([remain_users[k].b for k in remain_users]),
                                          T=sum([remain_users[k].T for k in remain_users]),
                                               V = sum([remain_users[k].V for k in remain_users]), users_begin = min(remain_users), d = self.d, user_num = len(remain_users), t=self.rounds,
                                               users = copy.deepcopy(remain_users),
                                               rewards= sum([remain_users[k].rewards for k in remain_users]), best_rewards= sum([remain_users[k].best_rewards for k in remain_users]),
                                               l_server_index= self.index, index= c)
            self.clusters[c] = tmp_cluster
            for k in remain_users:
                self.cluster_inds[k] = c

            c += 1



class Global_server:  # 最开始每个local_server中的user数目是已知的
    def __init__(self, L, n, userList, d, T):  # 这里n指的是user的总数量，假设最开始是已知的,每个cluster的user的数量已知，存储在numlist中
        self.l_server_list = []
        self.usernum = n
        self.rounds = T
        self.l_server_num = L
        self.g_cluster_num = 1 #即m的值
        self.d = d
        self.cluster_usernum = np.zeros(L*n,np.int64)  # 这个是记录每个global cluster的user数量的
        self.clusters = dict()
        self.regret = np.zeros(self.rounds)
        self.reward = np.zeros(self.rounds)
        self.best_reward = np.zeros(self.rounds)
        user_begin = 0
        #极端情况：cluster的数目等于user的数目，第一维表述global cluster，第二维表示global cluster里面的local cluster
        self.partition = np.zeros((self.usernum,self.usernum*2))
        self.partition.fill(-1)
        #这里假设local server与local cluster的index都是从0开始的
        for i in range(0, L*2, 2):
           self.partition[0][i] = i/2
           self.partition[0][i+1] = 0

        #print(self.partition[0])
        #此时只有一个global cluster
        self.clusters[0] = Base.DC_Cluster(b=np.zeros(self.d), T=0, V=np.zeros((self.d,self.d)), users_begin=0,d = self.d, user_num=self.usernum, t=self.rounds,users = {},rewards= np.zeros(self.rounds), best_rewards= np.zeros(self.rounds),l_server_index=-1,index= 0)
        self.cluster_inds = np.zeros(n,np.int64)   # 存储每个user对应的global cluster的index,下标索引值代表了user index
        self.l_server_inds = np.zeros(n,np.int64)  # 存储每个user所对应的local server的index,现在这种方法是否可行的关键在于user属local server index 的信息会不会传回global server
        user_index = 0
        j = 0
        for i in userList: # userlist中记录的的是每个local_server中的user的数目
            self.l_server_list.append(Local_server(i, d, user_index, self.rounds,server_index=j))
            self.cluster_usernum[j] = i
            self.cluster_inds[user_index:user_index + i] = 0
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


    def communicate(self):
        g_cluster_index = 0
        tmp_partition = np.zeros((self.usernum, self.usernum * 2))
        tmp_partition.fill(-1)
        for i in range(self.l_server_num):
            l_server = self.l_server_list[i]
            for cluster_index in l_server.clusters:
                self.clusters[g_cluster_index] = copy.deepcopy(l_server.clusters[cluster_index]);
                tmp_partition[g_cluster_index][0] = l_server.clusters[cluster_index].l_server_index
                tmp_partition[g_cluster_index][1] = l_server.clusters[cluster_index].index


                for user in l_server.cluster_inds:
                    if l_server.cluster_inds[user] == cluster_index:
                        self.cluster_inds[user] = g_cluster_index
                self.cluster_usernum[g_cluster_index] = l_server.clusters[cluster_index].user_num
                g_cluster_index += 1

        self.partition = tmp_partition



    def if_merge(self,cluster_id1, cluster_id2):
        cluster1 = self.clusters[cluster_id1]
        cluster2 = self.clusters[cluster_id2]
        T1 = cluster1.T
        T2 = cluster2.T
        fact_T1 = np.sqrt((1 + np.log(1 + T1)) / (1 + T1))
        fact_T2 = np.sqrt((1 + np.log(1 + T2)) / (1 + T2))
        #这里的theta需不需要重新算啊
        theta1 = cluster1.theta
        theta2 = cluster2.theta
        if (np.linalg.norm(theta1 - theta2) < alpha2 * (fact_T1 + fact_T2)):
            return False
        else:
            return True



    def merge(self, former_partition):
        done_merge = False
        cluster_node = list(self.clusters.keys())
        cluster_G = nx.complete_graph(cluster_node)
        # cluster_G = nx.gnp_random_graph(cluster_node, 0.8)
        nodes = cluster_G.nodes()
        cmax = max(self.clusters)
        for c1 in nodes:
            if c1 not in self.clusters:
                continue
            A = [a for a in cluster_G.neighbors(c1)]
            for c2 in A:
                if c2 not in self.clusters:
                    continue
                if self.if_merge(c1,c2):
                    #(c1,c2)
                    cluster_G.remove_edge(c1,c2)
                    done_merge = True


        if done_merge and (former_partition != self.partition).any():
            cluster_num = nx.number_connected_components(cluster_G)
            for cluster_set in nx.connected_components(cluster_G):
                global_l_cluster_num = 1
                cluster_list = list(cluster_set)
                C1 = cluster_list[0]
                for i in cluster_list[1:]:
                    self.clusters[C1].S += self.clusters[i].S
                    self.clusters[C1].u += self.clusters[i].u
                    self.clusters[C1].T += self.clusters[i].T
                    self.clusters[C1].user_num += self.clusters[i].user_num
                    self.cluster_usernum[C1] += self.cluster_usernum[i]
                    #此处theta需要重新计算嘛？

                    self.partition[C1][global_l_cluster_num * 2] = self.clusters[i].l_server_index
                    self.partition[C1][global_l_cluster_num * 2 + 1] = self.clusters[i].index
                    self.partition[i][0] = -1
                    self.partition[i][1] = -1
                    global_l_cluster_num += 1
                    for j in range(self.usernum):
                        if self.cluster_inds[j] == i:
                            self.cluster_inds[j] = cluster_list[0]
                    for user in self.clusters[i].users:
                        self.clusters[cluster_list[0]].users.setdefault(user, self.clusters[i].users[user])
                    del self.clusters[i]
                self.clusters[C1].theta = np.matmul(np.linalg.inv(np.eye(self.d) + self.clusters[C1].S),
                                                    self.clusters[C1].u)




    def detection(self, t):
        for l_index in range(len(self.l_server_list)):
            check_server = self.l_server_list[l_index]
            for i in check_server.cluster_inds:
                user1_index = i
                check_server.check_update(user1_index, t)

            check_server.update_cluster()


        #Upload the local clustered information to the global server,maybe should have some time cost

        former_partition  = self.partition
        self.communicate()
        self.merge(former_partition)
        if (former_partition != self.partition).any():
            #Renew the cluster information and create new upload/download buffers
            #更新global cluster的S,u,T工作在merge 中已经完成
            for g_cluster_id in self.clusters:
                self.clusters[g_cluster_id].S = np.zeros((self.d,self.d))
                self.clusters[g_cluster_id].u = np.zeros(self.d)
                self.clusters[g_cluster_id].T = 0
                l_cluster_info = self.partition[g_cluster_id]
                for i in range(0,self.usernum*2,2):
                    l_server_id = l_cluster_info[i].astype(np.int)
                    l_cluster_id = l_cluster_info[i + 1].astype(np.int)
                    if l_cluster_id == -1 or l_server_id == -1:
                        continue
                    #print("l_cluster_info:",l_cluster_info)
                    #print("l_cluster_id:",l_cluster_id)
                    l_server = self.l_server_list[l_server_id]
                    l_cluster = l_server.clusters[l_cluster_id]
                    self.clusters[g_cluster_id].S += l_cluster.S
                    self.clusters[g_cluster_id].u += l_cluster.u
                    self.clusters[g_cluster_id].T += l_cluster.T

                self.clusters[g_cluster_id].theta = np.matmul(
                        np.linalg.inv(np.eye(self.d) + self.clusters[g_cluster_id].S), self.clusters[g_cluster_id].u)

                for i in range(0,self.usernum*2,2):
                    l_server_id = l_cluster_info[i].astype(np.int)
                    l_cluster_id = l_cluster_info[i + 1].astype(np.int)
                    if l_cluster_id == -1 or l_server_id == -1:
                        continue
                    #print("l_cluster_info:",l_cluster_info)
                    #print("l_cluster_id:",l_cluster_id)
                    l_server = self.l_server_list[l_server_id]
                    l_cluster = l_server.clusters[l_cluster_id]
                    l_cluster.S = self.clusters[g_cluster_id].S
                    l_cluster.u = self.clusters[g_cluster_id].u
                    l_cluster.T = self.clusters[g_cluster_id].T
                    if t >= 60000:
                        t = t
                    l_cluster.theta = np.matmul(
                        np.linalg.inv(np.eye(self.d) + l_cluster.S),l_cluster.u)

                    #create new buffers
                    l_cluster.S_up = np.zeros((self.d,self.d))
                    l_cluster.S_down = np.zeros((self.d,self.d))
                    l_cluster.u_up = np.zeros(self.d)
                    l_cluster.u_down = np.zeros(self.d)
                    l_cluster.T_up = 0
                    l_cluster.T_down = 0

    def find_global_cluster(self, l_server_id, l_cluster_id):
        g_cluster_want = -1
        for g_cluster_id in self.clusters:
            #("exist value:", self.clusters.keys())
            g_cluster_want = g_cluster_id
            l_cluster_info = self.partition[g_cluster_id]
            for i in range(0, self.usernum*2, 2):
                l_server_id_tmp = l_cluster_info[i]
                l_cluster_id_tmp = l_cluster_info[i + 1]
                if l_server_id_tmp == l_server_id and l_cluster_id_tmp == l_cluster_id:
                    #print("g_cluster_want:", g_cluster_want)
                    return g_cluster_want

        return -1


    def check_upload(self, l_server_id, l_cluster_id):
        l_server = self.l_server_list[l_server_id.astype(np.int)]
        l_cluster = l_server.clusters[l_cluster_id]
        S1 = l_cluster.S
        S2 = l_cluster.S_up
        if np.linalg.det(S1 + S2) / np.linalg.det(S1) >= U:
            g_cluster_id = self.find_global_cluster(l_server_id, l_cluster_id)
            if g_cluster_id == -1:
                #zhe一步没有输出，说明找partition的操作是对的
                print("l_server_id",l_server_id)
                print("l_cluster_id", l_cluster_id)
                print(self.partition)
                self.clusters[9].S += S2
            if g_cluster_id != -1:
                #print("exist value1:", self.clusters.keys())
                self.clusters[g_cluster_id].S += S2
                self.clusters[g_cluster_id].u += l_cluster.u_up
                self.clusters[g_cluster_id].T += l_cluster.T_up
                self.clusters[g_cluster_id].theta = np.matmul(np.linalg.inv(np.eye(self.d) + self.clusters[g_cluster_id].S), self.clusters[g_cluster_id].u)
                l_cluster_info = self.partition[g_cluster_id]
                l_cluster_other_num = 0
                for i in range(0, self.usernum*2, 2):
                    l_server_id_other = l_cluster_info[i]
                    l_cluster_id_other = l_cluster_info[i + 1]
                    if l_server_id_other == l_server_id and l_cluster_id_other == l_cluster_id:
                        continue
                    if l_server_id_other == -1 or l_cluster_id_other == -1:
                        continue
                    l_server_other = self.l_server_list[l_server_id_other.astype(np.int)]
                    l_cluster_other = l_server_other.clusters[l_cluster_id_other.astype(np.int)]
                    l_cluster_other.S_down += S2
                    l_cluster_other.u_down += l_cluster.u_up
                    l_cluster_other.T_down += l_cluster.T_up
                    l_cluster_other_num += 1

                print(l_cluster_other_num)

                #Local server cleans the buffer
                l_cluster.S += l_cluster.S_up
                l_cluster.u += l_cluster.u_up
                l_cluster.T += l_cluster.T_up
                l_cluster.theta = np.matmul(np.linalg.inv(np.eye(self.d) + l_cluster.S), l_cluster.u)
                l_cluster.S_up = np.zeros((self.d,self.d))
                l_cluster.u_up = np.zeros(self.d)
                l_cluster.T_up = 0


    def check_download(self,g_cluster_id):
        l_cluster_info = self.partition[g_cluster_id]
        V_g = self.clusters[g_cluster_id].S
        for i in range(0, self.usernum*2, 2):
            l_server_id = l_cluster_info[i]
            l_cluster_id = l_cluster_info[i + 1]
            if l_server_id == -1 or l_cluster_id == -1:
                continue
            l_server = self.l_server_list[l_server_id.astype(np.int)]
            l_cluster = l_server.clusters[l_cluster_id.astype(np.int)]
            if np.linalg.det(V_g) / np.linalg.det(l_cluster.S) >= D:
                l_cluster.S += l_cluster.S_down
                l_cluster.u += l_cluster.u_down
                l_cluster.T += l_cluster.T_down
                l_cluster.theta = np.matmul(np.linalg.inv(np.eye(self.d) + l_cluster.S), l_cluster.u)

                l_cluster.S_down = np.zeros((self.d,self.d))
                l_cluster.u_down = np.zeros(self.d)
                l_cluster.T_down = 0


    def run(self, envir, phase,number, all_round):
        theta_exp = dict()
        result_tmp = list()
        for s in range(1, phase + 1):
            if s == 16:
                s = s
            self.detection(phase_cardinality**s - 1)
            for i in range(1, phase_cardinality**s + 1):
                t = (phase_cardinality**s - 1)//(phase_cardinality - 1) + i - 1
                if t >= all_round:
                    break
                if t == 40000:
                    t = t
                if t % 5000 == 0:
                    print(t)
                user_all = envir.generate_users()
                user_index = user_all[0]
                l_server_index, g_cluster_index = self.locate_user_index(user_index)
                l_server = self.l_server_list[l_server_index]
                g_cluster = self.clusters[g_cluster_index]
                l_cluster_index = l_server.locate_user_index(user_index)
                l_cluster = l_server.clusters[l_cluster_index]
                # the context set
                items = envir.get_items()
                r_item_index = l_server.recommend(l_cluster_index=l_cluster_index,T= t,items= items)
                x = items[r_item_index]
                #这个地方加了一个round，但是其实只是出于打印的需要，回头删掉
                self.reward[t - 1], y, self.best_reward[t - 1], ksi_noise, B_noise = envir.feedback_Local(items= items,i= user_index,k= r_item_index,d= self.d, now_round= t)
                l_cluster.users[user_index].store_info(x, y, t - 1, self.reward[t - 1],self.best_reward[t - 1], ksi_noise[0], B_noise)
                l_cluster.store_info(x, y, t - 1, self.reward[t - 1],self.best_reward[t - 1], ksi_noise[0], B_noise)
                self.check_upload(l_server_index, l_cluster_index)
                self.check_download(g_cluster_index)
                self.regret[t - 1] = self.best_reward[t - 1] - self.reward[t - 1]

                if t % 100000 ==  0:
                    for clst in self.clusters:
                        now_clus = self.clusters[clst]
                        for user in now_clus.users:
                            theta_exp[now_clus.users[user].index] = now_clus.users[user].theta

                    result = dict(sorted(theta_exp.items(), key=lambda k: k[0]))
                    result_tmp = list(result.values())
                    # print('result:',result)
                    # print('result_tmp:',result_tmp)

                if t % 100000 ==  0:
                    npzname = "no_"+str(number)+"_DC_1_5_20_user_" + str(t)
                    print(i)
                    np.savez(npzname, nu=self.usernum, d=self.d, L=len(self.clusters), T=t, G_server_regret=self.regret,
                             cluster_num=len(self.clusters), theta_exp= result_tmp, theta_theo=envir.theta)

        return self.regret,result_tmp,self.reward