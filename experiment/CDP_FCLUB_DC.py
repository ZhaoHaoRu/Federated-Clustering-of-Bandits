# -*- coding: utf-8 -*-
# version 3


import networkx as nx
import numpy as np

import Base
import Environment as Envi
import copy
from Environment import alpha, delt, epsi, sigma, alpha2, U, D

S = 1
phase_cardinality = 2
beta_scaling = 0.005


class Local_server:
    def __init__(self, nl, d, begin_num, T, server_index, edge_probability=0.8):
        self.nl = nl  # the number of users in a server
        self.d = d  # dimension
        self.rounds = T  # the number of all rounds
        user_index_list = list(range(begin_num, begin_num + nl))  # the index of users in this server
        self.G = nx.generators.classic.complete_graph(
            user_index_list)  # Generate undirected complete graph，user indexes range from begin_num to begin_num + nl
        self.clusters = {0: Base.CDP_Cluster(b=np.zeros(d), t=0, V=np.zeros((d, d)), users_begin=begin_num, d=d,
                                             user_num=nl, rounds=self.rounds, rewards=np.zeros(self.rounds),
                                             best_rewards=np.zeros(self.rounds), l_server_index=server_index, index=0)}
        self.index = server_index  # the server of this index in global
        self.cluster_inds = dict()  # Record the index of the cluster to which each user belongs, key:user_index, value:cluster_index
        self.begin_num = begin_num  # the beginning of the users' index in this server
        for i in range(begin_num, begin_num + nl):
            self.cluster_inds[
                i] = 0  # index of the cluster to which each user belongs, key:user_index ,value:cluster_index
        self.num_clusters = np.zeros(self.rounds,
                                     np.int64)  # the total number of clusters in each round , which recorded for a total of `round` times
        self.num_clusters[0] = 1  # only one cluster in the beginning

    # Determine which local cluster the user belongs to
    def locate_user_index(self, user_index):
        l_cluster_index = self.cluster_inds[user_index]
        # 确定user属于哪个global cluster
        return l_cluster_index

    # Decide what items should be recommended at present
    def recommend(self, l_cluster_index, items, L_num):
        cluster = self.clusters[l_cluster_index]
        V_t, b_t, T_t = cluster.get_info()
        M_t = V_t + np.eye(self.d)
        # assume S = 1
        beta_t = Envi.beta_CDP(T_t, self.d, L_num)
        beta_t = beta_t * beta_scaling  # Scale to accelerate learning
        print("beta in CDP_FCLUB_DC:", beta_t)
        Minv = np.linalg.inv(M_t)
        theta = np.dot(Minv, b_t)
        r_item_index = np.argmax(np.dot(items, theta) + beta_t * (np.matmul(items, Minv) * items).sum(axis=1))
        return r_item_index

    # Judge whether the edge between the two users in this cluster needs to be deleted
    def if_delete(self, user_index1, user_index2, cluster):
        t1 = cluster.users[user_index1].t
        t2 = cluster.users[user_index2].t
        fact_T1 = np.sqrt((1 + np.log(1 + t1)) / (1 + t1))
        fact_T2 = np.sqrt((1 + np.log(1 + t2)) / (1 + t2))
        gamma_1 = Envi.gamma(t1, self.d, alpha, sigma)
        gamma_2 = Envi.gamma(t2, self.d, alpha, sigma)

        # calculate and update the user's theta
        theta1 = np.matmul(np.linalg.inv(gamma_1 * 2 * np.eye(self.d) + cluster.users[user_index1].V),
                           cluster.users[user_index1].b)
        cluster.users[user_index1].theta = theta1
        theta2 = np.matmul(np.linalg.inv(gamma_2 * 2 * np.eye(self.d) + cluster.users[user_index2].V),
                           cluster.users[user_index2].b)
        cluster.users[user_index2].theta = theta2
        return np.linalg.norm(theta1 - theta2) > alpha * (fact_T1 + fact_T2)

    # Delete edges in this user's cluster
    def check_update(self, user_index):
        c = self.cluster_inds[user_index]  # Find the local cluster to which the updated user belongs
        i = user_index
        A = [a for a in self.G.neighbors(i)]
        for j in A:
            user2_index = j
            c2 = self.cluster_inds[user2_index]
            user1 = self.clusters[c].users[i]
            user2 = self.clusters[c2].users[user2_index]
            if user1.t != 0 and user2.t != 0 and self.if_delete(i, user2_index, self.clusters[c]):
                self.G.remove_edge(i, j)  # delete the edge

    # update clusters in this server after several delete edge operation
    def update_cluster(self, t):
        user_dict = dict()

        # Delete all previous clusters and save the users' information (since many edges may have been deleted, we regenerate all clusters for convenience)
        for j in list(self.clusters.keys()):
            for i in self.clusters[j].users:
                user_dict[i] = copy.deepcopy(self.clusters[j].users[i])
            del self.clusters[j]

        c = 0  # the cluster index in this server
        # Redivide the clusters according to the current connected_components
        for cluster_set in nx.connected_components(self.G):
            all_user = list(cluster_set)
            remain_users = dict()
            for k in all_user:
                remain_users[k] = user_dict[k]

            # Generate new cluster based on the connected_components
            tmp_cluster = Base.CDP_Cluster(b=sum([remain_users[k].b for k in remain_users]),
                                           t=sum([remain_users[k].t for k in remain_users]),
                                           V=sum([remain_users[k].V for k in remain_users]),
                                           users_begin=min(remain_users),
                                           d=self.d, user_num=len(remain_users), rounds=self.rounds,
                                           users=copy.deepcopy(remain_users),
                                           rewards=sum([remain_users[k].rewards for k in remain_users]),
                                           best_rewards=sum([remain_users[k].best_rewards for k in remain_users]),
                                           l_server_index=self.index, index=c)
            self.clusters[c] = tmp_cluster
            for k in remain_users:
                self.cluster_inds[k] = c

            c += 1

            # Generate new perturbation
            for cluster_index in self.clusters:
                cluster = self.clusters[cluster_index]
                self.clusters[cluster_index].phase_update()
                self.clusters[cluster_index].privatizer(t, delt, epsi)
                cluster.h_former = cluster.h_now
                cluster.H_former = cluster.H_now


class Global_server:
    def __init__(self, L, n, userList, d, T):
        self.l_server_list = []
        self.usernum = n  # the total number of users
        self.rounds = T
        self.l_server_num = L  # the number of local server
        self.g_cluster_num = 1  # only one cluster in the beginning
        self.d = d
        self.cluster_usernum = np.zeros(L * n, np.int64)  # Record the number of users in each global cluster in each round
        self.clusters = dict()
        self.regret = np.zeros(self.rounds)
        self.reward = np.zeros(self.rounds)
        self.best_reward = np.zeros(self.rounds)

        # Record the partition, the first dimension represents global cluster, the second dimension represents local clusters in this global cluster
        # the expression of local cluster in the second dimension: (local server index，local cluster index in local server)
        self.partition = np.zeros((self.usernum, self.usernum * 2))
        self.partition.fill(-1)  # initialization
        self.communicate_cost = 0
        # the initial partition
        for i in range(0, L * 2, 2):
            self.partition[0][i] = i / 2
            self.partition[0][i + 1] = 0

        # only one cluster in the beginning
        self.clusters[0] = Base.CDP_Cluster(b=np.zeros(self.d), t=0, V=np.zeros((self.d, self.d)), users_begin=0,
                                            d=self.d, user_num=self.usernum, rounds=self.rounds, users={},
                                            rewards=np.zeros(self.rounds), best_rewards=np.zeros(self.rounds),
                                            l_server_index=-1, index=0)
        self.cluster_inds = np.zeros(n, np.int64)  # index of the global cluster to which each user belongs, value: user index
        self.l_server_inds = np.zeros(n, np.int64)  # index of the local server to which each user belongs

        user_index = 0
        j = 0  # the local server index
        # initialize local server
        for i in userList:  # userList records the number of users in each local server
            self.l_server_list.append(Local_server(i, d, user_index, self.rounds, server_index=j))
            self.cluster_usernum[j] = i
            self.cluster_inds[user_index:user_index + i] = 0
            self.l_server_inds[user_index:user_index + i] = j
            user_index = user_index + i
            j = j + 1

    # Locate the local server and global cluster
    def locate_user_index(self, user_index):
        l_server_index = self.l_server_inds[user_index]
        g_cluster_index = self.cluster_inds[user_index]
        return l_server_index, g_cluster_index

    # communicate between global server and local server and update the partition
    def communicate(self):
        g_cluster_index = 0
        tmp_partition = np.zeros((self.usernum, self.usernum * 2))
        tmp_partition.fill(-1)
        for i in range(self.l_server_num):
            l_server = self.l_server_list[i]
            for cluster_index in l_server.clusters:    # for convenience, upload all local clusters and remerge
                self.clusters[g_cluster_index] = copy.deepcopy(l_server.clusters[cluster_index])
                self.clusters[g_cluster_index].S += l_server.clusters[cluster_index].H_now
                self.clusters[g_cluster_index].u += l_server.clusters[cluster_index].h_now
                tmp_partition[g_cluster_index][0] = l_server.clusters[cluster_index].l_server_index
                tmp_partition[g_cluster_index][1] = l_server.clusters[cluster_index].index

                for user in l_server.cluster_inds:
                    if l_server.cluster_inds[user] == cluster_index:
                        self.cluster_inds[user] = g_cluster_index
                self.cluster_usernum[g_cluster_index] = l_server.clusters[cluster_index].user_num
                g_cluster_index += 1

        self.partition = tmp_partition

    # determine whether the two clusters need to merge or not
    def if_merge(self, cluster_id1, cluster_id2):
        cluster1 = self.clusters[cluster_id1]
        cluster2 = self.clusters[cluster_id2]
        t1 = cluster1.t
        t2 = cluster2.t
        fact_T1 = np.sqrt((1 + np.log(1 + t1)) / (1 + t1))
        fact_T2 = np.sqrt((1 + np.log(1 + t2)) / (1 + t2))
        theta1 = cluster1.theta
        theta2 = cluster2.theta
        if (np.linalg.norm(theta1 - theta2) < alpha2 * (fact_T1 + fact_T2)):
            return True
        else:
            return False

    # merge the global clusters and update the information
    def merge(self, former_partition):
        done_change = False   # if no change, don't need to update the partition
        cluster_node = list(self.clusters.keys())
        cluster_G = nx.complete_graph(cluster_node)   # all global clusters generate a complete graph
        nodes = cluster_G.nodes()   # all global clusters
        for c1 in nodes:
            if c1 not in self.clusters:
                continue
            A = [a for a in cluster_G.neighbors(c1)]
            for c2 in A:
                if c2 not in self.clusters:
                    continue
                if not self.if_merge(c1, c2):
                    cluster_G.remove_edge(c1, c2)   # remove the edge if two clusters can't merge together
                    done_change = True

        # if the structure of global cluster has changed, update the info of global cluster
        if done_change and (former_partition != self.partition).any():
            for cluster_set in nx.connected_components(cluster_G):
                global_l_cluster_num = 1
                cluster_list = list(cluster_set)
                c1 = cluster_list[0]    # choose a cluster, other clusters merge to it
                # after merge, update the global clusters' information and the partition
                for i in cluster_list[1:]:
                    self.clusters[c1].S += self.clusters[i].S
                    self.clusters[c1].u += self.clusters[i].u
                    self.clusters[c1].t += self.clusters[i].t
                    self.clusters[c1].user_num += self.clusters[i].user_num
                    self.cluster_usernum[c1] += self.cluster_usernum[i]
                    self.partition[c1][global_l_cluster_num * 2] = self.clusters[i].l_server_index
                    self.partition[c1][global_l_cluster_num * 2 + 1] = self.clusters[i].index
                    self.partition[i][0] = -1
                    self.partition[i][1] = -1
                    global_l_cluster_num += 1
                    for j in range(self.usernum):
                        if self.cluster_inds[j] == i:
                            self.cluster_inds[j] = cluster_list[0]
                    for user in self.clusters[i].users:
                        self.clusters[cluster_list[0]].users.setdefault(user, self.clusters[i].users[user])
                    del self.clusters[i]

                # recompute theta
                self.clusters[c1].theta = np.matmul(np.linalg.inv(np.eye(self.d) + self.clusters[c1].S),
                                                    self.clusters[c1].u)

    # phase-based cluster detection and adjustment
    def detection(self, t):
        # Upload the local clustered information to the global server, should have some time cost
        for l_index in range(len(self.l_server_list)):
            check_server = self.l_server_list[l_index]
            for i in check_server.cluster_inds:
                user1_index = i
                check_server.check_update(user1_index)  #
            check_server.update_cluster(t)

        # Upload the local clustered information to the global server
        former_partition = self.partition
        self.communicate()     # update global clusters by uploading local clusters' information
        self.merge(former_partition)    # merge the global clusters
        if (former_partition != self.partition).any():
            self.communicate_cost += 1     # update the cumulative communication cost

            # Renew the cluster information and create new upload/download buffers
            for g_cluster_id in self.clusters:
                self.clusters[g_cluster_id].S = np.zeros((self.d, self.d))
                self.clusters[g_cluster_id].u = np.zeros(self.d)
                self.clusters[g_cluster_id].t = 0
                l_cluster_info = self.partition[g_cluster_id]
                for i in range(0, self.usernum * 2, 2):
                    l_server_id = l_cluster_info[i].astype(np.int)
                    l_cluster_id = l_cluster_info[i + 1].astype(np.int)
                    if l_cluster_id == -1 or l_server_id == -1:
                        continue
                    l_server = self.l_server_list[l_server_id]
                    l_cluster = l_server.clusters[l_cluster_id]
                    self.clusters[g_cluster_id].S += l_cluster.S
                    self.clusters[g_cluster_id].u += l_cluster.u
                    self.clusters[g_cluster_id].t += l_cluster.t

                self.clusters[g_cluster_id].theta = np.matmul(
                    np.linalg.inv(np.eye(self.d) + self.clusters[g_cluster_id].S), self.clusters[g_cluster_id].u)

                # update local cluster's information using global cluster's information
                for i in range(0, self.usernum * 2, 2):
                    l_server_id = l_cluster_info[i].astype(np.int)
                    l_cluster_id = l_cluster_info[i + 1].astype(np.int)
                    if l_cluster_id == -1 or l_server_id == -1:
                        continue
                    l_server = self.l_server_list[l_server_id]
                    l_cluster = l_server.clusters[l_cluster_id]
                    l_cluster.S = self.clusters[g_cluster_id].S
                    l_cluster.u = self.clusters[g_cluster_id].u
                    l_cluster.t = self.clusters[g_cluster_id].t
                    l_cluster.theta = np.matmul(
                        np.linalg.inv(np.eye(self.d) + l_cluster.S), l_cluster.u)

                    # create new buffers
                    l_cluster.S_up = np.zeros((self.d, self.d))
                    l_cluster.S_down = np.zeros((self.d, self.d))
                    l_cluster.u_up = np.zeros(self.d)
                    l_cluster.u_down = np.zeros(self.d)
                    l_cluster.T_up = 0
                    l_cluster.T_down = 0

                    # create new perturbation
                    l_cluster.privatizer(t, delt, epsi)

    # determine global cluster according to the local cluster
    def find_global_cluster(self, l_server_id, l_cluster_id):
        for g_cluster_id in self.clusters:
            g_cluster_want = g_cluster_id
            l_cluster_info = self.partition[g_cluster_id]
            for i in range(0, self.usernum * 2, 2):
                l_server_id_tmp = l_cluster_info[i]
                l_cluster_id_tmp = l_cluster_info[i + 1]
                if l_server_id_tmp == l_server_id and l_cluster_id_tmp == l_cluster_id:
                    return g_cluster_want

        return -1   # not found

    # Check upload event
    def check_upload(self, l_server_id, l_cluster_id, t):
        l_server = self.l_server_list[l_server_id.astype(np.int)]
        l_cluster = l_server.clusters[l_cluster_id]
        S = l_cluster.S
        S_up = l_cluster.S_up
        H = l_cluster.H_now - l_cluster.H_former
        if np.linalg.det(S) != 0 and np.linalg.det(S + S_up + H) / np.linalg.det(S) >= U:    # Check whether the upload conditions are met
            self.communicate_cost += 1    # update the cumulative communication cost
            g_cluster_id = self.find_global_cluster(l_server_id, l_cluster_id)

            # update global information
            if g_cluster_id != -1:
                self.clusters[g_cluster_id].S += S_up
                self.clusters[g_cluster_id].u += l_cluster.u_up
                self.clusters[g_cluster_id].t += l_cluster.T_up
                self.clusters[g_cluster_id].theta = np.matmul(
                    np.linalg.inv(np.eye(self.d) + self.clusters[g_cluster_id].S), self.clusters[g_cluster_id].u)

                # global server updates other servers' download buffer
                l_cluster_info = self.partition[g_cluster_id]   # all local clusters that generate this global cluster
                for i in range(0, self.usernum * 2, 2):
                    l_server_id_other = l_cluster_info[i]
                    l_cluster_id_other = l_cluster_info[i + 1]
                    if l_server_id_other == l_server_id and l_cluster_id_other == l_cluster_id:
                        continue
                    if l_server_id_other == -1 or l_cluster_id_other == -1:
                        continue
                    l_server_other = self.l_server_list[l_server_id_other.astype(np.int)]
                    l_cluster_other = l_server_other.clusters[l_cluster_id_other.astype(np.int)]
                    l_cluster_other.S_down += S_up
                    l_cluster_other.u_down += l_cluster.u_up
                    l_cluster_other.T_down += l_cluster.T_up

                # Local cluster cleans the buffer
                l_cluster.S += l_cluster.S_up
                l_cluster.u += l_cluster.u_up
                l_cluster.t += l_cluster.T_up
                l_cluster.theta = np.matmul(np.linalg.inv(np.eye(self.d) + l_cluster.S), l_cluster.u)
                l_cluster.S_up = np.zeros((self.d, self.d))
                l_cluster.u_up = np.zeros(self.d)
                l_cluster.T_up = 0
                l_cluster.H_former = l_cluster.H_now
                l_cluster.h_former = l_cluster.h_now

                # create new perturbation
                l_cluster.privatizer(t, delt, epsi)

    # Check download event
    def check_download(self, g_cluster_id):
        l_cluster_info = self.partition[g_cluster_id]   # all local clusters that generate this global cluster
        V_g = self.clusters[g_cluster_id].S
        for i in range(0, self.usernum * 2, 2):
            l_server_id = l_cluster_info[i]
            l_cluster_id = l_cluster_info[i + 1]
            if l_server_id == -1 or l_cluster_id == -1:
                continue
            l_server = self.l_server_list[l_server_id.astype(np.int)]
            l_cluster = l_server.clusters[l_cluster_id.astype(np.int)]
            if np.linalg.det(l_cluster.S) != 0 and np.linalg.det(V_g) / np.linalg.det(l_cluster.S) >= D:    # # Check whether the download conditions are met
                self.communicate_cost += 1      # update the cumulative communication cost

                # update local cluster's information
                l_cluster.S += l_cluster.S_down
                l_cluster.u += l_cluster.u_down
                l_cluster.t += l_cluster.T_down
                l_cluster.theta = np.matmul(np.linalg.inv(np.eye(self.d) + l_cluster.S), l_cluster.u)

                # clean the download buffer
                l_cluster.S_down = np.zeros((self.d, self.d))
                l_cluster.u_down = np.zeros(self.d)
                l_cluster.T_down = 0

    # Phase-based FCLUB with CDP
    def run(self, envir, phase, number, all_round):
        result_final = list()   # to save the users' final theta information
        communication_cost = list()    # to save the cumulative communication cost
        for s in range(1, phase + 1):
            # detect and adjust clusters
            self.detection(phase_cardinality ** s - 1)
            for i in range(1, phase_cardinality ** s + 1):
                # compute the total time step
                t = (phase_cardinality ** s - 1) // (phase_cardinality - 1) + i - 1
                if t >= all_round:
                    break
                user_all = envir.generate_users()     # random user arrives
                user_index = user_all[0]
                l_server_index, g_cluster_index = self.locate_user_index(user_index)
                l_server = self.l_server_list[l_server_index]
                l_cluster_index = l_server.locate_user_index(user_index)
                l_cluster = l_server.clusters[l_cluster_index]
                # the context set
                items = envir.get_items()
                r_item_index = l_server.recommend(l_cluster_index=l_cluster_index, items=items, L_num=self.l_server_num)
                x = items[r_item_index]
                # receive the feedback and update the user's information
                self.reward[t - 1], y, self.best_reward[t - 1], ksi_noise, B_noise = envir.feedback_Local(items=items,
                                                                                                          i=user_index,
                                                                                                          k=r_item_index,
                                                                                                          d=self.d)
                l_cluster.users[user_index].store_info(x, y, t - 1, self.reward[t - 1], self.best_reward[t - 1], ksi_noise[0], B_noise)
                l_cluster.store_info(x, y, t - 1, self.reward[t - 1], self.best_reward[t - 1])
                # check upload
                self.check_upload(l_server_index, l_cluster_index, phase_cardinality ** s - 1)
                # check download
                self.check_download(g_cluster_index)
                # calculate regret
                self.regret[t - 1] = self.best_reward[t - 1] - self.reward[t - 1]
                communication_cost.append(self.communicate_cost)

                if t % all_round == 0:
                    theta_exp = dict()  # to save all users' theta
                    for cluster_index in self.clusters:
                        now_clus = self.clusters[cluster_index]
                        for user in now_clus.users:
                            theta_exp[now_clus.users[user].index] = now_clus.users[user].theta

                    result = dict(sorted(theta_exp.items(), key=lambda k: k[0]))
                    result_final = list(result.values())

        return self.regret, result_final, self.reward, communication_cost
