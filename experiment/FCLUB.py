# -*- coding: utf-8 -*-
# final version


import networkx as nx
import numpy as np
import Base
import Environment as Envi
import copy
from Environment import alpha, sigma, alpha2

S = 1


class Local_server:
    def __init__(self, nl, d, begin_num, T, edge_probability=0.8):
        self.nl = nl    # the number of users in a server
        self.d = d     # dimension
        self.rounds = T  # the number of all rounds
        user_index_list = list(range(begin_num, begin_num + nl))    # the index of users in this server
        self.G = nx.generators.classic.complete_graph(user_index_list)   # Generate undirected complete graph，user indexes range from begin_num to begin_num + nl
        self.clusters = {
            0: Base.Cluster(b=np.zeros(d), t=0, V=np.zeros((d, d)), users_begin=begin_num, d=d, user_num=nl,
                            rounds=self.rounds, rewards=np.zeros(self.rounds),
                            best_rewards=np.zeros(self.rounds))}
        self.cluster_inds = dict()  # Record the index of the cluster to which each user belongs, key:user_index, value:cluster_index
        self.begin_num = begin_num  # the beginning of the users' index in this server
        for i in range(begin_num, begin_num + nl):
            self.cluster_inds[i] = 0   # index of the cluster to which each user belongs, key:user_index ,value:cluster_index
        self.num_clusters = np.zeros(self.rounds, np.int64)  # the total number of clusters in each round , which recorded for a total of `round` times
        self.num_clusters[0] = 1    # only one cluster in the beginning

    # calculate the item to recommend
    def recommend(self, M, b, beta, items):
        Minv = np.linalg.inv(M)
        theta = np.dot(Minv, b)
        r_item_index = np.argmax(np.dot(items, theta) + beta * (np.matmul(items, Minv) * items).sum(axis=1))
        return r_item_index

    # Judge whether the edge between the two users in this cluster needs to be deleted
    def if_delete(self, user_index1, user_index2, cluster):
        t1 = cluster.users[user_index1].t
        t2 = cluster.users[user_index2].t
        fact_T1 = np.sqrt((1 + np.log(1 + t1)) / (1 + t1))
        fact_T2 = np.sqrt((1 + np.log(1 + t2)) / (1 + t2))
        theta1 = cluster.users[user_index1].theta
        theta2 = cluster.users[user_index2].theta
        return np.linalg.norm(theta1 - theta2) > alpha * (fact_T1 + fact_T2)

    # update cluster due to edge deleting
    def update(self, user_index, t):
        update_cluster = False
        c = self.cluster_inds[user_index]  # Find the local cluster to which the updated user belongs
        i = user_index
        # store the origin cluster
        origin_cluster = self.clusters[c]
        A = [a for a in self.G.neighbors(i)]    # find the connected-component of this user
        for j in A:
            user2_index = j
            c2 = self.cluster_inds[user2_index]
            user1 = self.clusters[c].users[i]
            user2 = self.clusters[c2].users[user2_index]
            if user1.t != 0 and user2.t != 0 and self.if_delete(i, user2_index, self.clusters[c]):
                self.G.remove_edge(i, j)    # delete the edge if the user should split
                update_cluster = True

        # may split the cluster
        if update_cluster:
            C = nx.node_connected_component(self.G, i)     # current user in the updated cluster
            remain_users = dict()     # user waiting to be assigned to a new cluster
            for m in C:
                remain_users[m] = self.clusters[c].get_user(m)

            if len(C) < len(self.clusters[c].users):    # if the cluster  has been split
                all_users_index = set(self.clusters[c].users)
                all_users = dict()  # all users in the origin cluster
                for user_index_all in all_users_index:
                    all_users[user_index_all] = self.clusters[c].get_user(user_index_all)
                # generate new cluster
                tmp_cluster = Base.Cluster(b=sum([remain_users[k].b for k in remain_users]),
                                           t=sum([remain_users[k].t for k in remain_users]),
                                           V=sum([remain_users[k].V for k in remain_users]),
                                           users_begin=min(remain_users), d=self.d, user_num=len(remain_users),
                                           rounds=self.rounds,
                                           users=copy.deepcopy(remain_users),
                                           rewards=sum([remain_users[k].rewards for k in remain_users]),
                                           best_rewards=sum([remain_users[k].best_rewards for k in remain_users]))
                self.clusters[c] = tmp_cluster

                # Remove the users constituting the new cluster from the origin cluster's userlist
                for user_index3 in all_users_index:
                    if remain_users.__contains__(user_index3):
                        all_users.pop(user_index3)

                c = max(self.clusters) + 1
                while len(all_users) > 0:   # some users haven't been put into cluster
                    j = np.random.choice(list(all_users))
                    C = nx.node_connected_component(self.G, j)
                    new_cluster_users = dict()
                    for k in C:
                        new_cluster_users[k] = origin_cluster.get_user(k)
                    self.clusters[c] = Base.Cluster(b=sum([new_cluster_users[n].b for n in new_cluster_users]),
                                                    t=sum([new_cluster_users[n].t for n in new_cluster_users]),
                                                    V=sum([new_cluster_users[n].V for n in new_cluster_users]),
                                                    users_begin=min(new_cluster_users), d=self.d,
                                                    user_num=len(new_cluster_users),
                                                    rounds=self.rounds, users=copy.deepcopy(new_cluster_users),
                                                    rewards=sum(
                                                        [new_cluster_users[k].rewards for k in new_cluster_users]),
                                                    best_rewards=sum(
                                                        [new_cluster_users[k].best_rewards for k in new_cluster_users]))
                    for k in C:
                        self.cluster_inds[k] = c

                    c += 1
                    for user_index in all_users_index:
                        if new_cluster_users.__contains__(user_index):
                            all_users.pop(user_index)

        self.num_clusters[t] = len(self.clusters)


class Global_server:
    def __init__(self, L, n, userList, d, T):
        self.l_server_list = []
        self.usernum = n    # the total number of users
        self.rounds = T
        self.l_server_num = L   # the number of local server
        self.d = d
        self.cluster_usernum = np.zeros(L * n, np.int64)  # Record the number of users in each global cluster in each round
        self.clusters = dict()  # global cluster
        self.regret = np.zeros(self.rounds)
        self.reward = np.zeros(self.rounds)
        self.best_reward = np.zeros(self.rounds)
        user_begin = 0     # the first user's index in a cluster
        # initialize global cluster, there are L cluster at first, corresponding to L local server
        for i in range(L):
            self.clusters[i] = Base.Cluster(b=np.zeros(self.d), t=0, V=np.zeros((d, d)), users_begin=user_begin,
                                            d=self.d, user_num=userList[i], rounds=self.rounds, users={},
                                            rewards=np.zeros(self.rounds), best_rewards=np.zeros(
                    self.rounds))
            user_begin += userList[i]
        self.cluster_inds = np.zeros(n, np.int64)  # index of the global cluster to which each user belongs, value: user index
        self.l_server_inds = np.zeros(n,
                                      np.int64)  # index of the local server to which each user belongs
        # initialize local server
        user_index = 0     # the first user's index in the local server
        j = 0   # the local server index
        for i in userList:  # userList records the number of users in each local server
            self.l_server_list.append(Local_server(i, d, user_index, self.rounds))
            self.cluster_usernum[j] = i
            self.cluster_inds[user_index:user_index + i] = j
            self.l_server_inds[user_index:user_index + i] = j
            user_index = user_index + i
            j = j + 1

    # Locate the local server and global cluster
    def locate_user_index(self, user_index):
        # which local server
        l_server_index = self.l_server_inds[user_index]
        # which global cluster
        g_cluster_index = self.cluster_inds[user_index]
        return l_server_index, g_cluster_index

    # get the global cluster's information for recommendation
    def global_info(self, g_cluster_index):
        g_cluster = self.clusters[g_cluster_index]
        V = g_cluster.V
        b = g_cluster.b
        T = g_cluster.t
        gamma_t = Envi.gamma(T + 1, self.d, alpha, sigma)
        lambda_t = gamma_t * 2
        # S=L=1
        beta_t = Envi.beta(sigma, alpha, gamma_t, S, self.d, T + 1, self.l_server_num)
        M_t = np.eye(self.d) * np.float_(lambda_t) + V
        return M_t, b, beta_t

    # communicate between local server and global server
    def communicate(self):
        g_cluster_index = 0
        for i in range(self.l_server_num):
            l_server = self.l_server_list[i]
            for cluster_index in l_server.clusters:
                self.clusters[g_cluster_index] = copy.deepcopy(l_server.clusters[cluster_index])
                for user in l_server.cluster_inds:
                    if l_server.cluster_inds[user] == cluster_index:
                        self.cluster_inds[user] = g_cluster_index
                self.cluster_usernum[g_cluster_index] = l_server.clusters[cluster_index].user_num
                g_cluster_index += 1

    # merge two global clusters if they are close enough
    def merge(self):
        cmax = max(self.clusters)
        for c1 in range(cmax - 1):
            if c1 not in self.clusters:
                continue
            for c2 in range(c1 + 1, cmax):
                if c2 not in self.clusters:
                    continue
                T1 = self.clusters[c1].t
                T2 = self.clusters[c2].t
                fact_T1 = np.sqrt((1 + np.log(1 + T1)) / (1 + T1))
                fact_T2 = np.sqrt((1 + np.log(1 + T2)) / (1 + T2))
                theta1 = self.clusters[c1].theta
                theta2 = self.clusters[c2].theta
                # judge if two cluster should be merged
                if (np.linalg.norm(theta1 - theta2) >= alpha2 * (fact_T1 + fact_T2)):
                    continue
                else:
                    # merge two clusters and update the cluster's information
                    for i in range(self.usernum):
                        if self.cluster_inds[i] == c2:
                            self.cluster_inds[i] = c1

                    self.cluster_usernum[c1] = self.cluster_usernum[c1] + self.cluster_usernum[c2]
                    self.cluster_usernum[c2] = 0

                    self.clusters[c1].V = self.clusters[c1].V + self.clusters[c2].V
                    self.clusters[c1].b = self.clusters[c1].b + self.clusters[c2].b
                    self.clusters[c1].t = self.clusters[c1].t + self.clusters[c2].t
                    self.clusters[c1].user_num = self.clusters[c1].user_num + self.clusters[c2].user_num
                    # compute theta
                    self.clusters[c1].theta = np.matmul(np.linalg.inv(np.eye(self.d) + self.clusters[c1].V),
                                                        self.clusters[c1].b)
                    for user in self.clusters[c2].users:
                        self.clusters[c1].users.setdefault(user, self.clusters[c2].users[user])
                    del self.clusters[c2]

    # FCLUB with full communication and no differential privacy
    def run(self, envir, T, number, user_num):    # T: all_round
        theta_exp = dict()
        y_list = list()    # to save feedback in each round
        x_list = list()    # to save the recommended item in each round
        result_final = dict()     # to save the users' final theta information
        communication_cost = list()    # the cumulative communication cost
        for i in range(1, T + 1):
            if i % 5000 == 0:
                print(i)
            user_all = envir.generate_users()    # random user arrives
            user_index = user_all[0]
            l_server_index, g_cluster_index = self.locate_user_index(user_index)
            M_t, b, beta_t = self.global_info(g_cluster_index)
            l_server = self.l_server_list[l_server_index]
            g_cluster = self.clusters[g_cluster_index]
            l_cluster = l_server.clusters[l_server.cluster_inds[user_index]]
            # the context set
            items = envir.get_items()
            r_item_index = l_server.recommend(M_t, b, beta_t, items)
            x = items[r_item_index]
            x_list.append(x)
            # receive the feedback and update the user's information
            self.reward[i - 1], y, self.best_reward[i - 1], ksi_noise, B_noise = envir.feedback(items, user_index, b,
                                                                                                M_t, r_item_index,
                                                                                                self.d)
            y_list.append(y)
            l_cluster.users[user_index].store_info(x, y, i - 1, self.reward[i - 1], self.best_reward[i - 1],
                                                   ksi_noise[0], B_noise)
            l_cluster.store_info(x, y, i - 1, self.reward[i - 1], self.best_reward[i - 1], ksi_noise[0], B_noise)
            # delete edge and aggregated information
            l_server.update(user_index, i - 1)
            # update the global information
            g_cluster.store_info(x, y, i - 1, self.reward[i - 1], self.best_reward[i - 1], ksi_noise[0], B_noise)
            self.communicate()
            self.merge()
            self.regret[i - 1] = self.best_reward[i - 1] - self.reward[i - 1]
            communication_cost.append(i)

            if i % T == 0:
                for clst in self.clusters:
                    now_clus = self.clusters[clst]
                    for user in now_clus.users:
                        theta_exp[now_clus.users[user].index] = now_clus.users[user].theta
                result_final = dict(sorted(theta_exp.items(), key=lambda k: k[0]))

        return self.regret, result_final, self.reward, x_list, y_list, communication_cost
