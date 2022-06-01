# -*- coding: utf-8 -*-
# final version


import networkx as nx
import numpy as np

import Base
import Environment as Envi
import copy
from Environment import sigma

S = 1
alpha = 0.25  # parameter for edge deletion


class Local_server:
    def __init__(self, nl, d, begin_num, T):
        self.nl = nl    # the number of users in a server
        self.d = d      # dimension
        self.rounds = T  # the number of all rounds
        user_index_list = list(range(begin_num, begin_num + nl))     # the index of users in this server
        self.G = nx.generators.classic.complete_graph(user_index_list)  # Generate undirected complete graph，user indexes range from begin_num to begin_num + nl
        self.clusters = {
            0: Base.Cluster(b=np.zeros(d), t=0, V=np.zeros((d, d)), users_begin=begin_num, d=d, user_num=nl,
                            rounds=self.rounds, rewards=np.zeros(self.rounds),
                            best_rewards=np.zeros(self.rounds))}  # Initialize the cluster, there is only one at the beginning
        self.cluster_inds = dict()    # Record the index of the cluster to which each user belongs, key:user_index, value:cluster_index
        self.begin_num = begin_num
        for i in range(begin_num, begin_num + nl):
            self.cluster_inds[i] = 0
        self.num_clusters = np.zeros(self.rounds, np.int64)   # the total number of clusters in each round , which recorded for a total of `round` times
        self.num_clusters[0] = 1

    # Determine which local cluster the user belongs to
    def locate_user_index(self, user_index):
        l_cluster_index = self.cluster_inds[user_index]
        return l_cluster_index

    # decide which items should be recommended at present
    def recommend(self, l_cluster_index, items):
        cluster = self.clusters[l_cluster_index]
        V_t, b_t, T_t = cluster.get_info()
        gamma_t = Envi.gamma(T_t, self.d, alpha, sigma)
        lambda_t = gamma_t * 2
        M_t = np.eye(self.d) * np.float_(lambda_t) + V_t
        # S = 1
        beta_t = Envi.beta(sigma, alpha, gamma_t, S, self.d, T_t)
        Minv = np.linalg.inv(M_t)
        theta = np.dot(Minv, b_t)
        # calculate the best item
        r_item_index = np.argmax(np.dot(items, theta) + beta_t * (np.matmul(items, Minv) * items).sum(axis=1))
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

    # Delete edges in this user's cluster
    def update(self, user_index, t):
        update_cluster = False   # if cluster updates, may split
        c = self.cluster_inds[user_index]  # Find the local cluster to which the updated user belongs
        i = user_index
        origin_cluster = self.clusters[c]
        A = [a for a in self.G.neighbors(i)]
        for j in A:
            user2_index = j
            c2 = self.cluster_inds[user2_index]
            user1 = self.clusters[c].users[i]
            user2 = self.clusters[c2].users[user2_index]
            if user1.t != 0 and user2.t != 0 and self.if_delete(i, user2_index, self.clusters[c]):
                self.G.remove_edge(i, j)    # delete the edge
                update_cluster = True

        # may split the cluster
        if update_cluster:
            C = nx.node_connected_component(self.G, i)  # user in the updated cluster now
            remain_users = dict()   # user waiting to be assigned to a new cluster
            for m in C:
                remain_users[m] = self.clusters[c].get_user(m)

            if len(C) < len(self.clusters[c].users):
                all_users_index = set(self.clusters[c].users)  # all users in the origin cluster
                all_users = dict()
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
                for user_index_used in all_users_index:
                    if remain_users.__contains__(user_index_used):
                        all_users.pop(user_index_used)

                c = max(self.clusters) + 1
                while len(all_users) > 0:  # having users left
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

        self.num_clusters[t] = len(self.clusters)   # update the number of cluster


# Actually there is no global cluster and global server in CLUB, We use global server here for the sake of interface consistency.
# It only serves as a learning agent.
class Global_server:
    def __init__(self, L, n, userList, d, T):
        self.l_server_list = []
        self.usernum = n    # the total number of users
        self.rounds = T
        self.l_server_num = L    # the number of local server
        self.d = d
        self.regret = np.zeros(self.rounds)
        self.reward = np.zeros(self.rounds)
        self.best_reward = np.zeros(self.rounds)
        self.l_server_inds = np.zeros(n, np.int64)  # index of the local server to which each user belongs

        # initialize local server
        user_index = 0  # the first user's index in the local server
        j = 0   # the local server index
        for i in userList:  # userList records the number of users in each local server
            self.l_server_list.append(Local_server(i, d, user_index, self.rounds))
            self.l_server_inds[user_index:user_index + i] = j
            user_index = user_index + i
            j = j + 1

    # Locate the local server
    def locate_user_index(self, user_index):
        l_server_index = self.l_server_inds[user_index]
        return l_server_index

    # CLUB
    def run(self, envir, T, number):
        y_list = list()    # to save feedback in each round
        x_list = list()    # to save the recommended item in each round
        result_final = dict()   # to save the users' final theta information
        for i in range(1, T + 1):
            if i % 5000 == 0:
                print(i)
            user_all = envir.generate_users()   # random user arrives
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
            self.reward[i - 1], y, self.best_reward[i - 1], ksi_noise, B_noise = envir.feedback_Local(items=items,
                                                                                                      i=user_index,
                                                                                                      k=r_item_index,
                                                                                                      d=self.d)
            y_list.append(y)
            # update the user's information
            cluster.users[user_index].store_info(x, y, i - 1, self.reward[i - 1], self.best_reward[i - 1], ksi_noise[0],
                                                 B_noise)
            # update the cluster's information
            cluster.store_info(x, y, i - 1, self.reward[i - 1], self.best_reward[i - 1], ksi_noise[0], B_noise)
            l_server.update(user_index, i - 1)
            self.regret[i - 1] = self.best_reward[i - 1] - self.reward[i - 1]

            # get all users' theta
            cluster_num = 0
            if i % T == 0:
                theta_exp = dict()
                for server in self.l_server_list:
                    cluster_num += len(server.clusters)
                    for clst in server.clusters:
                        now_clus = server.clusters[clst]
                        for user in now_clus.users:
                            theta_exp[now_clus.users[user].index] = now_clus.users[user].theta
                    result_final = dict(sorted(theta_exp.items(), key=lambda k: k[0]))

        return self.regret, result_final, self.reward, x_list, y_list, cluster_num
