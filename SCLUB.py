# -*- coding: utf-8 -*-
# final version

import numpy as np

import Base
import Environment as Envi
from Environment import sigma

S = 1
phase_cardinality = 2
alpha_p = 1
alpha = 1.5


class Local_server:
    def __init__(self, nl, d, begin_num, T):
        self.nl = nl    # the number of users in a server
        self.d = d  # dimension
        self.rounds = T  # the number of all rounds
        self.clusters = {
            0: Base.sclub_Cluster(b=np.zeros(d), t=0, V=np.zeros((d, d)), users_begin=begin_num, d=d, user_num=nl,
                                  rounds=self.rounds, rewards=np.zeros(self.rounds), T_phase=0,
                                  theta_phase=np.zeros(self.d),
                                  best_rewards=np.zeros(self.rounds))} # Initialize the cluster, there is only one at the beginning
        self.T_phase = 0
        self.theta = np.zeros(self.d)
        self.init_each_stage()
        self.cluster_inds = dict()
        self.begin_num = begin_num
        for i in range(begin_num, begin_num + nl):
            self.cluster_inds[i] = 0  # Record the index of the cluster to which each user belongs, key:user_index, value:cluster_index
        self.num_clusters = np.zeros(self.rounds, np.int64)  # the total number of clusters in each round , which recorded for a total of `round` times
        self.num_clusters[0] = 1

    # preparation per stage
    def init_each_stage(self):
        # mark every user unchecked for each cluster, update T and theta
        for i in self.clusters:
            cluster = self.clusters[i]
            cluster.checks = {j: False for j in cluster.users}
            cluster.checked = False
            cluster.phase_update()

    # compute the cluster's frequency
    def cluster_aver_freq(self, c, t):
        if len(self.clusters[c].users) == 0:
            return 0
        return self.clusters[c].t / (len(self.clusters[c].users) * t)

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

    # check whether the user should be split from the cluster
    def if_split(self, user_index1, cluster, t):
        t1 = cluster.users[user_index1].t
        t2 = cluster.T_phase
        fact_T1 = np.sqrt((1 + np.log(1 + t1)) / (1 + t1))
        fact_T2 = np.sqrt((1 + np.log(1 + t2)) / (1 + t2))
        fact_t = np.sqrt((1 + np.log(1 + t)) / (1 + t))
        theta1 = cluster.users[user_index1].theta
        theta2 = cluster.theta_phase
        if np.linalg.norm(theta1 - theta2) > alpha * (fact_T1 + fact_T2):
            return True

        p1 = t1 / t
        for user_index2 in cluster.users:
            if user_index2 == user_index1:
                continue
            p2 = cluster.users[user_index2].t / t
            if np.abs(p1 - p2) > alpha_p * 2 * fact_t:
                return True

        return False

    # whether the two checked cluster should be merged together
    def if_merge(self, c1, c2, t):
        t1 = self.clusters[c1].t
        t2 = self.clusters[c2].t
        fact_T1 = np.sqrt((1 + np.log(1 + t1)) / (1 + t1))
        fact_T2 = np.sqrt((1 + np.log(1 + t2)) / (1 + t2))
        fact_t = np.sqrt((1 + np.log(1 + t)) / (1 + t))
        theta1 = self.clusters[c1].theta
        theta2 = self.clusters[c2].theta
        p1 = self.cluster_aver_freq(c1, t)
        p2 = self.cluster_aver_freq(c2, t)

        if np.linalg.norm(theta1 - theta2) >= (alpha / 2) * (fact_T1 + fact_T2):
            return False
        if np.abs(p1 - p2) >= alpha_p * fact_t:
            return False

        return True

    # find the available index for the new cluster
    def find_available_index(self):
        cmax = max(self.clusters)
        for c1 in range(cmax + 1):
            if c1 not in self.clusters:
                return c1
        return cmax + 1

    # split user from the cluster
    def update(self, user_index, t):
        c = self.cluster_inds[user_index]   # Find the cluster to which the updated user belongs
        cluster = self.clusters[c]
        cluster.update_check(user_index)
        now_user = cluster.users[user_index]    # all users in this cluster
        if self.if_split(user_index, cluster, t):
            # form a new cluster
            cnew = self.find_available_index()
            tmp_cluster = Base.sclub_Cluster(b=now_user.b, t=now_user.t, V=now_user.V, users_begin=user_index, d=self.d,
                                             user_num=1,
                                             rounds=self.rounds, users={user_index: now_user}, rewards=now_user.rewards,
                                             best_rewards=now_user.best_rewards, T_phase=cluster.T_phase,
                                             theta_phase=cluster.theta_phase)
            self.clusters[cnew] = tmp_cluster
            self.cluster_inds[user_index] = cnew

            del cluster.users[user_index]
            cluster.V = cluster.V - now_user.V
            cluster.b = cluster.b - now_user.b
            cluster.t = cluster.t - now_user.t
            del cluster.checks[user_index]

        self.num_clusters[t - 1] = len(self.clusters)

    # merge two clusters
    def merge(self, t):
        cmax = max(self.clusters)
        for c1 in range(cmax - 1):
            if c1 not in self.clusters or self.clusters[c1].checked == False:
                continue
            for c2 in range(c1 + 1, cmax):
                if c2 not in self.clusters or self.clusters[c2].checked == False:
                    continue
                if not self.if_merge(c1, c2, t):
                    continue
                else:
                    # update the cluster's information
                    for i in self.clusters[c2].users:
                        self.cluster_inds[i] = c1

                    self.clusters[c1].V = self.clusters[c1].V + self.clusters[c2].V
                    self.clusters[c1].b = self.clusters[c1].b + self.clusters[c2].b
                    self.clusters[c1].t = self.clusters[c1].t + self.clusters[c2].t
                    self.clusters[c1].theta = np.matmul(np.linalg.inv(np.eye(self.d) + self.clusters[c1].V),
                                                        self.clusters[c1].b)
                    for user in self.clusters[c2].users:
                        self.clusters[c1].users.setdefault(user, self.clusters[c2].users[user])
                    self.clusters[c1].checks = {**self.clusters[c1].checks, **self.clusters[c2].checks}
                    del self.clusters[c2]

        self.num_clusters[t - 1] = len(self.clusters)


# Actually there is no global cluster and global server in CLUB, We use global server here for the sake of interface consistency.
# It only serves as a learning agent.
class Global_server:
    def __init__(self, L, n, userList, d, T):
        self.l_server_list = []
        self.usernum = n    # the total number of users
        self.rounds = T
        self.l_server_num = L   # the number of local server
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

    # SCLUB
    def run(self, envir, phase, number):
        result_final = dict()    # to save the users' final theta information
        for s in range(1, phase + 1):
            for l_server in self.l_server_list:
                l_server.init_each_stage()

            for i in range(1, phase_cardinality ** s + 1):
                t = (phase_cardinality ** s - 1) // (phase_cardinality - 1) + i - 1
                if t % 5000 == 0:
                    print(t)
                user_all = envir.generate_users()   # random user arrives
                user_index = user_all[0]
                l_server_index = self.locate_user_index(user_index)
                l_server = self.l_server_list[l_server_index]
                l_cluster_index = l_server.locate_user_index(user_index)
                l_cluster = l_server.clusters[l_cluster_index]
                # the context set
                items = envir.get_items()
                r_item_index = l_server.recommend(l_cluster_index=l_cluster_index, items=items)
                x = items[r_item_index]
                # get feedback
                self.reward[t - 1], y, self.best_reward[t - 1], ksi_noise, B_noise = envir.feedback_Local(items=items,
                                                                                                          i=user_index,
                                                                                                          k=r_item_index,
                                                                                                          d=self.d)
                # update the user's information
                l_cluster.users[user_index].store_info(x, y, t - 1, self.reward[t - 1], self.best_reward[t - 1],
                                                       ksi_noise[0], B_noise)
                # update the local cluster's information
                l_cluster.store_info(x, y, t - 1, self.reward[t - 1], self.best_reward[t - 1], ksi_noise[0], B_noise)
                # check update
                l_server.update(user_index, t)
                # check merge
                l_server.merge(t)
                self.regret[t - 1] = self.best_reward[t - 1] - self.reward[t - 1]

                # get all users' theta
                cluster_num = 0
                if t % 100000 == 0:
                    theta_exp = dict()
                    for server in self.l_server_list:
                        cluster_num += len(server.clusters)
                        for clst in server.clusters:
                            now_clus = server.clusters[clst]
                            for user in now_clus.users:
                                theta_exp[now_clus.users[user].index] = now_clus.users[user].theta
                        result_final = dict(sorted(theta_exp.items(), key=lambda k: k[0]))

        return self.regret, result_final, self.reward, cluster_num
