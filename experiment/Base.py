# -*- coding: utf-8 -*-
# final version


import numpy as np
import copy


class User:
    def __init__(self, d, user_index, T):
        self.d = d  # dimension
        self.index = user_index  # the user's index, and it's unique
        self.t = 0  # rounds that pick the user
        self.b = np.zeros(self.d)
        self.V = np.zeros((self.d, self.d))
        self.rewards = np.zeros(T)  # T: the total round
        self.best_rewards = np.zeros(T)
        self.theta = np.zeros(d)

    def store_info(self, x, y, t, r, br, ksi_noise, B_noise):
        self.t += 1
        # self.V = self.V + np.outer(x,x) + B_noise
        self.V = self.V + np.outer(x, x)
        # self.b = self.b + y*x + ksi_noise
        self.b = self.b + y * x
        self.rewards[t] += r
        self.best_rewards[t] += br
        # c_t=1
        self.theta = np.matmul(np.linalg.inv(np.eye(self.d) + self.V), self.b)

    def get_info(self):
        return self.V, self.b, self.t


# Base cluster
class Cluster(User):
    def __init__(self, b, V, users_begin, d, user_num, rounds, rewards, best_rewards, users={}, t=0):
        self.d = d
        if not users:  # initialization at the beginning or a split/merged new cluster
            self.users = dict()
            for i in range(users_begin, users_begin + user_num):
                self.users[i] = User(self.d, i, rounds)  # a list/array of users
        else:
            self.users = copy.deepcopy(users)
        self.users_begin = users_begin
        self.user_num = user_num
        self.b = b
        self.t = t  # the current pick round
        self.V = V
        self.rewards = rewards
        self.best_rewards = best_rewards
        self.theta = np.matmul(np.linalg.inv(np.eye(self.d) + self.V), self.b)  # now c_t = 1

    def get_user(self, user_index):
        return self.users[user_index]

    # ksi_noise and B_noise are LDP noise parameter, in our experiment we don't add it
    def store_info(self, x, y, t, r, br, ksi_noise, B_noise):
        # self.V = self.V + np.outer(x, x) + B_noise
        self.V = self.V + np.outer(x, x)
        # self.b = self.b + y * x + ksi_noise
        self.b = self.b + y * x
        self.t += 1
        self.best_rewards[t] += br
        self.rewards[t] += r
        self.theta = np.matmul(np.linalg.inv(np.eye(self.d) + self.V), self.b)

    def get_info(self):
        V_t = self.V
        b_t = self.b
        t = self.t
        return V_t, b_t, t


# cluster delay communication version
class DC_Cluster(User):
    # Base cluster
    def __init__(self, b, V, users_begin, d, user_num, rounds, rewards, best_rewards, l_server_index, index, users={}, t=0):
        self.d = d
        if not users:  # initialization at the beginning or a split/merged new cluster
            self.users = dict()
            for i in range(users_begin, users_begin + user_num):
                self.users[i] = User(self.d, i, rounds)  # a list/array of users
        else:
            self.users = copy.deepcopy(users)
        self.users_begin = users_begin
        self.user_num = user_num
        self.u = b  # np.eye(d)
        self.t = t  # initial 0
        self.l_server_index = l_server_index    # l_server_index: which local server the DC_cluster belongs to
        self.index = index  # index: the cluster's index in the local server
        self.S = V  # synchronized gram matrix

        # upload buffer
        self.S_up = np.zeros((d, d))
        self.u_up = np.zeros(d)
        self.T_up = 0

        # download buffer
        self.S_down = np.zeros((d, d))
        self.u_down = np.zeros(d)
        self.T_down = 0

        self.rewards = rewards
        self.best_rewards = best_rewards

        # assume c_t = 1
        self.theta = np.matmul(np.linalg.inv(np.eye(self.d) + self.S), self.u)

    def get_user(self, user_index):
        return self.users[user_index]

    # update upload buffer
    def store_info(self, x, y, t, r, br, ksi_noise, B_noise):
        # self.V = self.V + np.outer(x, x) + B_noise
        self.S_up += np.outer(x, x)
        # self.b = self.b + y * x + ksi_noise
        self.u_up += y * x
        self.T_up += 1
        self.best_rewards[t] += br
        self.rewards[t] += r
        self.theta = np.matmul(np.linalg.inv(np.eye(self.d) + self.S), self.u)

    def get_info(self):
        V_t = self.S_up + self.S
        b_t = self.u + self.u_up
        t = self.t + self.T_up
        return V_t, b_t, t

# cluster delay communication version with CDP
class CDP_Cluster(DC_Cluster):
    def __init__(self, b, V, users_begin, d, user_num, rounds, rewards, best_rewards, l_server_index, index, users={},
                 t=0):
        super(CDP_Cluster, self).__init__(b, V, users_begin, d, user_num, rounds, rewards, best_rewards, l_server_index,
                                          index, users, t)
        self.H_queue = dict()
        self.h_queue = dict()
        self.serial_num = 1
        self.H_now = np.zeros((self.d, self.d))
        self.h_now = np.zeros(self.d)
        self.H_former = np.zeros((self.d, self.d))
        self.h_former = np.zeros(self.d)

    # update upload buffer
    def store_info(self, x, y, t, r, br):
        self.S_up += np.outer(x, x)
        self.u_up += y * x
        self.T_up += 1
        self.best_rewards[t] += br
        self.rewards[t] += r
        self.theta = np.matmul(np.linalg.inv(np.eye(self.d) + self.S), self.u)

    def countBits(self, n):
        List = list()
        bit_num = 0
        t = n
        while n != 0:
            if n % 2 != 0:
                left = t - 2 ** bit_num + 1
                right = t
                List.append(tuple([left, right]))
                t = left - 1

            n = n // 2
            bit_num += 1

        return List

    def phase_update(self):
        self.H_now = np.zeros((self.d, self.d))
        self.h_now = np.zeros(self.d)
        self.H_former = np.zeros((self.d, self.d))
        self.h_former = np.zeros(self.d)
        self.H_queue.clear()
        self.h_queue.clear()
        self.serial_num = 1

    def privatizer(self, t, delt, epsi):
        m = np.log(t + 1).astype(np.int) + 1
        num_list = self.countBits(self.serial_num)
        H_tmp = np.zeros((self.d, self.d))
        h_tmp = np.zeros(self.d)
        for i in num_list:
            if i in self.H_queue.keys():
                H_tmp += self.H_queue[i]
                h_tmp += self.h_queue[i]
            else:
                N_tmp = np.random.normal(0, 64 * m * (np.log(2 / delt)) ** 2 / epsi ** 2, (self.d + 1, self.d + 1))
                N = (N_tmp + N_tmp.T) / np.sqrt(2)
                H = N[0:self.d, 0:self.d]
                h = N[0:self.d, self.d - 1:self.d]
                self.H_queue[i] = H
                self.h_queue[i] = np.squeeze(h)

                H_tmp += self.H_queue[i]
                h_tmp += self.h_queue[i]

        self.h_now = h_tmp
        self.H_now = H_tmp

        self.serial_num += 1

# cluster in SCLUB
class sclub_Cluster(Cluster):
    def __init__(self, b, V, users_begin, d, user_num, rounds, rewards, best_rewards, theta_phase, users={}, t=0, T_phase=0):
        super(sclub_Cluster, self).__init__(b, V, users_begin, d, user_num, rounds, rewards, best_rewards, users, t)
        self.T_phase = T_phase
        self.theta_phase = theta_phase
        self.checks = {i: False for i in self.users}
        self.checked = len(self.users) == sum(self.checks.values())

    def phase_update(self):
        self.T_phase = self.t
        self.theta_phase = self.theta

    def update_check(self, i):
        self.checks[i] = True
        self.checked = len(self.users) == sum(self.checks.values())
