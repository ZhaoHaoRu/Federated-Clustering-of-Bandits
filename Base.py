# -*- coding: utf-8 -*-
import numpy as np
import copy
import Environment as Envi
#import sys

class User:
    def __init__(self, d, user_index, t):
        self.d = d  # dimension
        self.index = user_index
        self.T = 0  # round
        self.b = np.zeros(self.d)
        self.V = np.zeros((self.d, self.d))
        print("t", type(t))
        self.rewards = np.zeros(t)
        print("t", type(t))
        self.best_rewards = np.zeros(t)
        self.theta = np.zeros(d)

    def store_info(self,x, y, t, r, br, ksi_noise, B_noise):
        self.T += 1
        #self.V = self.V + np.outer(x,x) + B_noise
        self.V = self.V + np.outer(x, x)
        #self.b = self.b + y*x + ksi_noise
        self.b = self.b + y * x
        self.rewards[t] += r
        self.best_rewards[t] += br
        # 暂时设c_t=1
        self.theta = np.matmul(np.linalg.inv(np.eye(self.d) + self.V), self.b)

    def get_info(self):
        return self.V, self.b, self.T

class Cluster(User):
    # Base cluster
    def __init__(self, b, V, users_begin, d, user_num, t, rewards ,best_rewards, users = {}, T = 0):
        self.d = d
        if not users: #如果user为空的话说明是最开始的初始化，否则传入一个字典
            self.users = dict()
            for i in range(users_begin, users_begin + user_num):
                self.users[i]= User(self.d, i, t)  # a list/array of users, 类型就是users,i 是对应的user index
        else:
            self.users = copy.deepcopy(users)
        self.users_begin = users_begin
        self.user_num = user_num
        self.b = b  # np.eye(d)
        self.T = T
        self.V = V
        self.rewards = rewards
        # 不知道在cluster的reward和best reward还需不需要
        self.best_rewards = best_rewards
        #此时theta中c_t = 1
        self.theta = np.matmul(np.linalg.inv(np.eye(self.d) + self.V), self.b)

    def get_user(self,user_index):
        return self.users[user_index]

    #noise array
    def store_info(self,x, y, t, r, br, ksi_noise, B_noise):
        #self.V = self.V + np.outer(x, x) + B_noise
        self.V = self.V + np.outer(x, x)
        #self.b = self.b + y * x + ksi_noise
        self.b = self.b + y * x
        self.T += 1
        self.best_rewards[t] += br
        self.rewards[t] += r
        self.theta = np.matmul(np.linalg.inv(np.eye(self.d) + self.V), self.b)

    def get_info(self):
        V_t = self.V
        b_t = self.b
        T_t = self.T
        return V_t, b_t , T_t


class DC_Cluster(User):
    # Base cluster
    def __init__(self, b, V, users_begin, d, user_num, t, rewards ,best_rewards,l_server_index, index,users = {}, T = 0):
        self.d = d
        if not users: #如果user为空的话说明是最开始的初始化，否则传入一个字典
            self.users = dict()
            for i in range(users_begin, users_begin + user_num):
                self.users[i]= User(self.d, i, t)  # a list/array of users, 类型就是users,i 是对应的user index
        else:
            self.users = copy.deepcopy(users)
        self.users_begin = users_begin
        self.user_num = user_num
        self.u = b  # np.eye(d)
        self.T = T  # 0
        #在每个DC_cluster中记录它所属于的local server的index和它在local server中的index，以便更新global cluster的时候使用
        self.l_server_index = l_server_index
        self.index = index
        # synchronized gram matrix
        self.S = V  #np.zeros((d,d))

        #upload buffer
        self.S_up = np.zeros((d, d))
        self.u_up = np.zeros(d)
        self.T_up = 0

        #download buffer
        self.S_down = np.zeros((d, d))
        self.u_down = np.zeros(d)
        self.T_down = 0

        self.rewards = rewards
        # 不知道在cluster的reward和best reward还需不需要
        self.best_rewards = best_rewards
        #此时theta中c_t = 1
        self.theta = np.matmul(np.linalg.inv(np.eye(self.d) + self.S), self.u)

    def get_user(self,user_index):
        return self.users[user_index]

    #这里的store_info 相当于update upload buffer
    def store_info(self,x, y, t, r, br, ksi_noise, B_noise):
        #self.V = self.V + np.outer(x, x) + B_noise
        self.S_up += np.outer(x, x)
        #self.b = self.b + y * x + ksi_noise
        self.u_up += y * x
        self.T_up += 1
        self.best_rewards[t] += br
        self.rewards[t] += r
        self.theta = np.matmul(np.linalg.inv(np.eye(self.d) + self.S), self.u)

    def get_info(self):
        V_t = self.S_up + self.S
        b_t = self.u + self.u_up
        T_t = self.T + self.T_up
        return V_t, b_t , T_t

class CDP_Cluster(DC_Cluster):
    def __init__(self, b, V, users_begin, d, user_num, t, rewards ,best_rewards,l_server_index, index,users = {}, T = 0, ):
        super(DC_Cluster, self).__init__(self, b, V, users_begin, d, user_num, t, rewards ,best_rewards,l_server_index, index,users, T)
        self.H_queue = dict()
        self.h_queue = dict()
        self.serial_num = 1
        self.H_now = np.zeros(self.d,self.d)
        self.h_now = np.zeros(self.d)
        self.H_former = np.zeros(self.d, self.d)
        self.h_former = np.zeros(self.d)

    def countBits(self,n):
        count = 0;
        while n != 0 :
            n = n & (n-1)
            count += 1
        return count

    def privatizer(self, t, delt, epsi):
        m = np.log(t + 1).astype(np.int) + 1
        n = self.countBits(self.serial_num)
        while len(self.H_queue) < n:
            N_tmp = np.random.normal(0, 64 * m * (np.log(2 / delt)) ** 2 / epsi ** 2)
            N = (N_tmp + N_tmp.T) / np.sqrt(2)
            H = N[0:self.d, 0:self.d]
            h = N[0:self.d,self.d - 1:self.d]
            self.H_queue[len(self.H_queue)] = H
            self.h_queue[len(self.h_queue)] = h

        H_tmp = np.zeros(self.d,self.d)
        h_tmp = np.zeros(self.d)
        for i in range(n):
            H_tmp += self.H_queue[i]
            h_tmp += self.h_queue[i]

        self.H_former = self.H_now
        self.h_former = self.h_now








class sclub_Cluster(Cluster):
    def __init__(self, b, V, users_begin, d, user_num, t, rewards ,best_rewards, theta_phase, users = {}, T = 0, T_phase = 0):
        super(sclub_Cluster,self).__init__(b, V, users_begin, d, user_num, t, rewards , best_rewards, users, T)
        self.T_phase = T_phase
        self.theta_phase = theta_phase
        self.checks = {i: False for i in self.users}
        self.checked = len(self.users) == sum(self.checks.values())

    def phase_update(self):
        self.T_phase = self.T
        self.theta_phase = self.theta

    def update_check(self, i):
        self.checks[i] = True
        self.checked = len(self.users) == sum(self.checks.values())
