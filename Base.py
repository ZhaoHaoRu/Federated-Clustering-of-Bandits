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
        self.rewards = np.zeros(t)
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
        self.T = T  # np.zeros(d)
        self.V = V
        self.rewards = rewards
        # 不知道在cluster的reward和best reward还需不需要
        self.best_rewards = best_rewards
        #此时theta中c_t = 1
        self.theta = np.matmul(np.linalg.inv(np.eye(self.d) + self.V), self.b)

    def get_user(self,user_index):
        return self.users[user_index]

    def store_info(self,x, y, t, r, br, ksi_noise, B_noise):
        #self.V = self.V + np.outer(x, x) + B_noise
        self.V = self.V + np.outer(x, x)
        #self.b = self.b + y * x + ksi_noise
        self.b = self.b + y * x
        self.T += 1
        self.best_rewards[t] += br
        self.rewards[t] += r
        self.theta = np.matmul(np.linalg.inv(np.eye(self.d) + self.V), self.b)


class DC_Cluster(User):
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
        self.T = T  # np.zeros(d)
        self.V = V
        #synchronized gram matrix
        self.S = np.zeros((d,d))
        self.u = np.zeros(d)
        self.N = 0

        #upload buffer
        self.S_up = np.zeros((d, d))
        self.u_up = np.zeros(d)
        self.N_up = 0

        #download buffer
        self.S_down = np.zeros((d, d))
        self.u_down = np.zeros(d)
        self.N_down = 0

        self.rewards = rewards
        # 不知道在cluster的reward和best reward还需不需要
        self.best_rewards = best_rewards
        #此时theta中c_t = 1
        self.theta = np.matmul(np.linalg.inv(np.eye(self.d) + self.V), self.b)

    def get_user(self,user_index):
        return self.users[user_index]

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
        V_t = self.S_up + self.S
        b_t = self.u + self.u_up
        T_t = self.N + self.N_up
        return V_t, b_t , T_t
