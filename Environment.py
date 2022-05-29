# -*- coding: utf-8 -*-
# final version

import cmath

import numpy as np
import random
import sys

# Some constant
# c = 0.01

# 自定参数
# alpha = 4
# alpha2 = 3.5
# alpha = 241
alpha = 1.5
# alpha2 = 250
alpha2 = 2
delt = 0.1
alpha1 = 1
# alpha1 = 0.01
# epsi = 0.1
# epsi = 0.5
epsi = 1
# epsi = 2
# epsi = 4
# epsi = 6
# epsi = 8
# epsi = 10
U = 1.01
D = 1.01


# --------------------------------generate some parameters-------------------------------------- #

def isInvertible(S):
    return np.linalg.cond(S) < 1 / sys.float_info.epsilon


# generate items to recommend
def generate_items(num_items, d):
    # return a ndarray of num_items * d
    x = np.random.normal(0, 1, (num_items, d - 1))
    x = np.concatenate((np.divide(x, np.outer(np.linalg.norm(x, axis=1), np.ones(np.shape(x)[1]))) / np.sqrt(2),
                        np.ones((num_items, 1)) / np.sqrt(2)), axis=1)
    return x


# generate sigma
def sigm(delta, epsilon):
    tmp = np.power(2 * np.log(2.5 / delta), 0.5)
    # print("sigma:",6 * tmp / epsilon)
    return 6 * tmp / epsilon


# generate gamma
def gamma(t, d, alpha, sigma):
    tmp = 4 * cmath.sqrt(d) + 2 * np.log(2 * t / alpha)
    # return sigma * cmath.sqrt(rounds) * tmp
    return 1


# generate beta
def beta(sigma, alpha, gamma, S, d, t, L=1):
    # tmp1 = cmath.sqrt(2 * np.log(2 / alpha) + d * np.log(3 + rounds * np.power(L, 2) / (d * gamma)))
    # tmp2 = cmath.sqrt(3 * gamma)
    # tmp3 = cmath.sqrt((1/gamma) * d * rounds)
    # #print("beta:", sigma * tmp1 + S * tmp2 + sigma * tmp3)
    # return sigma * tmp1 + S * tmp2 + sigma * tmp3 * 0.5
    tmp1 = cmath.sqrt(2 * np.log(2 / alpha) + d * np.log(3 + t * np.power(L, 2) / d))
    return tmp1 * 0.5


# L_1 = 1, generate sigma in CDP version
def sigma_CDP(t):
    m = np.log(t + 1e-6) + 1
    tmp1 = cmath.sqrt(m * np.log(16 / (delt ** 2)))
    return 4 * (1 + 1) * tmp1 / epsi


# Intermediate variables for CDP calculation
def rou_min(t, d):
    m = np.log(t + 1e-6) + 1
    tmp1 = 4 * cmath.sqrt(d) + 2 * np.log(2 * t / alpha1)
    return cmath.sqrt(32) * m * 2 * np.log(4 / delt) * tmp1 / epsi


# Intermediate variables for CDP calculation
def rou_max(t, d):
    return rou_min(t, d)


# Intermediate variables for CDP calculation
def upsilon(t, d):
    m = np.log(t + 1e-6) + 1
    tmp1 = cmath.sqrt(d) + 2 * np.log(2 * t / alpha1)
    return cmath.sqrt(m * 2 * tmp1 / cmath.sqrt(2 * epsi))


# generate beta in CDP version, L: the number of local server
def beta_CDP(t, d, L):
    rou_min1 = rou_min(t, d)
    rou_max1 = rou_max(t, d)
    tmp1 = cmath.sqrt(2 * np.log(2 / alpha1 + 1e-6) + d * np.log(rou_max1 / rou_min1 + t / (d * rou_min1)))
    sigm = 1
    upsi = upsilon(t, d)
    return sigm * tmp1 + cmath.sqrt(L * rou_max1) + cmath.sqrt(L) * upsi


sigma = sigm(delt, epsi);

# ---------------------------------- Environment: generate user, item and feedback ----------------------------------------- #

class Environment:
    def __init__(self, d, num_users, theta, L=10):
        self.L = L  # the number of items
        self.d = d
        self.user_num = num_users
        self.theta = theta

    def get_items(self):
        self.items = generate_items(self.L, self.d)
        return self.items

    # get reward, best reward and then compute regret
    def feedback_Local(self, items, i, k, d):  # k: the chosen item's index , i: user_index
        x = items[k, :]  # select item from item array
        B_noise = np.random.normal(0, sigma ** 2, (d, d))
        reward = np.dot(x, self.theta[i])
        if reward < 0 or reward > 1:    # if reward is illegal
            y = 0
        else:
            y = np.random.binomial(1, reward)
        ksi_noise = np.random.normal(np.zeros(d), np.eye(d), (d, d))
        best_reward = np.max(np.dot(items, self.theta[i]))
        return reward, y, best_reward, ksi_noise, B_noise

    # get reward, best reward and then compute regret
    def feedback(self, items, i, b, M, k, d):   # k: the chosen item's index , i: user_index
        x = items[k, :]  # select item from item array
        Minv = np.linalg.inv(M)
        theta = np.matmul(Minv, b)
        B_noise = np.random.normal(0, sigma ** 2, (d, d))
        reward = np.dot(self.theta[i], x)
        y = np.random.binomial(1, reward)
        ksi_noise = np.random.normal(np.zeros(d), np.eye(d), (d, d))
        best_reward = np.max(np.dot(items, self.theta[i]))
        return reward, y, best_reward, ksi_noise, B_noise

    def generate_users(self):  # user selection is uniform
        X = np.random.multinomial(1, [1 / self.user_num] * self.user_num)  # X: 1*d array
        I = np.nonzero(X)[0]  # I: user_index
        return I
