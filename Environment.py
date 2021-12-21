import cmath

import numpy as np
import random
import sys

# Some constant
# c = 0.01

#自定参数
# alpha = 4
# alpha2 = 3.5
# alpha = 241
alpha = 1.5
#alpha2 = 250
alpha2 = 2
delt = 0.99
# alpha1 =
epsi = 0.9


def isInvertible(S):
    return np.linalg.cond(S) < 1 / sys.float_info.epsilon


# 这个应该是一个生成item 和context的时候随机取样的一个函数，没怎么看懂内部的逻辑，直接从老师的代码里抄下来了
def generate_items(num_items, d):
    # return a ndarray of num_items * d
    x = np.random.normal(0, 1, (num_items, d - 1))
    x = np.concatenate((np.divide(x, np.outer(np.linalg.norm(x, axis=1), np.ones(np.shape(x)[1]))) / np.sqrt(2),
                        np.ones((num_items, 1)) / np.sqrt(2)), axis=1)
    return x


def sigm(delta, epsilon):
    tmp = np.power(2 * np.log(2.5 / delta), 0.5)
    #print("sigma:",6 * tmp / epsilon)
    return 6 * tmp / epsilon


def gamma(t, d, alpha, sigma):
    tmp = 4 * cmath.sqrt(d) + 2 * np.log(2 * t / alpha)
    # print(cmath.sqrt(d))
    # print(2 * t / alpha)
    # print(np.log(2 * t / alpha))
    return sigma * cmath.sqrt(t) * tmp


def beta(sigma, alpha, gamma, S, d, t, L = 1):
    # tmp1 = cmath.sqrt(2 * np.log(2 / alpha) + d * np.log(3 + t * np.power(L, 2) / (d * gamma)))
    # tmp2 = cmath.sqrt(3 * gamma)
    # tmp3 = cmath.sqrt((1/gamma) * d * t)
    # #print("beta:", sigma * tmp1 + S * tmp2 + sigma * tmp3)
    # return sigma * tmp1 + S * tmp2 + sigma * tmp3 * 0.5
    tmp1 = cmath.sqrt(2 * np.log(2 / alpha) + d * np.log(3 + t * np.power(L, 2) / d))
    return tmp1*5


sigma = sigm(delt, epsi);


class Environment:
    def __init__(self, d, num_users, theta, L = 10):
        self.L = L  # the number of items
        self.d = d
        #self.p = p  # probability distribution over users,在我们的paper中有用吗？还是假设所有user都是uniform的,p应该是一个Rd的数组，如果是uniform的，那么p[i]=1/n
        self.user_num = num_users
        # self.items = generate_items(num_items=L, d=d)
        self.theta = theta

    def get_items(self):
        self.items = generate_items(self.L, self.d)
        return self.items

    #这里的i是user index
    def feedback(self, items,i, b, M, k, d):  # 寻找regret的操作，k是实际选取的item的index
        x = items[k, :]  # 将选取的item从item_array中选取出来
        Minv = np.linalg.inv(M)
        theta = np.matmul(Minv, b)
        B_noise = np.random.normal(0, sigma ** 2, (d,d))
        reward = np.dot(self.theta[i], x)
        # if reward >= 0 and reward <= 1:
        #   print("reward:", reward)
        y = np.random.binomial(1, reward)  # 在19 cascading中介绍的一个概念，标记recommend item是否被click，在update b的值的时候会用到
        # else:
        ksi_noise = np.random.normal(np.zeros(d), np.eye(d), (d, d))
        best_reward = np.max(np.dot(items, self.theta[i]))
        # print("best reward:",best_reward)
        return reward, y, best_reward, ksi_noise, B_noise

    def generate_users(self): # 假设user的选取是uniform的
        X = np.random.multinomial(1, [1/self.user_num]*self.user_num)  # X也应该是一个1*d的数组
        I = np.nonzero(X)[0]  # I是选取的user的index
        return I
