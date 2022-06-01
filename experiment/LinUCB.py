# -*- coding: utf-8 -*-
# final version

import numpy as np

import Base
import Environment as Envi
from Environment import sigma

S = 1

alpha = 15

# Actually there is no cluster and server in LinUCB, We use global server here for the sake of interface consistency.
# It only serves as a learning agent.
class Global_server:
    def __init__(self, n, d, T):
        self.usernum = n    # the number of users in a server
        self.rounds = T     # the number of all rounds
        self.d = d      # dimension
        self.regret = np.zeros(self.rounds)
        self.reward = np.zeros(self.rounds)
        self.best_reward = np.zeros(self.rounds)
        self.users = dict()     # all users' information
        # initialize every user
        for i in range(self.usernum):
            self.users[i] = Base.User(self.d, i, self.rounds)

    # recommend the item to the user
    def recommend(self, user_index, items):
        user = self.users[user_index]
        V_t, b_t, T_t = user.get_info()
        gamma_t = Envi.gamma(T_t, self.d, alpha, sigma)
        lambda_t = gamma_t * 2
        M_t = np.eye(self.d) * np.float_(lambda_t) + V_t
        # S = 1
        beta_t = Envi.beta(sigma, alpha, gamma_t, S, self.d, T_t)
        Minv = np.linalg.inv(M_t)
        theta = np.dot(Minv, b_t)
        r_item_index = np.argmax(np.dot(items, theta) + beta_t * (np.matmul(items, Minv) * items).sum(axis=1))
        return r_item_index

    # LinUCB for per user
    def run(self, envir, T, number):
        y_list = list()    # to save feedback in each round
        x_list = list()    # to save the recommended item in each round
        result_final = dict()   # to save the users' final theta information
        for i in range(1, T + 1):
            if i % 5000 == 0:
                print(i)
            user_all = envir.generate_users()   # random user arrives
            user_index = user_all[0]
            # the context set
            items = envir.get_items()
            r_item_index = self.recommend(user_index, items)
            x = items[r_item_index]
            x_list.append(x)
            # get feedback
            self.reward[i - 1], y, self.best_reward[i - 1], ksi_noise, B_noise = envir.feedback_Local(items=items,
                                                                                                      i=user_index,
                                                                                                      k=r_item_index, d=self.d)
            y_list.append(y)
            # update the user's information
            self.users[user_index].store_info(x, y, i - 1, self.reward[i - 1], self.best_reward[i - 1], ksi_noise[0],
                                              B_noise)
            self.regret[i - 1] = self.best_reward[i - 1] - self.reward[i - 1]

            if i % T == 0:
                for i in range(self.usernum):
                    result_final[i] = self.users[i].theta

        return self.regret, result_final, self.reward, x_list, y_list
