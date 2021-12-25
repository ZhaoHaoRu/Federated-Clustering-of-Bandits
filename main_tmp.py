import numpy as np

import LDP_FCLUB_DC
import time
import random
import Environment as Envi



theta1 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
theta2 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
theta3 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
d = 10 #dimension
user_num = 10  # the number of all users
I = 10  # the number of items
T = 100 # the number of rounds
L = 3  # the number of local server
T = 20000
userList = [3, 3, 4]
theta_tmp= np.vstack((theta1, theta2, theta1, theta1, theta2, theta3, theta1, theta2, theta3, theta3))

#L:item的数量
phase = (np.log2(T)).astype(np.int)
G_server = LDP_FCLUB_DC.Global_server(L, user_num, userList= userList, d=d,T=2**(phase + 1) - 2)
envi = Envi.Environment(d=d, num_users=user_num, L = I, theta=theta_tmp)
regret, result_tmp, reward = G_server.run(envi, phase)
np.savez('LDP_FCLUB_DC_12_25_user_10_1', nu=user_num, d=d, L=L, T=2**(phase + 1) - 2, G_server_regret=regret,
                  cluster_num=len(G_server.clusters))
print("finish")