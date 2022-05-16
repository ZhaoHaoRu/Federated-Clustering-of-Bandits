import numpy as np

import LDP_FCLUB_DC
import CDP_FCLUB_DC
import SCLUB
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
T = 50000
userList = [3, 3, 4]
theta_tmp= np.vstack((theta1, theta2, theta1, theta1, theta2, theta3, theta1, theta2, theta3, theta3))

#L:item的数量
seed = int(time.time() * 100) % 399
print("Seed = %d" % seed)
np.random.seed(seed)
random.seed(seed)
phase = (np.log2(T)).astype(np.int)
all_round = 2**(phase + 1) - 2
G_server = CDP_FCLUB_DC.Global_server(L, user_num, userList= userList, d=d,T=2**(phase + 1) - 2)
# G_server = SCLUB.Global_server(L, user_num, userList= userList, d=d,T=2**(phase + 1) - 2)
envi = Envi.Environment(d=d, num_users=user_num, L = I, theta=theta_tmp)
start_time = time.time()
regret, result_tmp, reward = G_server.run(envi, phase,1, all_round)
run_time = time.time() - start_time
# np.savez('1_17_DC_debug_user_10', nu=user_num, d=d, L=L, T=2**(phase + 1) - 2, G_server_regret=regret,
#                   cluster_num=len(G_server.clusters))
# np.savez('1_19_DC_debug_user_10_no1', nu=user_num, d=d, L=L, T=2**(phase + 1) - 2, seed=seed, G_server_regret=regret, run_time=run_time,
#                  cluster_num= cluster_num, reward=reward)
np.savez('1_20_CDP_debug_user_10_no1', nu=user_num, d=d, L=L, T=2**(phase + 1) - 2, seed=seed, G_server_regret=regret, run_time=run_time,
                  reward=reward)
print("finish")