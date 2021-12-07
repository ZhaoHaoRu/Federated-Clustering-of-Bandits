import numpy as np

import FCLUB
import time
import random
import Environment as Envi
import matplotlib.pyplot as plt

d = 10 #dimension
user_num = 10  # the number of all users
I = 10  # the number of items
T = 10000 # the number of rounds
L = 3  # the number of local server
userList = [3, 3, 4]
theta1 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
theta2 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
theta3 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
theta_tmp= np.vstack((theta1, theta2, theta1, theta1, theta2, theta3, theta1, theta2, theta3, theta3))


def drawResult(regret):
    fig = plt.figure()
    regret_range = np.arange(1,T + 1)
    Cumulative_regret = 0
    for i in range(T):
        Cumulative_regret += regret[i]
        regret[i] = Cumulative_regret/(i + 1)

    ax = fig.add_subplot(111)
    plt.plot(regret_range, regret, 'r.-', ms=2, label="regret")
    # ax.scatter(regret_range, regret, color='r', label="regret")
    ax.set_ylabel('regret in each round')
    my_x_ticks = np.arange(0, 10000, 1000)
    plt.xticks(my_x_ticks)
    my_y_ticks = np.arange(0, 1, 0.1)
    plt.yticks(my_y_ticks)
    plt.xlabel("round")
    plt.ylabel("regret")
    plt.legend()
    plt.grid()
    plt.show()

def main(num_users, d, m, L, l_server_num, filename=''):
    # seed = int(time.time() * 100) % 399
    # print("Seed = %d" % seed)
    # np.random.seed(seed)
    # random.seed(seed)
    #
    # #set up theta vector
    # def _get_theta(thetam, num_users, m):
    #     k = int(num_users / m)
    #     theta = {i: thetam[0] for i in range(k)}
    #     for j in range(1, m):
    #         theta.update({i: thetam[j] for i in range(k * j, k * (j + 1))})
    #     return theta
    #
    # if filename == '':
    #     thetam = Envi.generate_items(num_items=m, d=d)
    #     # thetam = np.concatenate((np.dot(np.concatenate((np.eye(m), np.zeros((m,d-m-1))), axis=1), ortho_group.rvs(d-1))/np.sqrt(2), np.ones((m,1))/np.sqrt(2)), axis=1)
    #     print(thetam, [[np.linalg.norm(thetam[i,:]-thetam[j,:]) for j in range(i)] for i in range(1,m)])
    #     theta = _get_theta(thetam, num_users, m)
    #     # print([np.linalg.norm(theta[0]-theta[i]) for i in range(num_users)])
    # else:
    #     theta = np.load(filename)
    #
    # envi = Envi.Environment(d, num_users, theta, L)
    # if num_users % l_server_num == 0:
    #     userList = [num_users//l_server_num] * l_server_num
    # else:
    #     userList = [num_users//l_server_num] * (l_server_num - 1)
    #     userList[l_server_num - 1] = num_users - (num_users//l_server_num)*(l_server_num - 1)
    # #edge_probability的问题
    # G_server = FCLUB.Global_server(l_server_num, num_users, userList, d = 100, T = 5000)
    # start_time = time.time()
    # regret = G_server.run(envi, T)
    # run_time = time.time() - start_time
    # np.savez('FCLUB_:' + '_nu' + str(num_users) + 'd' + str(d) + 'm' + str(m) + 'L' + str(L) + '_' + str(seed),
    #     seed, G_server.reward, G_server.best_reward, run_time, len(G_server.clusters))

    G_server = FCLUB.Global_server(L, user_num, userList, d, T)
    envi = Envi.Environment(d=d, num_users=user_num, L = I, theta=theta_tmp)
    regret = G_server.run(envi, T)
    print("finish")
    drawResult(regret)


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    main(num_users = 10, d = 10, m = 3, L = 3, l_server_num = 5, filename= '')



