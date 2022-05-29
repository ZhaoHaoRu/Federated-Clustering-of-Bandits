# -*- coding: utf-8 -*-
import numpy as np

import CLUB
import SCLUB
import FCLUB
import LDP_FCLUB_DC
import LinUCB
import CDP_FCLUB_DC
import homogeneous_version
import DC_homogenenous_version
import time
import random
import math
import Environment as Envi
import matplotlib.pyplot as plt

# d = 10 #dimension
# user_num = 10  # the number of all users
# I = 10  # the number of items
# rounds = 100 # the number of rounds
# L = 3  # the number of local server
#userList = [3, 3, 4]
#theta_tmp= np.vstack((theta1, theta2, theta1, theta1, theta2, theta3, theta1, theta2, theta3, theta3))
theta1 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
theta2 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
theta3 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
theta4 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
theta5 = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
theta6 = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
d = 10 #dimension
user_num = 30  # the number of all users
I = 10  # the number of items
T = 50000 # the number of rounds
L = 5  # the number of local server
phase_cardinality = 2
# userList = [6, 6, 6, 6, 6]
theta_tmp1= np.vstack((theta1, theta1, theta2, theta2, theta3, theta3))
theta_tmp2= np.vstack((theta4, theta4, theta5, theta5, theta6, theta6))
theta_tmp3= np.vstack((theta2, theta2, theta4, theta4, theta6, theta6))
theta_tmp4= np.vstack((theta1, theta1, theta3, theta3, theta5, theta5))
theta_tmp5= np.vstack((theta1, theta1, theta3, theta3, theta6, theta6))
theta_tmp=np.vstack((theta_tmp1, theta_tmp2, theta_tmp3, theta_tmp4, theta_tmp5))



#在DC的版本中，T为phase
#回头传参两个npzname
def main(number, num_users, d, m, L, l_server_num,T, filename='',npzname=''):
    seed = int(time.time() * 100) % 399
    # seed = 262
    print("Seed = %d" % seed)
    np.random.seed(seed)
    random.seed(seed)

    #set up theta vector，把thetam分配给每个具体的user
    def _get_theta(thetam, num_users, m):
        #下一步确定m的含义，
        k = int(num_users / m)
        theta = {i: thetam[0] for i in range(k)}
        for j in range(1, m):
            theta.update({i: thetam[j] for i in range(k * j, k * (j + 1))})
        return theta

    if filename == '':
        #thetam是一个theta数组，m*d维
        thetam = Envi.generate_items(num_items=m, d=d)
        # thetam = np.concatenate((np.dot(np.concatenate((np.eye(m), np.zeros((m,d-m-1))), axis=1), ortho_group.rvs(d-1))/np.sqrt(2), np.ones((m,1))/np.sqrt(2)), axis=1)
        print(thetam, [[np.linalg.norm(thetam[i,:]-thetam[j,:]) for j in range(i)] for i in range(1,m)])
        theta = _get_theta(thetam, num_users, m)
        theta = list(theta.values())
        print("theta:",theta)
        np.save(str(num_users) + '_theta.npy', theta)
        #theta 是一个dict
        # print([np.linalg.norm(theta[0]-theta[i]) for i in range(num_users)])
    else:
        theta = np.load(filename)

    #envi = Envi.Environment(d, num_users, theta, L)
    if num_users % l_server_num == 0:
        userList = [num_users//l_server_num] * l_server_num
    else:
        userList = [num_users//l_server_num] * (l_server_num - 1)
        userList[l_server_num - 1] = num_users - (num_users//l_server_num)*(l_server_num - 1)


    # main_FCLUB_DC(number,num_users= num_users, d=d, m=m, L=L, l_server_num=l_server_num,theta= theta,T=T, filename=filename,npzname=npzname, seed= seed,userList=userList)
    # Homogeneous(number, num_users=num_users, d=d, m=m, L=L, l_server_num=l_server_num, theta=theta, T=T,
    #               filename=filename, npzname=npzname, seed=seed, userList=userList)
    # DC_Homogenenous(number, num_users=num_users, d=d, m=m, L=L, l_server_num=l_server_num, theta=theta, T=T,
    #               filename=filename, npzname=npzname, seed=seed, userList=userList)
    # main_CLUB(number, num_users=num_users, d=d, m=m, L=L, l_server_num=l_server_num, theta=theta, T=T,
    #               filename=filename, npzname=npzname, seed=seed, userList=userList)
    # main_SCLUB(number, num_users=num_users, d=d, m=m, L=L, l_server_num=l_server_num, theta=theta, T=T,
    #               filename=filename, npzname=npzname, seed=seed, userList=userList)
    main_LinUCB(number, num_users=num_users, d=d, theta=theta, T=T, L= L,
                   filename=filename, npzname=npzname, seed=seed)
    # main_CDP_FCLUB_DC(number,num_users= num_users, d=d, m=m, L=L, l_server_num=l_server_num,theta= theta,T=T, filename=filename,npzname=npzname, seed= seed,userList=userList)
    # main_FCLUB(number, num_users=num_users, d=d, m=m, L=L, l_server_num=l_server_num, theta=theta, T=T,
    #            filename=filename, npzname=npzname, seed=seed, userList=userList)





    #former interface
    # G_server = FCLUB.Global_server(L, user_num, userList, d, rounds)
    # envi = Envi.Environment(d=d, num_users=user_num, L = I, theta=theta_tmp)
    # regret = G_server.run(envi, rounds)
    print("finish")
    # drawResult(regret,rounds)



def main_FCLUB(number, num_users, d, m, L, l_server_num,T,theta,seed,userList,filename='',npzname=''):
    G_server = FCLUB.Global_server(l_server_num, num_users, userList, d = d, T = T)
    envi = Envi.Environment(d=d, num_users=num_users, L= L, theta=theta)
    start_time = time.time()
    regret, theta_get, reward, x_list, y_list, comu_cost= G_server.run(envi, T,number, user_num = num_users)
    run_time = time.time() - start_time

    if npzname == '':
        print(len(regret))
        print(len(theta))
        np.savez('FCLUB_4_12_22_user_1_3', nu= num_users, d=d, L= L,T=T,seed=seed,G_server_regret=regret, run_time= run_time,cluster_num=len(G_server.clusters),theta_exp=theta_get,theta_theo= theta, reward= reward,x= x_list, y= y_list, comu_cost= comu_cost)
    else:
        np.savez("FCLUB_"+npzname, nu=num_users, d=d, L=L, T=T, seed=seed, G_server_regret=regret, run_time=run_time,
                 cluster_num=len(G_server.clusters), theta_exp=theta_get, theta_theo=theta, reward= reward,x= x_list, y= y_list, comu_cost= comu_cost)


def Homogeneous(number, num_users, d, m, L, l_server_num,T,theta,seed,userList,filename='',npzname=''):
    G_server = homogeneous_version.Global_server(l_server_num, num_users, userList, d=d, T=T)
    envi = Envi.Environment(d=d, num_users=num_users, L= L, theta=theta)
    start_time = time.time()
    regret, theta_get, reward, x_list, y_list = G_server.run(envi, T, number, user_num = num_users)
    run_time = time.time() - start_time

    if npzname == '':
        print(len(regret))
        print(len(theta))
        np.savez('Homo_1_1_16_user_1_3', nu=num_users, d=d, L=L, T=T, seed=seed, G_server_regret=regret,
                 run_time=run_time, cluster_num=len(G_server.clusters), theta_exp=theta_get, theta_theo=theta,
                 reward=reward, x=x_list, y=y_list)
    else:
        np.savez("Homo_" + npzname, nu=num_users, d=d, L=L, T=T, seed=seed, G_server_regret=regret, run_time=run_time,
                 cluster_num=len(G_server.clusters), theta_exp=theta_get, theta_theo=theta, reward=reward, x=x_list,
                 y=y_list)



def main_FCLUB_DC(number, num_users, d, m, L, l_server_num,T,theta, seed, userList, filename='',npzname=''):
    #对于DC版本而言T是phase
    phase = (np.log(T)/np.log(phase_cardinality)).astype(np.int64)
    round = (phase_cardinality**(phase + 1) - 1)//(phase_cardinality - 1)
    print("phase:", phase)
    G_server_DC = LDP_FCLUB_DC.Global_server(l_server_num, num_users, userList, d=d,T=(phase_cardinality**(phase + 1) - 1)//(phase_cardinality - 1))
    print("round:", (phase_cardinality**(phase + 1) - 1)//(phase_cardinality - 1))
    envi = Envi.Environment(d=d, num_users=num_users, L=L, theta=theta)
    start_time = time.time()
    regret, theta_get, reward, comu_cost= G_server_DC.run(envi,phase=phase, number= number, all_round = (phase_cardinality**(phase + 1) - 1)//(phase_cardinality - 1))
    run_time = time.time() - start_time

    if npzname == '':
        print(len(regret))
        print(len(theta))
        np.savez('LDP_FCLUB_DC_1_2_user_20_1', nu=num_users, d=d, L=L, T=round, seed=seed, G_server_regret=regret,
                 run_time=run_time, cluster_num=len(G_server_DC.clusters), reward=reward, comu_cost= comu_cost)
    else:
        np.savez("LDP_FCLUB_DC_"+npzname, nu=num_users, d=d, L=L, T=round, seed=seed, G_server_regret=regret, run_time=run_time,
                 cluster_num=len(G_server_DC.clusters), reward=reward, comu_cost= comu_cost)


def DC_Homogenenous(number, num_users, d, m, L, l_server_num,T,theta, seed, userList, filename='',npzname=''):
    #对于DC版本而言T是phase
    phase = (np.log(T)/np.log(phase_cardinality)).astype(np.int64)
    round = (phase_cardinality**(phase + 1) - 1)//(phase_cardinality - 1)
    print("phase:", phase)
    G_server_DC = DC_homogenenous_version.Global_server(l_server_num, num_users, userList, d=d,T=(phase_cardinality**(phase + 1) - 1)//(phase_cardinality - 1))
    print("round:", (phase_cardinality**(phase + 1) - 1)//(phase_cardinality - 1))
    envi = Envi.Environment(d=d, num_users=num_users, L=L, theta=theta)
    start_time = time.time()
    regret, theta_get, reward = G_server_DC.run(envi,phase=phase, number= number, all_round = (phase_cardinality**(phase + 1) - 1)//(phase_cardinality - 1))
    run_time = time.time() - start_time

    if npzname == '':
        print(len(regret))
        print(len(theta))
        np.savez('homo_DC_1_2_user_20_1', nu=num_users, d=d, L=L, T=round, seed=seed, G_server_regret=regret,
                 run_time=run_time, cluster_num=len(G_server_DC.clusters), reward=reward)
    else:
        np.savez("homo_DC_"+npzname, nu=num_users, d=d, L=L, T=round, seed=seed, G_server_regret=regret, run_time=run_time,
                 cluster_num=len(G_server_DC.clusters), reward=reward)


def main_CLUB(number, num_users, d, m, L, l_server_num,T,theta, seed, userList, filename='',npzname=''):
    G_server = CLUB.Global_server(l_server_num, num_users, userList, d=d, T=T)
    envi = Envi.Environment(d=d, num_users=num_users, L= L, theta=theta)
    start_time = time.time()
    regret, theta_get, reward, x_list, y_list, cluster_num = G_server.run(envi, T, number)
    run_time = time.time() - start_time

    if npzname == '':
        print(len(regret))
        print(len(theta))
        np.savez('CLUB_4_1_16_user_1_3', nu=num_users, d=d, L=L, T=T, seed=seed, G_server_regret=regret,
                 run_time=run_time, cluster_num=len(G_server.clusters), theta_exp=theta_get, theta_theo=theta,
                 reward=reward, x=x_list, y=y_list)
    else:
        np.savez("CLUB_" + npzname, nu=num_users, d=d, L=L, T=T, seed=seed, G_server_regret=regret, run_time=run_time,
                 cluster_num= cluster_num, theta_exp=theta_get, theta_theo=theta, reward=reward, x=x_list,
                 y=y_list)

def main_LinUCB(number, num_users, d, theta, T, L, seed, filename = '', npzname= ''):
    G_server = LinUCB.Global_server(num_users, d=d, T=T)
    envi = Envi.Environment(d=d, num_users=num_users, L= L, theta=theta)
    start_time = time.time()
    regret, theta_get, reward, x_list, y_list = G_server.run(envi, T, number)
    run_time = time.time() - start_time

    if npzname == '':
        print(len(regret))
        print(len(theta))
        np.savez('LinUCB_4_1_16_user_1_3', nu=num_users, d=d, L=L, T=T, seed=seed, G_server_regret=regret,
                 run_time=run_time, theta_exp=theta_get, theta_theo=theta,
                 reward=reward, x=x_list, y=y_list)
    else:
        np.savez("LinUCB_" + npzname, nu=num_users, d=d, L=L, T=T, seed=seed, G_server_regret=regret, run_time=run_time,
                 theta_exp=theta_get, theta_theo=theta, reward=reward, x=x_list,
                 y=y_list)

def main_SCLUB(number, num_users, d, m, L, l_server_num, T, theta, seed, userList, filename='', npzname=''):
    # 对于DC版本而言T是phase
    phase = (np.log(T) / np.log(phase_cardinality)).astype(np.int64)
    round = (phase_cardinality ** (phase + 1) - 1) // (phase_cardinality - 1)
    G_server = SCLUB.Global_server(l_server_num, num_users, userList, d=d, T= round)
    print("round:", (phase_cardinality ** (phase + 1) - 1) // (phase_cardinality - 1))
    envi = Envi.Environment(d=d, num_users=num_users, L=L, theta=theta)
    start_time = time.time()
    regret, theta_get, reward, cluster_num = G_server.run(envi, phase=phase, number=number)
    run_time = time.time() - start_time

    if npzname == '':
        print(len(regret))
        print(len(theta))
        np.savez('SCLUB_1_19_user_20_1', nu=num_users, d=d, L=L, T=round, seed=seed, G_server_regret=regret,
                 run_time=run_time, cluster_num=len(G_server_DC.clusters), reward=reward)
    else:
        np.savez("SCLUB_" + npzname, nu=num_users, d=d, L=L, T=round, seed=seed, G_server_regret=regret,
                 cluster_num= cluster_num, run_time=run_time, reward=reward)


def main_CDP_FCLUB_DC(number, num_users, d, m, L, l_server_num,T,theta, seed, userList, filename='',npzname=''):
    #对于DC版本而言T是phase
    phase = (np.log(T)/np.log(phase_cardinality)).astype(np.int64)
    round = (phase_cardinality**(phase + 1) - 1)//(phase_cardinality - 1)
    print("phase:", phase)
    G_server_DC = CDP_FCLUB_DC.Global_server(l_server_num, num_users, userList, d=d,T=(phase_cardinality**(phase + 1) - 1)//(phase_cardinality - 1))
    print("round:", (phase_cardinality**(phase + 1) - 1)//(phase_cardinality - 1))
    envi = Envi.Environment(d=d, num_users=num_users, L=L, theta=theta)
    start_time = time.time()
    regret, theta_get, reward , comu_cost = G_server_DC.run(envi,phase=phase, number= number, all_round = (phase_cardinality**(phase + 1) - 1)//(phase_cardinality - 1))
    run_time = time.time() - start_time

    if npzname == '':
        print(len(regret))
        print(len(theta))
        np.savez('CDP_FCLUB_DC_1_2_user_20_1', nu=num_users, d=d, L=L, T=round, seed=seed, G_server_regret=regret,
                 run_time= run_time, cluster_num=len(G_server_DC.clusters), reward=reward, comu_cost= comu_cost)
    else:
        np.savez("CDP_FCLUB_DC_"+npzname, nu=num_users, d=d, L=L, T=round, seed=seed, G_server_regret=regret, run_time=run_time,
                 cluster_num=len(G_server_DC.clusters), reward=reward, comu_cost= comu_cost)


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    number = eval(input())
    num_users = eval(input())
    #synthetic dataset
    if num_users == 20:
        # main(number, num_users=20, d=10, m=4, L=10, l_server_num=5, rounds=100000, filename='20_theta.npy',
        #      npzname='no' + str(number) + '_2_23_user_20_100000round_1_0.1'+'_m_4'+'_d_10')
        # main(number, num_users=60, d=10, m=5, L=10, l_server_num=5, rounds=100000, filename='',
        #      npzname='no' + str(number) + '_1_23_user_60_100000round_1_0.1' + '_m_5' + '_d_10')
        # main(number, num_users=80, d=10, m=5, L=10, l_server_num=5, rounds=100000, filename='',
        #      npzname='no' + str(number) + '_1_23_user_80_100000round_1_0.1' + '_m_5' + '_d_10')
        main(number, num_users=20, d=10, m=4, L=10, l_server_num=5, T=100000, filename='yelp_1000user_d10_m10.npy',
             npzname='3_no' + str(number) + '_5_25_user_20_100000round_yelp')
    elif num_users == 40:
        #m对比实验
        # main(number, num_users=50, d=10, m=10, L=10, l_server_num=5, rounds=100000, filename='m_10_50.npy',
        #  npzname='no' + str(number) + '_2_13_user_40_100000round_alpha_2_0.1'+'_m_10'+ '_d_10')
        # main(number, num_users=25, d=10, m=5, L=10, l_server_num=5, rounds=100000, filename='m_5_25.npy',
        #      npzname='no' + str(number) + '_2_13_user_40_100000round_alpha_2_0.1' + '_m_5' + '_d_10')
        #n对比实验
        main(number, num_users=20, d=10, m=4, L=10, l_server_num=5, T=100000, filename='20_theta.npy',
             npzname='no' + str(number) + '_4_10_user_20_100000round_1_0.1'+'_m_4'+'_d_10')
        main(number, num_users=20, d=10, m=4, L=10, l_server_num=5, T=100000, filename='m_4_20.npy',
             npzname='no' + str(number) + '_4_10_user_40_100000round_1_0.1' + '_m_4' + '_d_10')
        main(number, num_users=60, d=10, m=4, L=10, l_server_num=5, T=100000, filename='60_theta.npy',
             npzname='no' + str(number) + '_4_10_user_60_100000round_1_0.1' + '_m_4' + '_d_10')
        main(number, num_users=80, d=10, m=4, L=10, l_server_num=5, T=100000, filename='80_theta.npy',
             npzname='no' + str(number) + '_4_10_user_80_100000round_1_0.1' + '_m_4' + '_d_10')
        # main(number, num_users=15, d=10, m=3, L=10, l_server_num=5, rounds=100000, filename='m_3_15.npy',
        #      npzname='no' + str(number) + '_2_13_user_40_100000round_alpha_2_0.1' + '_m_3' + '_d_10')
        # main(number, num_users=30, d=10, m=6, L=10, l_server_num=5, rounds=100000, filename='m_6_30.npy',
        #      npzname='no' + str(number) + '_2_13_user_40_100000round_alpha_2_0.1' + '_m_6' + '_d_10')
        # main(number, num_users=10, d=10, m=2, L=10, l_server_num=5, rounds=100000, filename='m_2_10.npy',
        #      npzname='no' + str(number) + '_2_13_user_40_100000round_alpha_2_0.1' + '_m_2' + '_d_10')
        # main(number, num_users=40, d=10, m=8, L=10, l_server_num=5, rounds=100000, filename='m_8_40.npy',
        #      npzname='no' + str(number) + '_2_13_user_40_100000round_alpha_2_0.1' + '_m_8' + '_d_10')
        #d对比试验
        # main(number, num_users=40, d=5, m=4, L=10, l_server_num=5, rounds=100000, filename='5_theta.npy',
        #      npzname='no' + str(number) + '_2_16_user_40_100000round_alpha_1_0.1' + '_m_4' + '_d_5')
        # main(number, num_users=40, d=10, m=4, L=10, l_server_num=5, rounds=100000, filename='10_theta.npy',
        #      npzname='no' + str(number) + '_2_16_user_40_100000round_alpha_1_0.1' + '_m_4' + '_d_10_comu_cost')
        # main(number, num_users=40, d=15, m=4, L=10, l_server_num=5, rounds=100000, filename='15_theta.npy',
        #      npzname='no' + str(number) + '_2_16_user_40_100000round_alpha_1_0.1' + '_m_4' + '_d_15')
        # main(number, num_users=40, d=20, m=4, L=10, l_server_num=5, rounds=100000, filename='20_theta.npy',
        #      npzname='no' + str(number) + '_2_16_user_40_100000round_alpha_1_0.1' + '_m_4' + '_d_20')
        #epsi对比实验
        # main(number, num_users=40, d=10, m=4, L=10, l_server_num=5, rounds=100000, filename='m_theta.npy',
        #      npzname='no' + str(number) + '_2_16_user_40_100000round_alpha_0.1_0.1' + '_m_4' + '_d_10')
        # main(number, num_users=40, d=10, m=4, L=10, l_server_num=5, rounds=100000, filename='m_theta.npy',
        #      npzname='no' + str(number) + '_2_16_user_40_100000round_alpha_0.5_0.1' + '_m_4' + '_d_10')
        # main(number, num_users=40, d=10, m=4, L=10, l_server_num=5, rounds=100000, filename='m_theta.npy',
        #      npzname='no' + str(number) + '_2_16_user_40_100000round_alpha_1_0.1' + '_m_4' + '_d_10')
        # main(number, num_users=40, d=10, m=4, L=10, l_server_num=5, rounds=100000, filename='m_theta.npy',
        #      npzname='no' + str(number) + '_2_16_user_40_100000round_alpha_2_0.1' + '_m_4' + '_d_10')
        # main(number, num_users=40, d=10, m=4, L=10, l_server_num=5, rounds=100000, filename='m_theta.npy',
        #      npzname='no' + str(number) + '_2_16_user_40_100000round_alpha_4_0.1' + '_m_4' + '_d_10')
        # main(number, num_users=40, d=10, m=4, L=10, l_server_num=5, rounds=100000, filename='m_theta.npy',
        #      npzname='no' + str(number) + '_2_16_user_40_100000round_alpha_6_0.1' + '_m_4' + '_d_10')
        # main(number, num_users=40, d=10, m=4, L=10, l_server_num=5, rounds=100000, filename='m_theta.npy',
        #      npzname='no' + str(number) + '_2_16_user_40_100000round_alpha_8_0.1' + '_m_4' + '_d_10')
        # main(number, num_users=40, d=10, m=4, L=10, l_server_num=5, rounds=100000, filename='m_theta.npy',
        #      npzname='no' + str(number) + '_2_16_user_40_100000round_alpha_10_0.1' + '_m_4' + '_d_10')
        #L对比实验
        # main(number, num_users=10, d=10, m=5, L=10, l_server_num=2, rounds=100000, filename='m_2_10.npy',
        #      npzname='no' + str(number) + '_2_15_user_40_100000round_alpha_1_0.1' + '_L_2' + '_d_10')
        # main(number, num_users=15, d=10, m=5, L=10, l_server_num=3, rounds=100000, filename='m_3_15.npy',
        #      npzname='no' + str(number) + '_2_15_user_40_100000round_alpha_1_0.1' + '_L_3' + '_d_10')
        # main(number, num_users=20, d=10, m=5, L=10, l_server_num=4, rounds=100000, filename='m_4_20.npy',
        #      npzname='no' + str(number) + '_2_15_user_40_100000round_alpha_1_0.1' + '_L_4' + '_d_10')
        # main(number, num_users=25, d=10, m=5, L=10, l_server_num=5, rounds=100000, filename='m_5_25.npy',
        #      npzname='no' + str(number) + '_2_15_user_40_100000round_alpha_1_0.1' + '_L_5' + '_d_10')
        # main(number, num_users=30, d=10, m=5, L=10, l_server_num=6, rounds=100000, filename='m_6_30.npy',
        #      npzname='no' + str(number) + '_2_15_user_40_100000round_alpha_1_0.1' + '_L_8' + '_d_10')
        # main(number, num_users=40, d=10, m=5, L=10, l_server_num=8, rounds=100000, filename='m_8_40.npy',
        #      npzname='no' + str(number) + '_2_15_user_40_100000round_alpha_1_0.1' + '_L_57' + '_d_10')
        # main(number, num_users=50, d=10, m=5, L=10, l_server_num=8, rounds=100000, filename='m_10_50.npy',
        #      npzname='no' + str(number) + '_2_15_user_40_100000round_alpha_1_0.1' + '_L_57' + '_d_10')
    elif num_users == 60:
        main(number, num_users=60, d=10, m=5, L=10, l_server_num=5, T=100000, filename='60_theta.npy',
             npzname='no' + str(number) + '_2_23_user_60_100000round_1_0.1'+'_m_4'+ '_d_10')
    elif num_users == 50:
        # main(number, num_users=40, d=10, m=4, L=10, l_server_num=5, rounds=200000, filename='ml_1000user_d10.npy',
        #      npzname='_no' + str(number) + '_4_5_user_50_100000round_movielens')
        # main(number, num_users=40, d=10, m=4, L=10, l_server_num=5, rounds=100000, filename='m_theta.npy',
        #      npzname='no' + str(number) + '_2_20_user_50_100000round_synthetic')
        # main(number, num_users=40, d=10, m=4, L=10, l_server_num=5, rounds=300000, filename='yelp_1000user_d10_m10.npy',
        #      npzname= '_no' + str(number) + '_4_21_user_50_300000round_yelp_20')
        main(number, num_users=40, d=10, m=4, L=10, l_server_num=5, rounds=300000, filename='yelp_1000user_d10_tmp.npy',
             npzname= '_no' + str(number) + '_4_21_user_50_300000round_yelp_20')
        # main(number, num_users=40, d=10, m=4, L=10, l_server_num=5, T=200000, filename='ml_1000user_d10_tmp.npy',
        #      npzname='_no' + str(number) + '_4_5_user_50_100000round_movielens')
    elif num_users == 100:
        # main(number, num_users=100, d=10, m=10, L=10, l_server_num=5, rounds=200000, filename='ml_100user_d10.npy',
        #      npzname='no' + str(number) + '_1_26_user_100_200000round_movielens')
        main(number, num_users=100, d=10, m=10, L=10, l_server_num=5, T=200000, filename='',
             npzname='no' + str(number) + '_1_26_user_100_200000round_synthetic')
    elif num_users == 200:
        # main(number, num_users=200, d=10, m=10, L=10, l_server_num=5, rounds=200000, filename='ml_200user_d10.npy',
        #      npzname='no' + str(number) + '_1_26_user_200_200000round_movielens')
        main(number, num_users=200, d=10, m=10, L=10, l_server_num=5, T=200000, filename='',
             npzname='no' + str(number) + '_1_26_user_200_200000round_synthetic')
    elif num_users == 500:
        # main(number, num_users=500, d=10, m=10, L=10, l_server_num=5, rounds=500000, filename='ml_500user_d10.npy',
        #      npzname='no' + str(number) + '_1_26_user_500_500000round_movielens')
        main(number, num_users=500, d=10, m=10, L=10, l_server_num=5, T=500000, filename='',
             npzname='no' + str(number) + '_1_26_user_500_500000round_synthetic')




    #L : item的数量
    #main(num_users = 1, d = 10, m = 1, L = 1, l_server_num = 1, rounds = 1000000, filename= '')
    #m相当于cluster的数目？应该是num_user/theta的个数
    #main(num_users=100, d=10, m = 6 , L = 5, l_server_num=5, rounds = 100000, filename='')
    # main(number = 10,num_users=1000, d=10, m=20, L=8, l_server_num=10, rounds=2000000, filename='ml_1000user_d10.npy', npzname= "no10_user_1000_alpha_1.5_2000000round")
    # main(number = 10,num_users=100, d=10, m=10, L=8, l_server_num=10, rounds=1000000, filename='ml_100user_d10.npy',npzname='no10_user_100_1000000round_alpha_1.5')
    # main(number = 10,num_users=50, d=10, m=5, L=8, l_server_num=5, rounds=1000000, filename='ml_50user_d10.npy',
    #        npzname='no10_user_50_1000000round_alpha_1.5')
    # main(number = 10,num_users=20, d=10, m=4, L=8, l_server_num=4, rounds=100000, filename='ml_20user_d10.npy',
    #      npzname='no4_1_18_user_20_1000000round_alpha_1.5')





