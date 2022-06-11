import numpy as np
import pandas as pd

# ------------------- process the data for communication cost plot --------------------------- #

# Filename suffix
variable = 'comu_cost_0.01_true'
data = dict()
def open_file():
    for i in range(10):
        data[i] = np.load('CDP_FCLUB_DC__no' + str(i + 1) + '_4_1_user_50_100000round_movielens' + '.npz', allow_pickle=True)


def read_information(reward, round, regret, comu_cost):
    for i in range(10):
        reward[i] = data[i]['reward']
        round[i] = data[i]['T']
        regret[i] = data[i]['G_server_regret']
        comu_cost[i] = data[i]['comu_cost']

    user_num = data[0]['nu']
    return user_num


def process_data(reward, round, regret, comu_cost):
    #T = round[0]
    T = 100000
    round[0] = 100000
    reward_result = list()
    regret_result = list()
    comu_cost_result = list()
    tmp_reward = 0
    tmp_regret = 0
    tmp_cumulative_cost = 0
    std_comu = list()
    tmp_std_comu = list()
    for i in range(T):
        for j in range(10):
            tmp_reward += reward[j][i]
            tmp_regret += regret[j][i]
            tmp_cumulative_cost += comu_cost[j][i]
            tmp_std_comu.append(comu_cost[j][i])
        reward_result.append(tmp_reward/10)
        regret_result.append(tmp_regret/10)
        comu_cost_result.append(tmp_cumulative_cost/10)
        std_comu.append(np.std(tmp_std_comu))
        tmp_reward = 0
        tmp_regret = 0
        tmp_cumulative_cost = 0
        tmp_std_comu.clear()

    return regret_result, reward_result, comu_cost_result, std_comu



def main():
    open_file()
    reward = dict()
    regret = dict()
    round = dict()
    comu_cost = dict()
    user_num = read_information(reward, round, regret, comu_cost)
    regret_result, reward_result, comu_cost_result, std_comu = process_data(reward, round, regret, comu_cost)
    npzname = 'CDP_FCLUB_DC' + '_4_4_' + '40_user' + '10_round_' + variable
    # npzname = 'FCLUB' + '_4_4_' + '40_user' + '10_round_' + variable
    # npzname = 'FCLUB_DC' + '_4_4_' + '40_user' + '10_round_' + variable
    np.savez(npzname, nu=user_num, T= round[0], G_server_regret= regret_result,
             reward=reward_result, comu_cost= comu_cost_result, std_comu= std_comu)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


