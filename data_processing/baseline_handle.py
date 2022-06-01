import numpy as np
import pandas as pd

# -----------------------------process data for comparative experiment with baselines ------------------------ #

data = dict()
def open_file():
    for i in range(10):
        # communication cost file
        # data[i] = np.load('CLUB_no'+ str(i + 1) +'_2_16_user_40_100000round_alpha_1_0.1_m_4_d_10_comu_cost' + '.npz', allow_pickle=True)
        # data[i] = np.load('FCLUB_no' + str(i + 1) + '_2_16_user_40_100000round_alpha_1_0.1_m_4_d_10_comu_cost' + '.npz', allow_pickle=True)
        # data[i] = np.load('LinUCB_no' + str(i + 1) + '_2_16_user_40_100000round_alpha_1_0.1_m_4_d_10_comu_cost' + '.npz', allow_pickle=True)
        # data[i] = np.load('Homo_no' + str(i + 1) + '_2_16_user_40_100000round_alpha_1_0.1_m_4_d_10_comu_cost' + '.npz', allow_pickle=True)
        # data[i] = np.load('LDP_FCLUB_DC_no' + str(i + 1) + '_2_16_user_40_100000round_alpha_1_0.1_m_4_d_10_comu_cost' + '.npz', allow_pickle=True)
        # data[i] = np.load('homo_DC_no' + str(i + 1) + '_2_16_user_40_100000round_alpha_1_0.1_m_4_d_10_comu_cost' + '.npz', allow_pickle=True)
        # data[i] = np.load('SCLUB_no' + str(i + 1) + '_2_16_user_40_100000round_alpha_1_0.1_m_4_d_10_comu_cost' + '.npz', allow_pickle=True)
        # data[i] = np.load('CDP_FCLUB_DC_no' + str(i + 1) + '_2_16_user_40_100000round_alpha_1_0.1_m_4_d_10_comu_cost' + '.npz', allow_pickle=True)

        # movielens file
        # data[i] = np.load('CLUB__no'+ str(i + 1) +'_4_5_user_50_100000round_movielens' + '.npz', allow_pickle=True)
        # data[i] = np.load('FCLUB__no'+ str(i + 1) +'_4_5_user_50_100000round_movielens' + '.npz', allow_pickle=True)
        # data[i] = np.load('LinUCB__no'+ str(i + 1) +'_4_5_user_50_100000round_movielens' + '.npz', allow_pickle=True)
        # data[i] = np.load('Homo__no' + str(i + 1) + '_4_5_user_50_100000round_movielens' + '.npz', allow_pickle=True)
        # data[i] = np.load('LDP_FCLUB_DC__no'+ str(i + 1) +'_4_5_user_50_100000round_movielens' + '.npz', allow_pickle=True)
        # data[i] = np.load('homo_DC__no'+ str(i + 1) +'_4_5_user_50_100000round_movielens' + '.npz', allow_pickle=True)
        # data[i] = np.load('SCLUB__no'+ str(i + 1) +'_4_5_user_50_100000round_movielens' + '.npz', allow_pickle=True)
        data[i] = np.load('CDP_FCLUB_DC__no'+ str(i + 1) +'_4_5_user_50_100000round_movielens' + '.npz', allow_pickle=True)

        # yelp files
        # data[i] = np.load('CLUB__no'+ str(i + 1) +'_4_21_user_50_300000round_yelp_20' + '.npz', allow_pickle=True)
        # data[i] = np.load('FCLUB__no'+ str(i + 1) +'_4_21_user_50_300000round_yelp_20' + '.npz', allow_pickle=True)
        # data[i] = np.load('LinUCB__no'+ str(i + 1) +'_4_21_user_50_300000round_yelp_20' + '.npz', allow_pickle=True)
        # data[i] = np.load('Homo__no' + str(i + 1) + '_4_21_user_50_300000round_yelp_20' + '.npz', allow_pickle=True)
        # data[i] = np.load('LDP_FCLUB_DC__no'+ str(i + 1) +'_4_21_user_50_300000round_yelp_20' + '.npz', allow_pickle=True)
        # data[i] = np.load('homo_DC__no'+ str(i + 1) +'_4_21_user_50_300000round_yelp_20' + '.npz', allow_pickle=True)
        # data[i] = np.load('SCLUB__no'+ str(i + 1) +'_4_21_user_50_300000round_yelp_20' + '.npz', allow_pickle=True)
        # data[i] = np.load('CDP_FCLUB_DC__no' + str(i + 1) + '_4_21_user_50_300000round_yelp_20' + '.npz',
        #                   allow_pickle=True)


def read_information(reward, round, regret):
    for i in range(10):
        reward[i] = data[i]['reward']
        round[i] = data[i]['T']
        regret[i] = data[i]['G_server_regret']

    user_num = data[0]['nu']
    T = data[0]['T']
    return user_num, T


def process_data(reward, round, regret, T):
    # T = 200000
    # round[0] = 200000
    reward_result = list()
    regret_result = list()
    std_regret = list()
    tmp_reward = 0
    tmp_regret = 0
    tmp_std_regret = list()
    cumulitive_regret = [0 for i in range(0, 10)]
    for i in range(T):
        for j in range(10):
            tmp_reward += reward[j][i]
            tmp_regret += regret[j][i]
            cumulitive_regret[j] += regret[j][i]
            tmp_std_regret.append(regret[j][i])
        reward_result.append(tmp_reward/10)
        regret_result.append(tmp_regret/10)
        std_regret.append(np.std(tmp_std_regret))
        tmp_reward = 0
        tmp_regret = 0
        tmp_std_regret.clear()

    return regret_result, reward_result, std_regret



def main():
    open_file()
    reward = dict()
    regret = dict()
    round = dict()
    user_num, T = read_information(reward, round, regret)
    regret_result, reward_result, std_regret = process_data(reward, round, regret, T)
    # npzname = 'CLUB' + '5_22_' + '40_user' + '10_round'
    # npzname = 'FCLUB' + '5_22_' + '40_user' + '10_round'
    # npzname = 'LinUCB' + '5_22_' + '40_user' + '10_round'
    # npzname = 'Homo' + '5_22_' + '40_user' + '10_round'
    # npzname = 'FCLUB_DC' + '5_22_' + '40_user' + '10_round'
    # npzname = 'homo_DC' + '5_22_' + '40_user' + '10_round'
    # npzname = 'SCLUB' + '5_22_' + '40_user' + '10_round'
    npzname = 'CDP_FCLUB_DC' + '5_22_' + '40_user' + '10_round'
    np.savez(npzname, nu=user_num, T= round[0], G_server_regret= regret_result,
             reward=reward_result, std_regret= std_regret)





if __name__ == '__main__':
    main()


