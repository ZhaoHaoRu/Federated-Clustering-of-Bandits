import numpy as np


# ---------------------- process data for comparative experiment on synthetic dataset ------------------------- #


# vairiable changed, also the filename suffix
# variable = 'epsi_0.1'
# variable = 'epsi_0.5'
# variable = 'epsi_1'
# variable = 'epsi_2'
# variable = 'epsi_4'
# variable = 'epsi_6'
# variable = 'epsi_8'
# variable = 'epsi_10'
# variable = 'L_2'
# variable = 'L_3'
# variable = 'L_4'
# variable = 'L_5'
# variable = 'L_6'
# variable = 'L_8'
# variable = 'default'
# variable = 'm_2'
# variable = 'm_3'
# variable = 'm_5'
# variable = 'm_6'
# variable = 'm_8'
# variable = 'm_10'
# variable = 'user_20'
# variable = 'user_40'
# variable = 'user_60'
variable = 'user_80'
# variable = 'd_5'
# variable = 'd_15'
# variable = 'd_10'
# variable = 'd_20'

data = dict()


def open_file():
    for i in range(10):
        # Vary privacy budget ϵ and fix other variables
        # epsi = 0.1
        # data[i] = np.load('CDP_FCLUB_DC_no' + str(i + 1) + '_2_7_user_40_100000round_alpha_0.1_0.1_m_5_d_10' + '.npz', allow_pickle=True)
        # data[i] = np.load('CDP_FCLUB_DC_no' + str(i + 1) + '_2_16_user_40_100000round_alpha_0.1_0.1_m_4_d_10' + '.npz', allow_pickle=True)
        # epsi = 0.5
        # data[i] = np.load('CDP_FCLUB_DC_no' + str(i + 1) + '_2_16_user_40_100000round_alpha_0.5_0.1_m_4_d_10' + '.npz', allow_pickle=True)
        # data[i] = np.load('LDP_FCLUB_DC_no' + str(i + 1) + '_2_7_user_40_100000round_alpha_0.5_0.1_m_5_d_10' + '.npz', allow_pickle=True)
        # epsi = 1
        # data[i] = np.load('CDP_FCLUB_DC_no' + str(i + 1) + '_2_16_user_40_100000round_alpha_1_0.1_m_4_d_10' + '.npz', allow_pickle=True)
        # data[i] = np.load('LDP_FCLUB_DC_no' + str(i + 1) + '_2_7_user_40_100000round_alpha_1_0.1_m_5_d_10' + '.npz', allow_pickle=True)
        # epsi = 2
        # data[i] = np.load('CDP_FCLUB_DC_no' + str(i + 1) + '_2_16_user_40_100000round_alpha_2_0.1_m_4_d_10' + '.npz', allow_pickle=True)
        # data[i] = np.load('LDP_FCLUB_DC_no' + str(i + 1) + '_2_7_user_40_100000round_alpha_2_0.1_m_5_d_10' + '.npz', allow_pickle=True)
        # epsi = 4
        # data[i] = np.load('CDP_FCLUB_DC_no' + str(i + 1) + '_2_16_user_40_100000round_alpha_4_0.1_m_4_d_10' + '.npz', allow_pickle=True)
        # data[i] = np.load('LDP_FCLUB_DC_no' + str(i + 1) + '_2_7_user_40_100000round_alpha_4_0.1_m_5_d_10' + '.npz', allow_pickle=True)
        # epsi = 6
        # data[i] = np.load('CDP_FCLUB_DC_no' + str(i + 1) + '_2_16_user_40_100000round_alpha_6_0.1_m_4_d_10' + '.npz', allow_pickle=True)
        # data[i] = np.load('LDP_FCLUB_DC_no' + str(i + 1) + '_2_7_user_40_100000round_alpha_6_0.1_m_5_d_10' + '.npz', allow_pickle=True)
        # epsi = 8
        # data[i] = np.load('CDP_FCLUB_DC_no' + str(i + 1) + '_2_16_user_40_100000round_alpha_8_0.1_m_4_d_10' + '.npz', allow_pickle=True)
        # data[i] = np.load('LDP_FCLUB_DC_no' + str(i + 1) + '_2_7_user_40_100000round_alpha_8_0.1_m_5_d_10' + '.npz', allow_pickle=True)
        # epsi = 10
        # data[i] = np.load('CDP_FCLUB_DC_no' + str(i + 1) + '_2_12_user_40_100000round_alpha_10_0.1_m_5_d_10' + '.npz', allow_pickle=True)
        # data[i] = np.load('LDP_FCLUB_DC_no' + str(i + 1) + '_2_7_user_40_100000round_alpha_10_0.1_m_5_d_10' + '.npz', allow_pickle=True)
        # default
        # data[i] = np.load('CDP_FCLUB_DC_no' + str(i + 1) + '_2_12_user_40_100000round_alpha_1_0.1_m_5_d_10' + '.npz', allow_pickle=True)

        # Vary the number of global cluster m and fix other variables
        # m=2
        # data[i] = np.load('CDP_FCLUB_DC_no' + str(i + 1) + '_2_13_user_40_100000round_alpha_2_0.1_m_2_d_10' + '.npz', allow_pickle=True)
        # m=3
        # data[i] = np.load('CDP_FCLUB_DC_no' + str(i + 1) + '_2_13_user_40_100000round_alpha_2_0.1_m_3_d_10' + '.npz', allow_pickle=True)
        # m=5
        # data[i] = np.load('CDP_FCLUB_DC_no' + str(i + 1) + '_2_13_user_40_100000round_alpha_2_0.1_m_5_d_10' + '.npz', allow_pickle=True)
        # m=6
        # data[i] = np.load('CDP_FCLUB_DC_no' + str(i + 1) + '_2_13_user_40_100000round_alpha_2_0.1_m_6_d_10' + '.npz', allow_pickle=True)
        # m=8
        # data[i] = np.load('CDP_FCLUB_DC_no' + str(i + 1) + '_2_13_user_40_100000round_alpha_2_0.1_m_8_d_10' + '.npz', allow_pickle=True)
        # m = 10
        # data[i] = np.load('CDP_FCLUB_DC_no' + str(i + 1) + '_1_23_user_40_100000round_alpha_1_0.1_m_10_d_10' + '.npz', allow_pickle=True)

        # Vary the number of user n and fix other variables
        # user = 20
        # CDP_FCLUB_DC_no7_4_10_user_20_100000round_1_0.1_m_4_d_10
        # data[i] = np.load('CDP_FCLUB_DC_no' + str(i + 1) + '_4_10_user_20_100000round_1_0.1_m_4_d_10' + '.npz', allow_pickle=True)
        # user = 40
        # CDP_FCLUB_DC_no7_4_10_user_40_100000round_1_0.1_m_4_d_10
        # data[i] = np.load('CDP_FCLUB_DC_no' + str(i + 1) + '_4_10_user_40_100000round_1_0.1_m_4_d_10' + '.npz', allow_pickle=True)
        # user= 60
        # data[i] = np.load('CDP_FCLUB_DC_no' + str(i + 1) + '_4_10_user_60_100000round_1_0.1_m_4_d_10' + '.npz', allow_pickle=True)
        # user= 80
        data[i] = np.load('CDP_FCLUB_DC_no' + str(i + 1) + '_4_10_user_80_100000round_1_0.1_m_4_d_10' + '.npz',
                          allow_pickle=True)

        # Vary dimension d and fix other variables
        # d = 5
        # data[i] = np.load('CDP_FCLUB_DC_no' + str(i + 1) + '_2_16_user_40_100000round_alpha_1_0.1_m_4_d_5' + '.npz', allow_pickle=True)
        # d = 20
        # data[i] = np.load('CDP_FCLUB_DC_no' + str(i + 1) + '_2_16_user_40_100000round_alpha_1_0.1_m_4_d_20' + '.npz', allow_pickle=True)
        # d = 15
        # data[i] = np.load('CDP_FCLUB_DC_no' + str(i + 1) + '_2_16_user_40_100000round_alpha_1_0.1_m_4_d_15' + '.npz', allow_pickle=True)
        # d = 10
        # data[i] = np.load('CDP_FCLUB_DC_no' + str(i + 1) + '_2_16_user_40_100000round_alpha_1_0.1_m_4_d_10' + '.npz', allow_pickle=True)

        # Vary the number of local servers L and fix other variables
        # L = 2
        # data[i] = np.load('CDP_FCLUB_DC_no' + str(i + 1) + '_2_15_user_40_100000round_alpha_1_0.1_L_2_d_10' + '.npz',
        #                   allow_pickle=True)
        # L = 3
        # data[i] = np.load('CDP_FCLUB_DC_no' + str(i + 1) + '_2_15_user_40_100000round_alpha_1_0.1_L_3_d_10' + '.npz',
        #                   allow_pickle=True)
        # L = 4
        # data[i] = np.load('CDP_FCLUB_DC_no' + str(i + 1) + '_2_15_user_40_100000round_alpha_1_0.1_L_4_d_10' + '.npz',
        #                   allow_pickle=True)
        # L = 5
        # data[i] = np.load('CDP_FCLUB_DC_no' + str(i + 1) + '_2_15_user_40_100000round_alpha_1_0.1_L_5_d_10' + '.npz',
        #                   allow_pickle=True)
        # L = 6
        # data[i] = np.load('CDP_FCLUB_DC_no' + str(i + 1) + '_2_15_user_40_100000round_alpha_1_0.1_L_8_d_10' + '.npz',
        #                   allow_pickle=True)
        # L = 8
        # data[i] = np.load('CDP_FCLUB_DC_no' + str(i + 1) + '_2_15_user_40_100000round_alpha_1_0.1_L_57_d_10' + '.npz',
        #                   allow_pickle=True)


def read_information(reward, round, regret):
    for i in range(10):
        reward[i] = data[i]['reward']
        round[i] = data[i]['T']
        regret[i] = data[i]['G_server_regret']

    T = data[0]['T']
    user_num = data[0]['nu']
    return user_num, T


def process_data(reward, round, regret, T):
    # T = round[0]
    round[0] = T
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
        reward_result.append(tmp_reward / 10)
        regret_result.append(tmp_regret / 10)
        std_regret.append(np.std(cumulitive_regret))
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
    # npzname = 'CDP_FCLUB_DC' + '_2_9_' + '40_user' + '10_round_' + variable
    npzname = 'CDP_FCLUB_DC' + '_4_21_' + '40_user' + '10_round_' + variable
    np.savez(npzname, nu=user_num, T=round[0], G_server_regret=regret_result,
             reward=reward_result, std_regret=std_regret)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
