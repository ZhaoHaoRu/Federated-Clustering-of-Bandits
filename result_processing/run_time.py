import numpy as np
import pandas as pd

data = dict()

# --------------------------------- read and calculate the average run time -------------------------------------- #

def open_file():
    for i in range(10):
        # data[i] = np.load('CLUB_no'+ str(i + 1) +'_2_16_user_40_100000round_alpha_1_0.1_m_4_d_10_comu_cost' + '.npz', allow_pickle=True)
        # data[i] = np.load('FCLUB_no' + str(i + 1) + '_2_16_user_40_100000round_alpha_1_0.1_m_4_d_10_comu_cost' + '.npz', allow_pickle=True)
        # data[i] = np.load('LinUCB_no' + str(i + 1) + '_2_16_user_40_100000round_alpha_1_0.1_m_4_d_10_comu_cost' + '.npz', allow_pickle=True)
        # data[i] = np.load('Homo_no' + str(i + 1) + '_2_16_user_40_100000round_alpha_1_0.1_m_4_d_10_comu_cost' + '.npz', allow_pickle=True)
        # data[i] = np.load('LDP_FCLUB_DC_no' + str(i + 1) + '_2_16_user_40_100000round_alpha_1_0.1_m_4_d_10_comu_cost' + '.npz', allow_pickle=True)
        # data[i] = np.load('homo_DC_no' + str(i + 1) + '_2_16_user_40_100000round_alpha_1_0.1_m_4_d_10_comu_cost' + '.npz', allow_pickle=True)
        # data[i] = np.load('SCLUB_no' + str(i + 1) + '_2_16_user_40_100000round_alpha_1_0.1_m_4_d_10_comu_cost' + '.npz', allow_pickle=True)
        # data[i] = np.load('CDP_FCLUB_DC_no' + str(i + 1) + '_2_16_user_40_100000round_alpha_1_0.1_m_4_d_10_comu_cost' + '.npz', allow_pickle=True)

        # data[i] = np.load('CLUB_no'+ str(i + 1) +'_2_20_user_50_100000round_movielens' + '.npz', allow_pickle=True)
        # data[i] = np.load('FCLUB_no'+ str(i + 1) +'_2_18_user_50_100000round_movielens' + '.npz', allow_pickle=True)
        # data[i] = np.load('LinUCB_no'+ str(i + 1) +'_2_18_user_50_100000round_movielens' + '.npz', allow_pickle=True)
        # data[i] = np.load('Homo_no' + str(i + 1) + '_2_18_user_50_100000round_movielens' + '.npz', allow_pickle=True)
        # data[i] = np.load('LDP_FCLUB_DC_no'+ str(i + 1) +'_2_18_user_50_100000round_movielens' + '.npz', allow_pickle=True)
        # data[i] = np.load('homo_DC_no'+ str(i + 1) +'_2_18_user_50_100000round_movielens' + '.npz', allow_pickle=True)
        # data[i] = np.load('SCLUB_no'+ str(i + 1) +'_2_18_user_50_100000round_movielens' + '.npz', allow_pickle=True)
        data[i] = np.load('CDP_FCLUB_DC_no' + str(i + 1) + '_2_18_user_50_100000round_movielens' + '.npz',
                          allow_pickle=True)


def read_information():
    run_time_aver = 0
    for i in range(10):
        run_time_aver += data[i]['run_time']
    return run_time_aver / 10


def main():
    open_file()
    run_time_aver = read_information()
    print(run_time_aver)


if __name__ == '__main__':
    main()
