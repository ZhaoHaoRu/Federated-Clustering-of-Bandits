import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# FCLUB_data = np.load('FCLUB_no1_1_19_user_20_1000000round_alpha_1.5.npz', allow_pickle=True)
# FCLUB_DC_data = np.load('no_1_DC_1_1920_user_100000.npz', allow_pickle=True)
# Homo_data = np.load('Homo_no1_1_19_user_20_1000000round_alpha_1.5.npz', allow_pickle=True)
# Homo_DC_data = np.load('no_1homo_DC_1_1920_user_100000.npz', allow_pickle=True)
# CLUB_data = np.load('no_1_CLUB_1_1920_user_100000.npz', allow_pickle=True)
# LinUCB_data = np.load('LinUCB_no1_1_19_user_20_1000000round_alpha_1.5.npz', allow_pickle=True)
#SCLUB_data = np.load('no_1_SCLUB_1_1920_user_34466.npz', allow_pickle=True )

# FCLUB_3_no2_3_31_user_20_100000round_yelp
##FCLUB_3_no1_3_2_user_20_100000round_yelp
# pdf = PdfPages('yelp.pdf')
Label = ['FCLUB', 'FCLUB_DC', 'Homo', 'Homo_DC', 'CLUB', 'LinUCB','SCLUB','CDP_FCLUB_DC']
# Label = ['FCLUB', 'LDP_FCLUB_DC', 'Homo', 'Homo_DC', 'CLUB', 'LinUCB','SCLUB','CDP_FCLUB_DC']
data = dict()
for i in range(8):
    if i == 5:
        #CDP_FCLUB_DC2_18_40_user10_round_1
        data[i] = np.load(Label[i] + '5_21_40_user10_round' + '.npz')
        continue
    # elif i == 4:
    #     data[i] = np.load(Label[i] + '2_20_40_user10_round' + '.npz')
    #     data_tmp = np.load(Label[i] + '2_18_40_user10_round_2' + '.npz')
    # else:
    #CDP_FCLUB_DC3_21_40_user10_round
    data[i] = np.load(Label[i] + '5_21_40_user10_round' + '.npz')

fig = plt.figure()
# seed = FCLUB_data['seed']
# print(seed)
# fig_th = plt.figure()
# fig_ac = plt.figure()
# T = data[0]['rounds']
T = 300000
# print("rounds", T)

color_map = list()
color_map.append('xkcd:blue')
color_map.append('xkcd:green')
color_map.append('xkcd:magenta')
color_map.append('xkcd:pink')
color_map.append('xkcd:brown')
color_map.append('xkcd:purple')
color_map.append('xkcd:orange')
color_map.append('xkcd:red')

marker = ['d', '*', 's', 'x', 'p', '2' , '1', '^']
line_style = ['solid', 'solid', '-.', '--', ':', 'dashdot', '-.', 'solid']
markevery=[i*30000 for i in range(1,10)]
#handle regret
# regret1 = FCLUB_data['G_server_regret']
# print(type(regret1))
# regret2 = FCLUB_DC_data['G_server_regret']
# regret3 = Homo_data['G_server_regret']
# regret4 = Homo_DC_data['G_server_regret']
# regret5 = CLUB_data['G_server_regret']
# regret6 = LinUCB_data['G_server_regret']
# regret7 = SCLUB_data['G_server_regret']
# regret = [regret1, regret2, regret3, regret4, regret5, regret6, regret7]
regret = list()
reward = list()
std_reward = list()
min_reward = list()
max_reward = list()
n = 300000
for i in range(8):
    regret.append(data[i]['G_server_regret'])
    reward.append(data[i]['reward'])
    # std_reward.append(data[i]['std_regret'])
    # if i == 4:
    #     for j in range(n):
    #         regret[i][j] = (regret[i][j] + data_tmp['G_server_regret'][j]) / 2
regret_range = np.arange(1,T + 1)
accumulate_regret = list()
Cumulative_regret = list()

#handle reward
# reward1 = FCLUB_data['reward']
# # reward2 = FCLUB_DC_data['reward']
# reward2 = FCLUB_data['reward']
# reward3 = Homo_data['reward']
# #reward4 = Homo_DC_data['reward']
# reward4 = FCLUB_data['reward']
# reward5 = CLUB_data['reward']
# reward6 = LinUCB_data['reward']
# reward7 = SCLUB_data['reward']
# reward = [reward1, reward2, reward3, reward4, reward5, reward6, reward7]
cumulative_reward = list()
# theta_hat = data['theta_exp']
# theta = data['theta_theo']
nu = data[0]['nu']
theta_norm = list()


# def handle_regret(regret, std_regret, rounds):
#     accumulate_regret = list()
#     Cumulative_regret = 0
#     min_reward = list()
#     max_reward = list()
#     cumulative_std = 0
#     for i in range(rounds):
#         Cumulative_regret += regret[i]
#         accumulate_regret.append(Cumulative_regret)
#         regret[i] = Cumulative_regret / (i + 1)
#         # if i == 10000:
#         #     print(std_regret[i])
#         # min_reward.append(Cumulative_regret - std_regret[i])
#         # max_reward.append(Cumulative_regret + std_regret[i])
#     return accumulate_regret, Cumulative_regret, min_reward, max_reward

def handle_regret(regret, T):
    accumulate_regret = list()
    Cumulative_regret = 0
    min_reward = list()
    max_reward = list()
    cumulative_std = 0
    for i in range(T):
        Cumulative_regret += regret[i]
        accumulate_regret.append(Cumulative_regret)
        regret[i] = Cumulative_regret / (i + 1)
        # if i == 10000:
        #     print(std_regret[i])
        # min_reward.append(Cumulative_regret - std_regret[i])
        # max_reward.append(Cumulative_regret + std_regret[i])
    return accumulate_regret, Cumulative_regret, min_reward, max_reward

# def Draw_aver_regret():
#     ax = fig.add_subplot(111)
#     for i in range(8):
#         accumulate_regret_tmp, Cumulative_regret_tmp = handle_regret(regret[i], rounds)
#         accumulate_regret.append(accumulate_regret)
#         Cumulative_regret.append(Cumulative_regret_tmp)
#         plt.plot(regret_range, regret[i][:rounds:], '.-', color= color_map[i], ms=2, label=Label[i])
#     ax.set_ylabel('regret in each round')
#     my_x_ticks = np.arange(0, rounds + 1, rounds/5)
#     plt.xticks(my_x_ticks)
#     my_y_ticks = np.arange(0, 0.1, 0.01)
#     plt.yticks(my_y_ticks)
#     plt.xlabel("round")
#     plt.ylabel("aver_regret")
#     plt.legend()
#     plt.grid()
#     plt.show()


#accumulate regret
def Draw_accumulate_regret():
    ax = fig.add_subplot(111)
    max_regret = max(Cumulative_regret)
    for i in range(8):
        # print(accumulate_regret[i])
        plt.plot(regret_range, accumulate_regret[i], marker= marker[i], markevery= markevery, linestyle= line_style[i], color=color_map[i], ms=5, label=Label[i])
        # plt.fill_between(regret_range, max_reward[i], min_reward[i], facecolor=color_map[i], alpha=0.2)
    # plt.plot(regret_range, accumulate_regret, 'r.-', ms=2, label="cumulative regret")
    ax.set_ylabel('regret in each round')
    my_x_ticks = np.arange(0, T + 1, T / 5)
    plt.xticks(my_x_ticks)
    plt.ylim((0, 3501))
    my_y_ticks = np.arange(0, 3501, 350)
    plt.yticks(my_y_ticks)
    plt.xlabel("round")
    plt.ylabel("cumulative regret")
    plt.legend()
    plt.grid()
    fig1 = plt.gcf()
    fig1.savefig('yelp.pdf')
    plt.show()


for i in range(8):
    accumulate_regret_tmp, Cumulative_regret_tmp, min_reward_tmp, max_reward_tmp = handle_regret(regret[i], T)
    accumulate_regret.append(accumulate_regret_tmp)
    Cumulative_regret.append(Cumulative_regret_tmp)
    min_reward.append(min_reward_tmp)
    max_reward.append(max_reward_tmp)


# Draw_aver_regret()
Draw_accumulate_regret()
print("finish")
