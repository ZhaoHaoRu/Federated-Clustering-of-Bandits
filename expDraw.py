import numpy as np
import matplotlib.pyplot as plt


# FCLUB_data = np.load('FCLUB_no1_1_19_user_20_1000000round_alpha_1.5.npz', allow_pickle=True)
# FCLUB_DC_data = np.load('no_1_DC_1_1920_user_100000.npz', allow_pickle=True)
# Homo_data = np.load('Homo_no1_1_19_user_20_1000000round_alpha_1.5.npz', allow_pickle=True)
# Homo_DC_data = np.load('no_1homo_DC_1_1920_user_100000.npz', allow_pickle=True)
# CLUB_data = np.load('no_1_CLUB_1_1920_user_100000.npz', allow_pickle=True)
# LinUCB_data = np.load('LinUCB_no1_1_19_user_20_1000000round_alpha_1.5.npz', allow_pickle=True)
#SCLUB_data = np.load('no_1_SCLUB_1_1920_user_34466.npz', allow_pickle=True )

Label = ['FCLUB', 'FCLUB_DC', 'Homo', 'Homo_DC', 'CLUB', 'LinUCB','SCLUB']
data = dict()
for i in range(6):
    data[i] = np.load(Label[i] + '1_18_' + '50_user' + '20_round'+'.npz')
fig = plt.figure()
# seed = FCLUB_data['seed']
# print(seed)
# fig_th = plt.figure()
# fig_ac = plt.figure()
T = data[0]['T']
print("T", T)

color_map = list()
color_map.append('xkcd:red')
color_map.append('xkcd:green')
color_map.append('xkcd:blue')
color_map.append('xkcd:pink')
color_map.append('xkcd:brown')
color_map.append('xkcd:purple')
color_map.append('xkcd:orange')



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
for i in range(6):
    regret.append(data[i]['G_server_regret'])
    reward.append(data[i]['reward'])
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


def handle_regret(regret, T):
    accumulate_regret = list()
    Cumulative_regret = 0
    for i in range(T):
        Cumulative_regret += regret[i]
        accumulate_regret.append(Cumulative_regret)
        regret[i] = Cumulative_regret / (i + 1)
    return accumulate_regret, Cumulative_regret


def Draw_aver_regret():
    ax = fig.add_subplot(111)
    for i in range(6):
        accumulate_regret_tmp, Cumulative_regret_tmp = handle_regret(regret[i], T)
        accumulate_regret.append(accumulate_regret)
        Cumulative_regret.append(Cumulative_regret_tmp)
        plt.plot(regret_range, regret[i][:T:], '.-', color= color_map[i], ms=2, label=Label[i])
    ax.set_ylabel('regret in each round')
    my_x_ticks = np.arange(0, T + 1, T/5)
    plt.xticks(my_x_ticks)
    my_y_ticks = np.arange(0, 0.1, 0.01)
    plt.yticks(my_y_ticks)
    plt.xlabel("round")
    plt.ylabel("aver_regret")
    plt.legend()
    plt.grid()
    plt.show()


#accumulate regret
def Draw_accumulate_regret():
    ax = fig.add_subplot(111)
    max_regret = max(Cumulative_regret)
    for i in range(6):
        # print(accumulate_regret[i])
        plt.plot(regret_range, accumulate_regret[i], '.-', color=color_map[i], ms=2, label=Label[i])
    # plt.plot(regret_range, accumulate_regret, 'r.-', ms=2, label="cumulative regret")
    ax.set_ylabel('regret in each round')
    my_x_ticks = np.arange(0, T + 1, T / 5)
    plt.xticks(my_x_ticks)
    plt.ylim((0, 2001))
    my_y_ticks = np.arange(0, 2001, 200)
    plt.yticks(my_y_ticks)
    plt.xlabel("round")
    plt.ylabel("cumulative regret")
    plt.legend()
    plt.grid()
    plt.show()


for i in range(6):
    accumulate_regret_tmp, Cumulative_regret_tmp = handle_regret(regret[i], T)
    accumulate_regret.append(accumulate_regret_tmp)
    Cumulative_regret.append(Cumulative_regret_tmp)

Draw_aver_regret()
Draw_accumulate_regret()

