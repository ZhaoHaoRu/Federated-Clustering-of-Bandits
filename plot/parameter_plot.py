import numpy as np
import matplotlib.pyplot as plt


# ------------------------- plot functions for the Comparative experiment on synthetic dataset ---------------------------------- #

# Label = ['L = 2', 'L = 4', 'L = 5', 'L = 6', 'L = 8']
# Label = ['m = 2', 'm = 4', 'm = 6', 'm = 8']
# Label = ['epsilon = 0.5', 'epsilon = 1', 'epsilon = 2', 'epsilon = 4', 'epsilon = 8']
# Label = ['d = 5', 'd = 10', 'd = 15', 'd = 20']
Label = ['n = 20', 'n = 40', 'n = 60', 'n = 80']

# Filename suffix
file_name = ['user_20', 'user_40', 'user_60', 'user_80']
# file_name = ['L_2', 'L_4', 'L_5', 'L_6', 'L_8']
# file_name = ['m_2', 'm_4', 'm_6', 'm_8']
# file_name = ['epsi_0.5', 'epsi_1', 'epsi_2', 'epsi_4', 'epsi_8']
# file_name = ['d_5', 'd_10', 'd_15', 'd_20']
data = dict()
for i in range(4):
    data[i] = np.load('CDP_FCLUB_DC_4_21_40_user10_round_' + file_name[i] + '.npz')

# pdf = PdfPages('n.pdf')
fig = plt.figure()
T = data[0]['T']
print("T", T)

color_map = list()
color_map.append('xkcd:green')
color_map.append('xkcd:red')
color_map.append('xkcd:blue')
color_map.append('xkcd:purple')
color_map.append('xkcd:brown')
color_map.append('xkcd:orange')
color_map.append('xkcd:magenta')
color_map.append('xkcd:pink')

marker = ['d', '*', '^', 's', 'x', 'p']
line_style = ['-', 'solid', '-.', '--', ':', 'dashdot']
markevery = [i * 10000 for i in range(1, 10)]
regret = list()
reward = list()
std_reward = list()
min_reward = list()
max_reward = list()
for i in range(4):
    regret.append(data[i]['G_server_regret'])
    reward.append(data[i]['reward'])
    std_reward.append(data[i]['std_regret'])
regret_range = np.arange(1, T + 1)
accumulate_regret = list()
Cumulative_regret = list()

cumulative_reward = list()
theta_norm = list()


def handle_regret(regret, std_regret, T):
    accumulate_regret = list()
    min_reward = list()
    max_reward = list()
    Cumulative_regret = 0
    cumulative_std = 0
    for i in range(T):
        Cumulative_regret += regret[i]
        accumulate_regret.append(Cumulative_regret)
        regret[i] = Cumulative_regret / (i + 1)
        if i == 10000:
            print(std_regret[i])
        min_reward.append(Cumulative_regret - std_regret[i])
        max_reward.append(Cumulative_regret + std_regret[i])
    return accumulate_regret, Cumulative_regret, min_reward, max_reward


# def Draw_aver_regret():
#     ax = fig.add_subplot(111)
#     for i in range(6):
#         accumulate_regret_tmp, Cumulative_regret_tmp = handle_regret(regret[i], T)
#         accumulate_regret.append(accumulate_regret)
#         Cumulative_regret.append(Cumulative_regret_tmp)
#         plt.plot(regret_range, regret[i][:T:], marker= marker[i], markevery= markevery, linestyle= line_style[i], color= color_map[i], ms=5, label=Label[i])
#     ax.set_ylabel('regret in each round')
#     my_x_ticks = np.arange(0, T + 1, T/5)
#     plt.xticks(my_x_ticks)
#     my_y_ticks = np.arange(0, 0.1, 0.01)
#     plt.yticks(my_y_ticks)
#     plt.xlabel("round")
#     plt.ylabel("aver_regret")
#     plt.legend()
#     plt.grid()
#     plt.show()


# accumulate regret
def Draw_accumulate_regret():
    ax = fig.add_subplot(111)
    max_regret = max(Cumulative_regret)
    for i in range(4):
        # print(accumulate_regret[i])
        plt.plot(regret_range, accumulate_regret[i], marker=marker[i], markevery=markevery, linestyle=line_style[i],
                 color=color_map[i], ms=5, label=Label[i])
        plt.fill_between(regret_range, max_reward[i], min_reward[i], facecolor=color_map[i], alpha=0.2)
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
    fig1 = plt.gcf()
    fig1.savefig('n.pdf')
    plt.show()


for i in range(4):
    accumulate_regret_tmp, Cumulative_regret_tmp, min_reward_tmp, max_reward_tmp = handle_regret(regret[i],
                                                                                                 std_reward[i], T)
    accumulate_regret.append(accumulate_regret_tmp)
    Cumulative_regret.append(Cumulative_regret_tmp)
    min_reward.append(min_reward_tmp)
    max_reward.append(max_reward_tmp)

# Draw_aver_regret()
Draw_accumulate_regret()
