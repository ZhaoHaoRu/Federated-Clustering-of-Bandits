import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ------------------------- plot functions for the comparative plot ---------------------------------- #

Label = ['FCLUB', 'FCLUB_DC', 'CDP_FCLUB_DC']
Label2 = ['FCLUB', 'FCLUB_DC', 'CDP_FCLUB_DC']
data = dict()
for i in range(3):
    data[i] = np.load(Label[i] + '_4_4_40_user10_round_comu_cost_0.01_true'+'.npz')
fig = plt.figure()
T = data[0]['T']
print("T", T)

pdf = PdfPages('comu_cost.pdf')
marker = ['d', '*', '^', 's', 'x', 'p']
line_style = ['--', '-.','solid', '--', ':', 'dashdot']
markevery=[i*10000 for i in range(1,10)]
color_map = list()
color_map.append('xkcd:green')
color_map.append('xkcd:blue')
color_map.append('xkcd:red')

min_comu = list()
max_comu = list()
std_comu = list()
comu_cost = list()
for i in range(3):
    comu_cost.append(data[i]['comu_cost'])
    std_comu.append(data[i]['std_comu'])
regret_range = np.arange(1,T + 1)
Cumulative_comu_cost = list()

def handle_comu_cost(std_comu, comu_cost):
    min_comu = list()
    max_comu = list()
    for i in range(T):
        min_comu.append(comu_cost[i] - std_comu[i])
        max_comu.append(comu_cost[i] + std_comu[i])
        print("gap:", max_comu[i] - min_comu[i])
    return min_comu, max_comu


def Draw_accumulate_comu_cost():
    ax = fig.add_subplot(111)
    for i in range(3):
        min_comu_tmp, max_comu_tmp = handle_comu_cost(std_comu[i], comu_cost[i])
        print(min_comu_tmp)
        print(max_comu_tmp)
        min_comu.append(min_comu_tmp)
        max_comu.append(max_comu_tmp)
    for i in range(0,3):
        # print(accumulate_regret[i])
        plt.plot(regret_range, comu_cost[i], color=color_map[i], ms=5, label=Label2[i],  marker= marker[i], markevery= markevery, linestyle= line_style[i])
        # plt.fill_between(regret_range, max_comu[i], min_comu[i], facecolor=color_map[i], alpha=0.2)
    # plt.plot(regret_range, accumulate_regret, 'r.-', ms=2, label="cumulative regret")
        print(comu_cost[i].shape)
        print(regret_range.shape)
    my_x_ticks = np.arange(0, T + 1, T / 5)
    plt.xticks(my_x_ticks)
    # plt.ylim((0, 100001))
    print(comu_cost[0][-1])
    my_y_ticks = np.arange(0, comu_cost[0][-1], comu_cost[0][-1]/10)
    plt.yticks(my_y_ticks)
    plt.xlabel("round")
    plt.ylabel("cumulative communication cost")
    plt.legend()
    plt.grid()
    fig1 = plt.gcf()
    fig1.savefig('cumulative communication cost movielens.pdf')
    plt.show()

Draw_accumulate_comu_cost()



