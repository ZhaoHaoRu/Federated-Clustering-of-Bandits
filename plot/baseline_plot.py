import numpy as np
import matplotlib.pyplot as plt

# ---------------------------- comparison with baselines plot functions -------------------------------- #


Label = ['FCLUB', 'FCLUB_DC', 'Homo', 'Homo_DC', 'CLUB', 'LinUCB', 'SCLUB', 'CDP_FCLUB_DC']
color_map = list()
color_map.append('xkcd:blue')
color_map.append('xkcd:green')
color_map.append('xkcd:magenta')
color_map.append('xkcd:pink')
color_map.append('xkcd:brown')
color_map.append('xkcd:purple')
color_map.append('xkcd:orange')
color_map.append('xkcd:red')

marker = ['d', '*', 's', 'x', 'p', '2', '1', '^']
line_style = ['solid', 'solid', '-.', '--', ':', 'dashdot', '-.', 'solid']
# global markevery

data = dict()  # save all the files
fig = plt.figure()

regret = list()
reward = list()
std_reward = list()
lower_bounds = list()
upper_bounds = list()
T = 0

cumulative_regret_lists = list()
cumulative_regrets = list()


# process
def handle_regret(regret, T):
    cumulative_regret_list = list()
    cumulative_regret = 0
    regret_lower_bound = list()
    regret_upper_bound = list()
    # cumulative_std = 0
    for i in range(T):
        cumulative_regret += regret[i]
        cumulative_regret_list.append(cumulative_regret)
        regret[i] = cumulative_regret / (i + 1)
        # regret_lower_bound.append(cumulative_regret - std_regret[i])
        # regret_upper_bound.append(cumulative_regret + std_regret[i])
    return cumulative_regret_list, cumulative_regret, regret_lower_bound, regret_upper_bound


# accumulate regret
def plot_accumulate_regret(regret_range, T):
    ax = fig.add_subplot(111)
    max_regret = max(cumulative_regrets)
    for i in range(8):
        plt.plot(regret_range, cumulative_regret_lists[i], marker=marker[i], markevery=markevery,
                 linestyle=line_style[i],
                 color=color_map[i], ms=5, label=Label[i])
        # plt.fill_between(regret_range, upper_bounds[i], lower_bounds[i], facecolor=color_map[i], alpha=0.2)
    ax.set_ylabel('regret in each round')
    my_x_ticks = np.arange(0, T + 1, T / 5)
    plt.xticks(my_x_ticks)
    plt.ylim((0, 3001))
    my_y_ticks = np.arange(0, 3001, 300)
    plt.yticks(my_y_ticks)
    plt.xlabel("round")
    plt.ylabel("cumulative regret")
    plt.legend()
    plt.grid()
    fig1 = plt.gcf()
    fig1.savefig('yelp.pdf')
    plt.show()


def preparation():
    for i in range(8):
        data[i] = np.load(Label[i] + '6_1_40_user10_round' + '.npz')  # filename
    T = data[0]['T']
    global markevery
    markevery = [i * (int)(T / 10) for i in range(1, 10)]
    for i in range(8):
        regret.append(data[i]['G_server_regret'])
        reward.append(data[i]['reward'])
    nu = data[0]['nu']
    regret_range = np.arange(1, T + 1)

    for i in range(8):
        cumulative_regret_list, cumulative_regret, lower_bound, upper_bound = handle_regret(regret[i], T)
        cumulative_regret_lists.append(cumulative_regret_list)
        cumulative_regrets.append(cumulative_regret)
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)
    return regret_range, T


if __name__ == '__main__':
    regret_range, T = preparation()
    plot_accumulate_regret(regret_range, T)
    print("finish")
