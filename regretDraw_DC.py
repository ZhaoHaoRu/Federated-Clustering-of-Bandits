import numpy as np
import matplotlib.pyplot as plt

data = np.load('LDP_FCLUB_DC_12_25_user_10_1.npz',allow_pickle=True)
fig = plt.figure()
T = data['T']
print('T:',T)

regret = data['G_server_regret']
print(regret.shape)
regret_range = np.arange(1,T + 1)
# reward = data['reward']
cumulative_reward = list()
accumulate_regret = list()
Cumulative_regret = 0

for i in range(T):
    Cumulative_regret += regret[i]
    accumulate_regret.append(Cumulative_regret)
    regret[i] = Cumulative_regret/(i + 1)

#average regret
ax = fig.add_subplot(111)
plt.plot(regret_range, regret[:T:], 'r.-', ms=2, label="regret")
ax.set_ylabel('regret in each round')
my_x_ticks = np.arange(0, T + 1, T/5)
plt.xticks(my_x_ticks)
my_y_ticks = np.arange(0, 1, 0.1)
plt.yticks(my_y_ticks)
plt.xlabel("round")
plt.ylabel("regret")
plt.legend()
plt.grid()
plt.show()

#accumulate regret
ax = fig.add_subplot(111)
plt.plot(regret_range, accumulate_regret, 'r.-', ms=2, label="cumulative regret")
ax.set_ylabel('regret in each round')
my_x_ticks = np.arange(0, T + 1, T/5)
plt.xticks(my_x_ticks)
my_y_ticks = np.arange(0, accumulate_regret[-1],accumulate_regret[-1]/10)
plt.yticks(my_y_ticks)
plt.xlabel("round")
plt.ylabel("cumulative regret")
plt.legend()
plt.grid()
plt.show()


