import numpy as np
import matplotlib.pyplot as plt

data = np.load('FCLUB.npz')
fig = plt.figure()
T = data['T']
regret = data['G_server_regret']
regret_range = np.arange(1,T + 1)
Cumulative_regret = 0
for i in range(T):
    Cumulative_regret += regret[i]
    regret[i] = Cumulative_regret/(i + 1)
ax = fig.add_subplot(111)
plt.plot(regret_range, regret, 'r.-', ms=2, label="regret")
# ax.scatter(regret_range, regret, color='r', label="regret")
ax.set_ylabel('regret in each round')
my_x_ticks = np.arange(0, 500, 50)
plt.xticks(my_x_ticks)
my_y_ticks = np.arange(0, 1, 0.1)
plt.yticks(my_y_ticks)
plt.xlabel("round")
plt.ylabel("regret")
plt.legend()
plt.grid()
plt.show()