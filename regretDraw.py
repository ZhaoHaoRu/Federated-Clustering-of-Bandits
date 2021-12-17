import numpy as np
import matplotlib.pyplot as plt

data = np.load('beta_0.5_100user_200000round_alpha_1.npz',allow_pickle=True)
fig = plt.figure()
fig_th = plt.figure()
fig_ac = plt.figure()
T = data['T']
regret = data['G_server_regret']
regret_range = np.arange(1,T + 1)
theta_hat = data['theta_exp']
print(theta_hat.shape)
theta = data['theta_theo']
print(theta.shape)
nu = data['nu']
theta_norm = list()
accumulate_regret = list()
Cumulative_regret = 0
for i in range(T):
    Cumulative_regret += regret[i]
    accumulate_regret.append(Cumulative_regret)
    regret[i] = Cumulative_regret/(i + 1)
ax = fig.add_subplot(111)
plt.plot(regret_range, regret, 'r.-', ms=2, label="regret")
# ax.scatter(regret_range, regret, color='r', label="regret")
ax.set_ylabel('regret in each round')
my_x_ticks = np.arange(0, 220000, 20000)
plt.xticks(my_x_ticks)
my_y_ticks = np.arange(0, 1, 0.1)
plt.yticks(my_y_ticks)
plt.xlabel("round")
plt.ylabel("regret")
plt.legend()
plt.grid()
plt.show()

for j in range(nu):
    theta_norm.append(np.linalg.norm(theta[j]-theta_hat[j]))
ax = fig.add_subplot(111)
plt.plot(np.arange(0,nu), theta_norm, 'r.-', ms=2, label="theta_norm")
# ax.scatter(regret_range, regret, color='r', label="regret")
ax.set_ylabel('regret in each round')
my_x_ticks = np.arange(0, nu, 10)
plt.xticks(my_x_ticks)
my_y_ticks = np.arange(0, 1, 0.1)
plt.yticks(my_y_ticks)
plt.xlabel("user")
plt.ylabel("theta_norm")
plt.legend()
plt.grid()
plt.show()


#accumulate regret
ax = fig.add_subplot(111)
plt.plot(regret_range, accumulate_regret, 'r.-', ms=2, label="cumulative regret")
ax.set_ylabel('regret in each round')
my_x_ticks = np.arange(0, 220000, 20000)
plt.xticks(my_x_ticks)
my_y_ticks = np.arange(0, accumulate_regret[-1],accumulate_regret[-1]/10)
plt.yticks(my_y_ticks)
plt.xlabel("round")
plt.ylabel("cumulative regret")
plt.legend()
plt.grid()
plt.show()
