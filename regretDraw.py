import numpy as np
import matplotlib.pyplot as plt

#FCLUB_3_no1_3_2_user_20_100000round_yelp
data = np.load('LinUCB4_21_40_user10_round.npz',allow_pickle=True)
fig = plt.figure()
# seed = data['seed']
# communication_cost = data['comu_cost']
# print(type(communication_cost))
# np.append(communication_cost, communication_cost[-1])
# print("communication_cost:",communication_cost.shape)
# print(seed)
# fig_th = plt.figure()
# fig_ac = plt.figure()
T = data['T']
print("T", T)
regret = data['G_server_regret']
print(type(regret))
print(regret.shape)
regret_range = np.arange(1,T + 1)
print("regret_range:", regret_range.shape)
reward = data['reward']
cumulative_reward = list()
# theta_hat = data['theta_exp']
# theta = data['theta_theo']
nu = data['nu']
theta_norm = list()
accumulate_regret = list()
Cumulative_regret = 0

for i in range(T):
    Cumulative_regret += regret[i]
    accumulate_regret.append(Cumulative_regret)
    regret[i] = Cumulative_regret/(i + 1)
ax = fig.add_subplot(111)
plt.plot(regret_range, regret[:T:], 'r.-', ms=2, label="regret")
ax.set_ylabel('regret in each round')
my_x_ticks = np.arange(0, T + 1, T/5)
plt.xticks(my_x_ticks)
my_y_ticks = np.arange(0, 0.1, 0.01)
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
my_y_ticks = np.arange(0, accumulate_regret[-1] + 10,accumulate_regret[-1]/10)
plt.yticks(my_y_ticks)
plt.xlabel("round")
plt.ylabel("cumulative regret")
plt.legend()
plt.grid()
plt.show()

# print(theta.shape)
# print(theta_hat)

#communication cost
# ax = fig.add_subplot(111)
# regret_range = np.arange(1, T)
# plt.plot(regret_range, communication_cost[:T:], 'r.-', ms=2, label="communication cost")
# ax.set_ylabel('communication cost in each round')
# my_x_ticks = np.arange(0, T + 1, T/5)
# plt.xticks(my_x_ticks)
# my_y_ticks = np.arange(0, communication_cost[-1],communication_cost[-1]/10)
# plt.yticks(my_y_ticks)
# plt.xlabel("round")
# plt.ylabel("cumulative communication cost")
# plt.legend()
# plt.grid()
# print("get")
# plt.show()

#one user
# x = data['x']
# y = data['y']
cmu_reward = 0
aver_reward = list()
for j in range(T):
    # theta_norm.append(np.linalg.norm(theta[0]-theta_hat[j]))
    cmu_reward += reward[j]
    cumulative_reward.append(cmu_reward)
    aver_reward.append(cmu_reward/(j + 1))
ax = fig.add_subplot(111)
# plt.plot(np.arange(0,T), theta_norm, 'r.-', ms=2, label="theta_norm")
# ax.scatter(regret_range, regret, color='r', label="regret")
my_x_ticks = np.arange(0, T + 1, T/5)
theta1 = theta_hat[9999]
gap = list()
for i in range(10):
    gap.append(theta[0][i] - theta1[i])
# print(gap)
plt.xticks(my_x_ticks)
my_y_ticks = np.arange(0, 1, 0.1)
plt.yticks(my_y_ticks)
plt.xlabel("user")
plt.ylabel("theta_norm")
plt.legend()
plt.grid()
plt.show()

print(cumulative_reward)
# aver_reward
cmu_reward = 0
aver_reward = list()
for j in range(T):
    cmu_reward += reward[j]
    cumulative_reward.append(cmu_reward)
    aver_reward.append(cmu_reward / (j + 1))
ax = fig.add_subplot(111)
plt.plot(np.arange(0,T), aver_reward, 'r.-', ms=1, label="aver_reward")
my_x_ticks = np.arange(0, T + 1, T/5)
plt.xticks(my_x_ticks)
my_y_ticks = np.arange(0, 1, 0.1)
plt.yticks(my_y_ticks)
plt.xlabel("round")
plt.ylabel("aver_reward")
plt.grid()
plt.show()

# cmu_reward
plt.plot(np.arange(0,T), cumulative_reward, 'r.-', ms=1, label="cumulative_reward")
plt.xlabel("round")
plt.ylabel("cumulative_reward")
plt.grid()
plt.show()

# print(len(x))
# print(len(y))
# print(x[1])
# print(y)
# for i in range(T):
#     x[i] = np.linalg.norm(x[i])
#     y[i] = np.linalg.norm(y[i])
#
# print(x[1])
# print(y)
# plt.scatter(x,y,s= 1,label="x/y")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.grid()
# plt.show()

# for j in range(nu):
#     theta_norm.append(np.linalg.norm(theta[j]-theta_hat[j]))
# ax = fig.add_subplot(111)
# plt.plot(np.arange(0,nu), theta_norm, 'r.-', ms=2, label="theta_norm")
# # ax.scatter(regret_range, regret, color='r', label="regret")
# ax.set_ylabel('regret in each round')
# my_x_ticks = np.arange(0, nu, 10)
# plt.xticks(my_x_ticks)
# my_y_ticks = np.arange(0, 1, 0.1)
# plt.yticks(my_y_ticks)
# plt.xlabel("user")
# plt.ylabel("theta_norm")
# plt.legend()
# plt.grid()
# plt.show()



