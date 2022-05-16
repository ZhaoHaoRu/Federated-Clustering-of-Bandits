import numpy as np
import matplotlib.pyplot as plt


# theta= np.load("ml_100user_d10.npy")
# fig = plt.figure()
# print(type(theta))
# ax = fig.add_subplot(111)
# theta_norm = list()
# cluster1_theta_norm = list()
# for i in range(len(theta)):
#     theta_norm.append(np.linalg.norm(theta[i]))
#
# for i in range(20,30):
#     for j in range(20,30):
#         cluster1_theta_norm.append(np.linalg.norm(theta[i]-theta[j]))
#     print(i)
#     print(j)
#
# print(cluster1_theta_norm)
#
#
# #plt.plot(np.arange(0,len(theta)), theta_norm, 'r.-', ms=2, label="norm"
# plt.plot(np.arange(0,len(theta)), cluster1_theta_norm, 'r.-', ms=2, label="norm")
# ax.set_ylabel('regret in each round')
# my_x_ticks = np.arange(0, len(theta), 10)
# plt.xticks(my_x_ticks)
# my_y_ticks = np.arange(0, 1, 0.1)
# plt.yticks(my_y_ticks)
# plt.xlabel("user")
# plt.ylabel("theta_norm")
# plt.legend()

def generate_items(num_items, d):
    # return a ndarray of num_items * d
    x = np.random.normal(0, 1, (num_items, d - 1))
    x = np.concatenate((np.divide(x, np.outer(np.linalg.norm(x, axis=1), np.ones(np.shape(x)[1]))) / np.sqrt(2),
                        np.ones((num_items, 1)) / np.sqrt(2)), axis=1)
    return x

items_lamda = list()
mat_cal = np.zeros((10,10))
aver_lamda = 0
for i in range(1000):
    x = generate_items(10, 10)
    mat = np.matmul(x,x.T)
    mat_cal += mat
    eigenvalue = np.linalg.eigvalsh(mat)
    items_lamda.append(eigenvalue[0])
    print(eigenvalue[0])
    aver_lamda += eigenvalue[0]

print(aver_lamda/1000)
print(aver_lamda)
eigenvalue1 = np.linalg.eigvalsh((mat_cal/1000))
print(eigenvalue1[0])




