import networkx as nx
import numpy as np


# d = {1:'a',2:'d',3:'b',5:'c'}
# a = np.arange(25).reshape(5, 5)
# ixgrid = a[0:3, 4:5]
# ixgrid2 = a[0:3, 0:3]
# print(a)
# print(ixgrid)
# print(ixgrid2)
# users = {0:1, 2:3, 4:5, 6:7}
# checks = {i: False for i in users}
# print(type(checks))
# print(checks)

# a = (2,3)
# b = (3,2)
#
# if a == b:
#     print('True!')
# result= dict(sorted(d.items(), key=lambda k: k[0]))
# result_tmp=list(result.values())
#
# print(result_tmp)
# print(type(result_tmp))

def countBits(n):
    List = list()
    bit_num = 0
    t = n
    while n != 0:
        if n % 2 != 0:
            left = t - 2 ** bit_num + 1
            right = t
            List.append(tuple([left, right]))
            t = left - 1

        n = n // 2
        bit_num += 1

    return List

a = countBits(11)

print(a)