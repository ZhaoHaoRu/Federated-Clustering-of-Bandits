import networkx as nx
import numpy as np

import Base
import Environment as Envi
from Environment import alpha, delt, epsi, sigma

S = 1

#class Global_server

#Global_server G_server


class Local_server:
    def __init__(self, nl, d, begin_num, T, edge_probability=1):
        self.nl = nl
        self.d = d
        self.rounds = T  # the number of all rounds
        #self.G = nx.gnp_random_graph(nl, edge_probability)
        # 生成无向完全图，此时index是真正的user index
        user_index_list = list(range(begin_num,begin_num + nl))
        self.G = nx.generators.classic.complete_graph(user_index_list)
        self.clusters = {0: Base.Cluster(b=np.zeros(d), T=0, V=np.eye(d), users_begin = begin_num, d=d, user_num = nl, t= self.rounds)}  # users的初始化方法直接抄了老师里面的，不知道对不对
        self.cluster_inds = dict()
        self.begin_num = begin_num
        for i in range(begin_num, begin_num + nl):
            self.cluster_inds[i] = 0    # 每个user所属于的cluster的index, key:user_index ,value:cluster_index
        #self.cluster_inds = np.zeros(nl)  # 每个user所属于的cluster的index,这种方法貌似不可行，得用字典，因为user index是从global的维度而言的
        self.num_clusters = np.zeros(self.rounds,np.int64)  # 每一轮中cluster的数量，总共记录round T 次
        self.num_clusters[0] = 1
        self.V = np.zeros((d,d)) # 对于server而言V，b，T是否是必要的？
        self.b = np.zeros(d)
        self.T = 0

    def recommend(self, M, b , beta, user_index, items):
        Minv = np.linalg.inv(M)
        # 将global server上算得的M，b，β传给当前的local server
        r_item_index = np.argmax(np.dot(Minv,b) + beta * (np.matmul(items, Minv) * items).sum(axis=1))
        return r_item_index

    def if_delete(self, user_index1, user_index2, cluster): # 这个cluster是要进行删边操作的cluster
        T1 = cluster.users[user_index1].T
        T2 = cluster.users[user_index2].T
        fact_T1 = np.sqrt((1 + np.log(1 + T1)) / (1 + T1))
        fact_T2 = np.sqrt((1 + np.log(1 + T2)) / (1 + T2))
        theta1 = cluster.users[user_index1].theta
        theta2 = cluster.users[user_index2].theta
        return np.linalg.norm(theta1 - theta2) > alpha * (fact_T1 + fact_T2)

    def update(self, user_index, t): # 这里的user_index是从0到n的一个自然数
        update_cluster = False
        c = self.cluster_inds[user_index] # 找到被更新的user所属于的local cluster
        #i = user_index-self.begin_num # 感觉图中的node是从0开始排序的，但是user index其实并不是
        i = user_index
        # 存储最开始的cluster，以便下一步使用
        origin_cluster = self.clusters[c]
        A = [a for a in self.G.neighbors(i)]
        for j in A:
            #user2_index = j + self.begin_num
            user2_index = j
            c2 = self.cluster_inds[user2_index] # c和c2应该是同一个cluster
            user1 = self.clusters[c].users[user_index]
            user2 = self.clusters[c2].users[user2_index]
            if user1.T != 0 and user2.T != 0 and self.if_delete(user_index, user2_index, self.clusters[c]):
                self.G.remove_edge(i,j)
                update_cluster = True

        if update_cluster:
            C =set()
            C = nx.node_connected_component(self.G, i) #对应的user index,这是更新后的cluster应该有的user
            remain_users = dict()
            for i in C:
                remain_users[i] = self.clusters[c].get_user(user_index)


            if len(C) < len(self.clusters[c].users):
                all_users_index = set(self.clusters[c].users) # 这是原始的cluster中的所有user index
                all_users = dict()
                for user_index in all_users_index:
                    all_users[user_index] = self.clusters[c].get_user(user_index)
                # 将当前cluster应有的user放到这个cluster中
                tmp_cluster = Cluster(b=sum([k.b for k in remain_users]), T =sum([k.T for k in remain_users]),
                                           V = sum([k.V for k in remain_users]), user_begin = min(remain_users), d = self.d, user_num = len(remain_users), t=self.rounds, users = remain_users)
                self.clusters[c] = tmp_cluster

                # 将新的cluster中包含的user从原始的cluster中删除
                for user_index in all_users_index:
                    if remain_users.__contains__(user_index):
                        all_users.pop(user_index)

                c = max(self.clusters) + 1
                while len(all_users) > 0:
                    j = np.random.choice(list(all_users))
                    C = nx.node_connected_component(self.G, j)
                    new_cluster_users = dict()
                    for k in C:
                        new_cluster_users[k] = origin_cluster.get_user(user_index)
                    self.clusters[c] = Cluster(b=sum([n.b for n in new_cluster_users]), T=sum([n.T for n in new_cluster_users]),
                                          V=sum([n.V for n in new_cluster_users]), user_begin = min(new_cluster_users), d = self.d, user_num = len(new_cluster_users), t=self.rounds, users = new_cluster_users)
                    for k in C:
                        self.cluster_inds[k] = c

                    c += 1
                    for user_index in all_users_index:
                        if new_cluster_users.__contains__(user_index):
                            all_users.pop(user_index)

        self.num_clusters[t] = len(self.clusters)



class Global_server:  # 最开始每个local_server中的user数目是已知的
    def __init__(self, L, n, userList, d, T):  # 这里n指的是user的总数量，假设最开始是已知的,每个cluster的user的数量已知，存储在numlist中
        self.l_server_list = []
        self.usernum = n
        self.rounds = T
        self.l_server_num = L
        self.d = d
        self.cluster_usernum = np.zeros(L,np.int64)  # 这个是记录每个local——server的user数量的
        self.clusters = dict()
        user_begin = 0
        for i in range(L):
            self.clusters[i]= Base.Cluster(b=np.zeros(self.d), T=0, V=np.zeros((d,d)), users_begin=user_begin,d = self.d, user_num=userList[i], t=self.rounds, users = {}) # global_server上的cluster，最开始只有L个, 对应L个local server
            user_begin += userList[i]
        self.cluster_inds = np.zeros(n,np.int64)   # 存储每个user对应的global cluster的index,下标索引值代表了user index
        self.l_server_inds = np.zeros(n,np.int64)  # 存储每个user所对应的local server的index,现在这种方法是否可行的关键在于user属local server index 的信息会不会传回global server
        user_index = 0
        j = 0
        for i in userList: # userlist中记录的的是每个local_server中的user的数目
            self.l_server_list.append(Local_server(i, d, user_index, self.rounds))
            self.cluster_usernum[j] = i
            self.cluster_inds[user_index:user_index + i] = j
            self.l_server_inds[user_index:user_index + i] = j
            user_index = user_index + i
            j = j + 1

    def locate_user_index(self, user_index):
        # 如果这种方法不可行的话，只能强行遍历
        # 确定user属于哪个local server
        l_server_index = self.l_server_inds[user_index]
        #确定user属于哪个global cluster
        g_cluster_index = self.cluster_inds[user_index]
        return l_server_index, g_cluster_index

    def global_info(self,user_index, g_cluster_index):
       #l_server = self.l_server_list[l_server_index]
       g_cluster = self.clusters[g_cluster_index]
       V = g_cluster.V
       b = g_cluster.b
       T = g_cluster.T
       gamma_t = Envi.gamma(T+1, self.d, alpha, sigma)
       lambda_t = gamma_t * 2
       # 这里算β的S和L还不是很确定，S=L=1
       beta_t = Envi.beta(sigma, alpha, gamma_t, S, self.d, T+1, self.l_server_num)
       M_t = lambda_t*np.eye(self.d) + V
       # 接下来将M_t, b, beta传给local server，先直接调local server类中的函数
       #l_server.recommend(M_t, b, beta_t, user_index)
       return M_t, b, beta_t

    def merge(self):
        cmax = max(self.clusters)
        for c1 in range(cmax-1):
            if c1 not in self.clusters:
                continue
            for c2 in range(c1 + 1,cmax):
                if c2 not in self.clusters:
                    continue
                T1 = self.clusters[c1].T
                T2 = self.clusters[c2].T
                fact_T1 = np.sqrt((1 + np.log(1 + T1)) / (1 + T1))
                fact_T2 = np.sqrt((1 + np.log(1 + T2)) / (1 + T2))
                theta1 = self.clusters[c1].theta
                theta2 = self.clusters[c2].theta
                if(np.linalg.norm(theta1 - theta2) >= alpha * (fact_T1 + fact_T2)):
                    continue
                else:
                    for i in range(self.usernum):
                        if self.cluster_inds[i] == c2:
                            self.cluster_inds[i] = c1

                    self.clusters[c1].V = self.clusters[c1].V + self.clusters[c2].V
                    self.clusters[c1].b = self.clusters[c1].b + self.clusters[c2].b
                    self.clusters[c1].T = self.clusters[c1].T + self.clusters[c2].T
                    self.clusters[c1].user_num = self.clusters[c1].user_num + self.clusters[c2].user_num
                    for user in self.clusters[c2].users:
                        self.clusters[c1].users.setdefault(user, self.clusters[c2].users[user] )
                    del self.clusters[c2]

    def run(self, envir, T):
        for i in range(1, T+1):
            user_all = envir.generate_users()
            user_index = user_all[0]
            l_server_index, g_cluster_index = self.locate_user_index(user_index)
            M_t, b, beta_t = self.global_info(user_index, g_cluster_index)
            l_server = self.l_server_list[l_server_index]
            g_cluster = self.clusters[g_cluster_index]
            l_cluster = l_server.clusters[l_server.cluster_inds[user_index]]
            # the context set
            items = envir.get_items()
            r_item_index = l_server.recommend(M_t, b, beta_t, user_index, items)
            x = items[r_item_index]
            reward, y, best_reward, ksi_noise, B_noise = envir.feedback(items,i, b, M_t, r_item_index, self.d)
            l_cluster.users[user_index].store_info(x, y, i - 1, reward, best_reward, ksi_noise[0], B_noise)
            l_cluster.store_info(x, y, i - 1, reward, best_reward, ksi_noise[0], B_noise)
            # 这一步相当于delete edge 并计算 aggregated information，但是没有send to global server 这一步
            l_server.update(user_index,i - 1)
            g_cluster.store_info(x, y, i - 1, reward, best_reward, ksi_noise[0], B_noise)
            self.merge()



































