import FCLUB
import Environment as Envi

d = 10 #dimension
user_num = 10  # the number of all users
I = 10  # the number of items
T = 100 # the number of rounds
L = 3  # the number of local server
userList = [3, 3, 4]




def main():
    G_server = FCLUB.Global_server(L, user_num, userList, d, T)
    envi  = Envi.Environment(d, user_num, I)
    G_server.run(envi, T)


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    main()


