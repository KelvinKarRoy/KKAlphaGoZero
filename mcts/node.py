import mcts.edge;

import Go.playGo;

class Node(object):
    def __init__(self,playGo,color):
        self.__playGo = playGo;
        self.__color = color; # 接下来轮到谁
        acc_actions = playGo.getAcceptable(color); # 获取可执行action
        self.__edges = [mcts.Edge(playGo,color,self,[-1,-1])];
        for acc_action in acc_actions:
            self.__edges.append(mcts.Edge(playGo,color,self,acc_action));
        self.__access_num = 0; # 访问次数
        self.vlist = []; # 历次访问的v值


    def get_v(self):
        pass

    def get_max_u_plus_v_edge(self):
        pass

    def get_max_n_edge(self):
        pass


