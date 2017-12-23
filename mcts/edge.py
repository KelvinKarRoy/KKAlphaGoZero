import mcts.node;

class Edge(object):
    def __init__(self,playGo,color,pnode,action):
        self.__playGo = playGo;
        self.__color = color; # 此次action执行者
        self.__pnode = pnode; # 父节点
        self.__action = action;
        new_playGo = playGo.copy();
        new_playGo.save_move(action[0], action[1], color);
        self.__cnode = mcts.Node(new_playGo,-1 * color); # 子节点
        self.__access_num = 0;  # 访问次数
        self.__win_num = 0; # 胜利次数


    def get_u(self):
        pass

    def get_u_plus_v(self):
        pass


