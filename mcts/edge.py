import mcts.node;

class Edge(object):
    def __init__(self,rule,_mcts,color,pnode,action):
        self.__color = color; # 此次action执行者
        self.__pnode = pnode; # 父节点
        self.__action = action;
        new_rule = rule.copy();
        new_rule.save_move(action[0], action[1], color);
        self.__cnode = mcts.Node(new_rule,-1 * color,_mcts,pnode.get_depth() + 1); # 子节点 子节点对应颜色相反
        _mcts.node_map[pnode.get_depth() + 1][new_rule.get_context_copy()] = self.__cnode; # 放入树的字典
        self.__access_num = 0;  # 访问次数
        self.__win_num = 0; # 胜利次数

    def get_n(self):
        return self.__access_num;

    def get_win_num(self):
        return self.__win_num;

    def get_u(self):
        # TODO 计算U的值
        pass

    def get_u_plus_v(self):
        return self.get_u() + self.__cnode.get_v() ;

    def update_para(self,is_win = True):
        if is_win:
            self.__win_num += 1;
        self.__access_num += 1;


