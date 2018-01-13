# -*- coding: utf-8 -*-
import node;

from utils import Utils


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
        self.__mcts = _mcts;
        self.__pa = 0;

    def get_action(self):
        return self.__action;

    def get_n(self):
        return self.__access_num;

    def get_win_num(self):
        return self.__win_num;

    def get_u(self,tao):
        # TODO 计算U的值
        e = 1.0 / tao;
        pnode = self.__pnode;
        sum_n = 1; # 避免除以零
        for e in pnode.get_edges:
            sum_n += e.get_n() ** e;
        return self.__mcts.c_puct * self.get_psa() * self.get_n() ** e / sum_n;

    def get_psa(self):
        # 增加dirichlet分布噪音
        return (1 - self.__mcts.epsilon) * self.__pa + self.__mcts.epsilon * Utils.get_dirichlet(self.__mcts.dirichlet);

    def get_u_plus_v(self):
        return self.get_u() + self.__cnode.get_v() ;


    def update_para(self,is_win = True):
        if is_win:
            self.__win_num += 1;
        self.__access_num += 1;

    def set_pa(self,pa):
        self.__pa = pa;
