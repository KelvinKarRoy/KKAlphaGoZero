# -*- coding: utf-8 -*-
import random

import edge;
import rule;


class Node(object):
    def __init__(self, rule, color, _mcts, depth=0):
        self.__context = rule.get_context_copy();
        self.__color = color;  # 接下来轮到谁
        # 有条件扩展
        self.__edges = [];  # 这个节点向下展开的边
        self.__access_num = 0;  # 访问次数
        self.__v_mean = 0;  # 历次访问v值的平均
        self.__mcts = _mcts;  # 属于哪棵树
        self.__depth = depth;  # 深度

    """
        getter & setter
    """

    def get_depth(self):
        return self.__depth;
    def get_context(self):
        return self.__context;
    def get_edges(self):
        return self.__edges;

    """扩展节点"""

    def expand(self, rule, color):
        acc_actions = rule.getAcceptable(color);  # 获取可执行action
        self.__edges = [mcts.Edge(rule, self.__mcts, color, self, [-1, -1])];
        for acc_action in acc_actions:
            self.__edges.append(mcts.Edge(rule, self.__mcts, color, self, acc_action));

    """
        访问节点 计算并更新相关参数
        如果是叶子结点且能够扩展则扩展
    """
    def visit(self):
        # TODO 获取v值 并更新mean
        if self.__edges == []:
            temp_rule = self.__mcts.get_rule().copy();
            temp_rule.remove_all(self.__context);
            self.expand(rule=temp_rule,color=self.__color);


    """
        获取这个节点平均的v值
    """
    def get_v(self):
        return self.__v_mean;

    # 获取u+v最大的一条边 叶子节点返回[]
    def get_max_u_plus_v_edge(self):
        max_uv_edge = [];
        max_u_plus_v = -1;
        for edge in self.__edges:
            if edge.get_u_plus_v() > max_u_plus_v:
                max_u_plus_v = edge.get_u_plus_v();
        if max_u_plus_v >= 0:
            for edge in self.__edges:
                if edge.get_u_plus_v() == max_u_plus_v:
                    max_uv_edge.append(edge);
        else:
            return max_uv_edge;
        return random.sample(max_uv_edge, 1)[0];

    # 获取模拟次数最多的一条边 叶子节点返回[]
    def get_max_n_edge(self):
        max_n_edge = [];
        max_n = -1;
        for edge in self.__edges:
            if edge.get_n() > max_n:
                max_n = edge.get_n();
        if max_n >= 0:
            for edge in self.__edges:
                if edge.get_n() == max_n:
                    max_n_edge.append(edge);
        else:
            return max_n_edge;
        return random.sample(max_n_edge, 1)[0];

    """ 更新所有子节点的Pa """
    def update_child_pa(self):
        context = self.__context;
        size = self.__mcts.get_rule().get_size();
        pa = self.__mcts.model.get_p(context,self.__color);
        if self.__edges != []:
            for e in self.__edges:
                action = e.get_action();
                e.set_pa(pa[action[0] * size + action[1]]);