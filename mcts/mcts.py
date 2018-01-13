# -*- coding: utf-8 -*-
import rule;

import mcts.node;

class Mcts(object):
    # 狄利克雷噪音参数
    epsilon = 0.25;
    dirichlet = 0.03;

    # puct参数 极为重要
    c_puct = 1;

    def __init__(self,rule,model,color):
        # TODO
        self.__root_node = mcts.Node(rule,color,self);
        self.node_map[0] = self.__root_node;
        self.__root_rule = rule;
        self.model = model;

    def get_rule(self):
        return self.__root_rule;

