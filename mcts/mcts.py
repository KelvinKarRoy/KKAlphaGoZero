import rule;

import mcts.node;

class Mcts(object):
    def __init__(self,rule,color):
        # TODO
        self.__root_node = mcts.Node(rule,color,self);
        self.node_map[0] = self.__root_node;
        self.__root_rule = rule;

    def get_rule(self):
        return self.__root_rule;

