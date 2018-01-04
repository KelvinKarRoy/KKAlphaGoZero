from abc import abstractmethod, ABCMeta

# 规则抽象类
class Rule(object):
    _metaclass__ = ABCMeta;
    _num_history = 8;  # 考虑的历史次数

    def __init__(self,size,context):
        self._size = size;
        self._context = context;

    """
        getter & setter
    """
    def get_size(self):
        return self._size;
    def get_num_history(self):
        return self._num_history;
    def get_context_copy(self):
        return  self._context.copy();

    """
        落子 以及落子后的变化
        这里判断必须落子是否合法
    """
    @abstractmethod
    def save_move(self, x, y, color):pass

    """
        获取所有当前color合法的位置（可以下的位置）
        输出格式：[[0, 0], [0, 3], [1, 2], [2, 0], [2, 3], [3, 0], [3, 2], [3, 3]]
    """
    @abstractmethod
    def getAcceptable(self, color): pass

    """
        输入初始化
        将棋盘信息变为神经网络的输入格式
        维度为[size,size,(2*num_history+1)]
        可以参考围棋代码
    """
    @abstractmethod
    def process_input(self, color): pass

    """
        撤销一步
    """
    @abstractmethod
    def undo(self): pass

    """
        计算结果
        围棋：计算目数差（已让目）
        五子棋：看黑棋是否胜利 黑赢1 白棋赢-1 和局（无子可下）0
        Reversi： 计算棋盘上黑子减去白棋数目差
        为正黑旗赢
    """
    @abstractmethod
    def score(self): pass


    """
        判断是否结束
        返回 True or False
    """
    @abstractmethod
    def is_over(self): pass

    """
        深度copy函数
    """
    @abstractmethod
    def copy(self): pass

    """
       remove_all函数,初始化棋盘信息
    """
    @abstractmethod
    def remove_all(self,context): pass



