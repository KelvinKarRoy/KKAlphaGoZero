import numpy as np




class PlayGo(object):
    __size = 19;  # 棋盘大小
    __context = np.zeros([__size, __size]);  # 棋盘内容
    BLACK = 1;
    WHITE = -1;
    EMPTY = 0;

    """
        构造函数
        psize: 棋盘绘制大小
        size: 棋盘大小
        context: 棋盘内容
    """

    def __init__(self, size=19, context= np.zeros([__size, __size]),history = {BLACK: [], WHITE: []}):
        self.__size = size;
        self.__context = context;  # 棋盘内容
        self.__history = history;  # 黑子白子历史 若不下则为[-1,-1]
    """
        获取每个落子的连通所有点
        例如
         0 0 1 0 -
         0 0 1 1 0
         0 1 1 - 0
         0 0 - - 0
         0 0 0 0 0
        getNeighbor(1,2) --- [[1,2],[0,2],[1,3],[2,2],[2,1]] 顺序不保证一致
        getNeighbor(2,3) --- [[2,3],[3,3],[3,2]]
    """
    def getNeighbor(self,x,y):
        is_visited = np.zeros([self.__size,self.__size]);
        queue = [];# 空队列
        neighbor = [];
        queue.append([x,y]);
        neighbor.append([x,y]);
        is_visited[x][y] = 1;
        color = self.__context[x][y];
        while queue.__len__() !=0:
            cur_point = queue.pop(0);
            # 如果上下左右颜色一致则入队
            cur_x = cur_point[0];
            cur_y = cur_point[1];
            visited_change = [[-1,0],[1,0],[0,-1],[0,1]];
            for vc in visited_change:
                if  cur_x+vc[0]>=0 and cur_x+vc[0]<self.__size\
                    and cur_y+vc[1] >= 0 and cur_y+vc[1] < self.__size\
                    and is_visited[cur_x+vc[0]][cur_y+vc[1]] == 0\
                    and self.__context[cur_x+vc[0]][cur_y+vc[1]] == color:
                    queue.append([cur_x+vc[0],cur_y+vc[1]]);
                    neighbor.append([cur_x+vc[0],cur_y+vc[1]]);
                    is_visited[cur_x+vc[0]][cur_y+vc[1]] = 1;
        return neighbor;

    """
        得到每个落点气的数目
         例如
         0 0 1 0 -
         0 0 1 1 0
         0 1 1 - 0
         0 0 - - 0
         0 0 0 0 0
         将得到
         2 2 8 0 2
         3 2 8 8 1
         2 8 8 5 2
         3 2 5 5 2
         2 3 2 2 2
    """
    def getKyu(self):
        kyu = np.zeros([self.__size,self.__size]); # 气初始化为0
        for x in range(self.__size):
            for y in range(self.__size):
                if self.__context[x][y] == self.EMPTY :
                    # 若是一个为空 给周围不为空的邻居都加一 给周围空加一
                    visited_change = [[-1, 0], [1, 0], [0, -1], [0, 1]];
                    for vc in visited_change:
                        if x+vc[0] < 0 or x+vc[0] >= self.__size or y+vc[1] < 0 or y+vc[1] >= self.__size:
                            continue;
                        if self.__context[x+vc[0]][y+vc[1]] == self.EMPTY:
                            kyu[x+vc[0]][y+vc[1]] = kyu[x+vc[0]][y+vc[1]] + 1;
                        else:
                            neighbor = self.getNeighbor(x+vc[0],y+vc[1]);
                            for nei in neighbor:
                                kyu[nei[0]][nei[1]] = kyu[nei[0]][nei[1]] + 1;
        return kyu;


    """
        判断提子
        输入 x y color
        返回 可以提子的集合
        例如
        0 0 1 1 -
        0 1 - - -
        0 - 1 1 0
        0 - - - -
        0 0 0 0 0
        getCapture(2,4,1) --- [[0, 4], [1, 2], [1, 3], [1, 4]]
        getCapture(2,4,-1) --- [[2, 2], [2, 3]]
    """
    def getCapture(self,x,y,color):
        capture = [];
        if self.__context[x][y] != self.EMPTY:
            return capture;
        self.__context[x][y] = color;
        kyu = self.getKyu();
        for _x in range(self.__size):
            for _y in range(self.__size):
                if kyu[_x][_y] == 0 and self.__context[_x][_y] == -1*color:
                    capture.append([_x,_y]);
        self.__context[x][y] = self.EMPTY;# 引用类型 所以还原
        return capture;



    """
        判断某点是否可以落子
    """
    def is_acceptable(self,x,y,color):
        is_acceptable = False;
        # 循环打劫需要去掉 上次走过的子不能走
        last_step = [-1, -1];
        if self.__history[color] != []:
            last_step = self.__history[color][-1];
        if self.__context[x][y] == self.EMPTY:
            self.__context[x][y] = color;
            kyu = self.getKyu();
            self.__context[x][y] = self.EMPTY;
            if kyu[x][y] > 0 and (x != last_step[0] or y != last_step[1]):
                is_acceptable = True;
            else:
                capture = self.getCapture(x, y, color);
                if capture != [] and (x != last_step[0] or y != last_step[1]):
                    is_acceptable = True;
        return is_acceptable;

    """
        对于某种颜色 哪些子可以走
        输入color
        例1：
        1 1 0 1 1
        1 1 - - -
        1 - 0 1 -
        1 - 1 1 1
        1 - - - -
        注意中间那一点是都可以下的
        上面那一点是-能下的 1不能下
        getAcceptable(1) --- [[2,2]]
        getAcceptable(-1) --- [[0,2],[2,2]]
        例2：
        0 - 1 0
        - 1 0 1
        0 - 1 0
        0 1 0 0
        history={
                    black:[[1,1]],
                    white:[[1,2]]
                }
        注意[0,3]和[1,2]两处白旗是不能走的 特别是[1,2]出现了循环打劫
        playGo.getAcceptable(1) --- [[0, 0], [0, 3], [1, 2], [2, 0], [2, 3], [3, 0], [3, 2], [3, 3]]
        playGo.getAcceptable(-1) --- [[0, 0], [2, 0], [2, 3], [3, 0], [3, 2], [3, 3]]
    """
    def getAcceptable(self,color):
        acceptable = [];
        for x in range(self.__size):
            for y  in range(self.__size):
                if self.is_acceptable(x,y,color) == True :
                    acceptable.append([x,y]);
        return acceptable;



    """
        落子
        这里不做能不能落子的检测 
    """
    def __move(self,x,y,color):
        # 提子
        capture = self.getCapture(x,y,color);
        for cap in capture:
            self.__context[cap[0]][cap[1]] = self.EMPTY;
        self.__context[x][y] = color;
        self.__history[color].append([x,y]);

    """
        落子
        这里判断落子是否合法
    """
    def save_move(self,x,y,color):
        # 是否合法
        if self.is_acceptable(x,y,color):
            self.__move(self,x,y,color);
        else:
            print("落子不合法");
