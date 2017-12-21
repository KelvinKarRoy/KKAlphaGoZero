import numpy as np

class PlayGo(object):
    __size = 19;  # 棋盘大小
    __context = np.zeros([__size, __size]);  # 棋盘内容
    WHITE, EMPTY, BLACK, FILL, KO, UNKNOWN = range(-1, 5);
    __komi = 7.5; # 黑子让目
    __passes_white = 0; # pass次数
    __passes_black = 0;
    __num_history = 8;# 考虑的历史次数
    __context_history = [np.zeros([__size, __size])];
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
        self.__context = np.zeros([size, size]);  # 棋盘内容
        self.__context_history = [np.zeros([size, size])];
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
        self.__context_history.append(np.copy(self.__context));

    """
        pass 即不下棋
    """
    def __pass(self,color):
        self.__history[color].append([-1, -1]);
        self.__context_history.append(np.copy(self.__context_history[-1])); # 棋局不变
        if color == self.WHITE:
            self.__passes_white = self.__passes_white + 1;
        else:
            self.__passes_black = self.__passes_black + 1;

    """
        落子
        这里判断落子是否合法
    """
    def save_move(self,x,y,color):
        # pass
        if x == -1 and y == -1:
            self.__pass(color);
            return [x,y];
        # 是否合法
        if self.is_acceptable(x,y,color):
            self.__move(x,y,color);
            return [x,y];
        else:
            print("落子不合法，自动视为pass");
            self.__pass(color);
            return [-1,-1];

    """
        判断是否为该颜色的眼
    """
    def is_eyeish(self, x, y, color):
        """returns whether the position is empty and is surrounded by all stones of 'owner'
        """
        if self.__context[x, y] != self.EMPTY:
            return False

        for (nx, ny) in self.getNeighbor(x,y):
            if self.__context[nx, ny] != color:
                return False
        return True

    """
        计算目数差（已让目）
        为正黑旗赢
    """
    def score(self):
        score_black = 0;
        score_white = 0;
        for x in self.__size:
            for y in self.__size:
                if self.__context == self.WHITE:
                    score_white = score_white + 1;
                elif self.__context == self.BLACK:
                    score_black = score_black + 1;
                else:
                    if self.is_eyeish(x,y, self.BLACK):
                        score_black = score_black + 1;
                    elif self.is_eyeish(x,y, self.WHITE):
                        score_white = score_white + 1;

        score_white += self.__komi; # 让目
        score_white -= self.__passes_white; # 减去pass的步骤
        score_black -= self.__passes_black;

        return score_black - score_white;

    """
            获取最后几步
        """

    def get_last_context(self):
        num_histroy = len(self.__context_history)
        if num_histroy < self.__num_history:
            last_context = (self.__num_history - num_histroy) * [np.zeros([self.__size, self.__size])];
            for his in self.__context_history:
                last_context.append(his.copy());
        else:
            last_context = self.__context_history[num_histroy - self.__num_history: num_histroy].copy();
        return last_context;


    """
        输入初始化
    """
    def process_input(self, color):
        x = [];
        last_context = self.get_last_context();
        # 输入格式为 [batch, in_height, in_width, in_channels] 不要batch 即后面三项
        for lc in last_context:
            x.append ((lc == self.BLACK) * 1);
        for lc in last_context:
            x.append((lc == self.WHITE) * 1);
        if color == self.WHITE:
            x.append(np.zeros([self.__size,self.__size]));
        else:
            x.append(np.ones([self.__size,self.__size]));

        x = np.transpose(x, ( 1, 2, 0));
        return x;



