from PIL import Image
from pylab import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


"""
    用来可视化棋盘的类
"""
class ShowGo(object):
    __psize = 180; # 棋盘绘制大小
    __size = 19; # 棋盘大小
    __iter = int(__psize / (__size - 1));  # 像素间距

    """
        构造函数
        psize: 棋盘绘制大小
        size: 棋盘大小
    """
    def __init__(self,psize = 180,size = 19):
        self.__psize = psize;
        self.__size = size;
        self.__iter = int(psize / (size - 1));  # 像素间距

    """
        getter & setter
    """
    def get_size(self):
        return self.__size;

    """
        画格子
    """
    def __showCross(self):
        for x in range(0, self.__psize+self.__iter, self.__iter):
            plt.plot([x,x], [0, -1 * self.__psize], 'k', linewidth ='0.5')
            plt.plot([0,self.__psize], [-1 * x, -1 * x], 'k', linewidth ='0.5')

        # 隐藏坐标轴
        plt.axis('off')

    """
        显示棋盘内容
        context:棋盘内容
    """
    def __showContext(self, context=np.zeros([__size, __size])):
        blackPoint_x = [];
        blackPoint_y = [];
        whitePoint_x = [];
        whitePoint_y = [];
        for x in range(self.__size):
            for y in range(self.__size):
                if context[x][y] == 0:
                    continue
                elif int(context[x][y]) == 1:
                    blackPoint_x.append(-1 * x * self.__iter);
                    blackPoint_y.append(y * self.__iter);
                else:
                    whitePoint_x.append(-1 * x * self.__iter);
                    whitePoint_y.append(y * self.__iter);
        plt.scatter(blackPoint_y ,blackPoint_x, marker = 'o', c = 'k', s = 50);
        plt.scatter(whitePoint_y, whitePoint_x, marker = 'o', c = 'r', s = 50);
        plt.show()

    """
        显示整个棋盘（外部调用）
        context:棋盘内容
    """
    def show(self, context=np.zeros([__size, __size])):
        self.__showCross();
        self.__showContext(context);