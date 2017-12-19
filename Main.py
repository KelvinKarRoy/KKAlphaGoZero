from showGo import ShowGo, plt
from PlayGo import PlayGo
import numpy as np


#plt.scatter([32],[50], marker = 'o', c = 'k', s = 15);
#plt.show();



"""
    测试画图

showGo = ShowGo();

context = np.zeros([showGo.get_size(), showGo.get_size()])
context[5][4] = 1
context[12][10] = -1
context[7][11] = 1

showGo.show(context);

"""

"""
    测试 PlayGo.GetNeighbor
_context = [[0,0,1,0,-1],[0,0,1,1,0],[0,1,1,-1,0],[0,0,-1,-1,0],[0,0,0,0,0]];
playGo = PlayGo(size=5,context=_context);
a = playGo.getNeighbor(1,2);
b = playGo.getNeighbor(2,3);
c = playGo.getNeighbor(0,1);
"""


"""
    测试获取气

_context = [[0,0,1,0,-1],[0,0,1,1,0],[0,1,1,-1,0],[0,0,-1,-1,0],[0,0,0,0,0]];
playGo = PlayGo(size=5,context=_context);
kyu = playGo.getKyu();
"""

"""
    测试提子

_context = [[0,0,1,1,-1],[0,1,-1,-1,-1],[0,-1,1,1,0],[0,-1,-1,-1,-1],[0,0,0,0,0]];
playGo = PlayGo(size=5,context=_context);
a = playGo.getCapture(2,4,1);
b = playGo.getCapture(2,4,-1);
c = playGo.getCapture(2,0,1);
"""

"""
    测试可以下的地方

_context = [[1,1,0,1,1],[1,1,-1,-1,-1],[1,-1,0,1,-1],[1,-1,1,1,1],[1,-1,-1,-1,-1]];
playGo = PlayGo(size=5,context=_context);
a = playGo.getAcceptable(1);
b = playGo.getAcceptable(-1);
aa = playGo.is_acceptable(0,2,1);
bb = playGo.is_acceptable(0,2,-1);
"""

"""
    循环打劫测试

_context = [[0,-1,1,0],[-1,1,0,1],[0,-1,1,0],[0,1,0,0]];
playGo = PlayGo(size=4,context=_context,history={1:[[1,1]],-1:[[1,2]]});
a = playGo.getAcceptable(1);
b = playGo.getAcceptable(-1);
bb = playGo.is_acceptable(1,2,-1);
"""



pass