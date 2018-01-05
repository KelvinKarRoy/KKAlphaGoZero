# KKAlphaGoZero
2017年10月nature论文实现 基于TensorFlow  
开工日期 2017年12月19日  
## 项目结构
* rule.py ---------- rule父类
* go 围棋相关类  
    * playGo.py -------- 围棋规则类  
    * showGo.py -------- 提供可视化  
    * testGo.py -------- 相关测试  

* gobang 五子棋相关类  
    * gobang.py -------- 五子棋规则类  

* model 网络结构类  
    * resnet_model.py --------------- resNet的TensorFlow版本  
    * alphago_zero_resnet_model.py -- alphaGo zero的model  

* mcts 蒙特卡洛搜索树类  
    * mcts.py --------------- 蒙特卡洛搜索树  
    * node.py --------------- 树的节点  
    * edge.py --------------- 树的边  

* img 项目相关图片  
    * net_model.jpg ------ AlphaGo Zero的结构图  
     
## 任务清单
- [x] 围棋规则类（可以下）
- [x] 可视化类
- [x] resNet网络结构类（正向传播）
- [x] MCTS
- [ ] 自对弈
- [ ] 增强学习

## 网络结构
![网络结构](https://github.com/KelvinKarRoy/KKAlphaGoZero/blob/master/img/net_model.jpg)  

## 参考资料
### AlphaGo
* [AlphaGo Lee中文PPT](http://blog.csdn.net/songrotek/article/details/51065143)
* [AlphaGo Zero原论文](https://deepmind.com/documents/119/agz_unformatted_nature.pdf)
* [AlphaGo Zero中文详细解析](http://www.sohu.com/a/199892682_500659)
* [知乎大神的AlphaGo Zero解析](https://www.zhihu.com/question/66861459/answer/246844524)
* [Alpha Zero(2017年12月，本项目未采用)](https://arxiv.org/pdf/1712.01815.pdf)
### MCTS
* [蒙特卡洛搜索树](http://mcts.ai/)


## 更新日志
#### 2018年1月4日
  准备考试加上元旦休息了一阵，修改了围棋相关类的结构；mcts大体上实现。
#### 2017年12月24日
  晚上睡不着，起来写代码。完成了mcts相关类的结构设计。
#### 2017年12月23日
  AlphaGo Zero的model写完了！忙网络的实验忙了一下午也没做完，做这个调节一下心情。
#### 2017年12月22日
  对model类注释完毕，并针对此项目进行了部分修改，主要是参数部分。
#### 2017年12月21日
  完善了围棋相关类，增加了结算子目和判断真眼的功能；对围棋类增加了针对AlphaGo Zero输入的格式化；对model类增加了注释。
#### 2017年12月19日
  花了一个下午，完成PlayGo和ShowGo的编写，即对围棋规则类编写完成，并提供最简单的可视化，其后依据后续需要完善其功能。

