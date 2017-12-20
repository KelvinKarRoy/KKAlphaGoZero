# KKAlphaGoZero
2017年10月nature论文实现 基于TensorFlow<br>
开工日期 2017年12月19日
## 项目结构
> Go 围棋相关类  
>> PlayGo.py 围棋规则类  
>> ShowGo.py 提供可视化  
>> TestGo.py 相关测试

> model 网络结构类  
>> resnet_model.py resNet的TensorFlow版本  
>> alphago_zero_resnet_model.py alphaGo zero的model

## 任务清单
- [x] 围棋规则类（可以下）
- [x] 可视化类
- [ ] resNet网络结构类（正向传播）
- [ ] MCTS
- [ ] 增强学习

## 更新日志
#### 2017年12月20日
  完善了围棋相关类，增加了结算子目和判断真眼的功能，下午休息一下。
#### 2017年12月19日
  花了一个下午，完成PlayGo和ShowGo的编写，即对围棋规则类编写完成，并提供最简单的可视化，其后依据后续需要完善其功能。

## 参考资料
### AlphaGo
* [AlphaGo Lee中文PPT](http://blog.csdn.net/songrotek/article/details/51065143)
* [AlphaGo Zero原论文](https://deepmind.com/documents/119/agz_unformatted_nature.pdf)
* [AlphaGo Zero中文详细解析](http://www.sohu.com/a/199892682_500659)
* [Alpha Zero(2017年12月，本项目未采用)](https://arxiv.org/pdf/1712.01815.pdf)
### MCTS
* [蒙特卡洛搜索树](http://mcts.ai/)
