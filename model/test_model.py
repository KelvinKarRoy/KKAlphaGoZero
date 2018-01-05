# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from alphago_zero_resnet_model import AlphaGoZeroResNet
from playGo import PlayGo


"""
    测试走子
    0 0 0 0     0 0 0 0    0 - 0 0    0 - 1 0   - - 1 0    0 0 1 0    0 0 1 0
    0 0 0 0     0 1 0 0    0 1 0 0    0 1 0 0   0 1 0 0    1 1 0 0    1 1 0 0
    0 0 0 0     0 0 0 0    0 0 0 0    0 0 0 0   0 0 0 0    0 0 0 0    0 0 0 0
    0 0 0 0     0 0 0 0    0 0 0 0    0 0 0 0   0 0 0 0    0 0 0 0    0 0 0 0
    最后一步白旗pass
"""
input_a = [];

playGo = PlayGo(size=4);
input_a.append(playGo.process_input(1));
playGo.save_move(1,1,1);
playGo.save_move(0,1,-1);
input_a.append(playGo.process_input(1));
playGo.save_move(0,2,1);
playGo.save_move(0,0,-1);
input_a.append(playGo.process_input(1));
playGo.save_move(1,0,1);
playGo.save_move(-1,-1,-1);
input_a.append(playGo.process_input(1));

input = tf.placeholder(tf.float32,np.shape(input_a));

net = AlphaGoZeroResNet(_input=input,rule=playGo,mode='eval');

sess = tf.Session();
init_op = tf.initialize_all_variables();
sess.run(init_op);




pass