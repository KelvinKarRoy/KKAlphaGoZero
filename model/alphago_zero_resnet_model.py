from collections import namedtuple

import numpy as np
import tensorflow as tf

import Go.PlayGo

from tensorflow.python.training import moving_averages

HParams = namedtuple('HParams',
                     'batch_size, num_classes, min_lrn_rate, lrn_rate, '
                     'num_residual_units, use_bottleneck, weight_decay_rate, '
                     'relu_leakiness, optimizer')


class AlphaGoZeroResNet(object):
    """ResNet model."""




    def __init__(self, hps, images, labels,mode):
        """ResNet constructor.

        Args:
          hps: Hyperparameters.
          images: Batches of images. [batch_size, image_size, image_size, 3]
          labels: Batches of labels. [batch_size, num_classes]
          mode: One of 'train' and 'eval'.
        """
        self.hps = hps
        self._images = images
        self.labels = labels
        self.mode = mode

        self._extra_train_ops = []

    """
        其卷积层的定义。
        name 该卷基层的名字（拥有唯一的特性）
        x 输入
        filter_size 卷积大小
        in_filters 输入feature的数目
        out_filters 输出feature的数目
        stride 步长。
        中间的参数的维度即成了[filter_size, filter_size, in_filters, out_filters]。
        和一般卷积网络相同，这里的卷积核参数初始化采用了Xavier初始化。
        padding采用了SAME PADDING 使输入输出feature内部维度相等。
    """
    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
                name + 'DW', [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0 / n)))
            return tf.nn.conv2d(x, kernel, strides, padding='SAME')

    """
        ReLu层比较简单，采用普通ReLu。若想改成Leaky ReLu，使用上面一段代码。
    """
    def _relu(self, x, leakiness=0.0):
        # return tf.select(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')
        return tf.nn.relu(x)

    """
        全连接层也比较简单，首先将上一层的输入reshape到[batch_size, -1]上，变成二维变量。
        这一层的参数维度即[x.get_shape()[1], out_dim]。初始化采用uniform。
    """
    def _fully_connected(self, x, out_dim, name=''):
        with tf.variable_scope(name):
            x = tf.reshape(x, [self.hps.batch_size, -1])
            w = tf.get_variable(
                name+'DW', [x.get_shape()[1], out_dim],
                initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
            b = tf.get_variable(name+'biases', [out_dim],
                            initializer=tf.constant_initializer())
            return tf.nn.xw_plus_b(x, w, b)

    """
        Pooling层采用平均池化，直接使用了tf.nn.tf.nn.avg_pool。下面这个长得很像的并不是pool层，作用是统计每个feature的平均值。
        使用reduce_mean函数对x的第二、三维度（即对每一个feature，X的维度是[batch, in_height, in_width, in_channels]）。
        最终维度是[batch, in_channels]
    """
    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])


    """
        BN层
    """
    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable(
                'beta', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
                'gamma', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32))

            if self.mode == 'train':
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

                moving_mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                moving_variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)

                self._extra_train_ops.append(
                    moving_averages.assign_moving_average(
                        moving_mean, mean, 0.9))
                self._extra_train_ops.append(
                    moving_averages.assign_moving_average(
                        moving_variance, variance, 0.9))
            else:
                mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)
                tf.summary.histogram(mean.op.name, mean)
                tf.summary.histogram(variance.op.name, variance)
            # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
            y = tf.nn.batch_normalization(
                x, mean, variance, beta, gamma, 0.001)
            y.set_shape(x.get_shape())
            return y

    """
        残差运算单元（2层conv为一个单元）
        分为两种 用参数activate_before_residual控制
        当activate_before_residual为True时：
             x->BN层->ReLu层->conv->BN层->ReLu层->conv
                           ↓                     ↓
                            ->pool->pad-------->add-------->输出
                            (如果上下的filter数目不相等才进行pad)
        此结构下面用TR表示
        False时：
             x->BN层->ReLu层->conv->BN层->ReLu层->conv
             ↓                                   ↓
              ->pool->pad----------------------->add-------->输出
              (如果上下的filter数目不相等才进行pad)
        此结构下面用FR表示
        默认是FR        
    """
    def _residual(self, x, in_filter, out_filter, stride,
                  activate_before_residual=False):
        """Residual unit with 2 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
                orig_x = tf.pad(
                    orig_x, [[0, 0], [0, 0], [0, 0],
                             [(out_filter - in_filter) // 2,
                              (out_filter - in_filter) // 2]])
            x += orig_x

        tf.logging.info('image after unit %s', x.get_shape())
        return x


    """
       残差运算单元（3层conv为一个单元）
        分为两种 用参数activate_before_residual控制
        当activate_before_residual为True时：
             x->BN层->ReLu层->conv->BN层->ReLu层->conv->BN->ReLu->conv
                           ↓                                    ↓
                            ->pool->conv------------------------->add-------->输出
                            (如果In和Out的filter数目不相等才进行conv)
        此结构下面用TBR表示
        False时：
             x->BN层->ReLu层->conv->BN层->ReLu层->conv->BN->ReLu->conv
             ↓                                                   ↓
              ->pool->conv--------------------------------------->add-------->输出
              (如果In和Out的filter数目不相等才进行conv)
        此结构下面用FR表示
        默认是FBR     
    """
    def _bottleneck_residual(self, x, in_filter, out_filter, stride,
                             activate_before_residual=False):
        """Bottleneck resisual unit with 3 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('common_bn_relu'):
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)
                orig_x = x
        else:
            with tf.variable_scope('residual_bn_relu'):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 1, in_filter, out_filter / 4, stride)

        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._conv('conv2', x, 3, out_filter / 4, out_filter / 4,
                           [1, 1, 1, 1])

        with tf.variable_scope('sub3'):
            x = self._batch_norm('bn3', x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._conv('conv3', x, 1, out_filter / 4, out_filter,
                           [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = self._conv('project', orig_x, 1, in_filter,
                                    out_filter, stride)
            x += orig_x

        tf.logging.info('image after unit %s', x.get_shape())
        return x

    """
        计算L2正则项
    """
    def _decay(self):
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'DW') > 0:
                costs.append(tf.nn.l2_loss(var))
                # tf.histogram_summary(var.op.name, var)

        return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))

    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    """
        搭建model
        首先self._images作为输入 维度是[filter_size, filter_size, in_filters, out_filters]。
        self.hps.use_bottleneck代表用上述两层还是三层的残差单元(True为三层)
        x -> init conv -> 3 * (TR -> (n-1)*FR) -> BN -> ReLu -> global_pool -> FC -> softmax
                        或3 * (TBR-> (n-1)*FBR)
        n为self.hps.num_residual_units，即每个大单元多少小单元
        m为self.hps.num_classes，即FC层输出
        需要说明的是global_pool是将每个feature单独直接取平均，即变成feature数目个数。
    """
    def _build_model(self,playGo):
        """Build the core model within the graph."""
        with tf.variable_scope('init'):
            x = self._images
            """
            卷积核大小为3 
            输入有（2*历史数+1）个通道 如论文里面考虑前8次信息 则17通道
            输出神经元16个 即16个通道
          """
            x = self._conv('init_conv', x, 3, 2 * playGo.get_num_history()+1, 16, self._stride_arr(1))

        # 步长 三个大单元的第一个小单元的步长 后n-1个小单元步长均为1
        strides = [1, 2, 2]
        # 三个大单元的第一个小单元是 T(B)R还是F(B)R
        activate_before_residual = [True, False, False]
        if self.hps.use_bottleneck:
            res_func = self._bottleneck_residual
            filters = [16, 64, 128, 256] # 第一个是输入个数 小单元神经元个数
        else:
            res_func = self._residual
            filters = [16, 16, 32, 64]
            # Uncomment the following codes to use w28-10 wide residual network.
            # It is more memory efficient than very deep residual network and has
            # comparably good performance.
            # https://arxiv.org/pdf/1605.07146v1.pdf
            # filters = [16, 160, 320, 640]
            # Update hps.num_residual_units to 9

        with tf.variable_scope('unit_1_0'):
            x = res_func(x, filters[0], filters[1],
                         self._stride_arr(strides[0]),
                         activate_before_residual[0])
        for i in range(1, self.hps.num_residual_units):
            with tf.variable_scope('unit_1_%d' % i):
                x = res_func(x, filters[1], filters[1], self._stride_arr(1),
                             False)

        with tf.variable_scope('unit_2_0'):
            x = res_func(x, filters[1], filters[2],
                         self._stride_arr(strides[1]),
                         activate_before_residual[1])
        for i in range(1, self.hps.num_residual_units):
            with tf.variable_scope('unit_2_%d' % i):
                x = res_func(x, filters[2], filters[2], self._stride_arr(1),
                             False)

        with tf.variable_scope('unit_3_0'):
            x = res_func(x, filters[2], filters[3],
                         self._stride_arr(strides[2]),
                         activate_before_residual[2])
        for i in range(1, self.hps.num_residual_units):
            with tf.variable_scope('unit_3_%d' % i):
                x = res_func(x, filters[3], filters[3], self._stride_arr(1),
                             False)

        with tf.variable_scope('unit_last'):
            x = self._batch_norm('final_bn', x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._global_avg_pool(x)

        with tf.variable_scope('logit'):
            logits = self._fully_connected(x, self.hps.num_classes)
            self.predictions = tf.nn.softmax(logits)

        # TODO 计算loss 可以开始改了
        with tf.variable_scope('costs'):
            xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=self.labels)
            self.cost = tf.reduce_mean(xent, name='xent')
            self.cost += self._decay()

            tf.summary.scalar('cost', self.cost)

        # TODO 计算准确率
        with tf.variable_scope('acc'):
            correct_prediction = tf.equal(
                tf.cast(tf.argmax(logits, 1), tf.int32), self.labels)
            self.acc = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32), name='accu')

            tf.summary.scalar('accuracy', self.acc)

    """
        训练操作
        self.hps.lrn_rate 学习率
        self.hps.optimizer 'sgd' SGD 和 'mom' Momentum
    """
    def _build_train_op(self):
        """Build training specific ops for the graph."""
        self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
        tf.summary.scalar('learning rate', self.lrn_rate)

        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.cost, trainable_variables)

        if self.hps.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
        elif self.hps.optimizer == 'mom':
            optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)

        apply_op = optimizer.apply_gradients(
            zip(grads, trainable_variables),
            global_step=self.global_step, name='train_step')

        train_ops = [apply_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)
