from collections import namedtuple

import numpy as np
import tensorflow as tf

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
    """
    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])


    # TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py
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
