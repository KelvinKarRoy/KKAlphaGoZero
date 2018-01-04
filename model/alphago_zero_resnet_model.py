from collections import namedtuple

import numpy as np
import tensorflow as tf

import go.playGo

from tensorflow.python.training import moving_averages

HParams = namedtuple('HParams',
                     'batch_size, num_classes, min_lrn_rate, lrn_rate, '
                     'num_residual_units, use_bottleneck, weight_decay_rate, '
                     'relu_leakiness, optimizer')


class AlphaGoZeroResNet(object):
    """ResNet model."""




    def __init__(self, hps, _input, next_action,is_winner ,mode):
        """ResNet constructor.

        Args:
          hps: Hyperparameters.
          _imput: Batches of images. [batch_size, size, size, num_history * 2 + 1]
          next_action: 下一步采取的行动. [batch_size, size * size + 1]
          is_winner: 胜1 负-1 和0 [batch_size, 1]
          mode: One of 'train' and 'eval'.
        """
        self.hps = hps;
        self._input = _input;
        self.next_action = next_action;
        self.mode = mode
        self.is_winner = is_winner;
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

    """
               AlphaGo里的残差运算单元（2层conv为一个单元）
                    x->conv->BN层->ReLu->conv->BN层
                    ↓                           ↓
                     ------------(pad)------------->add---->ReLu---->输出
                     (如果上下的filter数目不相等才进行pad,在AlphaGo Zero中都是256，所以其实没有pad)
               此结构下面用AR表示        
    """
    def _my_residual(self, x, in_filter, out_filter, stride):
        """Residual unit with 2 sub layers."""
        orig_x = x;

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride);
            x = self._batch_norm('bn1',x);
            x = self._relu(x, self.hps.relu_leakiness);

        with tf.variable_scope('sub2'):
            x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1]);
            x = self._batch_norm('bn2', x);

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
                orig_x = tf.pad(
                    orig_x, [[0, 0], [0, 0], [0, 0],
                             [(out_filter - in_filter) // 2,
                              (out_filter - in_filter) // 2]])
            x += orig_x;
            x = self._relu(x, self.hps.relu_leakiness);

        tf.logging.info('image after unit %s', x.get_shape())
        return x


    """
        搭建AlphaGo Zero的model
        首先self._images作为输入 维度是[filter_size, filter_size, in_filters, out_filters]。
        self.hps.use_bottleneck代表用上述两层还是三层的残差单元(True为三层)
        x -> conv -> BN -> ReLu -> 19或39*AR -> conv -> BN -> ReLu -> FC -> softmax (Policy output=size*size+1)
                                      ↓
                                       -> conv -> BN -> ReLu -> FC(output=256) ->ReLu -> FC(output=1) -> Tanh (Value)
        最后的卷积核为1*1
    """
    def _my_build_model(self,playGo,num_filter=256,num_block=20):
        """Build the core model within the graph."""
        with tf.variable_scope('init'):
            x = self._input;
            """
            卷积核大小为3 
            输入有（2*历史数+1）个通道 如论文里面考虑前8次信息 则17通道
            输出神经元num_filter个 即num_filter个通道
          """
            x = self._conv('init_conv', x, 3, 2 * playGo.get_num_history()+1, num_filter, self._stride_arr(1));
            x = self._batch_norm("init_BN",x);
            x = self._relu(x);

        # 20或40次循环
        for i in range(num_block-1):
            with tf.variable_scope('unit_%d' % i):
                x = self._my_residual(x,num_filter,num_filter,self._stride_arr(1));

        org_x = x;

        # Policy
        with tf.variable_scope('unit_policy'):
            # 1*1小卷积 2个卷积核
            x = self._conv('policy_conv',x,1,num_filter,2,self._stride_arr(1));
            x = self._batch_norm('policy_bn', x);
            x = self._relu(x, self.hps.relu_leakiness);
            x = self._fully_connected(x,playGo.get_size() * playGo.get_size() + 1,"policy_FC");
            x = tf.nn.softmax(x);

        # Value
        with tf.variable_scope('unit_value'):
            # 1*1小卷积 1个卷积核
            y = self._conv('value_conv',org_x,1,num_filter,1,self._stride_arr(1));
            y = self._batch_norm('value_bn', y);
            y = self._relu(y, self.hps.relu_leakiness);
            y = self._fully_connected(y, 256, "value_FC1");
            y = self._relu(y, self.hps.relu_leakiness);
            y = self._fully_connected(y, 1, "value_FC2");
            y = tf.nn.tanh(y);



        # TODO 计算loss 以后加了MCTS 会改x这部分
        with tf.variable_scope('costs'):
            xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=x, labels=self.next_action);
            self.cost = tf.multiply(self.hps.policy_rate, tf.reduce_mean(xent, name='xent'));

            self.cost += tf.multiply(self.hps.value_rate, tf.square(y - self.is_winner));

            self.cost += self._decay(); # L2正则项


            tf.summary.scalar('cost', self.cost);
        """
        计算准确率
        with tf.variable_scope('acc'):
            correct_prediction = tf.equal(
                tf.cast(tf.argmax(logits, 1), tf.int32), self.labels)
            self.acc = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32), name='accu')

            tf.summary.scalar('accuracy', self.acc)
        """

