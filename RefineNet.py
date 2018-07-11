import tensorflow as tf
from tensorflow.contrib import slim
import tensorflow.contrib.metrics as tcm

from ResNetV1 import ResNetV1


class RefineNet:

    def __init__(self, data, conf):

        self.conf = conf
        # model
        self.weight_decay = self.conf.weight_decay
        self.is_training = self.conf.is_training

        # data
        self.data = data
        self.num_classes = self.data.num_classes
        self.image_height = self.data.image_height
        self.image_width = self.data.image_width
        self.image_channel = self.data.image_channel
        self.label_channel = self.data.label_channel
        self.label_channel = self.data.label_channel
        self.batch_size = self.data.batch_size

        # 学习
        self.learning_rate_tensor = tf.convert_to_tensor(self.conf.learning_rate)
        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                           trainable=False, dtype=tf.int32)
        # add summary
        tf.summary.scalar('learning_rate', self.learning_rate_tensor)

        # inputs
        self.images_tensor, self.labels_tensor = self.data.get_next_data()

        # output
        logits = self._network(self.images_tensor)

        self.prediction = tf.cast(tf.expand_dims(tf.argmax(logits, axis=3), dim=3), tf.uint8)

        # 评估
        pred = tf.reshape(self.prediction, [-1, ])
        gt = tf.reshape(self.labels_tensor, [-1, ])
        temp = tf.less_equal(gt, self.num_classes - 1)
        weights = tf.cast(temp, tf.int32)
        gt = tf.where(temp, gt, tf.cast(temp, tf.uint8))
        self.acc, self.acc_update_op = tcm.streaming_accuracy(pred, gt, weights=weights)
        # confusion matrix
        self.confusion_matrix = tcm.confusion_matrix(pred, gt, num_classes=self.num_classes, weights=weights)

        # loss
        self.loss = self._loss(logits)

        # save moving average
        variables_averages_op = tf.train.ExponentialMovingAverage(self.conf.moving_average_decay,
                                                                  self.global_step).apply(tf.trainable_variables())
        # 优化
        with tf.control_dependencies([variables_averages_op]):
            self.train_op = self._get_train_op()

        self.summary_op = tf.summary.merge_all()

        pass

    def _network(self, inputs):

        with slim.arg_scope(ResNetV1.resnet_arg_scope(weight_decay=self.weight_decay)):
            logits, end_points = ResNetV1(inputs, is_training=self.is_training, scope='resnet_v1_101')()

        with tf.variable_scope('feature_fusion', values=[end_points.values]):
            batch_norm_params = {'decay': self.weight_decay, 'epsilon': 1e-5,
                                 'scale': True, 'is_training': self.is_training}
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                weights_regularizer=slim.l2_regularizer(self.weight_decay)):

                f = [end_points['pool5'], end_points['pool4'], end_points['pool3'], end_points['pool2']]

                for i in range(4): print('Shape of f_{} {}'.format(i, f[i].shape))

                g = [None, None, None, None]
                h = [None, None, None, None]

                for i in range(4): h[i] = slim.conv2d(f[i], 256, 1)
                for i in range(4): print('Shape of h_{} {}'.format(i, h[i].shape))

                g[0] = self._refine_block(h[0])
                g[1] = self._refine_block(g[0], h[1])
                g[2] = self._refine_block(g[1], h[2])
                g[3] = self._refine_block(g[2], h[3])
                g[3] = self._unpool(g[3],scale=4)
                F_score = slim.conv2d(g[3], 21, 1, activation_fn=tf.nn.relu, normalizer_fn=None)

        return F_score

    @staticmethod
    def predict(inputs, weight_decay = 1e-5):
        with slim.arg_scope(ResNetV1.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = ResNetV1(inputs, is_training=False, scope='resnet_v1_101')()

        with tf.variable_scope('feature_fusion', values=[end_points.values]):
            batch_norm_params = {'decay': weight_decay, 'epsilon': 1e-5,
                                 'scale': True, 'is_training': False}
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                weights_regularizer=slim.l2_regularizer(weight_decay)):

                f = [end_points['pool5'], end_points['pool4'], end_points['pool3'], end_points['pool2']]

                for i in range(4): print('Shape of f_{} {}'.format(i, f[i].shape))

                g = [None, None, None, None]
                h = [None, None, None, None]

                for i in range(4): h[i] = slim.conv2d(f[i], 256, 1)
                for i in range(4): print('Shape of h_{} {}'.format(i, h[i].shape))

                g[0] = RefineNet._refine_block(h[0])
                g[1] = RefineNet._refine_block(g[0], h[1])
                g[2] = RefineNet._refine_block(g[1], h[2])
                g[3] = RefineNet._refine_block(g[2], h[3])
                g[3] = RefineNet._unpool(g[3],scale=4)
                F_score = slim.conv2d(g[3], 21, 1, activation_fn=tf.nn.relu, normalizer_fn=None)

        return F_score
        pass

    ### ------------------   model
    def _get_train_op(self):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate_tensor).minimize(self.loss, global_step=self.global_step)

    # loss
    def _loss(self, logits_tensor):

        # Groud Truth: ignoring all labels greater or equal than n_classes
        label_proc = tf.reshape(self.labels_tensor, [-1, ])

        # 去掉和合理的标签，不然交叉熵会出现nan
        indices = tf.squeeze(tf.where(tf.less_equal(label_proc, self.num_classes - 1)), 1)
        label_proc = tf.cast(tf.gather(label_proc, indices), tf.int32)
        logits_tensor = tf.reshape(logits_tensor, [-1, self.num_classes])
        logits_tensor = tf.gather(logits_tensor, indices)

        cross_entropies = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_tensor, labels=label_proc)
        # model_loss
        cross_entropy_sum = tf.reduce_mean(cross_entropies)
        tf.summary.scalar('cross_entropy_loss', cross_entropy_sum)
        return tf.add_n([cross_entropy_sum] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    ###-------------------------  RefineBlock
    @staticmethod
    def _unpool(inputs, scale):
        return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1] * scale, tf.shape(inputs)[2] * scale])

    # RCU模块， 注意这里不要batch_norm
    @staticmethod
    def _residual_conv_unit(inputs, features=256, kernel_size=3):
        net = tf.nn.relu(inputs)
        net = slim.conv2d(net, features, kernel_size)
        net = tf.nn.relu(net)
        net = slim.conv2d(net, features, kernel_size)
        return tf.add(net, inputs)

    # 链式残差模块
    @staticmethod
    def _chained_residual_pooling(inputs, features=256):
        net_relu = tf.nn.relu(inputs)
        net = slim.max_pool2d(net_relu, [5, 5], stride=1, padding='SAME')
        net = slim.conv2d(net, features, 3)
        return tf.add(net, net_relu)

    # 多个分辨率下特征融合
    @staticmethod
    def _multi_resolution_fusion(high_inputs=None, low_inputs=None, up0=2, up1=1, n_i=256):
        g0 = RefineNet._unpool(slim.conv2d(high_inputs, n_i, 3), scale=up0)

        if low_inputs is None:
            return g0
        g1 = RefineNet._unpool(slim.conv2d(low_inputs, n_i, 3), scale=up1)
        return tf.add(g0, g1)

    @staticmethod
    def _refine_block(high_inputs=None, low_inputs=None):
        if low_inputs is not None:
            print(high_inputs.shape)
            rcu_high = RefineNet._residual_conv_unit(high_inputs, features=256)
            rcu_low = RefineNet._residual_conv_unit(low_inputs, features=256)
            fuse = RefineNet._multi_resolution_fusion(rcu_high, rcu_low, up0=2, up1=1, n_i=256)
            fuse_pooling = RefineNet._chained_residual_pooling(fuse, features=256)
            output = RefineNet._residual_conv_unit(fuse_pooling, features=256)
            return output
        else:
            rcu_high = RefineNet._residual_conv_unit(high_inputs, features=256)
            fuse = RefineNet._multi_resolution_fusion(rcu_high, low_inputs=None, up0=1, n_i=256)
            fuse_pooling = RefineNet._chained_residual_pooling(fuse, features=256)
            output = RefineNet._residual_conv_unit(fuse_pooling, features=256)
            return output

    pass


