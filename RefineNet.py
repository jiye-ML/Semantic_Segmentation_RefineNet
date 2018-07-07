import tensorflow as tf
from tensorflow.contrib import slim

from ResNetV1 import ResNetV1


class RefineNet:

    def __init__(self, data, learning_rate, moving_average_decay, weight_decay=1e-5, is_training=True):

        # model
        self.weight_decay = weight_decay
        self.is_training = is_training

        # data
        self.data = data
        self.class_number = self.data.class_number
        self.image_height = self.data.image_height
        self.image_width = self.data.image_width
        self.image_channel = self.data.image_channel
        self.label_channel = self.data.label_channel
        self.label_channel = self.data.label_channel
        self.batch_size = self.data.batch_size
        self.class_labels = self.data.class_labels

        # 学习
        self.learning_rate_tensor = tf.Variable(learning_rate, trainable=False)
        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                           trainable=False, dtype=tf.int32)
        # add summary
        tf.summary.scalar('learning_rate', self.learning_rate_tensor)

        # inputs
        self.images_tensor, self.labels_tensor = self.data.get_next_data()
        # output
        logits = self._network(self.images_tensor)
        self.prediction = tf.cast(tf.argmax(logits, axis=3), tf.uint8) # 预测 输出
        self.is_correct = tf.cast(tf.equal(self.prediction, tf.cast(self.labels_tensor, tf.uint8)), tf.uint8)  # 是否正确
        self.accuracy = tf.reduce_mean(tf.cast(self.is_correct, tf.float32))  # 正确率

        # loss
        self.loss = self._loss(self.labels_tensor, logits)

        # save moving average
        variables_averages_op = tf.train.ExponentialMovingAverage(moving_average_decay, self.global_step).apply(tf.trainable_variables())

        # 优化
        with tf.control_dependencies([variables_averages_op]):
            self.train_op = self._get_train_op()

        self.summary_op = tf.summary.merge_all()

        pass

    def _network(self, inputs):
        with slim.arg_scope(ResNetV1.resnet_arg_scope(weight_decay=self.weight_decay)):
            logits, end_points = ResNetV1(inputs, is_training=self.is_training, scope='resnet_v1_101')()

        with tf.variable_scope('feature_fusisson', values=[end_points.values]):
            batch_norm_params = {'decay': self.weight_decay, 'epsilon': 1e-5,
                                 'scale': True, 'is_training': self.is_training}
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                weights_regularizer=slim.l2_regularizer(self.weight_decay),
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params):

                f = [end_points['pool5'], end_points['pool4'], end_points['pool3'], end_points['pool2']]

                for i in range(4): print('Shape of f_{} {}'.format(i, f[i].shape))

                g = [None, None, None, None]
                h = [None, None, None, None]

                for i in range(4): h[i] = slim.conv2d(f[i], 256, 1)
                for i in range(4): print('Shape of h_{} {}'.format(i, h[i].shape))

                g[0] = self._refine_block(high_inputs=None, low_inputs=h[0])
                g[1] = self._refine_block(g[0], h[1])
                g[2] = self._refine_block(g[1], h[2])
                g[3] = self._refine_block(g[2], h[3])

            return slim.conv2d(g[3], 21, 1, activation_fn=tf.nn.relu)

    ### ------------------   model
    def _get_train_op(self):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate_tensor).minimize(self.loss, global_step=self.global_step)

    # loss
    def _loss(self, labels_tensor, logits_tensor):
        # reshape label
        valid_labels_batch_tensor, valid_logits_batch_tensor = self._get_valid_logits_and_labels(labels_tensor, logits_tensor)

        cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=valid_logits_batch_tensor,
                                                                  labels=valid_labels_batch_tensor)
        # model_loss
        cross_entropy_sum = tf.reduce_mean(cross_entropies)
        tf.summary.scalar('cross_entropy_loss', cross_entropy_sum)
        return tf.add_n([cross_entropy_sum] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))


    ###-------------------------  RefineBlock
    def _unpool(self, inputs, scale):
        return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1] * scale, tf.shape(inputs)[2] * scale])

    # RCU模块， 注意这里不要batch_norm
    def _residual_conv_unit(self, inputs, features=256, kernel_size=3):
        net = tf.nn.relu(inputs)
        net = slim.conv2d(net, features, kernel_size)
        net = tf.nn.relu(net)
        net = slim.conv2d(net, features, kernel_size)
        net = tf.add(net, inputs)
        return net

    # 链式残差模块
    def _chained_residual_pooling(self, inputs, features=256):
        net_relu = tf.nn.relu(inputs)
        net = slim.max_pool2d(net_relu, [5, 5], stride=1, padding='SAME')
        net = slim.conv2d(net, features, 3, normalizer_fn=None)
        net_sum_1 = tf.add(net, net_relu)

        net = slim.max_pool2d(net, [5, 5], stride=1, padding='SAME')
        net = slim.conv2d(net, features, 3, normalizer_fn=None)
        net_sum_2 = tf.add(net, net_sum_1)

        return net_sum_2

    # 多个分辨率下特征融合
    def _multi_resolution_fusion(self, high_inputs=None, low_inputs=None, features=256):
        if high_inputs is None:
            rcu_low_1 = low_inputs[0]
            rcu_low_2 = low_inputs[1]

            rcu_low_1 = slim.conv2d(rcu_low_1, features, 3)
            rcu_low_2 = slim.conv2d(rcu_low_2, features, 3)

            return tf.add(rcu_low_1, rcu_low_2)

        else:
            rcu_low_1 = low_inputs[0]
            rcu_low_2 = low_inputs[1]

            rcu_low_1 = slim.conv2d(rcu_low_1, features, 3)
            rcu_low_2 = slim.conv2d(rcu_low_2, features, 3)

            rcu_low = tf.add(rcu_low_1, rcu_low_2)

            rcu_high_1 = high_inputs[0]
            rcu_high_2 = high_inputs[1]

            rcu_high_1 = self._unpool(slim.conv2d(rcu_high_1, features, 3), 2)
            rcu_high_2 = self._unpool(slim.conv2d(rcu_high_2, features, 3), 2)

            rcu_high = tf.add(rcu_high_1, rcu_high_2)

            return tf.add(rcu_low, rcu_high)

    def _refine_block(self, high_inputs=None, low_inputs=None):
        if high_inputs is None:  # block 4
            rcu_low_1 = self._residual_conv_unit(low_inputs, features=256)
            rcu_low_2 = self._residual_conv_unit(low_inputs, features=256)
            rcu_low = [rcu_low_1, rcu_low_2]

            fuse = self._multi_resolution_fusion(high_inputs=None, low_inputs=rcu_low, features=256)
            fuse_pooling = self._chained_residual_pooling(fuse, features=256)
            output = self._residual_conv_unit(fuse_pooling, features=256)
            return output
        else:
            rcu_low_1 = self._residual_conv_unit(low_inputs, features=256)
            rcu_low_2 = self._residual_conv_unit(low_inputs, features=256)
            rcu_low = [rcu_low_1, rcu_low_2]

            rcu_high_1 = self._residual_conv_unit(high_inputs, features=256)
            rcu_high_2 = self._residual_conv_unit(high_inputs, features=256)
            rcu_high = [rcu_high_1, rcu_high_2]

            fuse = self._multi_resolution_fusion(rcu_high, rcu_low, features=256)
            fuse_pooling = self._chained_residual_pooling(fuse, features=256)
            output = self._residual_conv_unit(fuse_pooling, features=256)
            return output


    ###------------------------- 对标签进行处理 begin

    # 先将标签one-hot 然后剔除掉模糊标签。
    def _get_valid_logits_and_labels(self, annotation_batch_tensor, logits_batch_tensor):
        # one-hot
        labels_batch_tensor = self._get_labels_from_annotation_batch(
            annotation_batch_tensor=annotation_batch_tensor)
        # 获得合理标签的下标
        valid_batch_indices = self._get_valid_entries_indices_from_annotation_batch(
            annotation_batch_tensor=annotation_batch_tensor)
        # 获得合理的数据
        valid_labels_batch_tensor = tf.gather_nd(params=labels_batch_tensor, indices=valid_batch_indices)
        valid_logits_batch_tensor = tf.gather_nd(params=logits_batch_tensor, indices=valid_batch_indices)

        return valid_labels_batch_tensor, valid_logits_batch_tensor

    # 对一个annotation实行，onehot
    def _get_labels_from_annotation(self, annotation_tensor):
        # 最后的value是的标签是不确定的， 在训练的时候不应该使用
        valid_entries_class_labels = list(self.class_labels)[:-1]

        # Stack the binary masks for each class
        labels_2d = list(map(lambda x: tf.equal(annotation_tensor, x), valid_entries_class_labels))

        # Perform the merging of all of the binary masks into one matrix
        labels_2d_stacked = tf.stack(labels_2d, axis=2)

        # Convert tf.bool to tf.float
        # Later on in the labels and logits will be used
        # in tf.softmax_cross_entropy_with_logits() function
        # where they have to be of the float type.
        labels_2d_stacked_float = tf.to_float(labels_2d_stacked)

        return labels_2d_stacked_float

    # 对一个batch的annotation实行one-hot
    def _get_labels_from_annotation_batch(self, annotation_batch_tensor):
        return tf.map_fn(fn=lambda x: self._get_labels_from_annotation(annotation_tensor=x),
                         elems=annotation_batch_tensor, dtype=tf.float32)

    # 获得剔除了模糊标签的下标
    def _get_valid_entries_indices_from_annotation_batch(self, annotation_batch_tensor):
        # 最后一个模糊标签训练的时候不使用
        mask_out_class_label = list(self.class_labels)[-1]

        # 剔除模糊标签
        valid_labels_indices = tf.where(tf.not_equal(annotation_batch_tensor, mask_out_class_label))

        return tf.to_int32(valid_labels_indices)

    ####----------------- 对标签进行处理 end


    pass


