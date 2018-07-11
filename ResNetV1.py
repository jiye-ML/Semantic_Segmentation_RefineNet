"""
ResNet v1 版本的实现
"""
import collections
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import utils

slim = tf.contrib.slim



class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """A named tuple describing a ResNet block.

    Its parts are:
      scope: The scope of the `Block`.
      unit_fn: The ResNet unit function which takes as input a `Tensor` and
        returns another `Tensor` with the output of the ResNet unit.
      args: A list of length equal to the number of units in the `Block`. The list
        contains one (depth, depth_bottleneck, stride) tuple for each unit in the
        block to serve as argument to unit_fn.
    """


class ResNetV1:
    '''
    实现残差网络
    '''
    def __init__(self, inputs, num_classes=None, is_training=True, global_pool=True,
                 output_stride=None, spatial_squeeze=True, reuse=None, scope="resnet_v1_101"):
        self.inputs = inputs
        self.num_classes = num_classes
        self.is_training = is_training
        self.global_pool = global_pool
        self.output_stride = output_stride
        self.spatial_squeeze = spatial_squeeze
        self.reuse = reuse
        self.scope = scope
        pass

    def __call__(self, *args, **kwargs):
        if self.scope not in ['resnet_v1_50', 'resnet_v1_101', 'resnet_v1_152', 'resnet_v1_200']:
            raise ("ResNet scope must be in ['resnet_v1_50', 'resnet_v1_101', 'resnet_v1_152', 'resnet_v1_200']")
        if self.scope == 'resnet_v1_50':
            return self.resnet_v1_50()
        elif self.scope == 'resnet_v1_101':
            return self.resnet_v1_101()
        elif self.scope == 'resnet_v1_151':
            return self.resnet_v1_152()
        else:
            return self.resnet_v1_200()
        pass

    # 残差网络中使用的参数封装
    @staticmethod
    def resnet_arg_scope(weight_decay=0.0001,
                         batch_norm_decay=0.997,
                         batch_norm_epsilon=1e-5,
                         batch_norm_scale=True):
        batch_norm_params = {'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon,
                             'scale': batch_norm_scale, 'updates_collections': tf.GraphKeys.UPDATE_OPS}
        with slim.arg_scope([slim.conv2d],
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            weights_initializer=slim.variance_scaling_initializer(),
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                    return arg_sc

    def resnet_v1_200(self):
        blocks = [Block('block1', self.bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
                  Block('block2', self.bottleneck, [(512, 128, 1)] * 23 + [(512, 128, 2)]),
                  Block('block3', self.bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
                  Block('block4', self.bottleneck, [(2048, 512, 1)] * 3)]
        return self.resnet_v1(blocks, include_root_block=True)

    def resnet_v1_152(self):
        blocks = [Block('block1', self.bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
                  Block('block2', self.bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
                  Block('block3', self.bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
                  Block('block4', self.bottleneck, [(2048, 512, 1)] * 3)]
        return self.resnet_v1(blocks, include_root_block=True)

    def resnet_v1_101(self):
        blocks = [Block('block1', self.bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
                  Block('block2', self.bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
                  Block('block3', self.bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
                  Block('block4', self.bottleneck, [(2048, 512, 1)] * 3)]
        return self.resnet_v1(blocks, include_root_block=True)

    def resnet_v1_50(self):
        blocks = [Block('block1', self.bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
                  Block('block2', self.bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
                  Block('block3', self.bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
                  Block('block4', self.bottleneck, [(2048, 512, 1)] * 3) ]
        return self.resnet_v1(blocks, include_root_block=True)

    # Generator for v1 ResNet models.
    def resnet_v1(self, blocks, include_root_block=True):
        """
        输出特征图大小 [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]

        Args:
          inputs: A tensor of size [batch, height_in, width_in, channels].
          blocks: A list of length equal to the number of ResNet blocks. Each element
            is a resnet_utils.Block object describing the units in the block.
          num_classes: Number of predicted classes for classification tasks. If None
            we return the features before the logit layer.
          is_training: whether is training or not.
          global_pool: If True, we perform global average pooling before computing the
            logits. Set to True for image classification, False for dense prediction.
          output_stride: If None, then the output will be computed at the nominal
            network stride. If output_stride is not None, it specifies the requested
            ratio of input to output spatial resolution.
          include_root_block: If True, include the initial convolution followed by
            max-pooling, if False excludes it.
          spatial_squeeze: if True, logits is of shape [B, C], if false logits is
              of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
          reuse: whether or not the network and its variables should be reused. To be
            able to reuse 'scope' must be given.
          scope: Optional variable_scope.

        Returns:
          net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
            If global_pool is False, then height_out and width_out are reduced by a
            factor of output_stride compared to the respective height_in and width_in,
            else both height_out and width_out equal one. If num_classes is None, then
            net is the output of the last ResNet block, potentially after global
            average pooling. If num_classes is not None, net contains the pre-softmax
            activations.
          end_points: A dictionary from components of the network to the corresponding
            activation.

        Raises:
          ValueError: If the target output_stride is not valid.
        """
        with tf.variable_scope(self.scope, 'resnet_v1', [self.inputs], reuse=self.reuse) as sc:
            end_points_collection = sc.name + '_end_points'
            with slim.arg_scope([slim.conv2d, self.bottleneck, self.stack_blocks_dense],
                                outputs_collections=end_points_collection):
                with slim.arg_scope([slim.batch_norm], is_training=self.is_training):
                    net = self.inputs  # 3x512x512x3
                    # 基本块
                    if include_root_block:
                        if self.output_stride is not None:
                            if self.output_stride % 4 != 0: raise ValueError('The output_stride needs to be a multiple of 4.')
                            self.output_stride /= 4
                        net = self.conv2d_same(net, 64, 7, stride=2, scope='conv1')  # 3x256x256x64
                        net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')  # 3x128x128x64
                        # 给输出起个别名
                        net = utils.collect_named_outputs(end_points_collection, 'pool2', net)
                    # 堆叠残差块
                    net = self.stack_blocks_dense(net, blocks)  # 3x16x16x2048
                    # 处理输出
                    end_points = utils.convert_collection_to_dict(end_points_collection)
                    end_points['pool3'] = end_points['{}/block1'.format(self.scope)]
                    end_points['pool4'] = end_points['{}/block2'.format(self.scope)]
                    end_points['pool5'] = net
                    return net, end_points

    # Bottleneck residual unit variant with BN after convolutions.
    @slim.add_arg_scope
    def bottleneck(self, inputs, depth, depth_bottleneck, stride, rate=1, outputs_collections=None, scope=None):

        with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
            depth_in = utils.last_dimension(inputs.get_shape(), min_rank=4)
            # 如果不能在深度上变化，就在宽度上变化
            if depth == depth_in:
                shortcut = self.subsample(inputs, stride, 'shortcut')
            else:
                shortcut = slim.conv2d(inputs, depth, [1, 1], stride=stride, activation_fn=None, scope='shortcut')  # 3x128x128x256
            residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1, scope='conv1')  # 3x128x128x64
            residual = self.conv2d_same(residual, depth_bottleneck, 3, stride, rate=rate, scope='conv2')  # 3x128x128x64
            residual = slim.conv2d(residual, depth, [1, 1], activation_fn=None, stride=1, scope='conv3')  # 3x128x128x256

            output = tf.nn.relu(shortcut + residual)
            return utils.collect_named_outputs(outputs_collections, sc.original_name_scope, output)

    # 下采样
    def subsample(self, inputs, factor, scope=None):
        if factor == 1:
            return inputs
        else:
            return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)

    # SAME 补边的卷积
    def conv2d_same(self, inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
        if stride == 1:
            return slim.conv2d(inputs, num_outputs, kernel_size, stride=1,
                               rate=rate, padding='SAME', scope=scope)
        else:
            kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
            return slim.conv2d(inputs, num_outputs, kernel_size,
                               stride=stride, rate=rate, padding='VALID', scope=scope)
        pass

    # 密集堆叠每一层，当步长累计超过输出步长的时候，使用空洞卷积代替下采样
    @slim.add_arg_scope
    def stack_blocks_dense(self, net, blocks, outputs_collections=None):
        # current_stride保持有效的步长目前激活的，当步长达到output_stride可以使用空洞卷积替代卷积
        current_stride = 1
        # The atrous convolution rate parameter.
        rate = 1

        for block in blocks:
            with tf.variable_scope(block.scope, 'block', [net]) as sc:
                for i, unit in enumerate(block.args):
                    if self.output_stride is not None and current_stride > self.output_stride:
                        raise ValueError('The target output_stride cannot be reached.')

                    with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                        unit_depth, unit_depth_bottleneck, unit_stride = unit
                        # 如果我们达到了output_stride目标，我们需要使用空洞卷积，stride=1在接下来的层中， rate=当前步长*rate在接下来的层中
                        if self.output_stride is not None and current_stride == self.output_stride:
                            net = block.unit_fn(net,
                                                depth=unit_depth,
                                                depth_bottleneck=unit_depth_bottleneck,
                                                stride=1, rate=rate)
                            rate *= unit_stride
                        else:
                            net = block.unit_fn(net,
                                                depth=unit_depth,
                                                depth_bottleneck=unit_depth_bottleneck,
                                                stride=unit_stride, rate=1)
                            current_stride *= unit_stride
                print(sc.name, net.shape)
                # 将当前输出输出加入到集合中，然后给当前集合别名
                net = utils.collect_named_outputs(outputs_collections, sc.name, net)

        if self.output_stride is not None and current_stride != self.output_stride:
            raise ValueError('The target output_stride cannot be reached.')

        return net

    pass

if __name__ == '__main__':

    input = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='input')
    with slim.arg_scope(ResNetV1.resnet_arg_scope()) as sc:
        resnet_v1 = ResNetV1(input)
        logits = resnet_v1()

    print()
