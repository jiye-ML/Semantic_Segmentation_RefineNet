import time
import numpy as np
import tensorflow as tf

from Data import PascalVocData
from RefineNet import RefineNet


IMG_MEAN = np.array((123.68, 116.78, 103.94), dtype=np.float32)


def configure():
    flags = tf.app.flags
    # 数据
    flags.DEFINE_string('data_path', 'data/pascal_val.tfrecords', '')
    flags.DEFINE_integer('train_size', 384, '')
    flags.DEFINE_integer('batch_size', 16, '')
    flags.DEFINE_integer('num_classes', 21, '')

    # 数据增强
    flags.DEFINE_boolean('random_scale', True, 'whether to perform random scaling data-augmentation')
    flags.DEFINE_boolean('random_mirror', True, 'whether to perform random left-right flipping data-augmentation')
    flags.DEFINE_integer('ignore_label', 255, 'label pixel value that should be ignored')

    # 训练
    flags.DEFINE_boolean('is_training', False, 'whether to training')
    flags.DEFINE_integer('max_steps', 60000, '')
    flags.DEFINE_float('moving_average_decay', 0.997, '')
    flags.DEFINE_float('learning_rate', 1e-4, '')
    flags.DEFINE_integer('decay_steps', 15000, '')
    flags.DEFINE_integer('decay_rate', 0.1, '')
    flags.DEFINE_integer('weight_decay', 1e-5, '')

    # 模型
    flags.DEFINE_string('checkpoint_path', 'checkpoints/', '')
    flags.DEFINE_string('-pretrained_model_path', 'data/resnet_v1_101.ckpt', '')

    # 日志
    flags.DEFINE_string('logs_path', 'logs/', '')
    flags.DEFINE_integer('save_checkpoint_steps', 1000, '')
    flags.DEFINE_integer('save_summary_steps', 500, '')

    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS


class Runner(object):

    def __init__(self, data, net, conf):
        self.data = data
        self.net = net
        self.conf = conf

        # 管理网络中创建的图
        self.config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        pass

    def test(self):

        # 图上下文
        with tf.Session(config=self.config) as sess:

            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=100)

            # 模型加载
            ckpt = tf.train.get_checkpoint_state(self.conf.checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                print('test from {}'.format(ckpt.model_checkpoint_path))
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("请先训练模型..., Train.py first")

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            # test
            confusion_matrix = np.zeros((self.data.num_classes, self.data.num_classes), dtype=np.int)
            for i in range(1449 // self.conf.batch_size):
                start = time.time()
                pred, _,  c_matrix = sess.run([self.net.prediction, self.net.acc_update_op, self.net.confusion_matrix])
                confusion_matrix += c_matrix
                _diff_time = time.time() - start
                print('{}: cost {:.0f}ms'.format(i, _diff_time * 1000))
            # 总体
            self.compute_IoU_per_class(confusion_matrix)
            print("Pascal VOC 2012 validation dataset pixel accuracy: " + str(sess.run(self.net.acc)))

            coord.request_stop()
            coord.join(threads)
        pass

    # 每一类的iou和 miou
    def compute_IoU_per_class(self, confusion_matrix):
        mIoU = 0
        for i in range(self.data.num_classes):
            # IoU = true_positive / (true_positive + false_positive + false_negative)
            TP = confusion_matrix[i, i]
            FP = np.sum(confusion_matrix[:, i]) - TP
            FN = np.sum(confusion_matrix[i]) - TP
            IoU = TP / (TP + FP + FN)
            print('class {}: {}'.format(i, IoU))
            mIoU += IoU / self.conf.num_classes
        print('mIoU: %.3f' % mIoU)
        pass


    pass



if __name__ == "__main__":

    # 数据
    data = PascalVocData(conf=configure())

    # net,
    refine_net = RefineNet(data, conf = data.conf)
    # runner
    runner = Runner(data=data, net = refine_net, conf=data.conf)
    runner.test()

    pass