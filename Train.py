import os
import time
import numpy as np
import tensorflow as tf

from Data import PascalVocData
from RefineNet import RefineNet
from Tools import Tools


IMG_MEAN = np.array((123.68, 116.78, 103.94), dtype=np.float32)

def configure():
    flags = tf.app.flags
    # 数据
    flags.DEFINE_string('data_path', 'data/pascal_train.tfrecords', '')
    flags.DEFINE_integer('batch_size', 8, '')
    flags.DEFINE_integer('train_size', 384, '')
    flags.DEFINE_integer('num_classes', 21, '')

    # 数据增强
    flags.DEFINE_boolean('random_scale', True, 'whether to perform random scaling data-augmentation')
    flags.DEFINE_boolean('random_mirror', True, 'whether to perform random left-right flipping data-augmentation')
    flags.DEFINE_integer('ignore_label', 255, 'label pixel value that should be ignored')

    # 训练
    flags.DEFINE_boolean('is_training', True, 'whether to training')
    flags.DEFINE_integer('max_steps', 60000, '')
    flags.DEFINE_float('moving_average_decay', 0.997, '')
    flags.DEFINE_float('learning_rate', 1e-6, '')
    flags.DEFINE_integer('decay_steps', 20000, '')
    flags.DEFINE_integer('decay_rate', 0.1, '')
    flags.DEFINE_integer('weight_decay', 5e-4, '')

    # 模型
    flags.DEFINE_string('checkpoint_path', 'checkpoints/', '')
    flags.DEFINE_string('pretrained_model_path', 'data/resnet_v1_101.ckpt', '')

    # 日志
    flags.DEFINE_string('logs_path', 'logs/', '')
    flags.DEFINE_integer('save_checkpoint_steps', 1000, '')
    flags.DEFINE_integer('save_summary_steps', 500, '')

    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS


# 保存图片到日志
def build_image_summary():
    log_image_data = tf.placeholder(tf.uint8, [None, None, None, 3])
    log_label_data = tf.placeholder(tf.uint8, [None, None, None, 3])
    log_pred_data = tf.placeholder(tf.uint8, [None, None, None, 3])
    total_summary = tf.summary.image("images",
                                     tf.concat(axis = 2, values=[log_image_data, log_label_data, log_pred_data]),
                                     max_outputs=5)  # Concatenate row-wise.
    return total_summary, log_image_data, log_label_data, log_pred_data

class Runner(object):

    def __init__(self, data, net, conf):
        self.data = data
        self.net = net
        self.conf = conf

        # 管理网络中创建的图
        self.config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        pass

    def train(self):

        # 图上下文
        with tf.Session(config=self.config) as sess:

            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=100)

            # 日志
            log_image, log_image_data, log_label_data, log_pred_data = build_image_summary()
            summary_writer = tf.summary.FileWriter(self.conf.logs_path, tf.get_default_graph())

            # 模型加载
            ckpt = tf.train.get_checkpoint_state(self.conf.checkpoint_path)
            load_step = 0
            if ckpt and ckpt.model_checkpoint_path:
                print('continue training from previous checkpoint')
                load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('_')[2].split('.')[0])
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                Tools.new_dir(self.conf.checkpoint_path)

            # train
            start = time.time()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            try:
                while not coord.should_stop():
                    for step in range(load_step, self.conf.max_steps):
                        loss, _, learning_rate = sess.run([self.net.loss, self.net.train_op, self.net.learning_rate_tensor])
                        # 打印中间过程
                        print_step = 1
                        if step % print_step == 0:
                            avg_time_per_step = (time.time() - start) / print_step
                            start = time.time()
                            print('Step {:06d}, loss {:.010f}, {:.08f} seconds/step, lr: {:.10f}'.
                                  format(step, loss, avg_time_per_step, learning_rate))

                        # 保存模型
                        if (step + 1) % self.conf.save_checkpoint_steps == 0:
                            filename = os.path.join(self.conf.checkpoint_path, 'RefineNet_step_{:d}.ckpt'.format(step + 1))
                            saver.save(sess, filename)
                            print('Write model to: {:s}'.format(filename))
                            # 学习率衰减
                        if step != 0 and step % self.conf.decay_steps == 0:
                            sess.run(tf.assign(self.net.learning_rate_tensor,
                                               self.net.learning_rate_tensor.eval() * self.conf.decay_rate))
                        # 保存日志
                        if step % self.conf.save_summary_steps == 0:
                            image, label, output_pred, summary = sess.run([self.net.images_tensor, self.net.labels_tensor,
                                                                           self.net.prediction, self.net.summary_op])
                            summary_writer.add_summary(summary, global_step=step)

                            annotation = np.squeeze(label)
                            output_pred = np.squeeze(output_pred)

                            # 标签可视化
                            color_seg = np.zeros((output_pred.shape[0], output_pred.shape[1], output_pred.shape[2], 3))
                            color_pred = np.zeros((output_pred.shape[0], output_pred.shape[1], output_pred.shape[2], 3))
                            for k in range(output_pred.shape[0]):
                                for i in range(output_pred.shape[1]):
                                    for j in range(output_pred.shape[2]):
                                        image[k, i, j, :] += IMG_MEAN
                                        if annotation[k][i][j] < self.data.num_classes:
                                            color_seg[k, i, j, :] = self.data.color_map[annotation[k][i][j]]
                                        if output_pred[k][i][j] < self.data.num_classes:
                                            color_pred[k, i, j, :] = self.data.color_map[output_pred[k][i][j]]

                            log_image_summary = sess.run(log_image, feed_dict={log_image_data: image,
                                                                               log_label_data: color_seg,
                                                                               log_pred_data: color_pred})
                            summary_writer.add_summary(log_image_summary, global_step=step)
            except tf.errors.OutOfRangeError:
                print('finish')
            finally:
                coord.request_stop()
            coord.join(threads)
        pass

    pass


def main(_):
    # 数据
    data = PascalVocData(conf=configure())

    # net,
    refine_net = RefineNet(data, conf = data.conf)
    # # runner
    runner = Runner(data=data, net = refine_net, conf = data.conf)
    runner.train()
    pass


if __name__ == "__main__":

    tf.app.run()

    pass