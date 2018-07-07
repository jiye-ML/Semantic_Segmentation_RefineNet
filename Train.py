import os
import time
import numpy as np
import tensorflow as tf

from Data import PascalVocData
from RefineNet import RefineNet


tf.app.flags.DEFINE_integer('batch_size', 4, '')
tf.app.flags.DEFINE_integer('train_size', 512, '')
tf.app.flags.DEFINE_float('learning_rate', 1e-4, '')
tf.app.flags.DEFINE_integer('max_steps', 6000, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_integer('num_classes', 21, '')
tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoints/', '')
tf.app.flags.DEFINE_string('logs_path', 'logs/', '')
tf.app.flags.DEFINE_boolean('restore', False, 'whether to resotre from checkpoint')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 1000, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
tf.app.flags.DEFINE_string('train_data_path', 'data/pascal_train.tfrecords', '')
tf.app.flags.DEFINE_string('val_data_path', 'data/pascal_val.tfrecords', '')
tf.app.flags.DEFINE_string('pretrained_model_path', 'data/resnet_v1_101.ckpt', '')
tf.app.flags.DEFINE_integer('decay_steps', 1500, '')
tf.app.flags.DEFINE_integer('decay_rate', 0.1, '')
FLAGS = tf.app.flags.FLAGS


# 保存图片到日志
def build_image_summary():
    log_label_data = tf.placeholder(tf.uint8, [None, None, None, 3])
    log_pred_data = tf.placeholder(tf.uint8, [None, None, None, 3])
    total_summary = tf.summary.image("images", tf.concat(axis=2, values=[log_label_data, log_pred_data]), max_outputs=FLAGS.batch_size)  # Concatenate row-wise.
    return total_summary, log_label_data, log_pred_data

class Runner(object):

    def __init__(self, data, net):
        self.data = data
        self.net = net

        # 管理网络中创建的图
        self.config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        pass

    def train(self):

        # 图上下文
        with tf.Session(config=self.config) as sess:

            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=100)
            log_image, log_label_data, log_pred_data = build_image_summary()
            # 日志目录
            summary_writer = tf.summary.FileWriter(FLAGS.logs_path, tf.get_default_graph())

            restore_step = 0
            if FLAGS.restore:
                print('continue training from previous checkpoint')
                ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
                restore_step = int(ckpt.split('.')[0].split('_')[-1])
                saver.restore(sess, ckpt)
            # elif FLAGS.pretrained_model_path is not None:
            #     saver.restore(sess, FLAGS.pretrained_model_path)

            # train
            start = time.time()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            try:
                while not coord.should_stop():
                    for step in range(restore_step, FLAGS.max_steps):
                        loss, _, learning_rate = sess.run([self.net.loss, self.net.train_op, self.net.learning_rate_tensor])
                        # 打印中间过程
                        if step % 10 == 0:
                            avg_time_per_step = (time.time() - start) / 10
                            start = time.time()
                            print('Step {:06d}, loss {:.010f}, {:.08f} seconds/step, lr: {:.10f}'.
                                  format(step, loss, avg_time_per_step, learning_rate))

                        # 保存模型
                        if (step + 1) % FLAGS.save_checkpoint_steps == 0:
                            filename = os.path.join(FLAGS.checkpoint_path, 'RefineNet_step_{:d}.ckpt'.format(step + 1))
                            saver.save(sess, filename)
                            print('Write model to: {:s}'.format(filename))
                            # 学习率衰减
                        if step != 0 and step % FLAGS.decay_steps == 0:
                            sess.run(tf.assign(self.net.learning_rate_tensor,
                                               self.net.learning_rate_tensor.eval() * FLAGS.decay_rate))
                        # 保存日志
                        if step % FLAGS.save_summary_steps == 0:
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
                                        color_seg[k, i, j, :] = self.data.color_map[str(annotation[k][i][j])]
                                        color_pred[k, i, j, :] = self.data.color_map[str(output_pred[k][i][j])]

                            log_image_summary = sess.run(log_image, feed_dict={log_label_data: color_seg,
                                                                               log_pred_data: color_pred})
                            summary_writer.add_summary(log_image_summary, global_step=step)
            except tf.errors.OutOfRangeError:
                print('finish')
            finally:
                coord.request_stop()
            coord.join(threads)
        pass

    pass


if __name__ == "__main__":

    # 数据
    train_data = PascalVocData(FLAGS.num_classes, FLAGS.train_size, FLAGS.train_size, FLAGS.batch_size, FLAGS.train_data_path)
    val_data = PascalVocData(FLAGS.num_classes, FLAGS.train_size, FLAGS.train_size, FLAGS.batch_size, FLAGS.val_data_path)
    # net
    refine_net = RefineNet(train_data, learning_rate = FLAGS.learning_rate, moving_average_decay = FLAGS.moving_average_decay)
    # runner
    runner = Runner(data=train_data, net = refine_net)
    runner.train()

    pass