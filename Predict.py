import re
import cv2
import time
import os,shutil
import sys
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim


from RefineNet import RefineNet
from Data import pascal_segmentation_lut
from Tools import Tools, Visualize


tf.app.flags.DEFINE_string('test_data_path', 'demo', '')
tf.app.flags.DEFINE_integer('num_classes', 21, '')
tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoints/', '')
tf.app.flags.DEFINE_string('result_path', 'result/', '')

FLAGS = tf.app.flags.FLAGS



def get_images():
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files

def resize_image(im, size=32, max_side_len=2400):
    h, w, _ = im.shape
    resize_w = w
    resize_h = h
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)
    resize_h = resize_h if resize_h % size == 0 else (resize_h // size) * size
    resize_w = resize_w if resize_w % size == 0 else (resize_w // size) * size
    im = cv2.resize(im, (int(resize_w), int(resize_h)))
    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)
    return im, (ratio_h, ratio_w)

def main(_):

    with tf.get_default_graph().as_default():

        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')

        logits = RefineNet.predict(input_images)
        pred = tf.argmax(logits, dimension=3)

        saver = tf.train.Saver(tf.global_variables())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im_fn_list = get_images()
            for im_fn in im_fn_list:
                im = cv2.imread(im_fn)[:, :, ::-1]
                im_resized, (ratio_h, ratio_w) = resize_image(im, size=32)

                start = time.time()
                pred_re = sess.run([pred], feed_dict={input_images: [im_resized]})
                pred_re = np.array(np.squeeze(pred_re))

                img = Visualize.visualize_segmentation_adaptive(pred_re, pascal_segmentation_lut())
                _diff_time = time.time() - start
                cv2.imwrite(os.path.join(FLAGS.result_path, os.path.basename(im_fn)), img)

                print('{}: cost {:.0f}ms'.format(im_fn, _diff_time * 1000))
                pass
            pass
        pass
    pass



if __name__ == '__main__':
    tf.app.run()