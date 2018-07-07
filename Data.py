import os
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle

from Augmentation import Augmentation
from Tools import Tools


class PascalVocData(object):

    def __init__(self, class_number, image_height, image_width, batch_size, training_data_path, is_training=True):
        # 数据形式
        self.image_height = image_height
        self.image_width = image_width
        self.image_channel = 3
        self.label_channel = 1
        self.class_number = class_number
        self.batch_size = batch_size

        # 加载标签颜色
        self.class_labels = self._pascal_segmentation_lut().keys()
        with open('data/color_map', 'rb') as f:
            self.color_map = pickle.load(f, encoding='latin-1')

        # 数据
        filename_queue = tf.train.string_input_producer([training_data_path], num_epochs=1000)
        image_op, annotation_op = PacalVocToTfrecords.read_tfrecord_and_decode_into_image_annotation_pair_tensors(filename_queue)
        # 数据增强
        self._image_batch_op = None
        self._annotation_batch_op = None

        image_op, annotation_op = Augmentation.flip_randomly_left_right_image_with_annotation(image_op, annotation_op)

        resized_image, resized_annotation = Augmentation.scale_randomly_image_with_annotation_with_fixed_size_output(
            image_op, annotation_op, [self.image_height, self.image_width])
        resized_annotation = tf.squeeze(resized_annotation)

        self._image_batch_op, self._annotation_batch_op = tf.train.shuffle_batch([resized_image, resized_annotation],
                                                                                 batch_size=self.batch_size,
                                                                                 capacity=2000,
                                                                                 num_threads=32,
                                                                                 min_after_dequeue=500)
        pass

    # 获得下一批数据
    def get_next_data(self):
        return [self._mean_image_subtraction(self._image_batch_op), self._annotation_batch_op]

    # 减去均值
    def _mean_image_subtraction(self, images, means = [123.68, 116.78, 103.94]):
        images = tf.to_float(images)
        num_channels = images.get_shape().as_list()[-1]
        if len(means) != num_channels:
            raise ValueError('len(means) must match the number of channels')
        channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
        for i in range(num_channels):
            channels[i] -= means[i]
        return tf.concat(axis=3, values=channels)


    def _pascal_segmentation_lut(self):
        """Return look-up table with number and correspondng class names
        for PASCAL VOC segmentation dataset. Two special classes are: 0 -
        background and 255 - ambigious region. All others are numerated from
        1 to 20.

        Returns
        -------
        classes_lut : dict
            look-up table with number and correspondng class names
        """

        class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                       'dog', 'horse', 'motorbike', 'person', 'potted-plant',
                       'sheep', 'sofa', 'train', 'tv/monitor', 'ambigious']

        classes_lut = list(enumerate(class_names[:-1]))

        # 加入 255表示不确定类别
        classes_lut.append((255, class_names[-1]))

        return dict(classes_lut)

    pass


class PacalVocToTfrecords:
    '''
    pascalvoc 数据解析为 tfrecords文件
    '''

    def __init__(self, pascal_root = '/home/z840/ALISURE/Data/VOC2012', tfrecords_filename='pascal_{}.tfrecords'):
        # 数据根目录
        self.pascal_root = pascal_root

        if not os.path.exists("data/{}".format(tfrecords_filename.format('train'))):
            Tools.print_info("转换原始数据到tfrecords...")
            train, val = self._get_augmented_pascal_image_annotation_filename_pairs()
            self.write_image_annotation_pairs_to_tfrecord(train, "data/{}".format(tfrecords_filename.format('train')))
            self.write_image_annotation_pairs_to_tfrecord(val, "data/{}".format(tfrecords_filename.format('val')))
            Tools.print_info("转换原始数据到tfrecords end")
            pass

        pass

    # 将给定的 image/annotation pairs 写入 the tfrecords file.
    def write_image_annotation_pairs_to_tfrecord(self, filename_pairs, tfrecords_filename):
        writer = tf.python_io.TFRecordWriter(tfrecords_filename)

        for img_path, annotation_path in filename_pairs:
            # 打开图片
            img = np.array(Image.open(img_path))
            annotation = np.array(Image.open(annotation_path))
            # temp = annotation[annotation > 21]
            # temp = temp[temp != 255]
            # 图片大小
            height = img.shape[0]
            width = img.shape[1]

            example = tf.train.Example(features=tf.train.Features(feature={
                'height': self._int64_feature(height),
                'width': self._int64_feature(width),
                'image_raw': self._bytes_feature(img.tostring()),
                'mask_raw': self._bytes_feature(annotation.tostring())}))

            writer.write(example.SerializeToString())
        writer.close()
        pass

    # 从tfrecords文件中返回 image/annotation对
    @staticmethod
    def read_image_annotation_pairs_from_tfrecord(tfrecords_filename):

        image_annotation_pairs = []

        record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

        for string_record in record_iterator:
            # 解析example
            example = tf.train.Example()
            example.ParseFromString(string_record)
            # 获得数据
            height = int(example.features.feature['height'].int64_list.value[0])
            width = int(example.features.feature['width'].int64_list.value[0])
            img_string = (example.features.feature['image_raw'].bytes_list.value[0])
            annotation_string = (example.features.feature['mask_raw'].bytes_list.value[0])
            # 解析图片
            img_1d = np.fromstring(img_string, dtype=np.uint8)
            img = img_1d.reshape((height, width, -1))
            # 解析注解
            annotation_1d = np.fromstring(annotation_string, dtype=np.uint8)

            # Annotations don't have depth (3rd dimension)
            # TODO: check if it works for other datasets
            annotation = annotation_1d.reshape((height, width))

            image_annotation_pairs.append((img, annotation))

        return image_annotation_pairs

    # 返回 image/annotation对从tfrecords中
    @staticmethod
    def read_tfrecord_and_decode_into_image_annotation_pair_tensors(tfrecord_filenames_queue):

        reader = tf.TFRecordReader()

        _, serialized_example = reader.read(tfrecord_filenames_queue)

        features = tf.parse_single_example(
            serialized_example,
            features={
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string),
                'mask_raw': tf.FixedLenFeature([], tf.string)
            })

        image = tf.decode_raw(features['image_raw'], tf.uint8)
        annotation = tf.decode_raw(features['mask_raw'], tf.uint8)

        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)

        image_shape = tf.stack([height, width, 3])
        annotation_shape = tf.stack([height, width, 1])

        return tf.reshape(image, image_shape), tf.reshape(annotation, annotation_shape)

    def _get_pascal_segmentation_image_annotation_filenames_pairs(self):

        pascal_relative_images_folder = 'JPEGImages'
        pascal_relative_class_annotations_folder = 'SegmentationClass'

        images_extention = 'jpg'
        annotations_extention = 'png'

        pascal_images_folder = os.path.join(self.pascal_root, pascal_relative_images_folder)
        pascal_class_annotations_folder = os.path.join(self.pascal_root, pascal_relative_class_annotations_folder)

        pascal_images_lists_txts = self._get_pascal_segmentation_images_lists_txts()

        pascal_image_names = self._readlines_with_strip_array_version(pascal_images_lists_txts)

        images_full_names = self._add_full_path_and_extention_to_filenames_array_version(pascal_image_names,
                                                                                   pascal_images_folder,
                                                                                   images_extention)

        annotations_full_names = self._add_full_path_and_extention_to_filenames_array_version(pascal_image_names,
                                                                                        pascal_class_annotations_folder,
                                                                                        annotations_extention)

        return map(lambda x: zip(*x), zip(images_full_names, annotations_full_names))

    def _get_augmented_pascal_image_annotation_filename_pairs(self):
        # 获得文件路径
        pascal_txts = self._get_pascal_segmentation_images_lists_txts()
        # 将文件内容保存成数组
        pascal_name_lists = self._readlines_with_strip_array_version(pascal_txts)

        ### 训练集和验证集重叠，每个集合中元素不重复
        # 每个文件去重复
        pascal_train_name_set, pascal_val_name_set, _ = map(lambda x: set(x), pascal_name_lists)
        # 连接train和val
        all_pascal = pascal_train_name_set | pascal_val_name_set
        validation = pascal_val_name_set
        train = all_pascal - validation

        return self._get_pascal_selected_image_annotation_filenames_pairs(self.pascal_root, train), \
               self._get_pascal_selected_image_annotation_filenames_pairs(self.pascal_root, validation)

    # 获得图片和标签的完整路径对
    def _get_pascal_selected_image_annotation_filenames_pairs(self, pascal_root, selected_names):

        pascal_relative_images_folder = 'JPEGImages'
        pascal_relative_class_annotations_folder = 'SegmentationClass'

        images_extention = 'jpg'
        annotations_extention = 'png'

        pascal_images_folder = os.path.join(pascal_root, pascal_relative_images_folder)
        pascal_class_annotations_folder = os.path.join(pascal_root, pascal_relative_class_annotations_folder)
        # 获得图片的完整路径
        images_full_names = self._add_full_path_and_extention_to_filenames(selected_names, pascal_images_folder,
                                                                     images_extention)
        # 获得标签的完整路径
        annotations_full_names = self._add_full_path_and_extention_to_filenames(selected_names,
                                                                          pascal_class_annotations_folder,
                                                                          annotations_extention)
        return zip(images_full_names, annotations_full_names)

    # 获得分割目录文件
    def _get_pascal_segmentation_images_lists_txts(self):
        segmentation_images_lists_relative_folder = 'ImageSets/Segmentation'

        segmentation_images_lists_folder = os.path.join(self.pascal_root, segmentation_images_lists_relative_folder)

        pascal_train_list_filename = os.path.join(segmentation_images_lists_folder, 'train.txt')

        pascal_validation_list_filename = os.path.join(segmentation_images_lists_folder, 'val.txt')

        pascal_trainval_list_filname = os.path.join(segmentation_images_lists_folder, 'trainval.txt')

        return [pascal_train_list_filename, pascal_validation_list_filename, pascal_trainval_list_filname]

    # 读取每个文件内容，并去掉空格和换行
    def _readlines_with_strip(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        # 去掉每一行的空格和换行
        return map(lambda x: x.strip(), lines)

    # 对所有文件，读取文件内容，去掉空格和换行
    def _readlines_with_strip_array_version(self, filenames_array):
        return map(self._readlines_with_strip, filenames_array)

    # 获得图片的完整路径
    def _add_full_path_and_extention_to_filenames(self, filenames_array, full_path, extention):
        return map(lambda x: os.path.join(full_path, x) + '.' + extention, filenames_array)

    def _add_full_path_and_extention_to_filenames_array_version(self, filenames_array_array, full_path, extention):
        return map(lambda x: self._add_full_path_and_extention_to_filenames(x, full_path, extention), filenames_array_array)

    # Helper functions for defining tf types
    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    pass




if __name__ == '__main__':

    '''
    voc 数据元数据域 tfrecords 的转化
    '''
    # 得到 (image, annotation) list (filename.jpg, filename.png)
    PacalVocToTfrecords()

