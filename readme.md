## RefineNet tensorflow 实现

* 使用ResNet作为特征提取器
* [paper](https://github.com/jiye-ML/Semantic_Segmentation_Review.git)

## prepare
- download the pretrain model of resnet_v1_101.ckpt, you can download it from [here](https://github.com/tensorflow/models/tree/master/slim)
- download the [pascal voc dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
- some dependence like cv2, numpy and etc. recommend to install Anaconda

## training
- first, run convert_pascal_voc_to_tfrecords.py to convert training data into .tfrecords, Or you can use the tfrecord I converted In [BaiduYun](http://pan.baidu.com/s/1kVefEIj).Currently, I only use the pascal voc 2012 for training.
- second, run python RefineNet/multi_gpu_train.py, also, you can change some hyper parameters in this file, like the batch size.

## eval
- if you have already got a model, or just download the model I trained on pascal voc.[model](http://pan.baidu.com/s/1kVefEIj).
- put images in demo/ and run python RefineNet/demo.py 

## roadmap
- [x] python2/3 compatibility
- [ ] Complete realization of refinenet model
- [ ] test on pascal voc, give the IoU result
- [ ] training on other datasets

* [参考github](https://github.com/eragonruan/refinenet-image-segmentation)