# -*- coding=UTF-8 -*-
"""
@Time : 2019/5/28
@Author : gzf
@File : DataGenerator.py
@brief:
"""

import tensorflow as tf
import numpy as np


IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)


# 训练测试数据生成
class DataGenerator(object):
    """
    Wrapper class around the new Tensorflows dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, trajs_file, mode='training', batch_size=64, shuffle=True, buffer_size=10000):
        """
        接收轨迹的npy文件，使用tf.data.Dataset处理数据

        :param trajs_file: trajs.npy
        :param mode: Either 'training' or 'validation'. Depending on this value,
                    different parsing functions will be used.
        :param batch_size: Number of data per batch.
        :param shuffle: Wether or not to shuffle the data in the dataset
                    and the initial file list.
        :param buffer_size: Number of data used as buffer for TensorFlows
                    shuffling of the dataset.

        Raises：
            ValueError: If an invalid mode is passed.
        """
        self.trajs_file = trajs_file

        # retrieve the data from the text file
        self._read_npy_file()

        # create dataset
        data = self._dataset_generator()

        # distinguish between train/infer. when calling the parsing functions
        if mode == 'training':
            data = data.map(self._parse_function_train)

            # create a new dataset with batches of trajs
            data = data.batch(batch_size, drop_remainder=True)

        elif mode == 'sample':
            data = data.map(self._parse_function_sample)
            # create a new dataset with batches of images
            data = data.batch(1, drop_remainder=True)
        else:
            raise ValueError("Invalid mode '%s'." % (mode))

        # shuffle the first `buffer_size` elements of the dataset
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)

        self.data = data
        self._show_data()

    def _read_npy_file(self):
        """Read the content of the npy file and store it into lists."""
        self.trajs_npy = np.load(self.trajs_file)

        self.data_size = np.shape(self.trajs_npy)[0]
        print(self.data_size)

    def _dataset_generator(self):
        data = tf.data.Dataset.from_tensor_slices(self.trajs_npy)

        return data

    def _parse_function_train(self, traj):
        """Input parser for samples of the training set."""
        # load and preprocess the json
        traj_s = traj[0]
        traj_t = traj[1]
        target = traj[2]
        return traj_s, traj_t, target

    def _parse_function_sample(self, traj):
        """Input parser for samples of the test set."""
        # load and preprocess the json
        traj_s = traj[0]
        target = traj[1]
        return traj_s, target

    # 显示数据内容
    def _show_data(self):
        iterator = self.data.make_one_shot_iterator()
        one_element = iterator.get_next()
        with tf.Session() as sess:
            for i in range(1):
                x, y, t = sess.run(one_element)
                print(x)
                print(y)
                print(t)
                # print(np.shape(out[1]))
                # print(out)

            # num = 0
            # try:
            #     while True:
            #         sess.run(one_element)
            #         num += 1
            # except tf.errors.OutOfRangeError:
            #     print("{} end!".format(num))


generator = DataGenerator("./raw_data/trajs.npy", mode='training')
# print(np.shape(np.load('trajs.npy')))
