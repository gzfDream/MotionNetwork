# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

import tensorflow as tf
import numpy as np

Py3 = sys.version_info[0] == 3


# 读取words
def _read_words(filename):
    with tf.io.gfile.GFile(filename, "r") as f:
        if Py3:
            return f.read().replace("\n", "<eos>").split()
        else:
            return f.read().decode("utf-8").replace("\n", "<eos>").split()


# 创建字符对应字典
def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
    """Load PTB raw data from data directory "data_path".

    Reads PTB text files, converts strings to integer ids,
    and performs mini-batching of the inputs.

    The PTB dataset comes from Tomas Mikolov's webpage:

    http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

    Args:
      data_path: string path to the directory where simple-examples.tgz has
        been extracted.

    Returns:
      tuple (train_data, valid_data, test_data, vocabulary)
      where each of the data objects can be passed to PTBIterator.
    """

    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary


def ptb_producer(raw_data, batch_size, num_steps, name=None):
    """Iterate on the raw PTB data.

    This chunks up raw_data into batches of examples and returns Tensors that
    are drawn from these batches.

    Args:
      raw_data: one of the raw data outputs from ptb_raw_data.
      batch_size: int, the batch size.
      num_steps: int, the number of unrolls.
      name: the name of this operation (optional).

    Returns:
      A pair of Tensors, each shaped [batch_size, num_steps]. The second element
      of the tuple is the same data time-shifted to the right by one.

    Raises:
      tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
    """
    with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(
            raw_data, name="raw_data", dtype=tf.int32)

        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0: batch_size * batch_len],
                          [batch_size, batch_len])

        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps],
                             [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1],
                             [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
        return x, y


class DataGenerator(object):
    """
    Wrapper class around the new Tensorflows dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(
            self,
            trajs_file,
            mode='training',
            batch_size=64,
            shuffle=True,
            buffer_size=10000):
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
        # self._show_data()

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


def _read_npy(trajs_file):
    return np.load(trajs_file)


def Traj_producer(raw_data_file, batch_size, num_steps, name=None):
    """Iterate on the raw PTB data.

    This chunks up raw_data into batches of examples and returns Tensors that
    are drawn from these batches.

    Args:
        raw_data: one of the raw data outputs from ptb_raw_data.
        batch_size: int, the batch size.
        num_steps: int, the number of unrolls.
        name: the name of this operation (optional).

    Returns:
        A pair of Tensors, each shaped [batch_size, num_steps]. The second element
        of the tuple is the same data time-shifted to the right by one.

    Raises:
        tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
    """
    with tf.name_scope(name, "TrajProducer", [raw_data_file, batch_size, num_steps]):
        data = DataGenerator(raw_data_file, batch_size, num_steps)




if __name__ == '__main__':
    train_data, valid_data, test_data, vocabulary = ptb_raw_data("./simple-examples/data/")
    sess = tf.compat.v1.Session()
    x_data, y_data = sess.run(ptb_producer(test_data, 64, 32))

    print(x_data)
    print("end")
