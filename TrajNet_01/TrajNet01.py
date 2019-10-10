# -*- coding=UTF-8 -*-
"""
@Time : 2019/5/28
@Author : gzf
@File : TrajNet.py
@brief:
"""

import sys
# sys.path.append('H:/OneDrive/code/MotionPlanning/MotionNetwork')

import numpy as np
import tensorflow as tf
import os
import time

from DataGenerator import DataGenerator

FLAGS = tf.flags.FLAGS


# RNN轨迹预测
class TrajNet:
    def __init__(
            self,
            input_x=None,
            input_y=None,
            targets=None,
            lstm_size=128,      # 隐藏层数目
            num_layers=3,       # LSTM层数
            batch_size=64,      # batch size
            timestep_size=50,   # 一条轨迹有多少个位姿
            training=True,      # train or sample
            keep_prob=0.5,      #
            num_pose=7):        # 一个位姿由七个值表示
        if training is False:
            batch_size, timestep_size = 1, 1
        else:
            batch_size, timestep_size = batch_size, timestep_size

        self._input_x = input_x
        self._input_y = input_y
        self._targets = targets
        self._batch_size = batch_size
        self._timestep_size = timestep_size
        self._lstm_size = lstm_size
        self._num_layers = num_layers
        self._keep_prob = keep_prob
        self._num_pose = num_pose

        # tf.reset_default_graph()
        self.build_lstm()

    def build_lstm(self):

        # 串联当前位姿和最终的目标位姿
        input_lstm = tf.concat([self._input_x, self._targets], axis=2)

        # 创建单个cell并堆叠多层
        def get_a_cell(lstm_size, keep_prob):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
            drop = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
                lstm, output_keep_prob=keep_prob)
            return drop

        with tf.name_scope('lstm'):
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [get_a_cell(self._lstm_size, self._keep_prob) for _ in range(self._num_layers)]
            )

            # 维度(num_layers, [self.batch_size, lstm_size])
            self.initial_state = cell.zero_state(self._batch_size, tf.float32)

            # 通过dynamic_rnn对cell展开时间维度
            # self.lstm_outputs的维度[batch_size, _timestep_size, lstm_size]
            # self.final_state的维度[batch_size, lstm_size]
            self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(
                cell, input_lstm, initial_state=self.initial_state)

            # 通过lstm_outputs得到预测
            # 维度[batch_size, _timestep_size, lstm_size]
            traj_output = tf.concat(self.lstm_outputs, 1)

            with tf.compat.v1.variable_scope('Output_FC'):
                softmax_w = tf.Variable(tf.random.truncated_normal(
                    [self._lstm_size, self._num_pose], stddev=0.1))
                softmax_b = tf.Variable(tf.zeros(self._num_pose))

                # 维度[batch_size, _timestep_size, _num_pose]
                self.logits = tf.matmul(traj_output, softmax_w) + softmax_b

            with tf.name_scope('Loss') as scope:
                self.cost = tf.compat.v1.losses.mean_squared_error(self._input_y, self.logits)


tf.flags.DEFINE_string('name', 'train02', 'name of the model')
tf.flags.DEFINE_integer('batch_size', 64, 'number of seqs in one batch')
tf.flags.DEFINE_integer('num_steps', 50, 'length of one seq')
tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 5, 'number of lstm layers')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
tf.flags.DEFINE_float('train_keep_prob', 0.5, 'dropout rate during training')
tf.flags.DEFINE_integer('num_epochs', 200, 'max steps to train')
tf.flags.DEFINE_integer('save_every_n', 20, 'save the model every n steps')
tf.flags.DEFINE_integer('log_every_n', 20, 'log to the screen every n steps')
tf.flags.DEFINE_integer("num_pose", 3, 'number of pose(x,y,z+pose)')


def train(argv):
    # Network params
    grad_clip = 5

    print("开始加载数据：")
    model_path = os.path.join('model', FLAGS.name)

    train_data = DataGenerator("./traj_position.npy",
                               mode="training",
                               batch_size=FLAGS.batch_size,
                               shuffle=True)

    with tf.name_scope('inputs'):
        input_X = tf.compat.v1.placeholder(dtype=tf.float32, shape=(
            FLAGS.batch_size, FLAGS.num_steps, FLAGS.num_pose), name='input_x')
        input_Y = tf.compat.v1.placeholder(dtype=tf.float32, shape=(
            FLAGS.batch_size, FLAGS.num_steps, FLAGS.num_pose), name='input_y')
        targets_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(
            FLAGS.batch_size, FLAGS.num_steps, FLAGS.num_pose), name='targets')

        # train_keep_prob = tf.compat.v1.placeholder(dtype=tf.float32, name='keep_prob')

    model = TrajNet(input_X,
                    input_Y,
                    targets_,
                    lstm_size=FLAGS.lstm_size,
                    num_layers=FLAGS.num_layers,
                    batch_size=FLAGS.batch_size,
                    timestep_size=FLAGS.num_steps,
                    training=True,
                    keep_prob=FLAGS.train_keep_prob,
                    num_pose=FLAGS.num_pose)

    loss = model.cost

    with tf.name_scope('optimizer'):
        tvars = tf.compat.v1.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
        train_op = tf.compat.v1.train.AdamOptimizer(FLAGS.learning_rate)
        optimizer = train_op.apply_gradients(zip(grads, tvars))

    train_batches_per_epoch = int(
        np.floor(
            train_data.data_size /
            FLAGS.batch_size))
    print('number of dataset: ', train_data.data_size)
    print('number of batch: ', train_batches_per_epoch)

    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session() as sess:
        print("开始训练：")
        sess.run(tf.compat.v1.global_variables_initializer())
        for epoch in range(FLAGS.num_epochs):
            iterator = train_data.data.make_one_shot_iterator()
            next_batch = iterator.get_next()

            new_state = sess.run(model.initial_state)
            for step in range(train_batches_per_epoch):
                x, y, t = sess.run(next_batch)
                start = time.time()

                feed = {input_X: x,
                        input_Y: y,
                        targets_: t
                        # train_keep_prob: FLAGS.train_keep_prob,
                        }
                for i, (c, h) in enumerate(model.initial_state):
                    feed[c] = new_state[i].c
                    feed[h] = new_state[i].h

                batch_loss, new_state, _ = sess.run([loss,
                                                     model.final_state,
                                                     optimizer],
                                                    feed_dict=feed)

                end = time.time()
                # control the print lines
                if step % FLAGS.log_every_n == 0:
                    print('step: {}/{}... '.format(step, epoch),
                          'loss: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))

            if (epoch % FLAGS.save_every_n == 0):
                saver.save(sess, os.path.join(
                    model_path,
                    'model'),
                    global_step=epoch)

        saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)


def sample(argv):
    checkpoint_path = 'model/train02'
    num_pose = 3
    lstm_size = 128
    num_layers = 5
    max_length = 150

    sample_data = DataGenerator("./traj.npy",
                                mode="sample",
                                shuffle=False)

    if os.path.isdir(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)

    with tf.name_scope('inputs'):
        input_x = tf.compat.v1.placeholder(tf.float32, shape=(
            1, 1, num_pose), name='inputs_x')

        input_y = tf.compat.v1.placeholder(tf.float32, shape=(
            1, 1, num_pose), name='inputs_y')

        target = tf.compat.v1.placeholder(tf.float32, shape=(
            1, 1, num_pose), name='inputs_target')

    model = TrajNet(input_x,
                    input_y,
                    target,
                    lstm_size=lstm_size,
                    num_layers=num_layers,
                    training=False,
                    num_pose=num_pose,
                    keep_prob=1)

    proba_prediction = model.logits
    final_state = model.final_state

    saver = tf.train.Saver()

    result_trajs = []
    iterator = sample_data.data.make_one_shot_iterator()
    next_batch = iterator.get_next()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_path)

        start_pose, target_pose = sess.run(next_batch)
        # print(np.shape(list(target_pose)))
        result_trajs.append(start_pose)

        feed = {input_x: start_pose,
                input_y: target_pose,
                target: target_pose}

        new_state = sess.run(model.initial_state)
        for i, (c, h) in enumerate(model.initial_state):
            feed[c] = new_state[i].c
            feed[h] = new_state[i].h

        preds, new_state = sess.run([proba_prediction,
                                     final_state],
                                    feed_dict=feed)
        result_trajs.append(preds)

        for i in range(max_length - 1):
            # print("new_state: ", new_state)
            feed = {input_x: preds,
                    target: target_pose}

            for j, (c, h) in enumerate(model.initial_state):
                feed[c] = new_state[j].c
                feed[h] = new_state[j].h

            preds, new_state = sess.run([proba_prediction,
                                         final_state],
                                        feed_dict=feed)

            result_trajs.append(preds)
            print(i)

    result_trajs = np.reshape(result_trajs, (max_length+1, num_pose))
    result_list = []
    for pose in result_trajs:
        l = []
        for i in pose:
            l.append(i)
        result_list.append(l)
    np.savetxt("results.txt", result_list, fmt='%f', delimiter=' ')


if __name__ == '__main__':
    tf.compat.v1.app.run(sample)
    # data = np.load("../raw_data/traj.npy")
    # print(np.shape(data[0][1]))
