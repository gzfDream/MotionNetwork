# -*- coding=UTF-8 -*-
"""
@Time : 2019/5/28
@Author : gzf
@File : TrajNet.py
@brief:
"""

import sys
sys.path.append('H:/OneDrive/code/MotionPlanning/MotionNetwork')
import numpy as np
import tensorflow as tf
from util_MDN import *


FLAGS = tf.flags.FLAGS


def scheduled_sampling_prob(global_step):
  def get_processed_step(step):
    step = tf.maximum(step, FLAGS.scheduled_sampling_starting_step)
    step = tf.minimum(step, FLAGS.scheduled_sampling_ending_step)
    step = tf.maximum(step - FLAGS.scheduled_sampling_starting_step, 0)
    step = tf.cast(step, tf.float32)
    return step

  def inverse_sigmoid_decay_fn(step):
    step = get_processed_step(step)
    k = float(FLAGS.inverse_sigmoid_decay_k)
    p = 1.0 - k / (k + tf.exp(step / k))
    return p

  def linear_decay_fn(step):
    step = get_processed_step(step)
    slope = (FLAGS.scheduled_sampling_ending_rate - FLAGS.scheduled_sampling_starting_rate) / (FLAGS.scheduled_sampling_ending_step - FLAGS.scheduled_sampling_starting_step)
    a = FLAGS.scheduled_sampling_starting_rate
    p = a + slope * step
    return p

  sampling_fn = {
      "linear": linear_decay_fn,
      "inverse_sigmoid": inverse_sigmoid_decay_fn
  }

  sampling_probability = sampling_fn[FLAGS.scheduled_sampling_method](global_step)
  tf.summary.scalar("scheduled_sampling/prob", sampling_probability)
  return sampling_probability


# RNN轨迹预测
class TrajNet:
    def __init__(
            self,
            input_x=None,
            input_y=None,
            targets=None,
            lstm_size=128,   #
            num_layers=1,    #
            batch_size=64,   #
            num_steps=50,    # 一条轨迹有多少个位姿
            training=True,   # train or
            keep_prob=0.5,
            num_pose=7):     # 一个位姿由七个值表示
        if training is False:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps

        self._input_x = input_x
        self._input_y = input_y
        self._targets = targets
        self._batch_size = batch_size
        self._num_steps = num_steps
        self._lstm_size = lstm_size
        self._num_layers = num_layers
        self._keep_prob = keep_prob
        self._num_pose = num_pose

        # tf.reset_default_graph()
        self.build_lstm()

    def build_lstm(self):

        # 串联当前位姿和最终的目标位姿
        print(self._input_x)
        print(self._targets)
        input_lstm = tf.concat([self._input_x, self._targets], axis=2)

        # 创建单个cell并堆叠多层
        def get_a_cell(lstm_size, keep_prob):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
            drop = tf.nn.rnn_cell.DropoutWrapper(
                lstm, output_keep_prob=keep_prob)
            return drop

        with tf.name_scope('lstm'):
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [get_a_cell(self._lstm_size, self._keep_prob) for _ in range(self._num_layers)]
            )

            # 维度(num_layers, [self.batch_size, self.lstm_size])
            self.initial_state = cell.zero_state(self._batch_size, tf.float32)

            # 通过dynamic_rnn对cell展开时间维度
            # self.lstm_outputs的维度[num_traj, num_steps, lstm_size]
            # self.final_state的维度[self.num_traj, self.lstm_size]
            self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(
                cell, input_lstm, initial_state=self.initial_state)

            # 通过lstm_outputs得到预测
            # 维度[num_traj, num_steps, lstm_size]
            traj_output = tf.concat(self.lstm_outputs, 1)
            # 维度[num_traj*num_steps, lstm_size]
            x = tf.reshape(traj_output, [-1, self._lstm_size])

            with tf.variable_scope('Output_MDN'):
                params = self._num_pose + 2  # mu(xyz+xyzw)+alpha+theta
                mixtures = 12  # 高斯核数量m
                output_units = mixtures * params  # m*(c+2)

                softmax_w = tf.Variable(tf.truncated_normal(
                    [self._lstm_size, output_units], stddev=0.1))
                softmax_b = tf.Variable(tf.zeros(output_units))

                # 维度[num_traj*num_steps, output_units]
                self.logits = tf.matmul(x, softmax_w) + softmax_b


            with tf.name_scope('MDN_Loss') as scope:
                # 维度[num_traj, num_steps, self.num_pose]
                self.proba_prediction = tf.reshape(
                     self.logits, (self._batch_size, self._num_steps, output_units))

                self.cost_mdn = mixture_density(self._input_y, self.proba_prediction, mixtures, self._num_pose)



