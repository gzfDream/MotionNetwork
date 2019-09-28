# -*- coding=UTF-8 -*-
"""
@Time : 2019/5/28
@Author : gzf
@File : sample.py
@brief:
"""

import numpy as np
import tensorflow as tf

from TrajNet import TrajNet
from DataGenerator import DataGenerator
import os

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_string('checkpoint_path', 'model/default', 'checkpoint path')
tf.flags.DEFINE_string('start_pose', '', 'use this pose to start generating')
tf.flags.DEFINE_string('start_img', '', 'use this img to start generating')
tf.flags.DEFINE_integer('max_length', 50, 'max length to generate')
tf.flags.DEFINE_integer("num_pose", 7, 'number of pose(x,y,z+pose)')


def main(_):

    sample_data = DataGenerator("traj.npy",
                                mode="sample",
                                shuffle=False)

    if os.path.isdir(FLAGS.checkpoint_path):
        FLAGS.checkpoint_path =\
            tf.train.latest_checkpoint(FLAGS.checkpoint_path)

    with tf.name_scope('inputs'):
        input_x = tf.placeholder(tf.float32, shape=(
            1, 1, FLAGS.num_pose), name='inputs')

        target = tf.placeholder(tf.float32, shape=(
            1, 1, FLAGS.num_pose), name='inputs')

    model = TrajNet(input_x,
                    None,
                    target,
                    lstm_size=FLAGS.lstm_size,
                    num_layers=FLAGS.num_layers,
                    training=False,
                    num_pose=FLAGS.num_pose,
                    keep_prob=1)

    proba_prediction = model.proba_prediction
    final_state = model.final_state

    saver = tf.train.Saver()

    result_trajs = []
    iterator = sample_data.data.make_one_shot_iterator()
    next_batch = iterator.get_next()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, FLAGS.checkpoint_path)

        start_pose, target_pose = sess.run(next_batch)

        feed = {input_x: start_pose,
                target: target_pose}

        new_state = sess.run(model.initial_state)
        for i, (c, h) in enumerate(model.initial_state):
            feed[c] = new_state[i].c
            feed[h] = new_state[i].h

        preds, new_state = sess.run([proba_prediction,
                                     final_state],
                                    feed_dict=feed)
        result_trajs.append(preds)

        for i in range(FLAGS.max_length - 1):
            # print("new_state: ", new_state)
            feed = {input_x: preds}

            for j, (c, h) in enumerate(model.initial_state):
                feed[c] = new_state[j].c
                feed[h] = new_state[j].h

            preds, new_state = sess.run([proba_prediction,
                                         final_state],
                                        feed_dict=feed)

            result_trajs.append(preds)
            print(i)


if __name__ == '__main__':
    tf.app.run()
