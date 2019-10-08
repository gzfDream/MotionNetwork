# -*- coding=UTF-8 -*-
"""
@Time : 2019/5/28
@Author : gzf
@File : train.py
@brief: 训练网络
"""

import numpy as np
import tensorflow as tf
import os
import time

from TrajNet import TrajNet
from DataGenerator import DataGenerator


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # 只显示 Error

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('name', 'default', 'name of the model')
tf.flags.DEFINE_integer('batch_size', 64, 'number of seqs in one batch')
tf.flags.DEFINE_integer('num_steps', 50, 'length of one seq')
tf.flags.DEFINE_integer('lstm_size', 64, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
tf.flags.DEFINE_float('train_keep_prob', 0.5, 'dropout rate during training')
tf.flags.DEFINE_integer('num_epochs', 10, 'max steps to train')
tf.flags.DEFINE_integer('save_every_n', 2, 'save the model every n steps')
tf.flags.DEFINE_integer('log_every_n', 2, 'log to the screen every n steps')
tf.flags.DEFINE_integer("num_pose", 7, 'number of pose(x,y,z+pose)')


def main(argv):
    # Network params
    grad_clip = 5

    print("开始加载数据：")
    model_path = os.path.join('model', FLAGS.name)

    train_data = DataGenerator("raw_data/trajs.npy",
                               mode="training",
                               batch_size=FLAGS.batch_size,
                               shuffle=True)

    with tf.name_scope('inputs'):
        input_x = tf.placeholder(dtype=tf.float32, shape=(
            FLAGS.batch_size, FLAGS.num_steps, FLAGS.num_pose), name='input_x')
        input_y = tf.placeholder(dtype=tf.float32, shape=(
            FLAGS.batch_size, FLAGS.num_steps, FLAGS.num_pose), name='input_y')
        targets = tf.placeholder(dtype=tf.float32, shape=(
            FLAGS.batch_size, FLAGS.num_steps, FLAGS.num_pose), name='targets')

        keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

    model = TrajNet(input_x,
                    input_y,
                    targets,
                    lstm_size=FLAGS.lstm_size,
                    num_layers=FLAGS.num_layers,
                    batch_size=FLAGS.batch_size,
                    timestep_size=FLAGS.num_steps,
                    training=True,
                    keep_prob=FLAGS.train_keep_prob,
                    num_pose=FLAGS.num_pose)

    loss = model.cost_mdn

    with tf.name_scope('optimizer'):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
        train_op = tf.train.AdamOptimizer(FLAGS.learning_rate)
        optimizer = train_op.apply_gradients(zip(grads, tvars))

    train_batches_per_epoch = int(
        np.floor(
            train_data.data_size /
            FLAGS.batch_size))
    print('number of dataset: ', train_data.data_size)
    print('number of batch: ', train_batches_per_epoch)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        print("开始训练：")
        sess.run(tf.global_variables_initializer())
        for epoch in range(FLAGS.num_epochs):
            iterator = train_data.data.make_one_shot_iterator()
            next_batch = iterator.get_next()

            new_state = sess.run(model.initial_state)
            for step in range(train_batches_per_epoch):
                x, y, targets = sess.run(next_batch)
                start = time.time()

                feed = {input_x: x,
                        input_y: y,
                        targets: targets,
                        keep_prob: FLAGS.train_keep_prob,
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


if __name__ == '__main__':
    tf.app.run()
