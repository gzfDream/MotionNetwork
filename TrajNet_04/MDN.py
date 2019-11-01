from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import math


def mdn_loss_xyz(y_true, y_pred):
    shape = np.shape(y_pred)
    out_size = y_true.shape[-1]
    components_size = 3  # 混合高斯数

    # [batch_size, time_steps, out_size * components_size]
    mu = tf.slice(y_pred, [0, 0, 0], [shape[0], shape[1], out_size * components_size])
    # [batch_size, components_size]
    sigma = tf.slice(y_pred, [0, 0, out_size * components_size], [shape[0], shape[1], components_size])
    # [batch_size, components_size]
    mix = tf.slice(y_pred, [0, 0, (out_size + 1) * components_size], [shape[0], shape[1], components_size])

    epsilon = 1e-10  #防止除数小于0
    # [batch_size, time_steps, out_size, components_size]
    mu = tf.reshape(mu, (mu.shape[0], mu.shape[1], out_size, components_size))
    # [components_size, batch_size, time_steps, out_size]
    mu = tf.transpose(mu, [3, 0, 1, 2])
    exponent_1 = tf.reduce_sum(tf.square(y_true - mu), axis=-1)  # ∑(x-μ)^2

    # [components_size, batch_size, time_steps]
    sigma = tf.transpose(sigma, [2, 0, 1])
    exponent_2 = (-2 * tf.maximum(sigma, epsilon))  # -2*(σ^2)
    exponent = exponent_1 / exponent_2  # 维度[components_size, batch_size, time_steps]
    normalizer = (2 * math.pi * sigma)
    y_normal = tf.exp(exponent) / tf.maximum(normalizer, epsilon)

    # 维度[batch_size, time_steps, components_size]
    y_normal = tf.transpose(y_normal, [1, 2, 0])

    # softmax all the mix's:
    max_mix = tf.reduce_max(mix, axis=-1)
    max_mix = tf.tile(tf.expand_dims(max_mix, -1), [1, 1, mix.shape[-1]])
    mix = tf.subtract(mix, max_mix)
    mix = tf.exp(mix)
    normalize_mix = 1/tf.maximum(tf.reduce_sum(mix, axis=-1), epsilon)
    normalize_mix = tf.tile(tf.expand_dims(normalize_mix, -1), [1, 1, mix.shape[-1]])
    # 维度[batch_size, time_steps, components_size]
    mix = tf.multiply(normalize_mix, mix)

    loss = tf.reduce_sum(tf.multiply(y_normal, mix), axis=-1)
    loss = -tf.math.log(tf.maximum(loss, epsilon))
    loss = tf.reduce_mean(loss)

    return loss


y_pred = np.array([
        [[1., 2.1, 3.2, 4.3, 5., 6., 11., 12.1, 13.2, 14.3, 15., 16.],
         [2., 2.1, 3.2, 4.3, 5., 6., 11., 12.1, 13.2, 14.3, 15., 16.],
         [3., 2.1, 3.2, 4.3, 5., 6., 11., 12.1, 13.2, 14.3, 15., 16.]],
        [[4., 2.1, 3.2, 4.3, 5., 6., 11., 12.1, 13.2, 14.3, 15., 16.],
         [5., 2.1, 3.2, 4.3, 5., 6., 11., 12.1, 13.2, 14.3, 15., 16.],
         [6., 2.1, 3.2, 4.3, 5., 6., 11., 12.1, 13.2, 14.3, 15., 16.]]
    ])
y_true = np.array([
        [[1., 2.1], [1., 2.1], [1., 2.1]],
        [[1.1, 2.1], [1., 2.1], [1., 2.1]]
    ])


class TrajNet(tf.keras.Model):
    def __init__(self, num_pose, num_components):
        super(TrajNet, self).__init__()
        self.num_pose = num_pose
        self.num_components = num_components
        self.cell_1 = tf.keras.layers.LSTMCell(units=128, activation='relu')
        self.cell_2 = tf.keras.layers.LSTMCell(units=128, activation='relu')
        self.cell_3 = tf.keras.layers.LSTMCell(units=128, activation='relu')
        self.dense_1 = tf.keras.layers.Dense(units=self.num_components*(self.num_pose+2), activation='relu')
        # self.dense_2 = tf.keras.layers.Dense(units=self.num_pose, activation='relu')

    # @tf.function
    def call(self, inputs, first_pose, usage_rate, pre_output):
        # inputs 维度：[batch_size, num_pose]
        state_1 = self.cell_1.get_initial_state(batch_size=inputs.shape[0], dtype=tf.float32)
        state_2 = self.cell_2.get_initial_state(batch_size=inputs.shape[0], dtype=tf.float32)
        state_3 = self.cell_3.get_initial_state(batch_size=inputs.shape[0], dtype=tf.float32)

        output_1, state_1 = self.cell_1(inputs, state_1)
        inputs_layer2 = tf.concat([inputs, output_1], axis=-1)
        output_2, state_2 = self.cell_2(inputs_layer2, state_2)

        inputs_layer3 = tf.concat([inputs, output_2], axis=-1)
        output_3, state_3 = self.cell_2(inputs_layer3, state_3)

        output = tf.concat([output_1, output_2, output_3], axis=-1)
        logit = self.dense_1(output)

        return logit


if __name__ == '__main__':
    print(y_pred.shape)
    print(y_true.shape)
    loss = mdn_loss_xyz(y_true, y_pred)
    print(loss)
