from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import math
import time

class DataLoaderTrajs:
    def __init__(self, file_path):
        self.file_path = file_path
        self.trajs = np.load(self.file_path)

    def get_batch(self, batch_size):
        seq = []
        next_pose = []

        for i in range(batch_size):
            index = np.random.randint(0, np.shape(self.trajs)[0])
            pose = np.concatenate(
                (self.trajs[index][0], self.trajs[index][2]), axis=2)
            seq.append(pose)
            next_pose.append(self.trajs[index][1])

        return np.array(
            seq, dtype=np.float32), np.array(
            next_pose, dtype=np.float32)

    def get_test_data(self, with_rotate=True):
        index = np.random.randint(0, np.shape(self.trajs)[0])
        # pose = np.concatenate((self.trajs[index][0][0], self.trajs[index][2][0]), axis=0)

        if with_rotate:
            pose = self.trajs[index][0][0]
            goal = self.trajs[index][2][0]

            ground_truth = [pose]
            for i in range(np.shape(self.trajs[index])[1]):
                ground_truth.append(self.trajs[index][1][i])
        else:
            pose = self.trajs[index, 0, 0, 0:3]
            goal = self.trajs[index, 2, 0, 0:3]

            ground_truth = [pose]
            for i in range(np.shape(self.trajs[index])[1]):
                ground_truth.append(self.trajs[index, 1, i, 0:3])

        return pose, goal, ground_truth

    def get_train_data(self, with_rotate=True):   # for keras functional API
        x_train = []
        y_train = []
        if with_rotate:
            for i in range(np.shape(self.trajs)[0]):
                pose = np.concatenate(
                    (self.trajs[i, 0], self.trajs[i, 2, :, 0:3]), axis=-1)
                x_train.append(pose)
                y_train.append(self.trajs[i, 2])
        else:
            for i in range(np.shape(self.trajs)[0]):
                pose = np.concatenate(
                    (self.trajs[i, 0, :, 0:3], self.trajs[i, 2, :, 0:3]), axis=-1)
                x_train.append(pose)
                y_train.append(self.trajs[i, 2, :, 0:3])

        return np.array(
            x_train, dtype=np.float32), np.array(
            y_train, dtype=np.float32)

    def get_train_data_without_target(self):   # for keras functional API
        x_train = []
        y_train = []
        for i in range(np.shape(self.trajs)[0]):
            x_train.append(self.trajs[i][0])
            y_train.append(self.trajs[i][1])

        return np.array(
            x_train, dtype=np.float32), np.array(
            y_train, dtype=np.float32)


def mdn_loss_xyz(y_true, y_pred):
    components_size = 3  # 混合高斯数

    shape = np.shape(y_pred)
    out_size = tf.cast(shape[-1] / components_size - 2, dtype=tf.int32)

    # [batch_size, time_steps, out_size * components_size]
    # mu = tf.slice(y_pred, [0, 0, 0], [batch_size, time_steps, out_size * components_size])
    mu = y_pred[:shape[0], :shape[1], :out_size * components_size]
    # [batch_size, components_size]
    # sigma = tf.slice(y_pred, [0, 0, out_size * components_size], [batch_size, time_steps, components_size])
    sigma = y_pred[:shape[0], :shape[1], out_size*components_size:(out_size+1)*components_size]
    # [batch_size, components_size]
    # mix = tf.slice(y_pred, [0, 0, (out_size + 1) * components_size], [batch_size, time_steps, components_size])
    mix = y_pred[:shape[0], :shape[1], (out_size+1) * components_size:(out_size + 2) * components_size]

    epsilon = 1e-10  # 防止除数小于0
    # [batch_size, time_steps, out_size, components_size]
    mu = tf.reshape(mu, (tf.shape(mu)[0], tf.shape(mu)[1], out_size, components_size))
    # [components_size, batch_size, time_steps, out_size]
    mu = tf.transpose(mu, [3, 0, 1, 2])
    exponent_1 = tf.reduce_sum(tf.square(y_true - mu), axis=-1)  # ∑(x-μ)^2

    # [components_size, batch_size, time_steps]
    sigma = tf.transpose(sigma, [2, 0, 1])
    exponent_2 = (-2 * tf.maximum(sigma, epsilon))  # -2*(σ^2)
    # 维度[components_size, batch_size, time_steps]
    exponent = exponent_1 / exponent_2
    normalizer = (2 * math.pi * sigma)
    y_normal = tf.exp(exponent) / tf.maximum(normalizer, epsilon)

    # 维度[batch_size, time_steps, components_size]
    y_normal = tf.transpose(y_normal, [1, 2, 0])

    # softmax all the mix's:
    max_mix = tf.reduce_max(mix, axis=-1)
    max_mix = tf.tile(tf.expand_dims(max_mix, -1), [1, 1, mix.shape[-1]])
    mix = tf.subtract(mix, max_mix)
    mix = tf.exp(mix)
    normalize_mix = 1 / tf.maximum(tf.reduce_sum(mix, axis=-1), epsilon)
    normalize_mix = tf.tile(tf.expand_dims(
        normalize_mix, -1), [1, 1, mix.shape[-1]])
    # 维度[batch_size, time_steps, components_size]
    mix = tf.multiply(normalize_mix, mix)

    loss = tf.reduce_sum(tf.multiply(y_normal, mix), axis=-1)
    loss = -tf.math.log(tf.maximum(loss, epsilon))
    loss = tf.reduce_mean(loss)

    return loss


# class MDNMetrics(keras.metrics.Metric):
#     def __init__(self, name='categorical_true_positives', **kwargs):
#       super(MDNMetrics, self).__init__(name=name, **kwargs)
#       self.true_positives = self.add_weight(name='tp', initializer='zeros')
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#       y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
#       values = tf.cast(y_true, 'int32') == tf.cast(y_pred, 'int32')
#       values = tf.cast(values, 'float32')
#       if sample_weight is not None:
#         sample_weight = tf.cast(sample_weight, 'float32')
#         values = tf.multiply(values, sample_weight)
#       self.true_positives.assign_add(tf.reduce_sum(values))
#
#     def result(self):
#       return self.true_positives
#
#     def reset_states(self):
#       # The state of the metric will be reset at the start of each epoch.
#       self.true_positives.assign(0.)


class SimpleRNN:
    def __new__(self, hidden, inp_dim, op_dim, stacked_lstm_layers):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.LSTM(units=hidden,
                                            activation='relu',
                                            return_sequences=True,
                                            input_shape=(None, inp_dim)))
        for j in range(stacked_lstm_layers - 1):
            self.model.add(
                tf.keras.layers.LSTM(
                    units=hidden,
                    return_sequences=True))

        self.model.add(tf.keras.layers.Dense(units=op_dim))
        self.model.add(tf.keras.layers.Activation('relu'))

        self.model.compile(loss=mdn_loss_xyz,
                           optimizer='adadelta',
                           metrics=['AUC'])
        return self.model


def train():
    # ------------- 加载数据 -------------
    print("开始加载数据！")
    data_filename = 'traj_position_rotate.npy'
    test_filename = 'traj_position_rotate_test.npy'
    trainingData = DataLoaderTrajs(file_path=data_filename)
    train_X, train_Y = trainingData.get_train_data(with_rotate=False)
    print("x_train:", train_X.shape)
    print("y_train: ", train_Y.shape)
    print("数据加载完毕！")

    # ------------- 设置训练参数 -------------
    components_size = 3  # 高斯混合个数
    inp_dim = train_X.shape[-1]  # 输入维数
    hid = 64   # 隐层节点数
    op_dim = (train_Y.shape[-1]+2)*components_size  # 输出维数
    batch_size = 128    # batch大小
    stacked_hidden_layers = 3   # LSTM层数

    # ------------- 构建模型 -------------
    print("构建模型中。。。")
    model = SimpleRNN(
        hidden=hid,
        inp_dim=inp_dim,
        op_dim=op_dim,
        stacked_lstm_layers=stacked_hidden_layers)
    model.summary()

    # ------------- 开始训练 -------------
    tensorboard_cbk = tf.keras.callbacks.TensorBoard(log_dir='./tensorboard_logs')

    print("训练开始！")
    n_iters = 100  # 迭代次数
    success_tracker = []
    model_name = './models/model_04.1.h5'    # 模型保存
    t1 = time.time()
    for k in range(n_iters):
        model.fit(train_X, train_Y, batch_size=batch_size,
                  epochs=10, validation_split=0.1, shuffle=True,
                  callbacks=[tensorboard_cbk])
        print(model_name + ' Epoch set:', k, ' for ', model_name)
        if k % 2 == 0:
            model.save(model_name)
            # path_pre = evaluate(model=model, data_filename=test_filename, from_file_model=False)

    t2 = time.time()
    print('time taken to train: ', t2 - t1)
    print('Training complete!')

def sample(model, data_filename, tiems_steps):

    components_size = 3  # 高斯混合个数
    pose_size = 3

    def sample_mix(mixes):
        stop = np.random.rand()  # random number to stop
        num_thetas = len(mixes)
        cum = 0.0  # cumulative probability
        for i in range(num_thetas):
            cum += mixes[i]
            if cum > stop:
              return i
        print('No theta is drawn, ERROR')
        return

    def sample_pose(output):
        idx_mix = sample_mix(output[0, 0][components_size*(pose_size+1):])
        mean = output[0][0][idx_mix*pose_size: (idx_mix+1)*pose_size]
        sigma = output[0][0][components_size*pose_size+idx_mix]

        position = mean+sigma*np.random.randn(pose_size)

        return position

    model_net = tf.keras.models.load_model(model)

    test_data = DataLoaderTrajs(file_path=data_filename)
    start_pose, goal_psoe, path = test_data.get_test_data(with_rotate=False)

    s_path = [start_pose]
    num_points = 0
    tstart = time.time()

    start_pred = start_pose[np.newaxis, np.newaxis, :]
    goal_pred = goal_psoe[np.newaxis, np.newaxis, :]
    s_pred = np.concatenate((start_pred, goal_pred), axis=-1)
    while True:
        out = model_net.predict(s_pred)
        out_position = sample_pose(out)

        s_path.append(out_position)
        s_pred = tf.reshape(out_position, (1, 1, pose_size))

        num_points += 1
        if np.linalg.norm(out[0, 0, 0:3] - goal_pred[0, 0, 0:3]) < 0.01 or num_points > tiems_steps:
            break
    tend = time.time()
    print('time elapsed for generated path: ', tend - tstart)
    s_path = np.asarray(s_path)


if __name__ == '__main__':
    train()
