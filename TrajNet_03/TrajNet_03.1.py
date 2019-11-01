from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import math


class DataLoaderTrajs:
    def __init__(self, file_path):
        self.file_path = file_path
        self.trajs = np.load(self.file_path)

    def get_batch(self, batch_size):
        seq = []
        next_pose = []

        for i in range(batch_size):
            index = np.random.randint(0, np.shape(self.trajs)[0])
            pose = np.concatenate((self.trajs[index][0], self.trajs[index][2]), axis=2)
            seq.append(pose)
            next_pose.append(self.trajs[index][1])

        return np.array(seq, dtype=np.float32), np.array(next_pose, dtype=np.float32)

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
                pose = np.concatenate((self.trajs[i, 0], self.trajs[i, 2, :, 0:3]), axis=-1)
                x_train.append(pose)
                y_train.append(self.trajs[i, 2])
        else:
            for i in range(np.shape(self.trajs)[0]):
                pose = np.concatenate((self.trajs[i, 0, :, 0:3], self.trajs[i, 2, :, 0:3]), axis=-1)
                x_train.append(pose)
                y_train.append(self.trajs[i, 2, :, 0:3])


        return np.array(x_train, dtype=np.float32), np.array(y_train, dtype=np.float32)

    def get_train_data_without_target(self):   # for keras functional API
        x_train = []
        y_train = []
        for i in range(np.shape(self.trajs)[0]):
            x_train.append(self.trajs[i][0])
            y_train.append(self.trajs[i][1])

        return np.array(x_train, dtype=np.float32), np.array(y_train, dtype=np.float32)


class MDN_output_layer(tf.keras.layers.Layer):
    def __init__(self, components_size, out_size):
        self.components_size = components_size
        self.out_size = out_size
        self.mu_linear = tf.keras.layers.Dense(units=out_size*components_size, activation='relu')
        self.sigma_linear = tf.keras.layers.Dense(units=components_size, activation='relu')
        self.mixing_linear = tf.keras.layers.Dense(units=components_size, activation='softmax')

    def call(self, inputs):
        mu = self.mu_linear(inputs)
        sigma = self.sigma_linear(inputs)
        mix = self.mixing_linear(inputs)

        return mu, sigma, mix


def MDN_LOSS(y_true, y_pred):
    shape = np.shape(y_pred)
    out_size = y_true.shape[-1]
    components_size = 3

    mu = tf.slice(y_pred, [0, 0, 0], [shape[0], shape[1], out_size*components_size])
    sigma = tf.slice(y_pred, [0, 0, out_size*components_size], [shape[0], shape[1], components_size])
    mix = tf.slice(y_pred, [0, 0, (out_size+1)*components_size], [shape[0], shape[1], components_size])

    print(mu)
    print(sigma)
    print(mix)

    epsilon = 1e-5
    # factor = 1 / math.sqrt(2 * math.pi)  # 系数

    mu = tf.reshape(mu, (mu.shape[0], mu.shape[1], out_size, components_size))
    mu = tf.transpose(mu, [3, 0, 1, 2])
    exponent_1 = tf.reduce_sum(tf.square(y_true - mu), axis=-1)
    sigma = tf.transpose(sigma, [2, 0, 1])
    exponent_2 = (-2 * tf.maximum(sigma, epsilon))
    exponent = exponent_1 / exponent_2  # 维度[components_size, batch_size, time_steps]
    # y_normal = factor * tf.exp(exponent) / tf.maximum(sigma, epsilon)
    mix = tf.transpose(mix, [2, 0, 1])
    normalizer = (2 * math.pi * sigma)
    exponent = exponent + tf.math.log(mix) - (out_size * .5) * tf.math.log(normalizer)
    exponent = tf.transpose(exponent, [1, 2, 0])

    max_exponent = np.max(exponent, axis=2, keepdims=True)
    mod_exponent = exponent - max_exponent
    gauss_mix = tf.reduce_sum(tf.exp(mod_exponent), axis=2, keepdims=True)
    log_gauss = tf.math.log(gauss_mix) + max_exponent

    cost = -tf.reduce_mean(log_gauss)

    return cost


class SimpleRNN:
    def __new__(self, hidden, inp_dim, op_dim, stacked_lstm_layers):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.LSTM(units=hidden,
                                            activation='relu',
                                            return_sequences=True,
                                            input_shape=(None, inp_dim)))
        for j in range(stacked_lstm_layers - 1):
            self.model.add(tf.keras.layers.LSTM(units=hidden, return_sequences=True))

        self.model.add(tf.keras.layers.Dense(units=op_dim))
        self.model.add(tf.keras.layers.Activation('relu'))

        self.model.add(MDN_output_layer(components_size=3, out_size=op_dim))

        self.model.compile(loss='mse',
                           optimizer='adadelta',
                           metrics=['accuracy'])
        return self.model


def evaluate(model, data_filename, from_file_model=False):
    if from_file_model:
        model_net = tf.keras.models.load_model(model)
    else:
        model_net = model

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
        s_path.append(out[0, 0, :])
        s_pred = np.concatenate((out, goal_pred), axis=-1)

        num_points += 1
        if np.linalg.norm(out[0, 0, 0:3] - goal_pred[0, 0, 0:3]) < 0.01 or num_points > 50:
            break
    tend = time.time()
    print('time elapsed for generated path: ', tend-tstart)
    s_path = np.asarray(s_path)

    return s_path


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
    inp_dim = train_X.shape[-1]  # 输入维数
    hid = 256   # 隐层节点数
    op_dim = train_Y.shape[-1]  # 输出维数
    batch_size = 128    # batch大小
    stacked_hidden_layers = 3   # LSTM层数

    # ------------- 构建模型 -------------
    print("构建模型中。。。")
    model = SimpleRNN(hidden=hid, inp_dim=inp_dim, op_dim=op_dim, stacked_lstm_layers=stacked_hidden_layers)
    model.summary()

    # ------------- 开始训练 -------------
    tensorboard_cbk = tf.keras.callbacks.TensorBoard(log_dir='./tensorboard_logs')

    print("训练开始！")
    n_iters = 100  #迭代次数
    success_tracker = []
    model_name = './models/model_03.1.h5'    # 模型保存
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

    ##---------------------------------------------------------------------

    # plt.figure()
    # success_tracker = np.asarray(success_tracker)
    # plt.plot(success_tracker[:, 0], success_tracker[:, 1])
    # plt.title('Success rate trends')
    # plt.xlabel('Training Iterations')
    # plt.ylabel('Success Rate over 100 trials')


if __name__ == '__main__':
    # train()
    # evaluate("./models/model_03.1.h5", "traj_position_rotate_test.npy", from_file_model=True)

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

    MDN_LOSS(y_true, y_pred)

