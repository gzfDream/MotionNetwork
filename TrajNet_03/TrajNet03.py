from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras


def save_trajectories(trajectories, path):
    result_list = []
    for pose in trajectories:
        l = []
        for i in pose:
            l.append(i)
        result_list.append(l)
    np.savetxt(path, result_list, fmt='%f', delimiter=' ')


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

    def get_batch_test(self):
        index = np.random.randint(0, np.shape(self.trajs)[0])
        # pose = np.concatenate((self.trajs[index][0][0], self.trajs[index][2][0]), axis=0)
        pose = self.trajs[index][0][0]
        return pose

    def get_train_data(self):   # for keras functional API
        x_train = []
        y_train = []
        for i in range(np.shape(self.trajs)[0]):
            pose = np.concatenate((self.trajs[i][0], self.trajs[i][2]), axis=1)
            x_train.append(pose)
            y_train.append(self.trajs[i][1])

        return np.array(x_train, dtype=np.float32), np.array(y_train, dtype=np.float32)

    def get_train_data_without_target(self):   # for keras functional API
        x_train = []
        y_train = []
        for i in range(np.shape(self.trajs)[0]):
            x_train.append(self.trajs[i][0])
            y_train.append(self.trajs[i][1])

        return np.array(x_train, dtype=np.float32), np.array(y_train, dtype=np.float32)


# 自定义损失函数（继承keras.losses.Loss）
class WeightedLossFunction(keras.losses.Loss):
    def __init__(self, weight=0.5, reduction=keras.losses.Reduction.AUTO, name="WeightedLossFunction"):
        super(WeightedLossFunction, self).__init__(reduction=reduction, name=name)
        self.weight = weight

    def call(self, y_true, y_pred):
        y_ture_position = y_true[:, :, 0:3]
        y_true_rotate = y_true[:, :, 3:7]

        y_pred_position = y_pred[:, :, 0:3]
        y_pred_rotate = y_pred[:, :, 3:7]

        loss_dis = tf.reduce_mean(tf.square(y_pred_position - y_ture_position))

        # 单位化四元数
        sqrt = tf.tile(tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(y_pred_rotate), 2)), -1), multiples=[1, 1, 4])
        sqrt_ = tf.where(sqrt > 0.00001, sqrt, 0.00001)
        norm_rot = y_pred_rotate / sqrt_

        multi = y_true_rotate * norm_rot
        reduce_sum = tf.abs(tf.reduce_sum(multi, 2))
        loss_rotate = 2 * tf.reduce_mean(tf.acos(reduce_sum))

        loss_total = self.weight * loss_dis + (1 - self.weight) * loss_rotate
        return loss_total


# 自定义损失函数
def basic_loss_function(y_true, y_pred):
    y_ture_position = y_true[:, :, 0:3]
    y_true_rotate = y_true[:, :, 3:7]

    y_pred_position = y_pred[:, :, 0:3]
    y_pred_rotate = y_pred[:, :, 3:7]

    # 均方误差计算位置的差异
    loss_dis = tf.reduce_mean(tf.square(y_pred_position - y_ture_position))

    # 单位化四元数
    sqrt = tf.tile(tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(y_pred_rotate), 2)), -1), multiples=[1, 1, 4])
    sqrt_ = tf.where(sqrt > 0.00001, sqrt, 0.00001)
    norm_rot = y_pred_rotate / sqrt_

    # 计算四元数的差异
    multi = y_true_rotate * norm_rot
    reduce_sum = tf.abs(tf.reduce_sum(multi, 2))
    loss_rotate = 2 * tf.reduce_mean(tf.acos(reduce_sum))

    return loss_dis + loss_rotate


def mdn_loss_xyz(y_true, y_pred):
    mixtures = 3  # 混合高斯数


batch_size = 64
time_steps = 50
pose_size = 7
learning_rate = 1e-3


def train():
    """
        构造模型
    """
    inputs = keras.Input(shape=(None, pose_size*2))

    lstm_layer_0 = layers.LSTM(256, return_sequences=True)
    lstm_layer_1 = layers.LSTM(256, return_sequences=True)
    lstm_layer_2 = layers.LSTM(256, return_sequences=True)
    lstm_layer_3 = layers.LSTM(256, return_sequences=False)
    dense_layer_0 = layers.Dense(128, activation='relu')
    dense_layer_1 = layers.Dense(pose_size, activation='relu')

    x_out0 = lstm_layer_0(inputs)
    x_out1 = lstm_layer_1(x_out0)
    x_out2 = lstm_layer_2(x_out1)
    x_out3 = lstm_layer_3(x_out2)
    x_out4 = dense_layer_0(x_out3)
    outputs = dense_layer_1(x_out4)

    model = keras.Model(inputs=inputs, outputs=outputs, name="motion_net")

    """
        模型结构
    """
    model.summary()
    # keras.utils.plot_model(model, 'rnn_model.png', show_shapes=True)

    """
        加载数据
    """
    print("加载数据")
    data_path = "traj_position_rotate.npy"
    dataset = DataLoaderTrajs(data_path)
    x_train, y_train = dataset.get_train_data_without_target()
    print("数据加载完毕！")
    print("x_train:", x_train.shape)
    print("y_train: ", y_train.shape)

    print("开始训练")
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
                  loss='mse',
                  metrics=['accuracy'])

    tensorboard_cbk = keras.callbacks.TensorBoard(log_dir='./tensorboard_logs')

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=10,
                        validation_split=0.2,
                        callbacks=[tensorboard_cbk])

    model.save('./models/path_to_my_model.h5')

    print("训练结束！")

    # The returned "history" object holds a record
    # of the loss values and metric values during training
    # print('\nhistory dict:', history.history)


def evaluate():
    test_data_path = "traj_position_rotate_test.npy"
    dataset_test = DataLoaderTrajs(test_data_path)
    x_test, y_test = dataset_test.get_train_data_without_target()

    model = keras.models.load_model('./models/path_to_my_model.h5',
                                    custom_objects={'basic_loss_function': basic_loss_function})

    # Evaluate the model on the test data using `evaluate`
    print('\n# Evaluate on test data')
    results = model.evaluate(x_test, y_test, batch_size=128)
    print('test loss, test acc:', results)


def predict():
    # 加载模型
    model = keras.models.load_model('./models/path_to_my_model.h5',
                                    custom_objects={'basic_loss_function': basic_loss_function})
    # 生成长度
    num_generate = 50

    # 数据加载
    test_data_path = "traj_position_rotate.npy"
    dataset_test = DataLoaderTrajs(test_data_path)
    # 起始位置
    start_pose = dataset_test.get_batch_test().reshape((1, 1, 7))

    # 生成轨迹
    result_trajectories = []

    for i in range(num_generate):
        if i == 0:
            predictions = start_pose
        prediction = model.predict(predictions)
        # remove the batch dimension

        # predictions = tf.concat([prediction, start_pose[:, :, 7:14]], axis=2)
        predictions = prediction
        prediction = tf.reshape(prediction, (7,))
        result_trajectories.append(prediction)
        print("生成位姿数目：", i)

    save_trajectories(result_trajectories, "result.txt")

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    # print('\n# Generate predictions for 3 samples')
    # predictions = model.predict(x_test[:3])
    # print('predictions shape:', predictions.shape)


if __name__ == '__main__':
    train()
    # evaluate()
    # predict()

