import tensorflow as tf
# from tensorflow.keras.utils import plot_model

import numpy as np
import math


class RNN(tf.keras.Model):
    def __init__(self, num_pose):
        super().__init__()
        self.num_pose = num_pose
        self.cell_1 = tf.keras.layers.LSTMCell(units=128, input_shape=(None, None, num_pose*2))
        self.cell_2 = tf.keras.layers.LSTMCell(units=128, )
        self.cell_3 = tf.keras.layers.LSTMCell(units=128)
        self.dense_1 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(units=self.num_pose, activation='relu')

    # @tf.function
    def call(self, inputs, first_pose, usage_rate, pre_output):
        # inputs 维度：[batch_size, num_pose]
        state_1 = self.cell_1.get_initial_state(batch_size=inputs.shape[0], dtype=tf.float32)
        state_2 = self.cell_2.get_initial_state(batch_size=inputs.shape[0], dtype=tf.float32)
        state_3 = self.cell_3.get_initial_state(batch_size=inputs.shape[0], dtype=tf.float32)

        if first_pose:
            output_1, state_1 = self.cell_1(inputs, state_1)
            inputs_layer2 = tf.concat([inputs, output_1], axis=-1)
            output_2, state_2 = self.cell_2(inputs_layer2, state_2)

            inputs_layer3 = tf.concat([inputs, output_2], axis=-1)
            output, state_3 = self.cell_2(inputs_layer3, state_3)
        else:
            inputs_layer1 = inputs[:, 0:7] * usage_rate + pre_output * (1 - usage_rate)
            inputs_layer1 = tf.concat([inputs_layer1, inputs[:, 7:14]], axis=1)
            output_1, state_1 = self.cell_1(inputs_layer1, state_1)

            inputs_layer2 = tf.concat([inputs_layer1, output_1], axis=-1)
            output_2, state_2 = self.cell_2(inputs_layer2, state_2)

            inputs_layer3 = tf.concat([inputs_layer1, output_2], axis=-1)
            output, state_3 = self.cell_2(inputs_layer3, state_3)

        logit_1 = self.dense_1(output)
        logit = self.dense_2(logit_1)

        return logit

    def predict(self, inputs):
        logits = self(inputs)

        return logits

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
        pose = np.concatenate((self.trajs[index][0][0], self.trajs[index][2][0]), axis=0)
        return pose


def loss_function(real, pred):
    loss_dis = tf.keras.losses.mean_squared_error(y_true=real[:, 0:3], y_pred=pred[:, 0:3])
    loss_dis = tf.reduce_mean(loss_dis)

    # 单位化四元数
    sqrt = tf.tile(tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(pred[:, 3:7]), 1)), -1), multiples=[1, 4])
    sqrt_ = tf.where(sqrt > 0.00001, sqrt, 0.00001)
    norm_rot = pred[:, 3:7] / sqrt_

    # 均方误差
    # loss_rotate = tf.keras.losses.mean_squared_error(y_true=real[:, 3:7], y_pred=norm_rot)
    # loss_rotate = tf.reduce_mean(loss_rotate)

    multi = real[:, 3:7] * norm_rot
    reduce_sum = tf.abs(tf.reduce_sum(multi, 1))
    loss_rotate = 2 * tf.reduce_mean(tf.acos(reduce_sum))

    return loss_dis + loss_rotate


def train():
    num_epochs = 100
    seq_length = 50
    batch_size = 64
    learning_rate = 1e-4
    num_pose = 7
    usage_rate = 1
    decay_rate = 0.9
    decay_steps = 100

    data_path = "traj_position_rotate.npy"
    data_loader = DataLoaderTrajs(data_path)

    summary_writer = tf.summary.create_file_writer('./tensorboard')  # 参数为记录文件所保存的目录

    model = RNN(num_pose=num_pose)
    # model.build(input_shape=[None, seq_length, num_pose])
    # plot_model(model, to_file='model.png', show_shapes=True)
    # model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    for batch_index in range(num_epochs):
        X, y = data_loader.get_batch(batch_size)
        loss = 0.
        usage_rate_epoch = usage_rate * tf.pow(decay_rate, (batch_index / decay_steps))
        with tf.GradientTape(persistent=True) as tape:
            y_pred = np.array([])
            for pose_index in range(seq_length):
                if pose_index == 0:
                    y_pred = model.call(X[:, pose_index, :], True,  usage_rate_epoch, y_pred)
                else:
                    y_pred = model.call(X[:, pose_index, :], False, usage_rate_epoch, y_pred)

                loss_pose = loss_function(real=y[:, pose_index, :], pred=y_pred)
                loss += loss_pose

            loss = loss / seq_length
            print("batch %d: loss %f" % (batch_index, loss.numpy()))
            grads = tape.gradient(loss, model.variables)
            # grads, global_norm = tf.clip_by_global_norm(grads, 5)
            # 限定导数值域-1到1
            capped_gradients = [tf.clip_by_value(grad, -1., 1.) for grad in grads if grad is not None]
            optimizer.apply_gradients(grads_and_vars=zip(capped_gradients, model.variables))

            with summary_writer.as_default():  # 希望使用的记录器
                tf.summary.scalar("loss", loss, step=batch_index)

    # tf.saved_model.save(model, "model/train01")
    model.save_weights("model/train01", save_format='tf')

def sample():
    test_path = "traj_position_rotate.npy"
    data_loader = DataLoaderTrajs(test_path)

    model = RNN(num_pose=7)

    model.compile(loss=loss_function,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))

    model.load_weights04n38("model/train01")

    X_ = data_loader.get_batch_test()
    for t in range(4):
        if t == 0:
            X = X_
        y_pred = model.call(X, False, [], 1)
        print(y_pred)
        X = np.concatenate([y_pred, X_[:, 7:14]], axis=-1)
    print("\n")


if __name__ == '__main__':
    train()
    # sample()


