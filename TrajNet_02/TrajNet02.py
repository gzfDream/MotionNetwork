import tensorflow as tf
import numpy as np


class RNN(tf.keras.Model):
    def __init__(self, num_pose, batch_size, seq_length):
        super().__init__()
        self.num_pose = num_pose
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.cell_1 = tf.keras.layers.LSTMCell(units=256, return_sequences=True)
        self.cell_2 = tf.keras.layers.LSTMCell(units=256, return_sequences=True)
        self.cell_3 = tf.keras.layers.LSTMCell(units=256, return_sequences=True)
        self.dense_1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(units=self.num_pose, activation='relu')

    @tf.function
    def call(self, inputs):
        # inputs = tf.one_hot(inputs, depth=self.num_chars)       # [batch_size, seq_length, num_chars]
        state_1 = self.cell_1.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)
        state_2 = self.cell_2.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)
        state_3 = self.cell_3.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)
        for t in range(self.seq_length):
            if t == 0:
                output_1, state_1 = self.cell_1(inputs[:, t, :], state_1)
                output_2, state_2 = self.cell_2(output_1, state_2)
                output, state_3 = self.cell_2(output_2, state_3)
            # else:
            #     output_1, state_1 = self.cell_1(inputs[:, t, :], state_1)
            #     state_12 = tf.concat([state_1, state_2], axis=1)
            #     output_2, state_2 = self.cell_2(inputs[:, t, :], state_12)
            #     state_23 = tf.concat(state_2, state_3, axis=1)
            #     output_3, state_3 = self.cell_2(inputs[:, t, :], state_23)

        # output = tf.concat([output_1, output_2, output_3], axis=2)
        print(output.shape)
        logits = self.dense_1(output)
        logits = self.dense_2(logits)
        # logits = tf.reshape(logits, (self.batch_size, self.seq_length, self.num_pose))

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
            pose = np.concatenate((self.trajs[index][0], self.trajs[index][2]), axis=1)
            seq.append(pose)
            next_pose.append(self.trajs[index][1][-1])

        return np.array(seq, dtype=np.float32), np.array(next_pose, dtype=np.float32)


def train():
    num_epochs = 1000
    seq_length = 50
    batch_size = 64
    learning_rate = 1e-3

    data_path = "traj_position.npy"
    data_loader = DataLoaderTrajs(data_path)

    model = RNN(num_pose=3, batch_size=batch_size, seq_length=seq_length)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    for batch_index in range(num_epochs):
        X, y = data_loader.get_batch(batch_size)
        with tf.GradientTape() as tape:
            y_pred = model.call(X)
            # loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.keras.losses.mean_squared_error(y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
            print("batch %d: loss %f" % (batch_index, loss.numpy()))
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    tf.saved_model.save(model, "model/train01")

    X_, _ = data_loader.get_batch(1)
    print(X_)
    for t in range(4):
        y_pred = model.call(X_)
        print(y_pred)
        # X_ = np.concatenate([X_[:, 1:], np.expand_dims(y_pred, axis=1)], axis=-1)
    print("\n")


def sample():
    test_path = "traj_position.npy"
    data_loader = DataLoaderTrajs(test_path)

    model = tf.saved_model.load("model/train01")

    X_, _ = data_loader.get_batch(1)
    for t in range(400):
        y_pred = model.call(X_, True)
        print(y_pred)
        X_ = np.concatenate([X_[:, 1:], np.expand_dims(y_pred, axis=1)], axis=-1)
    print("\n")


if __name__ == '__main__':
    train()
    # sample()


