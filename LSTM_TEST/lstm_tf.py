import time
import os
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()


# 训练测试数据生成
class DataGenerator(object):
    """
    Wrapper class around the new Tensorflows dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(
            self,
            trajs_file,
            mode='training',
            batch_size=64,
            shuffle=True,
            buffer_size=10000):
        """
        接收轨迹的npy文件，使用tf.data.Dataset处理数据

        :param trajs_file: trajs.npy
        :param mode: Either 'training' or 'validation'. Depending on this value,
                    different parsing functions will be used.
        :param batch_size: Number of data per batch.
        :param shuffle: Wether or not to shuffle the data in the dataset
                    and the initial file list.
        :param buffer_size: Number of data used as buffer for TensorFlows
                    shuffling of the dataset.

        Raises：
            ValueError: If an invalid mode is passed.
        """
        self.trajs_file = trajs_file

        # retrieve the data from the text file
        self._read_npy_file()

        # create dataset
        data = self._dataset_generator()

        # distinguish between train/infer. when calling the parsing functions
        if mode == 'training':
            data = data.map(self._parse_function_train)

            # create a new dataset with batches of trajs
            data = data.batch(batch_size, drop_remainder=True)

        elif mode == 'sample':
            data = data.map(self._parse_function_sample)
            # create a new dataset with batches of images
            data = data.batch(1, drop_remainder=True)
        else:
            raise ValueError("Invalid mode '%s'." % (mode))

        # shuffle the first `buffer_size` elements of the dataset
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)

        self.data = data
        # self._show_data()

    def _read_npy_file(self):
        """Read the content of the npy file and store it into lists."""
        self.trajs_npy = np.load(self.trajs_file)

        self.data_size = np.shape(self.trajs_npy)[0]
        print(self.data_size)

    def _dataset_generator(self):
        data = tf.data.Dataset.from_tensor_slices(self.trajs_npy)

        return data

    def _parse_function_train(self, traj):
        """Input parser for samples of the training set."""
        # load and preprocess the json
        traj_s = tf.cast(traj[0], dtype=tf.float32)
        traj_t = tf.cast(traj[1], dtype=tf.float32)
        return traj_s, traj_t

    def _parse_function_sample(self, traj):
        """Input parser for samples of the test set."""
        # load and preprocess the json
        traj_s = traj[0]
        target = traj[1]
        return traj_s, target

    # 显示数据内容
    def _show_data(self):
        iterator = self.data.make_one_shot_iterator()
        one_element = iterator.get_next()
        with tf.Session() as sess:
            for i in range(1):
                x, y = sess.run(one_element)
                print(x)
                print(y)


class Model(tf.keras.Model):
    def __init__(self, units, pose_size):
        super(Model, self).__init__()
        self.units = units

        if tf.test.is_gpu_available():
            self.gru = tf.keras.layers.CuDNNGRU(
                self.units,
                return_sequences=True,
                recurrent_initializer='glorot_uniform',
                stateful=True,
                dtype='float32')
        else:
            self.gru = tf.keras.layers.GRU(
                self.units,
                return_sequences=True,
                recurrent_activation='sigmoid',
                recurrent_initializer='glorot_uniform',
                stateful=True,
                dtype='float32')

        self.fc = tf.keras.layers.Dense(pose_size)

    def call(self, x):

        # output at every time step
        # output shape == (batch_size, seq_length, hidden_size)
        output = self.gru(x)

        # The dense layer will output predictions for every time_steps(seq_length)
        # output shape after the dense layer == (seq_length * batch_size,
        # vocab_size)
        prediction = self.fc(output)

        # states will be used to pass at every step to the model while training
        return prediction


"""
    我们将使用采用默认参数的 Adam 优化器，并用 softmax 交叉熵作为损失函数。
    此损失函数很重要，因为我们要训练模型预测下一个字符，而字符数是一种离散数据（类似于分类问题）。
"""
# Using adam optimizer with default arguments
optimizer = tf.train.AdamOptimizer()


# Using sparse_softmax_cross_entropy so that we don't have to create
# one-hot vectors
def loss_function(real, preds):
    # return tf.losses.sparse_softmax_cross_entropy(labels=real, logits=preds)
    # 位置误差
    # distance_loss = tf.losses.mean_squared_error(labels=real[:, :, 0:3], predictions=preds[:, :, 0:3])

    # 角度误差
    # multi = real[:, :, 3:7] * preds[:, :, 3:7]
    # red_sum = tf.abs(tf.reduce_sum(multi, 2))
    # rotation_loss = tf.reduce_mean(tf.acos(red_sum))
    return tf.losses.mean_squared_error(labels=real, predictions=preds)  # distance_loss + rotation_loss


# Number of RNN units
units = 1024
# size of pose
pose_size = 7

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'


def train():
    BATCH_SIZE = 64
    seq_length = 50

    model = Model(units, pose_size)

    model.build(tf.TensorShape([BATCH_SIZE, seq_length, pose_size]))
    model.summary()

    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    # 读取数据
    train_data = DataGenerator(
        "../raw_data/trajs_2.npy",
        mode="training",
        batch_size=BATCH_SIZE,
        shuffle=True)

    train_batches_per_epoch = int(
        np.floor(
            train_data.data_size /
            BATCH_SIZE))
    print('number of dataset: ', train_data.data_size)
    print('number of batch: ', train_batches_per_epoch)

    # 训练
    EPOCHS = 50

    # Training loop
    for epoch in range(EPOCHS):
        start = time.time()

        # initializing the hidden state at the start of every epoch
        # initally hidden is None
        hidden = model.reset_states()

        iterator = train_data.data.make_one_shot_iterator()

        for batch in range(train_batches_per_epoch):
            inp, target = iterator.get_next()
            with tf.GradientTape() as tape:
                # feeding the hidden state back into the model
                # This is the interesting step
                predictions = model(inp)
                loss = loss_function(target, predictions)

            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(zip(grads, model.variables))

            if batch % 20 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             loss))
        # saving (checkpoint) the model every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.save_weights(checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1, loss))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


def sample():
    # 加载模型
    model = Model(units, pose_size)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, 1, pose_size]))

    # Evaluation step (generating text using the learned model)
    # Number of characters to generate
    num_generate = 100

    # the start position to experiment
    start_position = [0.166521, 0.914289, 0.098687, 0.155130, -0.169848, -0.590939, 0.773225]
    start_position = tf.expand_dims([start_position], 0)

    # Empty string to store our results
    trajectories_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Evaluation loop.

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(start_position)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a multinomial distribution to predict the word returned by the model
        predictions = predictions / temperature
        # predicted_id = tf.multinomial(predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        start_position = tf.expand_dims(predictions, 0)
        trajectories_generated.append(predictions.numpy()[0])

    result_list = []
    for pose in trajectories_generated:
        l = []
        for i in pose:
            l.append(i)
        result_list.append(l)
    np.savetxt("results.txt", result_list, fmt='%f', delimiter=' ')


# train()
sample()
