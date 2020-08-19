import tensorflow as tf


class AtariDQNetwork(tf.keras.Model):
    def __init__(self, output_size):
        super(AtariDQNetwork, self).__init__()
        self.output_size = output_size
        self.conv2d_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
        self.conv2d_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
        self.conv2d_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)
        self.dense1 = tf.keras.layers.Dense(units=512, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=self.output_size, activation=tf.keras.activations.linear)

    def call(self, inputs, training=None, mask=None):
        x = self.conv2d_1(inputs)
        x = self.conv2d_2(x)
        x = self.conv2d_3(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.dense1(x)
        return self.dense2(x)
